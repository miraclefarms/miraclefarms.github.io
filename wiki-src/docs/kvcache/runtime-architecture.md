# 运行时架构

## KVCache 在推理引擎中的位置

一个典型的 LLM 推理引擎分为以下几层：

```
┌─────────────────────────────────────┐
│             API Server              │  ← 接收请求，管理 session
├─────────────────────────────────────┤
│             Scheduler               │  ← 决定哪些请求进入当前 batch
├─────────────────────────────────────┤
│          Block Manager              │  ← 管理 KVCache 的物理显存块
├─────────────────────────────────────┤
│         Model Executor              │  ← 执行 Prefill/Decode kernel
├─────────────────────────────────────┤
│      GPU HBM (KVCache Storage)      │  ← 物理存储 K/V 张量
└─────────────────────────────────────┘
```

KVCache 横跨多个层次：它的**分配**由 Block Manager 负责，**调度决策**（哪些请求能用多少 KV 空间）由 Scheduler 决定，**实际读写**发生在 Model Executor 内的 Attention kernel 中。

## 核心概念

### Sequence（序列）

一个请求对应一个 Sequence（或多个，如 beam search）。Sequence 的生命周期：
1. **Waiting**：请求到达，尚未进入 GPU
2. **Prefill**：Prompt 处理中，KVCache 正在写入
3. **Decoding**：逐 token 生成，KVCache 持续追加
4. **Finished**：生成结束或被中止，KVCache 可以释放

### Block / Page

Block（也称 Page）是 KVCache 的**分配单位**，而不是单个 token。一个 Block 存储固定数量（`block_size`）的 token 的 KV 数据。

Block 的意义：以 Block 为单位分配，而不是按精确 token 数，避免了每步解码都需要重新分配显存的开销，也为 Prefix Cache 的粒度提供了基础。

```
典型 block_size 选择：
- vLLM 默认：16 tokens/block
- SGLang 默认：16 tokens/block（部分场景可配置更大）
```

### Slot

Slot 是 Block 内的最小分配单位，对应一个 token 的 K/V 存储位置。`slot_id = block_id * block_size + token_offset_in_block`。

### Block Table

每个 Sequence 维护一张 **Block Table**，记录它的每个逻辑 Block 映射到哪个物理 Block。Block Table 由 Block Manager 管理，在 kernel 执行时作为索引传入 GPU。

```
逻辑 Block 序号:   0     1     2     3
物理 Block 地址:  [47]  [23]  [81]  [56]
```

这一间接层是实现 Paged KV 和 Prefix Cache 的关键机制。

## Block Manager

Block Manager 是推理引擎的**显存分配器**，负责：

1. **初始化**：将所有可用 HBM 按 block_size 划分为固定大小的物理 Block 池
2. **分配**：当 Sequence 需要新 Block 时，从空闲池中取出物理 Block，更新 Block Table
3. **释放**：Sequence 完成时，将其使用的物理 Block 归还空闲池
4. **复制（Copy-on-Write）**：用于 beam search 或 Prefix Cache 场景下的 Block 共享与分叉
5. **Eviction**：当显存不足时，决定驱逐哪些 Block（通常是优先驱逐最长时间未使用的）

### 空闲池状态

```
total_blocks = gpu_kvcache_memory / (block_size × per_token_kv_bytes)
free_blocks = total_blocks - allocated_blocks
```

Scheduler 在构建每个 batch 前会查询 Block Manager 的空闲状态，确保 batch 内所有请求的 KVCache 需求可以被满足。

## Scheduler 与 KVCache 的交互

Scheduler 决定哪些请求可以进入当前 batch，KVCache 可用量是关键约束之一。

典型调度流程：

1. 估算当前等待队列中每个请求的 KVCache 需求（已有 token 数 + 预期生成长度）
2. 根据 Block Manager 报告的空闲 Block 数，选择可以容纳的请求集合
3. 如果 GPU KVCache 耗尽，触发**抢占（Preemption）**：
   - **Swap**：将部分 Sequence 的 KVCache 换出到 CPU DRAM
   - **Recompute**：丢弃部分 Sequence 的 KVCache，之后重新 Prefill

抢占策略直接影响系统在高负载下的延迟尾部和吞吐。

## 连续内存 vs 分页管理

### 连续内存分配（早期方案）

早期推理系统对每个 Sequence 预分配连续显存（按最大序列长度），存在严重问题：

- **内部碎片**：预分配长度 > 实际生成长度，剩余空间浪费
- **外部碎片**：短序列释放后产生碎片，无法被较长序列利用
- **低并发**：每个 Sequence 持有大块连续显存，批量大小严重受限

### 分页管理（PagedAttention）

将 KVCache 划分为固定大小的 Block（类比 OS 分页），Sequence 按需申请 Block，Block 可以不连续。

- **无外部碎片**：物理 Block 池统一管理，按块复用
- **内部碎片最小化**：最后一个 Block 可能有未使用的 Slot，但总碎片 ≤ `(block_size - 1)` tokens/sequence
- **高并发**：每个 Sequence 按实际使用量占用空间，同等显存可服务更多并发请求

详见 [Paged KV](paged-kv.md) 章节。

## Fragmentation 与 Reuse

即使在分页管理下，仍然存在一些次要问题：

- **内部碎片**：每个 Sequence 最后一个 Block 可能未填满
- **Block 复用**：Prefix Cache 允许多个 Sequence 共享同一物理 Block（只读场景），需要引用计数管理
- **Eviction 策略**：当显存不足时，被驱逐的 Block 如果之后又需要，必须重算（recompute）或从 CPU 换入（swap）

这些问题是推理引擎 Block Manager 工程设计的核心复杂性来源。

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 框架搭建 |
