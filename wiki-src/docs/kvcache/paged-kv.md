# Paged KV

## 问题背景：连续显存分配的困境

在 PagedAttention 出现之前，主流推理系统为每个 Sequence 预分配**连续的显存块**，大小等于最大序列长度对应的 KVCache。这带来了严重的显存浪费：

- **内部碎片（Internal Fragmentation）**：实际生成长度通常远小于预分配的最大长度，多余的显存无法被其他请求使用
- **外部碎片（External Fragmentation）**：连续显存的分配和释放产生不规则的空洞，无法被需要连续块的新请求填入
- **过度保守**：为避免显存不足，系统必须提前按最差情况估算显存需求，进一步压低了实际并发数

实测数据表明，在这种方案下，真正用于 KVCache 存储的显存利用率可能低至 20-40%，其余都是各种形式的浪费。

## PagedAttention 的直觉

PagedAttention 的思想直接借鉴自**操作系统虚拟内存分页（Virtual Memory Paging）**：

| OS 分页 | PagedAttention |
|---------|----------------|
| 物理内存 → 固定大小的页帧 | HBM → 固定大小的 KV Block |
| 虚拟地址 → 页表 → 物理地址 | 逻辑 Block 序号 → Block Table → 物理 Block 地址 |
| 进程按需申请页帧 | Sequence 按需申请 KV Block |
| 不同进程共享只读页帧（COW） | 不同 Sequence 共享前缀 Block（Prefix Cache） |

核心区别：OS 的分页针对 CPU 缓存层次和 DRAM 访问，而 PagedAttention 针对的是 GPU HBM 上的大块显存分配问题，其"页面调度"单位（block_size）远大于 OS 页（OS 页通常 4KB，KV Block 可能几十 KB 到几 MB）。

## 实现原理

### 初始化阶段

引擎启动时，Block Manager 将 GPU 可用于 KVCache 的 HBM（扣除模型权重后剩余）划分为 $N$ 个大小相等的物理 Block：

```
每个 Block 存储:
  shape = [block_size, num_kv_heads, head_dim]  × 2（K 和 V）× num_layers
  bytes = block_size × num_kv_heads × head_dim × 2 × num_layers × dtype_bytes
```

这些物理 Block 的地址被记录在**空闲 Block 池**中。

### 运行阶段

每个 Sequence 维护一张 Block Table：

```
Block Table:  [ phy_blk_0, phy_blk_1, phy_blk_2, ... ]
逻辑 Block:       0           1           2
```

- Prefill 时：按顺序申请物理 Block，填入对应 K/V
- 每完成一个 Block（block_size tokens）后，如果开启 Prefix Cache，对该 Block 进行哈希存储
- Decode 时：每新生成一个 token，写入当前活跃 Block 的下一个 Slot；当 Block 满时申请新 Block

### Attention Kernel 的适配

传统 Attention kernel 假设 K/V 存储在连续内存中。PagedAttention 需要改写 kernel：

- 输入额外的 Block Table 和 block_size 参数
- Kernel 内部根据 Block Table 做间接寻址，逐 Block 读取 K/V 进行 Attention 计算
- 间接寻址带来一定的访存效率损失，但实际测试中往往被更高并发带来的吞吐提升所覆盖

## 收益分析

### 显存利用率提升

分页后，内部碎片上界为每个 Sequence `block_size - 1` 个 token 的 KV，通常可以忽略不计。外部碎片归零（Block 池统一分配，无碎片）。

在高并发场景下，显存利用率通常可以从连续分配的 20-40% 提升至 90%+。

### 并发数提升

同等显存下，分页管理可以服务更多并发请求：

- 短序列不再占据为长序列准备的连续大块
- 空闲 Block 可立即复用，无需等待连续空间出现

### 支持 Prefix Cache

Block 粒度的管理天然支持多 Sequence 共享物理 Block（只读引用），是实现 Prefix Cache 的基础。详见 [Prefix Cache](prefix-cache.md)。

## 局限性

- **Block Table 间接层**：额外的显存访问（虽然 Block Table 通常很小）
- **Attention kernel 复杂性**：需要为分页访问模式定制 kernel，与 FlashAttention 等标准实现的集成需要额外工作
- **block_size 选择**：过大 → 内部碎片增加；过小 → Block Table 变大，间接层开销增加
- **KV 量化与 block_size 的交互**：部分量化方案（如 KV FP8 with scaling per block）要求 block_size 是特定值的倍数

## 与 FlashAttention 的关系

FlashAttention 本身不直接处理分页 KV 存储——它优化的是 Attention 计算的 SRAM 使用效率（减少 HBM 读写次数）。PagedAttention 则处理 HBM 上的 KV 存储布局。

两者可以结合：FlashAttention 的 block-wise 计算思路与 PagedAttention 的 block-wise 存储可以协同，但需要专门的 kernel 实现（如 vLLM 的 FlashInfer 集成、SGLang 的 triton kernel 等）。
