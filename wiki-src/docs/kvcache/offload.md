# KV Offload

## 为什么需要 Offload

GPU HBM 是推理系统中最稀缺的资源之一。对于长上下文请求（如 32K、128K token），单个序列的 KVCache 可能达到几十 GB，轻易超出单卡显存上限。即便是常规请求，当并发量足够高时，KVCache 也会耗尽 HBM。

KV Offload 的思路：将部分 KVCache 从 GPU HBM 移出，存储到更便宜（但更慢）的存储介质上，在需要时换回 GPU。

## 存储层次

```
访问速度（快→慢）              存储层次                 典型带宽
─────────────────────────────────────────────────────────────
GPU HBM (e.g. H100 80GB)         最快            3.35 TB/s (HBM3e)
CPU DRAM (e.g. 512GB~2TB)         快             ~100-200 GB/s (DDR5)
NVMe SSD (e.g. 8TB)               中             ~10-20 GB/s (PCIe 5.0 x4)
远端内存池 (RDMA KV Store)        慢             ~100-400 GB/s (InfiniBand/NVLink)
```

实践中最常用的两级方案是 **HBM → CPU DRAM**，通过 PCIe 连接传输，带宽约 20-64 GB/s（PCIe 4.0/5.0 x16）。

## Offload 的基本机制

### Swap-out（换出）

当 GPU KVCache 空间不足时：

1. Scheduler 选择低优先级或长时间未使用的 Sequence
2. 将其部分或全部 KV Block 通过 PCIe 传输到 CPU DRAM
3. 释放 GPU 上对应的物理 Block，供其他 Sequence 使用
4. 标记该 Sequence 为"Swapped"状态，暂停其 Decode

### Swap-in（换入）

当 Swapped Sequence 需要继续生成时：

1. Scheduler 确认 GPU 有足够空闲 Block
2. 将 CPU DRAM 中的 KV Block 传回 GPU HBM
3. 恢复 Sequence 的 Decode

### Recompute（重算，替代方案）

不存储 KV，而是在需要时对 Sequence 重新 Prefill，重新生成 KV。

- **优点**：不占用 CPU DRAM，实现简单
- **缺点**：计算成本高，延迟大，不适合长序列或频繁抢占场景

## Offload 的收益与代价

Offload 的核心权衡：**带宽成本 vs 重算成本**

### 收益

- 支持更大的并发批量（batch size），提高 GPU 计算利用率
- 支持更长的上下文（在 HBM 不足时将历史 KV 换出）
- 避免抢占后的完整 Recompute 延迟

### 代价

- **PCIe 带宽是瓶颈**：PCIe 4.0 x16 理论带宽约 32 GB/s（双向各 16 GB/s），远低于 HBM 的 TB/s 级带宽。大量 KV 换入换出会成为系统瓶颈。
- **换入延迟**：当 Sequence 需要继续 Decode 时，必须等待 KV 从 CPU 换入完成，增加尾部延迟。
- **CPU DRAM 容量上限**：典型服务器 CPU DRAM 512GB~2TB，但同样有上限，且会与其他进程竞争。
- **调度复杂性**：Scheduler 需要维护 Swapped 状态，决定换出哪些 Sequence、何时换入。

## 何时 Offload 有价值

Offload 并不总是合算。以下场景收益明显：

- **长序列、低请求率**：单个序列的 KVCache 大，但请求并发不高，偶尔需要换出历史轮次
- **多轮对话 with idle time**：用户思考时间较长，Sequence 可以暂时换出 GPU，空出资源给新请求
- **批量推理**：对延迟不敏感，愿意接受换入延迟换取更高的 GPU 利用率

以下场景 Offload 效果差或得不偿失：

- **PCIe 带宽已成瓶颈**：换入换出频繁，PCIe 成为系统限速点
- **短序列、高并发**：KV 小，换出收益有限，但调度开销不减
- **重算成本低**：序列较短，重新 Prefill 比换入换出更快
- **延迟敏感场景**：换入 latency 会显著增加尾部延迟，破坏 SLA

## 多级 Offload：HBM → DRAM → SSD

在极端长上下文场景（如数百万 token 的文档处理），研究探索了三级甚至四级存储层次：

```
HBM → CPU DRAM → NVMe SSD
```

- SSD 的读写带宽（10-20 GB/s）远低于 DRAM，但容量可以达到几 TB
- 在 IO-bound 场景（如批量文档索引）中有一定意义
- 在延迟敏感场景几乎不可行，除非有大量预取时间

## 远端 KVCache（Remote KV Cache）

另一个方向是跨节点的远端 KVCache 池，例如通过 RDMA 将 KV 存储在专用的内存节点上。

- 优点：存储容量可弹性扩展，支持多推理节点共享 Prefix Cache（跨实例命中）
- 挑战：网络延迟（即便 InfiniBand 也比 PCIe 更慢）、一致性管理、复杂的调度协议

LMCache 等项目在探索这个方向。详见 [框架对比](frameworks.md)。

## CXL 作为 KV Offload 通道

CXL（Compute Express Link）是近年兴起的新型互联技术，对 KV Offload 的工程重心产生了根本性影响。

### CXL 在存储层次中的位置

CXL 连接的内存池处于 DDR 和 NVMe SSD 之间，作为 **warm KV tier**：

| 介质 | 典型容量 | 典型带宽 | 延迟 | KV 角色 |
|------|----------|----------|------|---------|
| GPU HBM | 80–192 GB/GPU | 3.35 TB/s | ~1 μs | Hot KV（正在 decode） |
| CPU DDR | 512 GB–2 TB | 100–200 GB/s | ~100 ns (local) | Warm KV（近期可能复用） |
| **CXL 内存池** | **2–8 TB** | **~1 TB/s（aggregate）** | **~2 μs（CXL-RPC）** | **Warm KV（跨节点复用）** |
| NVMe SSD | 4–16 TB | 10–20 GB/s | ~10 μs | Cold KV（存档） |
| RDMA 远端 | 弹性 | 100–400 GB/s | ~8 μs（RDMA-RC） | Remote KV（跨机传输） |

### Beluga：CXL 内存池化 KV Cache

Beluga（arxiv 2511.20172v2）使用 CXL 2.0 switch 连接的最大 8TB 内存池（1TB/s 聚合带宽），服务 16 台服务器：

- **Cache-hit 场景**：TTFT 1.36s vs RDMA（Mooncake）13.00s，QPS 11.32 vs 1.54 → **7.35× 提升**
- **关键优势**：load/store 语义可以处理细粒度的 KV cache scatter/gather（Qwen-32B GQA 下每个 16-token Block 含 128 个不连续片段），RDMA 控制开销无法高效处理这种碎片化访问
- **CXL-RPC**：往返 2.11μs vs RDMA-RC 8.39μs

来源：主站 reading [Beluga：CXL 内存池为什么会改变 KV Cache Offload 的工程重心](/notes/2026/05/13/beluga-cxl-kvcache-memory-pool/)

### CXL 的五层分类

CXL 在 KVCache 生态中的五个应用层次：

| 层次 | 场景 | 代表工作 |
|------|------|----------|
| L1: 扩容 | 单机 GPU HBM 不足时 CXL 扩展 | Dynamo KVBM、Penguin MemoryAI |
| L2: Warm Tier | HBM→CXL 作为多级存储的一层 | Beluga、Predictive Multi-Tier |
| L3: Prefix Cache | CXL 池作为跨实例共享的 Prefix Cache | TraCT |
| L4: PD Transfer | PD 分离中通过 CXL 传输 KV | TraCT（TTFT 最高 9.8×, P99 最高 6.2×） |
| L5: Cache Server | CXL 池作为独立 KVCache 服务 | CXL-SpecKV |

来源：主站 essay [CXL + KVCache 现状调研报告](/notes/2026/05/13/cxl-kvcache-survey/)

### CXL 的限制

- **延迟不适合 GPU 直接 Attention 读取**：CXL 的 ~2μs 延迟远高于 HBM 的 ~1μs，不适合在线 attention kernel 直接访问
- **适用场景是预取而非在线**：需要在 decode 之前提前将 KV 从 CXL 预取到 HBM
- **生态尚在早期**：CXL 2.0 switch（XConn XC50256）和软件栈（Dynamo KVBM）都在快速迭代中

一个简化的 Offload 收益模型：

```
节省的计算时间 = hit_rate × recompute_time_per_token × offloaded_tokens
换入带宽成本 = offloaded_kv_bytes / pcie_bandwidth

当 节省的计算时间 > 换入带宽成本 时，Offload 有收益
```

实际系统中还需要考虑 Decode 等待时间、批次填充效率、DRAM 占用等因素，判断是否 Offload 是一个多变量的在线决策问题。

## 关联章节

- CXL 在存储层次中的位置：[存储层级](storage-hierarchy.md)
- Beluga 与 RDMA 方案的对比：[弹性与故障](elasticity.md)
- PD 分离中的 KV 传输：[PD 分离](pd-disaggregation.md)
- Agent 场景的 offload 决策：[工作负载维度](workloads.md)

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 框架搭建 |
| v0.2 | 2026-05-14 | 新增 CXL 作为 KV Offload 通道（Beluga 实测数据、五层分类、限制）；补充多级 offload 生态进展
