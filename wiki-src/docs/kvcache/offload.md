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

## 带宽与命中率的权衡模型

一个简化的 Offload 收益模型：

```
节省的计算时间 = hit_rate × recompute_time_per_token × offloaded_tokens
换入带宽成本 = offloaded_kv_bytes / pcie_bandwidth

当 节省的计算时间 > 换入带宽成本 时，Offload 有收益
```

实际系统中还需要考虑 Decode 等待时间、批次填充效率、DRAM 占用等因素，判断是否 Offload 是一个多变量的在线决策问题。
