# 存储层级

> **系统维度（纵向）之一**：KV 在哪里存、各级介质的容量/带宽/延迟/成本量级。

KVCache 不是一个"放在显存里"的同质对象，而是横跨多级存储的分布式状态。理解层级结构是理解 [Offload](offload.md)、[PD 分离](pd-disaggregation.md)、跨实例 Prefix Cache 等所有"搬运型"问题的前提。

## 1. 四级层次

```
L1 — GPU HBM           最快 / 最贵 / 最稀缺
L2 — CPU DRAM          快 / 容量中等 / 经 PCIe 连接
L3 — 本地 NVMe / SSD   中速 / 大容量 / 持久化
L4 — 远端 / 分布式 KV 池  慢 / 极大容量 / 跨节点共享
```

### L1 — GPU HBM

最快的层，决定**在线请求的并发上限**。

- 典型容量：H100 80GB / H200 141GB / B200 192GB
- 典型带宽：HBM3 ≈ 3.35 TB/s，HBM3e ≈ 4.8 TB/s
- 与权重共享：模型权重占用一部分，剩余才是 KV 可用区
- 典型 KV 可用：8B 模型在 H100 上 ~50GB；70B 模型 ~10–20GB

L1 的特征是**算力侧"零延迟"访问**——attention kernel 直接读写。所有其他层都通过"搬到 L1 来"才能参与计算。

### L2 — CPU DRAM

第一站 offload 目标，[KV Offload](offload.md) 的最常见落点。

- 典型容量：单机 512GB ~ 2TB（DDR5 服务器配置）
- 带宽限制：实际吞吐受 **PCIe** 限制
    - PCIe 4.0 x16：单向 ~32 GB/s，双向 ~64 GB/s
    - PCIe 5.0 x16：单向 ~64 GB/s，双向 ~128 GB/s
- 延迟：μs 级（PCIe 传输 + 内存访问）

注意：DRAM 自身带宽（DDR5 ~100-200 GB/s）远高于 PCIe，所以 GPU 与 CPU 之间的 KV 搬运瓶颈**几乎总是 PCIe**。

### L3 — 本地 NVMe / SSD

单机大容量持久化层，主要用于：

- 极长上下文场景（百万级 token）的部分 KV 落盘
- 离线批量推理的 KV 复用（跨任务、跨次运行）
- 故障恢复的 checkpoint

典型参数：

- 容量：单盘 4–16 TB，RAID 后可达数十 TB
- 带宽：PCIe 5.0 NVMe ~14 GB/s，单机多盘聚合可上 50+ GB/s
- 延迟：~100 μs 量级

**GPUDirect Storage（GDS）** 让 GPU 可以零拷贝直接读 NVMe，绕过 CPU DRAM；在批量推理场景有意义。

### L4 — 远端存储 / 分布式 KV 池

跨节点共享的"KV 数据湖"，典型形态：

- **专用 KV 池**：Mooncake Store、LMCache 远端层
- **对象存储**：S3 / Ceph 类（成本最低，延迟最差）
- **远端内存池**：通过 RDMA/InfiniBand 直接访问其他节点的 DRAM

带宽参考：

- InfiniBand HDR：200 Gbps ≈ 25 GB/s
- InfiniBand NDR：400 Gbps ≈ 50 GB/s
- NVLink-C2C / NVSwitch（同一机柜内）：900 GB/s+

L4 是支持**跨实例 Prefix Cache**和**PD 分离 KV Transfer**的物理基础。

## 2. 量级对比速查表

| 层级 | 典型容量 | 带宽（顺序读） | 延迟 | 成本量级 |
|------|---------|----------------|------|---------|
| L1 HBM | 几十~百多 GB | 3–5 TB/s | ns | 极高 |
| L2 DRAM | TB 量级 | PCIe 限制 ~32–128 GB/s | μs | 中 |
| L3 NVMe | 10+ TB | 10–20 GB/s/盘 | ~100 μs | 低 |
| L4 远端 DRAM (RDMA) | 集群 TB+ | 25–50 GB/s/链路 | μs–ms | 中（看带宽配额） |
| L4 远端对象存储 | PB | 网络限制 | 10+ ms | 极低 |

> 数值会随硬件代际变化（H100 → H200 → B200，PCIe 4 → 5 → 6，IB HDR → NDR → XDR），具体部署时以实际链路实测为准。

## 3. 介质特性对设计的影响

不同层级的"性格"决定了它适合承担的角色：

| 角色 | 适合的层级 | 原因 |
|------|-----------|------|
| 在线请求的活跃 KV | L1 | 只有 HBM 能跟上 attention kernel |
| 暂时换出的会话 KV | L2 | 容量足够、PCIe 带宽尚可 |
| 长上下文的"冷"历史 | L3 | 容量大、不在乎 ms 级延迟 |
| 跨实例共享 prefix | L4 | 跨节点访问的唯一选项 |
| 离线批量预计算 KV | L3 / L4 | 持久化 + 高吞吐 |

## 4. 不要忽略的隐藏成本

在做层级搬运决策时，常被忽略但影响显著的因素：

- **PCIe 在多 GPU 间共享**：8 卡服务器总 PCIe 带宽要分配给 8 个 GPU，单卡可用带宽常远低于理论值
- **NUMA 不亲和**：CPU DRAM 跨 socket 访问带宽腰斩
- **小块传输的 startup latency**：每次 RDMA / PCIe 发起都有固定开销，KV 太碎反而吃亏
- **量化对带宽的实质性放大**：FP8 KV 在 PCIe 上等效带宽是 BF16 的 2×

## 5. 通道与协议

不同层级间的**搬运通道**：

| 通道 | 典型场景 |
|------|---------|
| PCIe | L1 ↔ L2，L1 ↔ L3（GDS） |
| NVLink / NVSwitch | 同机柜内 GPU ↔ GPU |
| RDMA over InfiniBand | 跨节点 L4 |
| RDMA over Ethernet (RoCE) | 数据中心通用网络 |
| TCP/IP | 兼容性场景，性能最差 |
| **CXL（前瞻）** | 内存语义跨设备共享，可能重塑 L2/L3 |

## 6. CXL 存储层（前瞻）

CXL（Compute Express Link）内存池正在成为介于 L2 DRAM 和 L4 远端 RDMA 池之间的新层级：

| 层级 | 典型容量 | 带宽 | 延迟 | 适合角色 |
|------|----------|------|------|----------|
| **CXL 内存池** | **2–8 TB**（CXL 2.0 switch） | **~1 TB/s（aggregate）** | **~2 μs（CXL-RPC）** | Warm KV tier，跨节点 prefix cache |

**Beluga 实测对比**（CXL 2.0 switch, 8TB 池, 16 服务器）：
- Cache-hit TTFT：1.36s（CXL） vs 13.00s（RDMA/Mooncake），7.35× 提升
- QPS：11.32 vs 1.54
- CXL-RPC 往返 2.11μs vs RDMA-RC 8.39μs

**关键差异：** CXL 的 load/store 语义天然适合 KV cache 的碎片化访问——Qwen-32B GQA 下每个 16-token Block 含 128 个不连续片段，RDMA 需多次控制消息，CXL 直接内存读写。

来源：主站 reading [Beluga](/notes/2026/05/13/beluga-cxl-kvcache-memory-pool/)，主站 essay [CXL + KVCache 调研](/notes/2026/05/13/cxl-kvcache-survey/)

## 关联章节

- 跨层级搬运的具体机制：[KV Offload](offload.md)
- CXL offload 详解及五层分类：[KV Offload](offload.md) §CXL
- 跨节点 KV 传输的部署侧讨论：[PD 分离](pd-disaggregation.md)
- L4 远端 KV 池的产品化前瞻：[未来方向](future.md)

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 框架搭建 |
| v0.2 | 2026-05-14 | 新增 CXL 存储层（前瞻），含 Beluga 实测数据与 RDMA 对比；更新关联章节
