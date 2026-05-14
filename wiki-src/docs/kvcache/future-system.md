# 未来方向 — 系统侧

> 系统侧的演化方向是：把 KVCache 从推理引擎的内部组件，变成独立的、可寻址的基础设施层。

## 1. CXL 与近存计算对存储层级的重塑

**CXL（Compute Express Link）** 是目前最受关注的存储层级变化：

- CXL 让 CPU、GPU、加速器以"内存语义"共享同一个更大的内存池
- 访问延迟比 PCIe 低（目标 < 100ns），比 HBM 高（~10ns）
- 容量可以做到 TB 级，远超单张 GPU 的 HBM

**对 KVCache 的影响：**

- L2/L3 的边界模糊：CPU DRAM + CXL Memory 在延迟上趋近，可以合并为一层
- Offload 决策简化：KV "换出"的目标从 PCIe + CPU DRAM 变成 CXL pool（无需跨总线传输）
- 多 GPU 共享 KV pool：同一 CXL domain 内的多张 GPU 可以直接访问同一块 KV 内存

**时间表（2025 视角）：**

- 硬件：CXL 1.1/2.0 设备已有商用产品（Samsung CMM-D、SK Hynix AiMX）
- 软件栈：Linux CXL subsystem 仍在完善，推理引擎的 CXL-aware allocator 尚不成熟
- 预计 2026–2027 才有生产级部署

**近存计算（Near-Memory Computing / PIM）：**

- 在内存芯片内嵌入计算单元，对 KV 做原地 attention 计算
- 避免 KV 从内存搬运到 GPU 的带宽消耗
- 仍处于研究阶段，距离大规模部署较远

---

## 2. KV 池作为独立的分布式存储产品

**类比：Redis / Memcached 在 Web 时代的角色**

当前推理引擎的 KVCache 是"内部实现"——不同引擎各自为政，无法互通。未来的可能路径：

| 阶段 | 描述 | 代表 |
|------|------|------|
| **现在** | 各引擎内置 KV 管理，LMCache/Mooncake 作为插件 | vLLM + Mooncake Store |
| **近期** | 标准化 KV cache 协议，多引擎兼容同一 KV 后端 | LMCache 协议草案 |
| **中期** | 托管 KV cache 服务（私有部署，类 Redis Cluster） | — |
| **远期** | 公有云 KV cache 产品（按 GB-hour 计费） | — |

**标准化的挑战：**

- 不同模型的 KV 格式不同（head 数、层数、精度）
- KV 的"有效性"依赖模型版本：模型更新则 KV 全部失效
- 跨引擎的 KV 传输需要统一的序列化格式

**已有早期形态：**

- [LMCache](https://github.com/LMCache/LMCache)：多后端（Valkey/S3/本地 FS）可插拔设计，接近"KV 存储即服务"
- Mooncake Store：分布式 RDMA KV 池，以 KV 为中心的调度思路

---

## 3. KV 与权重、激活的统一内存抽象

GPU 内存里有三类张量，目前分开管理：

| 张量类型 | 访问模式 | 生命周期 | 当前管理方式 |
|----------|---------|---------|------------|
| 模型权重 | 静态读取，高频 | 永久（model 级别） | 手动预分配 |
| KVCache | 增长式写入，中频 | 请求级别 | BlockManager |
| 激活值 | 短暂，高频读写 | 算子级别 | PyTorch allocator |

**研究方向：统一 GPU 内存管理器（Unified GPU Memory Manager）**

- 类似 OS 的 page-level 统一管理
- 在权重、KV、激活之间动态分配 HBM
- 当 KV 压力大时，可以把不常用的权重层临时 offload（权重 offload）

**工程难点：**

- 三者的访问模式差异巨大，统一调度策略难以优化
- PyTorch / CUDA 的现有 allocator 对权重和激活假设了不同的访问模式
- 权重的 offload 需要 lazy loading，与 KV 的实时 swap 交互复杂

**当前进展：**

- vLLM 的 BlockManager 已经实现了 KV 的统一管理
- 权重 offload 在 Serverless LLM（ServerlessLLM 项目）中有探索
- 完全统一的内存管理器尚未有生产级实现

---

## 4. 可观测性与 KV cache 调试

系统越复杂，可观测性越重要：

**当前痛点：**

- KV cache 命中率通常只有聚合统计，无法 per-request 诊断
- Swap-in / Swap-out 的延迟分布通常不暴露给用户
- 分布式 KV 池的 trace 链路（请求→KV查询→命中/miss→传输）缺乏标准

**未来方向：**

- **OpenTelemetry 标准接入**：KV 事件（cache hit/miss/evict/transfer）作为 span 暴露
- **Per-request KV trace**：每个请求可以看到自己的 KV 命中路径
- **KV cache profiler**：类似 CUDA Profiler，对 KV 的分配/释放/传输做细粒度分析

LMCache 的 BlendEngineV2 已引入 per-request root OTel span（PR #3062），是这个方向的早期实践。

---

## 关联章节

- 当前存储层级架构：[存储层级](storage-hierarchy.md)
- 当前分布式 KV 实现：[框架对比](frameworks.md)
- 算法侧的未来方向：[未来方向 — 算法侧](future-algorithm.md)
- 部署侧的未来方向：[未来方向 — 部署侧](future-deployment.md)

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 从未来方向总览拆分，补充 CXL 时间表、KV 存储产品路线图、统一内存管理细节 |
