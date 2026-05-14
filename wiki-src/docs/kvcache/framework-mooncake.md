# Mooncake

[Mooncake](https://github.com/kvcache-ai/Mooncake) 是 Kimi 团队开源的以 KVCache 为中心的调度系统，设计哲学是"以 KV 存储为中心重新思考 LLM 推理架构"。

!!! warning "内容时效性"
    Mooncake 更新极快。请以项目的最新文档和 Release Notes 为准。

---

## 设计哲学

与 vLLM / SGLang 这类以"推理引擎"为中心的框架不同，Mooncake 的出发点是：

> **KVCache 是系统的一等公民资源**，不是推理引擎的附属品。调度决策应该围绕 KV 的存储和复用来组织，而不是反过来。

这导致了一套与传统推理引擎不同的架构：

- 分布式 KV 存储池是中心，推理实例是"消费者"
- 请求路由的核心依据是"哪个实例已经有这个 prefix 的 KV"
- 跨实例的 KV 共享是设计目标，而非可选特性

---

## 架构

### 分布式 KV Store

- **RDMA-capable 分布式 DRAM 池**：多节点 DRAM 通过 RDMA 网络互联，构成统一的 KV 存储空间
- **block-hash 全局寻址**：每个 KV block 以哈希为 key，全局可寻址
- **零拷贝 GPU-to-RDMA 路径**：GPU 直接将 KV 写入 RDMA buffer，无 CPU 参与的内存拷贝

### 元数据层

- **ObjectDataType 枚举**（PR #1719）：元数据层知道每块存储的内容类型（KV block / 索引 / 配置），支持类型感知的管理策略
- **DSA 式分配策略**（PR #2080）：数据流向的分配决策由中心元数据服务协调

---

## 主要特性

### 高可用与恢复

- **Redis HA 领导者选举后端**（PR #1722）：元数据服务的高可用，避免单点故障
- **三阶段恢复流水线**（PR #1876）：client-based 恢复，按优先级依次恢复 hot keys → DRAM entries → storage tier

### 传输层

多传输后端支持：

- PG 与 TENT 集成（PR #1676）+ P2P 内存区域本地注册（PR #1690）
- EFA transport（AWS Elastic Fabric Adapter）：补全 `fi_read` / LRU eviction / multi-NIC striping（PR #1821）
- Get 路径 batch route query（PR #1970）：批量查询降低元数据服务压力

### SSD Offload

- 通过 Python setup 参数暴露（PR #1884）：从全局环境变量改为实例级参数，支持精细控制
- 三层：GPU HBM → DRAM → SSD

### 与 vLLM 集成

PR #40900（vLLM 侧）：`MooncakeStoreConnector`

- vLLM scheduler 查询 Mooncake Store，获取已有 block 的位置
- block 分配时优先复用 Store 中已有的 block，跳过 prefill
- 支持 PD 分离场景下的 cross-instance KV 复用

**Agentic workload 实测（610 条真实 trace）：**

| 指标 | 基线（vLLM 内置 APC） | +Mooncake Store |
|------|---------------------|----------------|
| Cache hit rate | 1.7% | 92.2% |
| Throughput | 1× | 3.8× |
| P50 TTFT | 基准 | −46× |
| 端到端延迟 | 基准 | −8.6× |

---

## 与 LMCache 的对比

Mooncake 和 LMCache 都是"分布式 KV 存储"定位，但侧重不同：

| 维度 | Mooncake | LMCache |
|------|---------|---------|
| 存储介质 | 主打 RDMA DRAM 池 | 多后端（Valkey/FS/S3） |
| 目标场景 | 大规模集群，高 RDMA 带宽 | 灵活部署，小到大规模 |
| 开发背景 | Kimi 生产系统 | 学术 + 开源社区 |
| 可观测性 | 元数据层追踪 | OTel span + binary trace |
| HA | Redis HA 领导者选举 | 依赖后端（Valkey cluster） |

---

## 关联章节

- 分布式 KV 存储的系统原理：[存储层级](storage-hierarchy.md)
- Mooncake Store + vLLM 集成的 Agent 场景收益：[Agent 协作](workload-agent.md)
- 框架总览与对比表：[框架对比](frameworks.md)

来源：主站 reading [vLLM × Mooncake Store](/notes/2026/05/07/vllm-mooncake-store-distributed-kv-cache/)，主站 briefs 2026-03-19 ~ 2026-05-14

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 从框架对比总览拆分，梳理设计哲学、分布式 Store、HA 恢复、vLLM 集成实测数据 |
