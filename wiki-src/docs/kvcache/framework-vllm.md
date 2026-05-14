# vLLM

[vLLM](https://github.com/vllm-project/vllm) 是目前生态最成熟的开源推理引擎之一，也是 PagedAttention 的原始提出者。

!!! warning "内容时效性"
    vLLM 更新极快，下述 PR 引用可能已合并或过时。请以项目的最新文档和 Release Notes 为准。

---

## KVCache 架构

### PagedAttention 与 BlockManager

vLLM 是 PagedAttention 的原始提出者（Kwon et al., NeurIPS 2023）。

- **BlockPool**：统一管理 prefix cache block、shared block、free block，通过单一 `ref_cnt` 生命周期状态机
- **Block 粒度**：默认 16 token/block，可配置
- **物理/逻辑 Block 分离**：逻辑 block 是请求视角的 KV 序列，物理 block 是 HBM 中的实际分配——多个逻辑 block 可以映射到同一物理 block（prefix cache 的核心机制）

### Automatic Prefix Caching（APC）

- 基于链式哈希的精确匹配：每个 block 的 hash = hash(前一个 block hash, 本 block token ids)
- Lazy 匹配：只有 sealed（填满）的 block 才参与 prefix cache
- Block 粒度命中：hit 后直接复用物理 block，跳过对应 token 的 prefill

---

## 主要特性

### KV Offload（多级 Tier）

PR #40020 引入多级 KV cache offloading 框架：

- 支持链式二级 tier：GPU HBM → CPU DRAM → NVMe
- OffloadingManager 注入 per-request 身份追踪（PR #42507）
- 换入/换出由调度器触发，对推理引擎透明

### Mooncake Store 集成（分布式 KV 池）

PR #40900 将分布式 KV cache 池接入 vLLM：

- `MooncakeStoreConnector`：在 scheduler 查询 Store 后进行 block 分配
- MultiConnector 支持 PD + Store 混合拓扑

**Agentic workload 实测（610 条真实 agent trace）：**

| 指标 | 基线 | +Mooncake Store |
|------|------|----------------|
| Cache hit rate | 1.7% | 92.2% |
| Throughput | 1× | 3.8× |
| P50 TTFT | 基准 | −46× |

### PD 分离（Disaggregated Prefill）

vLLM V1 架构推进中的 disaggregated prefill 支持：

- MultiConnector 统一管理 PD 拓扑和 Store 拓扑
- KV 传输协议持续演进中

### KV 量化

- FP8 KV Cache：支持（部分硬件）
- NVFP4 KV sliding window（PR #42464）
- TurboQuant 集成（PR #40396）

### Attention Backend

- 支持 FlashAttention、FlashInfer、Triton
- TOKENSPEED_MLA 覆盖 Blackwell MLA prefill/decode（含 FP8 KV cache，PR #41778）

### 多租户隔离

- `cache_salt` 参数（PR #39837）：按租户/用户隔离共享 KV block，防止跨租户命中

---

## 可观测性

vLLM 暴露的 KVCache 相关指标（通过 Prometheus / OpenMetrics）：

- `vllm:gpu_cache_usage_perc`：GPU KV cache 利用率
- `vllm:cpu_cache_usage_perc`：CPU KV cache 利用率
- `vllm:num_preemptions_total`：抢占次数
- Prefix cache 命中率（通过 `cache_hit_rate` 统计）

---

## 关联章节

- PagedAttention 的详细原理：[Paged KV](paged-kv.md)
- Prefix Cache 的 APC 实现细节：[Prefix Cache](prefix-cache.md)
- 分布式 KV 池（Mooncake Store）：[框架对比](frameworks.md)、[存储层级](storage-hierarchy.md)

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 从框架对比总览拆分，梳理 BlockManager / APC / Offload / Mooncake 集成等核心特性 |
