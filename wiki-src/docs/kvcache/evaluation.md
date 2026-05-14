# 评估方法

> KVCache 系统的性能不是单一数字。同一套配置，在不同的 Workload 特征下，性能表现可以截然不同。

正确的评估需要明确三件事：

1. **评什么**：选择正确的指标 → [指标体系](evaluation-metrics.md)
2. **用什么请求**：Workload 特征决定哪些指标有意义 → [场景评估矩阵](evaluation-scenarios.md)
3. **用什么工具**：Benchmark 框架与数据集的选择 → [Benchmark 与工具](evaluation-benchmarks.md)

---

## 本章结构

| 子页面 | 内容 |
|--------|------|
| [指标体系](evaluation-metrics.md) | TTFT / TPOT / Cache Hit Rate / Utilization / Recompute Rate 等核心指标的定义、公式、正常范围与组合解读 |
| [Benchmark 与工具](evaluation-benchmarks.md) | vLLM Benchmark、SGLang Benchmark、LLMPerf、SCBench 框架介绍；合成 Workload 生成；Agent 评估的研究空白 |
| [场景评估矩阵](evaluation-scenarios.md) | 各场景（API / 多轮 / RAG / Agent / Reasoning）的重点指标与目标值；混合负载注意事项 |

---

## 核心原则

### 不要只跑单点指标

不同 Benchmark 的结论可能相互矛盾，原因往往是 Workload 不同。在报告性能数据时，必须同时说明 Workload 特征（Prompt 长度分布、Output 长度分布、并发数、Prefix 共享率等）。

### 分层分场景统计

生产环境通常是多场景混合的。按 workload 类型分别采集指标，不要混合平均：

- system prompt 命中率应接近 100%
- docs / history 命中率差异大，需要区分诊断

### P99 与 P50 分开看

KVCache 管理导致的等待（换入、重算、网络抖动）往往显著拉高尾部延迟，即便平均值表现良好。

---

## 快速参考：SCBench KV 生命周期框架

SCBench（Microsoft, arxiv 2412.10319）是当前最系统化的 KV cache 评估基准：

| 生命周期阶段 | 评估内容 |
|-------------|---------|
| Generation | 首次 Prefill 和 Decode 的 KV 产生效率 |
| Compression | 压缩方法在多轮复用下的退化程度 |
| Retrieval | 稀疏/剪枝方法对远距离信息检索的影响 |
| Loading | 跨请求复用时 KV 的有效性 |

覆盖矩阵：8 种方法 × 6 个模型 × 12 个任务。

详细分析见 [Benchmark 与工具](evaluation-benchmarks.md)。

---

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 框架搭建 |
| v0.2 | 2026-05-14 | 新增 SCBench KV 生命周期评估框架及核心发现；新增 Agent + KV Cache 评估空白分析 |
| v0.3 | 2026-05-14 | 拆分为子页面，本页保留总览与导航 |
