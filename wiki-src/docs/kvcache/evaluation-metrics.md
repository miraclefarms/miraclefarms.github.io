# 指标体系

> KVCache 系统的性能不是单一数字——不同指标捕捉不同维度的瓶颈，需要组合看。

## 延迟类指标

### TTFT（Time to First Token）

首 token 延迟：从请求到达到第一个生成 token 返回的时间。

$$\text{TTFT} = T_{\text{prefill}} + T_{\text{schedule}} + T_{\text{kv\_transfer}}$$

- 主要受 Prefill 阶段影响
- 开启 Prefix Cache 时，命中部分跳过 Prefill，TTFT 显著降低——命中越深，降幅越大
- PD 分离架构中，TTFT 额外包含 KV 从 Prefill 节点传输到 Decode 节点的时间
- Chat 和代码补全场景的核心 SLA 指标

**正常范围（参考）：**
- 短 prompt（< 2K token）无 prefix cache：50–200ms
- 长 prompt（100K token）有 prefix cache 命中：100–500ms（取决于未命中部分长度）

### TPOT（Time per Output Token）

每个输出 token 的平均生成延迟：

$$\text{TPOT} = \frac{T_{\text{decode\_total}}}{\text{output\_length}}$$

- 主要受 Decode 阶段影响
- KV 读取是 memory-bound 操作：TPOT ∝ KV 大小 / HBM 带宽
- 大 batch 下，多个 sequence 共享 HBM 带宽，TPOT 上升
- 流式输出场景直接决定"打字速度"的用户感知

### 端到端延迟（E2E Latency）

$$\text{E2E} = \text{TTFT} + \text{TPOT} \times (\text{output\_length} - 1)$$

对短输出（< 100 token）场景，TTFT 主导；对 reasoning 场景（30K+ decode），TPOT × output_length 主导。

### 尾部延迟（Tail Latency，P95/P99）

评估系统稳定性。以下操作会显著拉高尾部延迟：

- KV swap-in 等待（DRAM → HBM 换入延迟）
- 抢占后的重算（recompute）
- 调度等待（batch slot 不足）
- 分布式 KV 池的网络抖动

!!! tip "监控原则"
    P99 TTFT 和 P99 TPOT 应与 P50 分开监控。两者差距大说明有"长尾"操作（换入、重算、网络）需要优化。

---

## 吞吐类指标

### 请求吞吐（Request Throughput）

$$\text{Req Throughput} = \frac{\text{完成请求数}}{\text{时间窗口（秒）}}$$

适合短输出、均匀负载的场景。长输出不均匀时，单个长请求会拉低整体数字。

### Token 吞吐（Token Throughput）

$$\text{Token Throughput} = \frac{\text{生成 token 总数}}{\text{时间窗口（秒）}}$$

更适合长输出场景（RAG decode 短、Reasoning decode 长，两者的请求吞吐无法直接比较）。GPU 利用率高时，token 吞吐更能反映系统效率。

---

## KVCache 效率类指标

### Cache Hit Rate（缓存命中率）

$$\text{Hit Rate} = \frac{\text{命中的 KV token 数（无需 Prefill）}}{\text{总请求 Prompt token 数}}$$

- 命中率直接反映系统对 Workload 的适配效果
- 高命中率 → 低 TTFT + 低计算开销 + 低 GPU 利用率（计算省了）
- 命中率应**按 workload 分层**统计：system prompt 命中率通常接近 100%，docs/history 命中率差异大

### KVCache Memory Utilization（显存利用率）

$$\text{Utilization} = \frac{\text{已分配的 KV Block 数}}{\text{总 KV Block 数}}$$

高利用率不等于好事：

- 利用率长期 > 90% → KVCache 成瓶颈，新请求等待或触发抢占
- 利用率长期 < 60% → KVCache 过度配置，可以减小 KV pool 或接受更多并发

### Block Fragmentation Rate

内部碎片率：最后一个（未填满）Block 的平均浪费比例。

$$\text{Frag Rate} = 1 - \frac{\text{实际 KV token 数}}{\text{已分配 Block 数} \times \text{block\_size}}$$

- Block size 越大，碎片率越高（尤其对短 prompt）
- Block size 越小，碎片率低，但元数据开销和调度复杂度上升

---

## Offload 专项指标

### Offload Bandwidth Utilization

PCIe / RDMA 实际传输带宽 vs 最大可用带宽的比值。

- 带宽利用率持续 > 80% → 传输成为瓶颈，KV offload 策略过于激进
- 带宽利用率 < 10% → offload 基础设施配置过剩，或 offload 频率太低

### Swap-in Latency

从触发换入（HBM 不足，需要从 DRAM/SSD 换回）到 KV 可用于 Decode 的时间。

- DRAM → HBM：~100–500ms（取决于 KV 大小和 PCIe 带宽）
- SSD → HBM：~1–5s

Swap-in latency 直接加到等待中的 sequence 的 TTFT 或 TPOT 上，是尾部延迟的主要来源之一。

### Recomputation Rate

因抢占导致的重算占总 Prefill 计算的比例。

$$\text{Recompute Rate} = \frac{\text{因抢占触发的 Prefill token 数}}{\text{总 Prefill token 数}}$$

- 高 Recompute Rate（> 5%）说明调度策略不合理或 KV 容量严重不足
- Reasoning 场景尤其需要关注：被抢占的长 decode 请求重算代价极高

---

## 指标组合解读

| 现象 | 可能原因 |
|------|---------|
| TTFT 高，吞吐正常 | Prefill 瓶颈，或 prefix cache 命中率低 |
| TPOT 高，TTFT 正常 | HBM 带宽不足，或 batch size 过大 |
| P99 >> P50 | swap-in 或 recompute 频发，需降低 offload 阈值或扩 KV 容量 |
| 命中率低 | Workload 不适合 prefix cache，或路由未做 session affinity |
| Utilization 持续 100% | KV pool 容量不足，考虑 offload 或减小 batch size |
| Recompute Rate 高 | 调度抢占过于激进，或 KV pool 太小 |

---

## 关联章节

- 各场景的重点指标组合：[场景评估矩阵](evaluation-scenarios.md)
- Benchmark 工具与测量方法：[Benchmark 与工具](evaluation-benchmarks.md)
- 可观测性与监控实现：各框架的 metrics 暴露见 [框架对比](frameworks.md)

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 从评估方法总览拆分，补充公式、正常范围参考与组合解读表 |
