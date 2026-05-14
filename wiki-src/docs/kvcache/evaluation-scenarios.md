# 场景评估矩阵

> 同一套配置，在不同 Workload 特征下，性能表现可以截然不同。正确的评估必须先明确场景。

## 各场景的重点指标

### 短请求 / API / 代码补全

- Prompt 通常较短（< 1K token），KV 小
- System prompt 固定，Prefix Cache 命中率高
- Decode 短（< 200 token），TTFT 主导用户感知

**重点指标：**

| 指标 | 目标值（参考） | 诊断意义 |
|------|------------|---------|
| TTFT P50 | < 100ms | 代码补全场景的可接受阈值 |
| TTFT P99 | < 500ms | P99 远高于 P50 说明有换入或重算 |
| Prefix Cache 命中率 | > 80%（system prompt） | 低于此值说明 routing 或 cache 有问题 |
| Requests/s | 依负载而定 | 关注 saturation 点 |

### 长上下文 / 文档分析

- 单个请求 KVCache 可能达到数十 GB
- KVCache 容量是主要瓶颈，可能触发 Offload 或抢占
- TTFT 较长是正常的（prefill 长），更关注 P99 的稳定性

**重点指标：**

| 指标 | 诊断意义 |
|------|---------|
| TTFT P50 / P99 | P99 受换入和抢占影响，关注两者比值 |
| KV Utilization | 长期 > 90% 说明容量不足，需要 offload 或降并发 |
| Swap-in Latency | 长上下文触发换入的代价 |
| Recomputation Rate | 被迫重算的频率，应控制在极低水平 |

### 多轮对话

- 历史轮次的 KV 可被 Prefix Cache 复用
- Session affinity 是否生效决定命中率的上限

**重点指标：**

| 指标 | 诊断意义 |
|------|---------|
| Per-turn Prefix Cache 命中率 | 应随轮次增加而提升；稳定后应 > 80% |
| TTFT per turn | 有 cache 命中的轮次应显著低于 miss 轮次 |
| Session affinity 成功率 | 同 session 的请求路由到同副本的比例 |

### RAG（检索增强生成）

- System prompt 部分应接近 100% 命中
- Docs 部分因位置编码问题，精确命中率通常低

**重点指标：**

| 指标 | 诊断意义 |
|------|---------|
| System prompt 命中率 | 应 100%；低于此值说明 cache 管理问题 |
| Docs 部分命中率 | 区分统计；若 > 30% 说明 prompt 结构设计良好（固定排序） |
| TTFT | RAG decode 短，TTFT 是主要用户感知指标 |

### Agent / 工具调用

- 多轮工具调用链的累积 KV 大小
- 跨实例命中率（分布式 KV 池的收益）

**重点指标：**

| 指标 | 诊断意义 |
|------|---------|
| Task 级命中率 | 同一 task 内跨调用的累积命中，比单次请求命中率更有意义 |
| cache_creation_tokens 占比 | > 20% 说明大量 prefix 未命中 |
| 跨实例命中率 | 分布式 KV 池的核心收益指标 |
| 工具调用后 TTFT | 注入 tool result 后重新 prefill 的效率 |

### Reasoning / Long CoT

- decode 极长，容量是核心瓶颈
- 抢占代价高，需要专项监控

**重点指标：**

| 指标 | 诊断意义 |
|------|---------|
| Peak KV per request | decode 末尾时单请求的 KV 大小，决定最大并发数 |
| TPOT | 长 decode 下的 HBM 带宽效率 |
| Swap-out / Swap-in 频率 | KV overflow 到 DRAM 的触发率 |
| Recomputation Rate | 应极低（< 1%）；reasoning 请求重算代价极高 |

---

## 评估矩阵：快速参考

| 场景 | TTFT | TPOT | 命中率 | Utilization | Recompute | Swap |
|------|------|------|--------|------------|-----------|------|
| API/代码补全 | ★★★ | ★ | ★★★ | ★ | ★ | ★ |
| 长上下文 | ★★ | ★★ | ★★ | ★★★ | ★★★ | ★★★ |
| 多轮对话 | ★★★ | ★ | ★★★ | ★★ | ★ | ★★ |
| RAG | ★★★ | ★ | ★★★ | ★ | ★ | ★ |
| Agent | ★★ | ★ | ★★★ | ★★ | ★ | ★★ |
| Reasoning | ★ | ★★★ | ★ | ★★★ | ★★★ | ★★★ |

★★★ = 核心指标，★★ = 重要，★ = 次要

---

## 多场景混合负载的注意事项

生产环境通常是多场景混合的。此时：

1. **分场景统计**：按 workload 类型（通过 prompt 长度、session 标记等区分）分别采集指标，不要混合平均
2. **关注 P99**：混合负载下，长请求（reasoning、长上下文）的尾部延迟会拉高整体 P99
3. **优先级分离**：考虑为不同 workload 分配独立的 decode 池或优先级队列，避免 reasoning 请求挤占短对话的 KV 容量

---

## 关联章节

- 各指标的定义与公式：[指标体系](evaluation-metrics.md)
- Benchmark 工具与数据集：[Benchmark 与工具](evaluation-benchmarks.md)
- 各 workload 的详细特征：[工作负载维度](workloads.md)

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 从评估方法总览拆分，重组为场景维度，补充混合负载注意事项 |
