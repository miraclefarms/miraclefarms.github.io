# 部署 × 工作负载

> 部署架构（PD 分离、并行切分、路由）不是中立的——它对不同工作负载的收益相差巨大。

## Agent 场景与 PD 分离的契合度

**Agent 的请求特征：**

- 单步 decode 通常 < 100 token（工具调用结果短、指令简洁）
- 每次 LLM 调用是独立的请求，请求频率高但单次体量小
- 多次调用之间有 prefill（注入 tool result），但 prefill 也不长

**PD 分离的固定开销：**

- KV 从 P 传输到 D：每次 prefill 后都有一次传输，无论 KV 大小
- 调度协议开销：P/D 之间的请求协调、连接复用等
- 当 decode 很短（< 100 token）时，固定开销占端到端时间的比例很高

**结论：**

| 负载类型 | PD 分离收益 | 原因 |
|---------|-----------|------|
| 纯 Agent（短 decode） | 低 | 固定开销摊薄不了，decode 挤占 prefill 的问题不严重 |
| 混合（Agent + 长对话） | 中等 | PD 分离保护长对话的 decode，Agent 混入损失可接受 |
| 长推理（Reasoning） | 低 | decode 远长于 prefill，挤占问题自然不严重 |
| API 服务（均匀混合） | 高 | 经典受益场景 |

纯 Agent 负载建议**不做 PD 分离，统一混合部署**，用 cache-aware routing + 分布式 KV 池替代。

---

## Coding 长上下文下 SP 与 Prefix Cache 的取舍

**冲突的来源：**

Sequence Parallelism（SP）/ Context Parallelism（CP）把 prompt 沿序列维度切到多张卡：

```
卡 0: token[0:32768]   → KV[0:32768]
卡 1: token[32768:65536] → KV[32768:65536]
```

Prefix cache 的命中逻辑是"从 token[0] 开始的连续 prefix"——但分切后，每张卡只有部分 prefix，无法独立判断命中。

**当前主流做法：优先保证 SP，prefix cache 命中率次要**

原因：

- 100K+ token 的 coding 请求，如果不用 SP，单卡 HBM 直接 OOM
- SP 是功能保证，prefix cache 是性能优化——功能优先
- 实践中，同一用户的请求通常路由到同一组 SP 副本（session affinity），避免 SP 分片不一致

**研究方向：SP-aware 的分布式 prefix cache**

- 把 prefix tree 建在"SP 分片"粒度上
- 每个 SP 副本组有自己的 prefix tree，跨组之间通过 L4 协调
- 当前尚无生产级实现

---

## Reasoning 长 Decode 场景是否还需要 PD 分离

**Reasoning 的资源画像：**

```
典型请求：prefill 1K token → decode 30K token
时间比：T_prefill ≈ 0.5s，T_decode ≈ 60s
```

PD 分离的核心价值是：当 prefill 和 decode 互相抢占 GPU 时，分开到不同节点，各自专注。

**当 decode 时间 ≫ prefill 时间时：**

- prefill 在 decode 资源上的占比极低（< 1%）
- 即便 prefill 偶尔抢占了 decode slot，影响也微乎其微
- PD 分离带来的固定 KV 传输开销（每次 prefill 后一次传输）却是真实成本

**结论：**

| 场景 | PD 分离建议 |
|------|-----------|
| 纯 Reasoning 负载 | **不推荐**——收益有限，传输开销显著 |
| 混合（Reasoning + Chat） | **推荐**——Chat 的 prefill 会严重影响 Reasoning 的 decode，PD 分离隔离两者 |

纯 Reasoning 集群建议用**大 batch decode 池 + DRAM offload**替代 PD 分离。

---

## 关联章节

- PD 分离的架构与 KV 传输开销：[PD 分离](pd-disaggregation.md)
- Coding 场景的 SP/CP 配置：[并行切分](parallelism.md)、[Coding 工作负载](workload-coding.md)
- Reasoning 场景的容量与抢占策略：[Reasoning 工作负载](workload-reasoning.md)
- Agent 场景的分布式 KV 池方案：[Agent 工作负载](workload-agent.md)

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 从维度交叉总览拆分，补充各场景 PD 分离收益对比表与 SP/prefix cache 冲突分析 |
