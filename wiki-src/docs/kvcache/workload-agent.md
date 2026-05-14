# Agent 协作

> **工作负载画像**：Prefix 复用度中高（task 级共享）｜上下文中长｜工具调用打断生成｜多次 LLM 调用构成单任务

## 特征

- 多 agent 间上下文部分共享：同一 task 的不同 step 共享 system prompt + task description
- 工具调用打断生成：每次 tool result 注入都改变 KV 末尾，产生新的 sealed block
- 长链路：单个 task 可能涉及十几次 LLM 调用，每次调用的 prompt 在上一次基础上追加
- 分支-合并模式：多 agent 并行执行后合并结果，KV 有"分叉"的访问结构

---

## 关键 KV 问题

### 1. 共享 Prefix 的去重

多个 agent 看到的 system prompt + task description 完全相同，但每个 agent 的 trailing 状态略有不同。

- 共享部分（system prompt）：prefix cache 可 100% 命中
- 差异部分（每个 agent 的独立历史）：各自独立，无法共享
- 关键设计：block 粒度要能对齐到 system prompt 的末尾，避免跨 block 污染共享部分

### 2. 工具调用后的 KV 续接

工具调用完成后，tool result 注入 prompt，整段输入改变。能否复用工具调用前的 KV？

- **严格意义上**：tool result 改变了 prompt，tool result 之后的所有 token 的 K/V 都与位置编码强绑定，必须重新 prefill
- **实践上可复用的部分**：截至工具调用**之前**的 prefix（包括 system prompt、历史对话、此前的工具调用链）可精确命中
- **sealed block 对齐**：只有已填满的 block 才能命中 prefix cache；tool result 注入后，末尾未填满的 block 需要重新计算

### 3. 多 Agent 的 KV 隔离

- 不同 agent 的 KV 必须隔离（prompt 结果不能混淆）
- 但 prefix 重叠的部分（system prompt、共享 task context）应当共享同一批 KV block——需要系统支持 ref-counted block 共享
- 多租户场景下还需考虑 cache_salt 隔离，避免不同用户的 agent 共享 KV

---

## 工程实证

### Claude Code 的 KV Cache 工程

主站 essay 对 Anthropic 的 prompt caching 接口进行了详细分析，揭示了 Agent 场景的几个关键洞察：

- **90% 的缓存失效来自服务器端路由/驱逐，而非客户端变化**——优化应集中在服务端调度而非客户端 prompt 设计
- **Cache 价格差 10×**：cache read $0.30/MTok vs full input $3/MTok，命中率是 Agent 推理成本的单变量决定性因素
- **缓存失效的诊断粒度**：Anthropic 将失效分为 5 类——eviction、ttl_expired、prompt_too_large、rate_limited、usage_capped——每类对应不同优化策略
- **Agent list 位置优化**：将可用工具列表从 tool description 移至 message attachment，消除了 10.2% 的 cache_creation_tokens

来源：主站 essay [Claude Code Context & KV Cache 工程](/notes/2026/05/08/claude-code-context-kvcache-engineering/)

### Mooncake Store 的 Agentic Trace 实测

在 610 条真实 agent trace 上，Mooncake Store 的分布式 KV 池实现：

| 指标 | 基线 | Mooncake Store | 倍数 |
|------|------|---------------|------|
| Cache hit rate | 1.7% | 92.2% | +54× |
| Throughput | 基准 | 3.8× | — |
| P50 TTFT | 基准 | −46× | — |
| 端到端延迟 | 基准 | −8.6× | — |

Agent 场景的跨请求前缀共享远高于最初预期——一旦系统支持分布式 KV 池，收益是阶跃式的。

来源：主站 reading [vLLM × Mooncake Store](/notes/2026/05/07/vllm-mooncake-store-distributed-kv-cache/)

---

## 工程配置建议

| 配置项 | 推荐值 / 策略 | 原因 |
|--------|-------------|------|
| Prefix Cache | **必开** | system prompt + task context 全员共享 |
| Block Size | 16（较细） | 工具调用断点处对齐，避免 sealed block 浪费 |
| 分布式 KV 池 | **推荐**（Mooncake Store / LMCache） | 跨实例命中，agent 路由不受限于单副本 |
| Cache-aware routing | **推荐** | 同 task 的多次调用路由到同一副本 |
| cache_salt 隔离 | 多租户场景必开 | 不同用户 agent 的 KV 隔离 |

---

## 关键指标

- **Task 级 Prefix Cache 命中率**：应关注单个 task 内跨调用的累积命中率，而非单次请求命中率
- **cache_creation_tokens 占比**：越低越好；超过 20% 说明大量 prefix 未命中
- **工具调用后的 TTFT**：反映工具注入后重新 prefill 的效率
- **分布式 KV 池命中率**：Mooncake Store / LMCache 的跨实例命中率

---

## 关联章节

- 共享 KV block 的引用计数机制：[Paged KV](paged-kv.md)
- 跨实例 KV 池的工程实现：[存储层级](storage-hierarchy.md)
- cache_salt 隔离与路由策略：[路由与亲和性](routing.md)
- Agent 评估框架与研究空白：[Benchmark 与工具](evaluation-benchmarks.md)

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 从工作负载总览拆分，保留 Claude Code 与 Mooncake Store 实证数据 |
