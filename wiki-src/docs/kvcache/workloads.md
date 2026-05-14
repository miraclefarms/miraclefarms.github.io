# 工作负载维度

> **维度四**：KV 的访问模式因场景而异。同一套引擎、同一个模型，在不同 workload 下的最优配置可以截然不同。

本章用统一的"画像维度"刻画五类典型工作负载，每种场景的 prefix 复用度、上下文长度、decode 长度、并发模式各不相同。

## 0. 统一画像维度

四个变量描述任意工作负载：

| 维度 | 范围 |
|------|------|
| **Prefix 复用度** | 0%（完全独立）→ 100%（System Prompt 全员共享） |
| **上下文长度** | 1K → 1M token |
| **Decode 长度** | 几十 token → 数十万 token（reasoning） |
| **并发模式** | 短稳定流 → 突发 burst → 长持续连接 |

落到这个 4D 空间里，KV 关注点完全不同。

---

## 1. 多轮对话

### 特征

- 单 session 长寿命：一次对话可能持续几小时
- prefix 单调增长：每轮新增的内容追加在历史末尾
- 跨轮高度复用：第 N 轮的输入是前 N-1 轮的拼接

### 关键 KV 问题

- **session 级 KV 持久化**：用户思考时 KV 是否要换出？换到哪一级？
- **路由亲和性**：同 session 的轮次必须落到同副本（否则 cache miss）
- **历史压缩**：会话足够长时，前面几轮的 KV 是否可以摘要式压缩

### 推荐配置

- prefix cache **必开**
- session affinity 路由
- 长 idle 的 session 主动 swap 到 L2，需要时再换回（详见 [Offload](offload.md)）

---

## 2. Agent 协作

### 特征

- 多 agent 间上下文部分共享（同一 task 的不同 step）
- 工具调用打断生成：每次 tool result 注入都改变 KV 末尾
- 长链路：单个 task 可能涉及十几次 LLM 调用

### 关键 KV 问题

- **共享 prefix 的去重**：多个 agent 看到的 system prompt + task description 重复巨大，但每个的 trailing 状态略有不同
- **tool result 注入后的 KV 续接**：tool 调用后整段 prompt 改变，能否复用未被影响的部分？
    - 严格意义上不行（K/V 与位置编码强绑定）
    - 实践上：**可以复用截至工具调用前的 prefix**，工具结果及其后的内容必须重新 prefill
- **多 agent 的 KV 隔离**：每个 agent 的 KV 是独立的，但 prefix 重叠的部分应共享

### 工程实证：Claude Code 的上下文与缓存工程

主站 essay 对 Anthropic 的 prompt caching 接口进行了详细分析，揭示了 Agent 场景的几个关键洞察：

- **90% 的缓存失效来自服务器端路由/驱逐，而非客户端变化**——优化应集中在服务端调度而非客户端 prompt 设计
- **Cache 价格差 10×**：cache read $0.30/MTok vs full input $3/MTok，命中率是 Agent 推理成本的单变量决定性因素
- **缓存失效的诊断粒度**：Anthropic 将失效分为 5 类——eviction, ttl_expired, prompt_too_large, rate_limited, usage_capped——每一类对应不同的优化策略
- **Agent list 位置优化**：将可用工具列表从 tool description 移至 message attachment，消除了 10.2% 的 cache_creation_tokens

来源：主站 essay [Claude Code Context & KV Cache 工程](/notes/2026/05/08/claude-code-context-kvcache-engineering/)

### Mooncake Store 的 Agentic Trace 实测

在 610 条真实 agent trace 上，Mooncake Store 的分布式 KV 池实现：
- Cache hit rate：1.7% → 92.2%（+54×）
- Throughput：3.8×
- P50 TTFT：降低 46×
- 端到端延迟：降低 8.6×

这说明 Agent 场景的跨请求前缀共享远高于最初预期——一旦系统支持，收益是阶跃式的。

来源：主站 reading [vLLM × Mooncake Store](/notes/2026/05/07/vllm-mooncake-store-distributed-kv-cache/)

### 推荐配置

- block 粒度细一点（如 16），便于在 tool 调用断点处对齐
- prefix cache 命中以 sealed block 为单位，部分块未填满时不能命中——这对 agent 类负载很关键
- 启用分布式 KV 池（Mooncake Store / LMCache）以最大化跨实例命中

---

## 3. Coding / 长上下文

### 特征

- 仓库级上下文（数十万到 100 万+ token）
- 小幅编辑后大量重用：用户改了一行代码，前后大段内容不变
- TTFT 是关键体验

### 关键 KV 问题

- **增量 KV 更新**：编辑一处后能否复用前后未改动部分的 KV？
    - 编辑点之前的 KV 完全可复用
    - 编辑点之后的 KV 必须重算（位置编码已变）
- **KV 与代码索引/检索的协同**：仓库级 prompt 通常是检索拼接的，文档块的 prefix cache 复用度极高
- **长上下文下的容量瓶颈**：单请求可能就吃掉一张卡的 KV 配额，需要 SP/CP 切分

### 推荐配置

- prefix cache 必开，且 block size 大些（32 / 64）以减少元数据开销
- 启用 SP/CP 应对 100K+ 上下文
- 配合 L3/L4 持久 KV 池缓存仓库级文件 KV

---

## 4. RAG / 检索增强

### 特征

- prompt 结构：system prompt + 检索文档块 + 用户问题
- 每次检索的文档块组合不同，但单个文档块在不同请求里反复出现
- 文档块在 prompt 中的位置可变（top-1 vs top-3 vs top-5）

### 关键 KV 问题

- **文档块级 KV 复用——位置编码难题**：同一文档块在不同 prompt 里位置不同，K/V（含 RoPE）也就不同，**精确复用不可能**
- 解法方向：
    - **位置无关压缩**：把文档块预先压成位置无关的 KV（Prompt Cache、CacheBlend 路线）
    - **位置编码再校准**：在加载时根据当前 prompt 位置调整 RoPE
    - **退而求其次**：固定文档块在 prompt 中的位置，让精确 prefix cache 仍能命中
- **System prompt 部分**：100% 命中
- **检索文档部分**：当前主流系统命中率低；研究方向之一是 [CacheBlend (2024)](https://arxiv.org/abs/2405.16444)、[Prompt Cache](https://arxiv.org/abs/2311.04934)

### 推荐配置

- 把 RAG prompt 设计成"system + docs + question"的固定顺序
- 检索结果限制为 top-1 或 top-3 + 固定排序，提升 cache 命中
- 关注研究层的 fuzzy / position-invariant prefix cache 方案

---

## 5. 推理类任务（Reasoning / Long CoT）

### 特征

- prefill 短、decode 极长（思维链可能输出几万 token）
- KV 单调膨胀：decode 阶段每步增加一个 token 的 K/V
- 单请求资源占用大、持续时间长

### 关键 KV 问题

- **长 decode 下的容量瓶颈**：30K+ decode token × 70B 模型 KV ≈ 9 GB+，单个请求就可能吃满 batch 的容量
- **KV 单调增长，无法 prefix cache 复用**：每个推理是一次性的，跨请求复用空间小
- **与推测解码的交互**：speculative decoding 的 draft model KV 与 main model KV 如何协同
- **抢占代价高**：reasoning 已生成几万 token，丢了重算非常贵——倾向 swap 而非 recompute

### 推荐配置

- 大 KV 池配置（offload 到 DRAM 给长 decode 留余量）
- batch size 不能太大（避免单请求占用导致整体吞吐下降）
- 对 PD 分离的需求**降低**——decode 长度远大于 prefill，PD 解耦的边际收益小

---

## 6. 工作负载的统一画像表

| 场景 | Prefix 复用度 | 上下文长度 | Decode 长度 | 主要 KV 痛点 |
|------|--------------|------------|------------|------------|
| API / 短对话 | 高（system prompt） | < 2K | 短 | TTFT |
| 多轮对话 | 高（历史轮次） | 增长式 | 中 | session 亲和、swap |
| Agent | 中高（task 级） | 中长 | 中 | tool 注入续接 |
| Coding | 极高（文件不变） | 极长 | 中 | SP/CP + L3/L4 持久 |
| RAG | 中（system 高、docs 低） | 中 | 短 | 位置无关 prefix |
| Reasoning | 低 | 短 | 极长 | 容量、抢占代价 |

## 7. 不同象限对算法 / 系统 / 部署的诉求

| 场景 | 算法首选 | 系统首选 | 部署首选 |
|------|---------|---------|---------|
| 多轮 | GQA、StreamingLLM 风格 | prefix cache + L2 swap | session affinity 路由 |
| Agent | GQA + 块对齐 prompt | 细 block size | cache-aware 多副本 |
| Coding | GQA + KV 量化 | SP/CP + L4 KV 池 | 持久 prefix cache |
| RAG | 位置无关压缩（研究中） | 文档块级 prefix tree | tenant-aware 路由 |
| Reasoning | KV 量化（FP8） | 大 L2 容量 + swap | 大 batch decode 池，PD 不分 |

## 关联章节

- 各算法路线的细节：[Attention 变体](attention-variants.md)、[稀疏化](sparsity.md)、[压缩与量化](compression-quantization.md)
- 路由的具体机制：[路由与亲和性](routing.md)
- 工作负载与维度交叉的进一步讨论：[维度交叉 §3.3 / §3.4](crossings.md)

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 框架搭建 |
| v0.2 | 2026-05-14 | Agent 章节纳入 Claude Code 缓存工程实证（90%失效来自服务端路由、10× cache 价格差、5 类失效诊断）及 Mooncake Store agentic trace 实测数据（hit 1.7%→92.2%, 3.8× 吞吐）
