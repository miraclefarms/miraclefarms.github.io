# 评估方法

## 评估 KVCache 系统的挑战

KVCache 系统的性能不是单一数字。同一套配置，在不同的 Workload 特征下，性能表现可以截然不同。正确的评估需要明确三件事：

1. **评什么**：选择正确的指标
2. **用什么请求**：Workload 特征决定哪些指标有意义
3. **在什么条件下**：负载强度、系统配置、资源约束

## 核心指标

### 延迟类指标

**TTFT（Time to First Token）**

首 token 延迟，从请求到达到第一个生成 token 返回的时间。

- 主要受 Prefill 阶段影响
- 在开启 Prefix Cache 时，命中部分不需要 Prefill，TTFT 显著降低
- 在 PD 分离架构中，TTFT 额外包含 KV 传输时间
- 对于有响应感知要求的 Chat 场景，TTFT 是关键 SLA 指标

**TPOT（Time per Output Token）**

每个输出 token 的平均生成延迟，等于总生成时间除以生成 token 数。

- 主要受 Decode 阶段影响
- 受 KVCache 读取带宽限制（memory-bound）
- 在大 batch 下，TPOT 因多个 Sequence 共享 HBM 带宽而上升
- 对于流式输出场景（streaming），TPOT 直接影响用户感知的"打字速度"

**端到端延迟（End-to-End Latency）**

从请求发出到最后一个 token 返回的总时间，等于 TTFT + TPOT × (output_length - 1)。

**尾部延迟（Tail Latency，P95/P99）**

评估系统的稳定性。P99 TTFT 或 P99 TPOT 过高意味着部分请求体验极差，即便平均值表现良好。KVCache 管理导致的等待、换入换出、重算等操作往往显著拉高尾部延迟。

### 吞吐类指标

**请求吞吐（Request Throughput，requests/s）**

单位时间内系统完成的请求数。

**Token 吞吐（Token Throughput，tokens/s）**

单位时间内系统生成的 token 总数。在长输出场景中比请求吞吐更有代表性。

### KVCache 效率类指标

**Cache Hit Rate（缓存命中率）**

$$\text{Hit Rate} = \frac{\text{命中的 KV token 数（无需 Prefill）}}{\text{总请求 Prompt token 数}}$$

对于开启 Prefix Cache 的系统，命中率直接反映系统对 Workload 的适配效果。高命中率 → 低 TTFT + 低计算开销。

**KVCache Memory Utilization（显存利用率）**

$$\text{Utilization} = \frac{\text{已分配的 KV Block 数}}{\text{总 KV Block 数}}$$

高利用率不等于好事：如果利用率长期 100%，说明 KVCache 已成瓶颈，新请求可能需要等待或触发抢占。

**Block Fragmentation Rate**

内部碎片（最后一个 Block 的未使用 Slot 比例）的均值。理论上，block_size 越小，碎片率越低，但调度开销上升。

### Offload 专项指标

**Offload Bandwidth Utilization**

PCIe/RDMA 实际传输带宽 vs 最大可用带宽的比值。带宽利用率过高意味着传输成为瓶颈。

**Swap-in Latency**

从触发换入到 KV 可用于 Decode 的时间。过长的换入延迟直接增加等待中的 Sequence 的尾部延迟。

**Recomputation Rate**

因抢占导致的 Recompute 占总 Prefill 计算的比例。高 Recompute 率意味着调度策略不合理或 KVCache 容量太小。

## Workload 的重要性

### 短请求（API / 代码补全）

- Prompt 通常较短（< 1K tokens），KVCache 小
- System Prompt 固定，Prefix Cache 命中率高
- 关注：TTFT（响应速度）、requests/s（吞吐）

### 长上下文（文档分析 / 长对话）

- 单个请求 KVCache 可能达到数十 GB
- KVCache 容量是主要瓶颈，可能触发 Offload 或抢占
- 关注：TTFT、TPOT、显存利用率、Offload 带宽

### 多轮对话

- 历史轮次的 KV 可能被 Prefix Cache 复用
- 关注：Prefix Cache 命中率、TTFT（含命中加速效果）

### RAG（检索增强生成）

- Prompt = 系统提示 + 检索文档 + 用户问题
- 检索文档每次不同，但系统提示固定 → 部分命中
- 关注：命中率的拆分（系统提示 vs 文档部分）

### Agent / 工具调用

- 多轮工具调用，每轮 Prompt 在上一轮基础上追加工具结果
- 关注：多轮累积的 KVCache 大小、每轮的增量 Prefill 效率

## Benchmark 建议

**不要只跑单点指标**。以下是一个合理的评估矩阵：

| 场景 | 关键指标 | 关键变量 |
|------|----------|----------|
| 低负载 | TTFT、TPOT | Prompt 长度 |
| 中等负载 | 吞吐、命中率 | 并发请求数 |
| 高负载 | 尾部延迟、Recompute 率 | KVCache 利用率 |
| Prefix Cache | 命中率 vs TTFT 降幅 | Prefix 共享比例 |
| Offload | 换入延迟、带宽利用率 | HBM 容量限制 |

常用工具：

- **vLLM Benchmark** (`benchmark_serving.py`, `benchmark_throughput.py`)
- **SGLang Benchmark**
- **LLMPerf**
- **ShareGPT 数据集**（模拟真实多轮对话分布）
- **自定义合成 Workload**（控制 Prompt 长度分布、共享前缀比例等）

!!! note "指标要结合 Workload 解读"
    不同 Benchmark 的结论可能相互矛盾，原因往往是 Workload 不同。在报告性能数据时，必须同时说明 Workload 特征（Prompt 长度分布、Output 长度分布、并发数、Prefix 共享率等）。

## SCBench：KV 生命周期视角的评估框架

SCBench（Microsoft, arxiv 2412.10319）将 KV cache 评估拆解为四个生命阶段，每个阶段对应不同的测试场景：

| 生命周期阶段 | SCBench 对应场景 | 评估内容 |
|-------------|-----------------|----------|
| **Generation** | 生成阶段的 KV 写入 | 首次 Prefill 和 Decode 的 KV 产生效率 |
| **Compression** | KV 压缩/剪枝后保留效果 | 压缩方法在多轮复用下的退化程度 |
| **Retrieval** | 长上下文中检索特定信息 | 压缩/剪枝方法对远距离信息检索的影响 |
| **Loading** | KV 重载到新请求 | 跨请求复用时 KV 的有效性 |

**核心发现：**
- Sub-O(n) 内存方法（StreamingLLM、SnapKV 等）在单轮测试中表现尚可，但在多轮场景下**系统性退化**——压缩是 query-conditioned 的，新 query 可能使之前丢弃的 KV 变得关键
- O(n) 稀疏注意力方法在多轮复用场景下保持或提升性能
- Attention 分布在长生成过程中发生漂移，远离早期上下文的注意力可能随时间衰减

**覆盖矩阵：** 8 种方法 × 6 个模型 × 12 个任务，是当前 KV cache 评估最系统化的基准。

来源：主站 reading [SCBench](/notes/2026/04/21/scbench-kv-cache-lifecycle-analysis/)

## Agent + KV Cache 长上下文评估

Agent 场景的 KV cache 评估需要同时关注系统指标和任务成功率：

| 指标类别 | 具体指标 | 重要性 |
|----------|---------|--------|
| 系统指标 | Cache hit rate、TTFT、压缩率 | Agent 多轮链路效率 |
| 任务指标 | Agent 成功率、工具调用准确率 | 功能正确性保证 |

**当前研究空白：** 尚无 benchmark 同时结合系统指标和 agent 成功率。现有工作各自独立——SCBench 侧重系统效率，LoCoBench-Agent 侧重多轮交互，WebAgent 侧重 web 操作——没有形成统一的 agent+cache 评估框架。

来源：主站 essay [KV Cache Agent 长上下文 Benchmark](/notes/2026/04/08/kvcache-agent-long-context-benchmark/)

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 框架搭建 |
| v0.2 | 2026-05-14 | 新增 SCBench KV 生命周期评估框架及核心发现；新增 Agent + KV Cache 评估空白分析
