# Benchmark 与工具

> 不同 Benchmark 的结论可能相互矛盾，原因往往是 Workload 不同。报告性能数据时，必须同时说明 Workload 特征。

## 常用工具

### vLLM Benchmark Suite

vLLM 自带的压测脚本，是目前最广泛使用的推理引擎基准工具。

- `benchmark_serving.py`：模拟真实服务场景，支持并发请求、固定 QPS、ShareGPT 数据集
- `benchmark_throughput.py`：离线吞吐测试，批量处理所有请求，专注最大 token 吞吐
- 支持采集：TTFT、TPOT、E2E latency、requests/s、tokens/s
- 使用 ShareGPT 数据集时可模拟真实多轮对话分布

```bash
python benchmark_serving.py \
  --model meta-llama/Llama-3-70b-instruct \
  --dataset-name sharegpt \
  --num-prompts 1000 \
  --request-rate 10
```

### SGLang Benchmark

SGLang 提供类似 vLLM 的 benchmark 脚本，额外支持：

- RadixAttention 的 prefix cache 命中率统计
- 多并发下的 RadixTree 效率分析

### LLMPerf

MLCommons 维护的标准化推理性能测试框架：

- 支持多种模型和推理引擎
- 输出结构化 JSON 报告，便于跨框架对比
- 适合合规测试和供应商对比场景

### 合成 Workload 生成

当真实数据不可用或需要控制变量时，合成 Workload 是必要手段：

```python
# 控制 prefix 共享比例的合成请求生成示意
def gen_requests(shared_prefix_len, unique_suffix_len, n_requests):
    shared = "X" * shared_prefix_len
    return [shared + random_suffix(unique_suffix_len) for _ in range(n_requests)]
```

关键变量：
- **Prompt 长度分布**：短（< 1K）、中（1K–10K）、长（10K+）
- **Output 长度分布**：短（< 100）、中（100–1K）、长（1K+）
- **Prefix 共享比例**：0%（完全随机）→ 100%（全员相同 system prompt）
- **并发数**：1 → N（揭示排队和抢占效应）

---

## SCBench：KV 生命周期视角的评估框架

SCBench（Microsoft, 2024, arxiv 2412.10319）是目前最系统化的 KV cache 评估基准，将 KV 评估拆解为四个生命阶段：

| 生命周期阶段 | SCBench 对应场景 | 评估内容 |
|-------------|-----------------|----------|
| **Generation** | 生成阶段的 KV 写入 | 首次 Prefill 和 Decode 的 KV 产生效率 |
| **Compression** | KV 压缩/剪枝后保留效果 | 压缩方法在多轮复用下的退化程度 |
| **Retrieval** | 长上下文中检索特定信息 | 压缩/剪枝方法对远距离信息检索的影响 |
| **Loading** | KV 重载到新请求 | 跨请求复用时 KV 的有效性 |

**核心发现：**

- Sub-O(n) 内存方法（StreamingLLM、SnapKV 等）在**单轮**测试中表现尚可，但在**多轮**场景下系统性退化——压缩是 query-conditioned 的，新 query 可能使之前丢弃的 KV 变得关键
- O(n) 稀疏注意力方法在多轮复用场景下保持或提升性能
- Attention 分布在长生成过程中发生漂移，早期上下文的注意力可能随时间衰减

**覆盖矩阵：** 8 种方法 × 6 个模型 × 12 个任务，是目前覆盖最广的 KV cache 评估基准。

来源：主站 reading [SCBench](/notes/2026/04/21/scbench-kv-cache-lifecycle-analysis/)

---

## Agent + KV Cache 评估：当前空白

Agent 场景的 KV cache 评估需要同时关注系统指标和任务成功率，但目前没有统一基准：

| 现有 Benchmark | 侧重 | 缺失 |
|---------------|------|------|
| SCBench | 系统效率（命中率、压缩率） | 任务成功率 |
| LoCoBench-Agent | 多轮交互任务成功率 | 系统指标 |
| WebAgent 系列 | Web 操作成功率 | KV cache 效率 |

**当前研究空白：** 尚无 benchmark 同时结合系统指标（cache hit rate、TTFT、压缩率）和 agent 成功率（工具调用准确率、任务完成率）。

来源：主站 essay [KV Cache Agent 长上下文 Benchmark](/notes/2026/04/08/kvcache-agent-long-context-benchmark/)

---

## Benchmark 使用建议

**不要只跑单点指标**。合理的评估矩阵：

| 场景 | 关键指标 | 关键变量 |
|------|----------|----------|
| 低负载单请求 | TTFT、TPOT | Prompt 长度 |
| 中等并发 | 吞吐、命中率 | 并发请求数 |
| 高负载压力 | 尾部延迟、Recompute 率 | KVCache 利用率 |
| Prefix Cache 专项 | 命中率 vs TTFT 降幅 | Prefix 共享比例 |
| Offload 专项 | 换入延迟、带宽利用率 | HBM 容量限制 |

**数据集选择：**

- ShareGPT：真实多轮对话分布，Prompt 长度分布接近生产
- 合成 Workload：控制变量，用于诊断单一因素的影响
- 私有生产 Trace：最真实，但难以公开对比

---

## 关联章节

- 各指标的定义与公式：[指标体系](evaluation-metrics.md)
- 各场景的推荐指标组合：[场景评估矩阵](evaluation-scenarios.md)

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 从评估方法总览拆分，补充工具使用示例与 Agent 评估空白分析 |
