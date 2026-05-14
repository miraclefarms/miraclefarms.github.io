# 算法 × 工作负载

> 算法优化不是通用的——稀疏、量化、位置编码的选择必须与目标工作负载的访问模式匹配，否则适得其反。

## 多轮对话适合哪类稀疏

**多轮对话的 KV 访问特征：**

- 历史轮次单调累积，早期轮次的内容重要性逐轮衰减
- 每一轮新消息到来时，最近几轮是高度相关的，远早期轮次的影响弱

**适配方案：StreamingLLM 风格**

- 保留 attention sink token（最早的几个 token，attention 权重异常高）+ 最近 K 个 token 的 KV
- 老内容自然丢弃，滑动窗口跟随对话推进
- 与 Paged KV 兼容性好：丢弃的 block 整块释放，无需压实

**不适配方案：H2O、SnapKV 等"重要性评分"稀疏**

- 重要性评分在单轮内很好（每层保留 attention score 高的 token）
- 但多轮场景的问题：**评分窗口很难定义**
    - 基于当前轮的 query 评分 → 对下一轮 query 来说，哪些 token 重要是未知的
    - 多轮后评分累积漂移，早期被"重要"保留的 token 可能对未来无关
- SCBench 研究证实：sub-O(n) 稀疏方案在多轮场景下系统性退化

**总结：**

| 稀疏方案 | 多轮适配度 | 原因 |
|---------|----------|------|
| StreamingLLM / SWA | ✅ 好 | 时间局部性自然对齐多轮衰减规律 |
| H2O / SnapKV | ⚠ 有风险 | 跨轮重要性估计不准，多轮后退化 |
| PyramidKV | ⚠ 有风险 | 同上，层间稀疏率固定不适应动态对话 |

---

## RAG 场景下位置不变性压缩 / 块级 KV 的可行性

**问题根源：RoPE 与位置的强耦合**

同一文档块在不同 prompt 中的位置（token offset）不同，导致 K/V 中的 RoPE 分量不同，精确 prefix cache 不可能在文档块级别命中。

**三个解法方向：**

### 方向一：预压缩为位置无关的 KV Summary

[Prompt Cache (2023)](https://arxiv.org/abs/2311.04934) 的路线：

- 把文档块通过一个专门的"压缩模型"转为位置无关的 KV summary
- 加载时不需要重算 RoPE，直接注入 attention
- **代价**：需要专门训练的压缩模型；summary 与原始 KV 语义不完全等价

### 方向二：加载时重算 RoPE（RoPE Recomposition）

[CacheBlend (2024)](https://arxiv.org/abs/2405.16444) 的路线：

- 预计算文档块的 KV（在某个固定的参考 offset，如 0）
- 加载时，根据当前 prompt 中的实际 offset，对 K 做 RoPE 变换校正
- **代价**：校正计算本身有开销（约为全量 prefill 的 5–15%）；近似误差对长文档可能积累

### 方向三：固定文档位置，退而求其次

最实用的工程方案：

- 把 RAG prompt 设计为固定结构：`system + doc1 + doc2 + ... + query`
- 文档按固定规则排序（如文档 ID 字母序），确保相同文档总在相同 offset
- **代价**：top-k 排名信息丢失；检索文档数量变化时 offset 改变，cache miss

**精度损失的现实：**

方向一和方向二都有 1–3 个 BLEU / EM 点的精度损失。实用前提：**业务能容忍轻微质量下降**。大多数生产 RAG 场景目前走方向三。

---

## Coding 场景下增量 KV 与跨层共享 KV 的组合

**已知的两个事实：**

1. 编辑点之前的 prefix 完全可复用——传统 prefix cache 覆盖
2. 编辑点之后必须重新 prefill——每层的 K/V 都要重算

**跨层共享 KV（CLA / YOCO）在这里的意外收益：**

[YOCO](attention-variants.md)（You Only Cache Once）和 [CLA](attention-variants.md)（Cross-Layer Attention）让相邻层（或所有层）共享同一份 K/V：

```
传统 Transformer：每层有独立的 K/V
  prefill 编辑点后的 N 层：N 次计算

CLA（层共享）：多层共用一份 K/V
  prefill 编辑点后的 N 层：1 次计算（其余层直接复用）
```

- 编辑点之前：prefix cache 命中，CLA 的收益与传统一样
- **编辑点之后：CLA 让重 prefill 的计算量从 O(N_layers) 降到 O(N_layers / sharing_ratio)**

**这是一个尚未被充分挖掘的组合方向**：Coding 工作负载（高频小幅编辑 + 超长 prefix）+ 跨层共享 KV，可能是精度和效率的好甜区。当前没有大规模生产部署案例，是值得关注的研究方向。

---

## 关联章节

- 稀疏化方法与多轮评估（SCBench）：[稀疏化](sparsity.md)、[Benchmark 与工具](evaluation-benchmarks.md)
- RAG 工作负载的工程配置：[RAG / 检索增强](workload-rag.md)
- CLA / YOCO 的算法细节：[Attention 变体](attention-variants.md)
- Coding 工作负载的 SP 配置：[Coding / 长上下文](workload-coding.md)

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 从维度交叉总览拆分，补充稀疏方案多轮适配表、RAG 三种解法方向对比、CLA+Coding 组合分析 |
