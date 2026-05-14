# KV 稀疏化

> **算法维度（横向）之二**：在已经训练好的模型上，推理时**丢掉一部分 KV**，以降低显存占用和读带宽压力。

与 [Attention 变体](attention-variants.md) 的区别：那里改的是模型本身的结构，本章讨论的是**给定模型不变**，在推理阶段动态决定哪些 K/V 可以扔掉。

## 1. 为什么可以丢

经验观察：attention 分布是**高度稀疏的**——大部分 query 对大部分 key 的 attention 权重接近 0。如果能识别出"对未来 decode 几乎不影响"的 K/V，丢掉它们对最终输出影响有限。

剪枝粒度可以分四类：

| 粒度 | 操作 | 代表方法 |
|------|------|----------|
| **Token 级** | 丢掉序列中某些 token 的 K/V | StreamingLLM、H2O、SnapKV、Scissorhands |
| **Head 级** | 不同 head 重要性不同，丢掉次要 head 的 K/V | Ada-KV、MoA |
| **Layer 级** | 跨层共享 K/V 或跳过部分层的缓存 | YOCO、CLA |
| **基于 query 的动态稀疏** | 每个 decode 步只读一小部分相关 K/V | Quest、Loki、RetrievalAttention |

## 2. Token 级稀疏

### Attention sink + 滑动窗口（StreamingLLM）

观察：模型在前 $k$ 个 token（"attention sink"）上集中了大量 attention 概率，简单丢弃会让分布塌陷。

策略：**永久保留前 $k$ 个 token + 滑动窗口最近 $w$ 个 token**，中间的全部丢掉。
代价：长程依赖可能丢失，适合"短期局部依赖为主"的对话场景。

### 重要性评分驱动（H2O / Heavy-Hitter Oracle）

观察：少数 token（"heavy hitters"）在 attention 中被频繁高分关注，它们的 K/V 不能丢。

策略：用累积 attention 分数为每个 token 打分，**保留 top-K 个高分 token**，其余驱逐。
工程难点：

- 评分需要在线维护，不能太重
- 与 paged 内存管理的整合：被剪掉的 K/V 释放后，块复用策略要重新设计
- 长序列下评分窗口的滑动逻辑

代表论文：[H2O (2023)](https://arxiv.org/abs/2306.14048)。

### 提示驱动（SnapKV）

观察：在 prefill 完成时，可以从最后几个 prefill token 的 attention 模式**预测**整个 prompt 中哪些 token 会被 decode 频繁关注。

策略：prefill 结束时一次性筛掉不重要的 K/V，进入 decode 时 KV 已是"瘦身版"。
优点：实现简单，与 paged attention 兼容性好。
代价：依赖 prefill 末尾的 attention 是否真能反映 decode 期间的需求——RAG / Coding 等场景可能不准。

代表论文：[SnapKV (2024)](https://arxiv.org/abs/2404.14469)。

## 3. Head 级稀疏

观察：同一层不同 attention head 的"信息容量"差异巨大，部分 head 高度重复或承担"分发"角色。

策略：

- 离线：用校准集评估每个 head 的重要性，对低重要性的 head 进行更激进的压缩或丢弃
- 在线：保留 head 数量随上下文长度自适应变化

代价：与 GQA / MLA 的"绑定"——head 已经被压缩过的模型上，进一步 head 级剪枝空间有限。

代表论文：[Ada-KV (2024)](https://arxiv.org/abs/2407.11550)。

## 4. Layer 级稀疏

### YOCO（You Only Cache Once）

策略：只在**少数几层**真正缓存 K/V，其余层共享或重计算。
效果：KVCache 体积可下降一个数量级。
代价：需要从训练阶段就考虑这一架构（属于"半改架构、半改运行时"）。

代表论文：[YOCO (2024)](https://arxiv.org/abs/2405.05254)、[CLA (Cross-Layer Attention)](https://arxiv.org/abs/2405.12981)。

## 5. 基于 query 的动态稀疏

前面几类是**与 query 无关**的剪枝——一次决定哪些 K/V 留下。Query-aware 的方法每个 decode 步都重新选 K/V：

### Quest

每个 decode 步：

1. 把 KV 划成块，为每块计算一个紧凑的"摘要"（min/max/平均向量）
2. 用 query 与摘要做粗筛，选出 top-$k$ 个块
3. 只对选中的块做完整 attention

效果：每步只读 $k$ 个块的 K/V，HBM 带宽压力大幅下降。
难点：摘要质量、top-$k$ 阈值、块大小选择。

代表论文：[Quest (2024)](https://arxiv.org/abs/2406.10774)、[Loki](https://arxiv.org/abs/2406.02542)、[RetrievalAttention](https://arxiv.org/abs/2409.10516)。

## 6. 稀疏化在系统层面的代价

理论上稀疏可以让 KV 变小，但工程上有几个常见陷阱：

| 陷阱 | 描述 |
|------|------|
| **与 paged 内存管理的不兼容** | 被剪掉的 K/V 在块内位置不连续，要么留空（碎片）要么压实（额外搬运） |
| **prefix cache 的语义破坏** | 跨请求 prefix 相同但稀疏决策不同 → 物理 KV 不一致，无法复用 |
| **kernel 改写成本** | 稀疏 attention 需要 indirection，kernel 复杂度上升、效率下降 |
| **质量回归不易评估** | 短任务上看不出差异，长任务、多轮、reasoning 上才暴露问题 |

工业落地里 SWA + attention sink 是最稳的，token 级重要性剪枝在特定场景部署，layer/head 级和 query-aware 还多停留在论文阶段。

## 关联章节

- 与 [Attention 变体](attention-variants.md) 的关系：算法维度的"减条目" vs "改形状"
- 与 [Paged KV](paged-kv.md) 的兼容性问题在 [维度交叉 §3.1](crossings.md) 详细讨论
- 与不同工作负载的匹配度：[工作负载维度](workloads.md) 中讨论各场景适合哪类稀疏

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 框架搭建 |
