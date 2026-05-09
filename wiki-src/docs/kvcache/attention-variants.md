# Attention 变体

> **算法维度（横向）之一**：KV 张量本身的形状如何被 attention 结构改写。

KVCache 的体积、读带宽、是否还存在"可缓存"语义，几乎完全由 attention 结构决定。本章梳理对 KV 影响最大的几条算法路线。

## 1. MHA → MQA → GQA：KV 头数的压缩谱

**Multi-Head Attention（MHA）**——每个 Q head 配独立的 K/V head。
$\text{KV size} \propto H_q$，KVCache 最大。

**Multi-Query Attention（MQA）**——所有 Q head 共享单组 K/V。
$\text{KV size} \propto 1$，KVCache 最小。代价：表达力下降，长上下文质量受影响。

**Grouped-Query Attention（GQA）**——折中方案，把 Q head 分成 $G$ 组，每组共享一组 K/V。
$\text{KV size} \propto G$。

```
MHA:  Q1 Q2 Q3 Q4 ... Qn        ← n 个 K/V head
MQA:  Q1 Q2 Q3 Q4 ... Qn        ← 1 个 K/V head
GQA:  [Q1 Q2 Q3 Q4][...][...]   ← G 个 K/V head（如 LLaMA-3 70B 用 8）
```

**对系统的影响**：

- KVCache 体积按倍数下降，可服务的并发请求数同比例上升
- Decode 阶段 HBM 读带宽压力等比例下降
- LLaMA-3、Mistral、Qwen 等主流开源模型已普遍采用 GQA(8)

代表论文：[GQA (2023)](https://arxiv.org/abs/2305.13245)、[MQA (2019)](https://arxiv.org/abs/1911.02150)。

## 2. MLA（Multi-head Latent Attention）

DeepSeek-V2/V3 的路线：将 K/V 投影到一个**低秩潜空间**存储，attention 计算时再"解压"。

核心机制：

- 缓存的不是 K/V 本身，而是一个低维 latent 向量 $c \in \mathbb{R}^{d_c}$（$d_c$ 远小于 $H_{kv} \times d$）
- 计算时通过两个 down/up projection 恢复出有效 K/V
- 配合 RoPE 的 decoupled 设计避免位置编码干扰潜空间

**对 KVCache 的影响**：

- KV 体积可压缩到 GQA 的 1/4 ~ 1/10
- 但 attention kernel 必须改写——传统的 paged attention 直接读写 K/V，MLA 要在读取时插入解压计算
- **prefix cache 的语义还在不在**？理论上 latent 向量也可以按 token 序列哈希复用，但具体引擎支持参差不齐

代表论文：[DeepSeek-V2 (2024)](https://arxiv.org/abs/2405.04434)。

## 3. 线性注意力 / 状态空间模型：从"长 KV"到"恒定状态"

这一类的共同特征：**没有传统意义上的 KVCache**，取而代之的是一个固定大小的状态向量。

| 模型族 | 代表 | 关键性质 |
|--------|------|----------|
| 线性注意力 | RetNet、RWKV | softmax 替换为可拆分的核函数，状态可递推 |
| 状态空间模型 | Mamba、Mamba2 | 用 SSM 离散化建模序列，状态恒定大小 |
| 混合 RNN | Hyena、StripedHyena | 长序列友好的全局算子 |

**对系统的影响**：

- 推理时显存占用与序列长度**无关**，长上下文场景显著友好
- 但 prefix cache、跨请求复用等基于"K/V 是关于 token 序列的可拼接结果"的假设全部失效
- decode 不再 memory-bound 于 KV 读取，瓶颈转移到状态更新本身

注意：这些方案在工业部署上还远不主流，主要原因是质量和兼容性（与现有 attention 优化栈、speculative decoding 等的协作）。

## 4. 滑动窗口 / 局部注意力

**Sliding Window Attention（SWA）**——每个 token 只关注最近的 $w$ 个 token，超出窗口的 K/V 可丢弃。

代表：Mistral、Gemma、Phi 系列。

**对 KVCache 的影响**：

- KV 总量有上界 $O(w)$，与上下文长度解耦
- 但"窗口外的 token 完全不可见"过于激进，许多模型采用**全局-局部混合**：少数几层 full attention 维持全局信息，其余层 SWA

**Attention sink** 的发现（[StreamingLLM](https://arxiv.org/abs/2309.17453)）补上了 SWA 的一个工程问题：模型在前几个 token 上集中了大量 attention 概率，简单丢弃会破坏分布。实践方案是**始终保留前 $k$ 个 token 的 K/V + 滑动窗口**。

## 5. 混合架构

更激进的做法：**部分层用全局 attention、部分层用线性 / 局部注意力**。

代表：

- **Jamba**——Transformer + Mamba 混合
- **Hymba / MiniMax-Text-01**——分层混合，KVCache 只在部分层产生
- **Gemma 3**——SWA 与 full attention 交替

**对系统的影响**：

- KVCache 总量按"全局层数"计而非"总层数"，可下降 70%+
- 但调度器需要**层级感知**：哪些层有 KV、哪些没有，块管理逻辑更复杂
- 与 PD 分离、跨节点 KV 传输的协议都要适配新的 KV 形态

## 6. 算法维度的统一视角

把上面的所有变体抽象一下，attention 维度对 KVCache 做的事可以归为三类：

| 类别 | 操作 | 代表 |
|------|------|------|
| **改形状** | 把 KV 张量本身的维度改小 | MQA、GQA、MLA |
| **减条目** | 在 token 维度上去掉一些 K/V | SWA、StreamingLLM |
| **改语义** | 把"K/V 序列"换成另一种状态 | Mamba、RWKV |

它们在系统层面的共同效应是：**减少 KV 的"有效字节数"**——但代价分布不同：

| 方法 | 精度风险 | kernel 兼容性 | prefix cache 友好度 |
|------|---------|---------------|---------------------|
| GQA | 低 | 高（与现有 paged attention 直接兼容） | 高 |
| MLA | 中（解压精度） | 低（需要专门 kernel） | 中（需要适配） |
| SWA | 中（长程信息） | 中 | 低（窗口滑动破坏 prefix） |
| Mamba | 高（与 Transformer 行为差异大） | 不兼容 | 不适用 |

## 关联章节

- 与稀疏化（[Sparsity](sparsity.md)）的边界：本章是"模型本身就这样设计"，稀疏化是"训练好的模型在推理时丢一部分 KV"
- 与压缩量化（[Compression & Quantization](compression-quantization.md)）的边界：本章改 KV 的"形状/语义"，压缩量化改 KV 的"位宽/秩"
- 在 [维度交叉 §3.1](crossings.md) 中讨论 attention 变体与 paged 内存管理、prefix cache 的兼容性
