---
title: 主流 Attention 算法全景图：从 Softmax 到 MLA 的分类与对比
date: 2026-04-20 12:00:00 +0800
author: Ethan
kind: essay
category: Essay
intro: Attention 是 LLM 最核心的计算单元，过去七年围绕它产生了大量变体。本文系统梳理主流 attention 算法的分类、创新点与适用场景，帮助读者建立结构化的理解框架。
---

> **版本声明**：本文系统性梳理截至 2026 年初的主流 attention 算法，以对应论文原始贡献为依据，重点注明创新点与落地情况。

所有 attention 变体的起点都是同一个问题：**当模型处理一个序列时，如何决定当前 token 应该关注序列中的哪些其他 token？** 原点答案来自《Attention is All You Need》提出的 scaled dot-product attention，用 Q/K/V 三元组将这个问题形式化为"查询向量与所有键向量的相似度计算 + 加权求和"——即 softmax(QK^T)V，复杂度 O(N²)。此后七年的工作，基本都在从三个方向上回答这个问题的某个侧面：**谁该关注谁**（连接模式）、**计算能否更快**（效率优化）、**多头如何配置**（头部架构）。

按照这三个维度，本文将主流算法归为三大类：**连接模式类**（改变 token 之间的 attention 连接方式）、**效率优化类**（改变计算复杂度或内存占用）、**多头架构类**（改变 Q/K/V head 的组织方式）。每类先概述共同规律，再分节详述各算法的具体设计。

## 一、连接模式类：谁该关注谁

### 1.1 共性规律

所有连接模式类的 attention 变体，都不改变 Q/K/V 的基本计算方式，而是在**是否允许两个 token 之间建立 attention 连接**这件事上做约束。这类算法的核心差异在于：哪些 (query, key) 对会被允许进入 softmax，哪些会被 mask 掉或重新路由。

### 1.2 Softmax Attention（标准 Attention）

**论文**：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)（Vaswani et al., 2017, NeurIPS）

**解决的问题**：RNN/LSTM 无法并行、难以建模长距离依赖的问题。

**创新点**：提出 scaled dot-product attention，用矩阵乘法实现全局两两互注，引入 multi-head attention 捕捉不同子空间的关联关系。核心公式为：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中 $\sqrt{d_k}$ 为缩放因子，防止点积过大导致 softmax 梯度消失。

![Transformer 整体架构图](/assets/attention-algorithms-overview/fig1-transformer-architecture.png)

*图 1：Transformer 整体架构。Encoder-Decoder 结构完全基于 attention 堆叠，消除了 RNN 和卷积，奠定了现代 LLM 的基础框架。<a href="https://arxiv.org/abs/1706.03762">[1]</a>*

**实现效果**：在 WMT 2014 En→De 翻译任务上达到 28.4 BLEU（超过此前最佳 2 BLEU），En→Fr 翻译达到 41.8 BLEU。训练周期仅 3.5 天 8 GPU。

**适用场景**：几乎所有序列到序列任务；是所有后续变体的基础baseline。

**落地模型**：GPT 系列、BERT、T5、LLaMA、Claude、Gemini 等几乎所有主流 LLM。

### 1.3 Cross-Attention

**论文**：[Multi-layer Cross-Attention is Provably Optimal for Multi-modal In-context Learning](https://arxiv.org/abs/2602.04872)（Barnfield et al., 2026）

**解决的问题**：跨模态信息融合问题——当模型需要根据一个序列的信息来 attending 另一个序列时（如 image captioning、multimodal understanding），传统 self-attention 无法直接表达这种跨序列依赖。

**创新点**：将 cross-attention 描述为多模态 in-context learning 的理论框架，证明多层 cross-attention 在大 context 场景下可达到 Bayes 最优。cross-attention 使用来自一个模块的 query 和来自另一个模块的 key-value 对，从而使信息在不同模态之间流动。

**实现效果**：在多模态 in-context learning 任务上，多层 cross-attention 显著优于单层 self-attention。

**适用场景**：多模态理解任务（图像描述、视觉问答）、 seq2seq 解码时 attending 到 encoder 输出。

**落地模型**：Perceiver、Emu、Flamingo、Gemini 的视觉编码部分。

### 1.4 Causal Attention

**解决的问题**：自回归生成模型中，当前 token 只能看到历史 token，不能"偷看"未来答案。

**创新点**：通过 causal mask（上三角遮罩）强制每个位置只能 attend 到当前及之前的位置，使 attention 过程严格满足自回归约束。这是语言模型 decoder 的标配机制。

**升级版本**：CAST（2024）提出带 lookahead keys 的因果注意力，允许 token 从后续 token 中获取隐藏信息，并在处理过程中动态更新 key。

**适用场景**：所有自回归语言模型的核心位置；推理时控制 token 生成顺序。

**落地模型**：GPT-2、LLaMA、Claude 等所有自回归 LLM。

---

## 二、效率优化类：让 attention 跑得更快

### 2.1 共性规律

效率优化类变体的共同目标是打破 O(N²) 的时间和内存复杂度瓶颈，手段可以归结为两类：**稀疏化**（不是所有 Q-K 对都要计算）和**核优化**（不改变数学等价性，但改变数据访问模式以利用硬件特性）。这两类并不互斥，实际系统往往同时采用。

### 2.2 Linear Attention

**论文**：[Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236)（Katharopoulos et al., 2020, ICML）

**解决的问题**：标准 attention 的 O(N²) 复杂度在长序列上成为瓶颈，RNN 虽然线性但表达能力受限。

**创新点**：将 self-attention 重写为线性核函数形式：使用特征映射 φ(·) 将 Q/K 映射到高维空间，利用矩阵乘法的结合律将计算顺序从 O(N²) 降至 O(N)。核心思想是将 softmax 的指数运算用核函数近似，从而把完整的相似度矩阵拆解成线性递归。

公式上将 attention 表示为：
$$\text{Attention}(x_t, X_{1:t}) = \frac{\sum_{i=1}^{t} \kappa(x_t, x_i) v_i}{\sum_{i=1}^{t} \kappa(x_t, x_i)}$$

其中 κ 是核函数。最终可转化为等价的 RNN 形式，实现 O(N) 增量计算。

**实现效果**：在长序列（4K+ tokens） autoregressive prediction 上比标准 transformer 快 **4000 倍**，且 perplexity 接近 vanilla transformer。

**适用场景**：超长序列建模（音乐生成、蛋白质序列）、需要替代 RNN 的增量推理场景。

**落地模型**：并行框架如 `linear-transformers` 库；Performer（Google）采用了类似随机投影的线性 attention。Reformer、 locality-sensitive hashing（LSH） attention 也属于稀疏 attention 的工程变种。

### 2.3 Sparse Attention（滑动窗口 + 全局注意力）

**论文**：[Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509)（Child et al., 2019）；[Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062)（Zaheer et al., 2020, NeurIPS）

**解决的问题**：全量 QK 计算在长序列上内存和计算量爆炸。

**创新点**：引入稀疏注意力模式——并非每个 token attend 所有其他 token，而是定义两类稀疏模式：

- **Sliding Window（局部注意力）**：每个 token 只 attend 窗口内的 w 个邻近 token，复杂度降为 O(Nw)。
- **Global Attention**：指定若干全局 token（如 [CLS]）attend 所有位置，所有 token 也 attend 这些全局 token，用于汇总信息。

Big Bird 在此基础上增加 **random attention**（随机连通），并从理论上证明稀疏注意力是通用逼近器，仍保持 Turing 等价性。

**实现效果**：Big Bird 在相同硬件上可将上下文长度提升至原来的 **8 倍**，在 question answering 和 summarization 任务上显著优于全量 attention。

**适用场景**：长上下文建模（基因组数据、超长文档）、需要线性或近线性 scaling 的场景。

**落地模型**：Longformer（AI2）、BigBird（Google）、SGMT 模型；后来 LLM 的长上下文版本多采用滑动窗口 + 全局 token 的混合模式。

### 2.4 FlashAttention

**论文**：[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)（Dao et al., 2022）

**解决的问题**：标准 attention 需要实例化完整的 N×N 注意力矩阵，内存 O(N²)，在长序列上成为硬件瓶颈。现有近似方法虽减少计算但往往无法获得 wall-clock 加速。

**创新点**：IO-aware exact attention：通过 **tiling（分块计算）** 将 attention 分为小块处理，每块保持在 GPU on-chip SRAM 中，只在最终写回时与 HBM 交互。以此减少 GPU HBM 带宽压力——这才是实际 bottlenecks 的根源。数学上完全等价于标准 softmax attention，无精度损失。

**实现效果**：
- BERT-large（seq=512）：比 MLPerf 1.1 训练速度记录快 **15%**
- GPT-2（seq=1K）：**3 倍**加速
- Long-range Arena（seq=1K-4K）：**2.4 倍**加速
- 首次在 Path-X（16K）和 Path-256（64K）上达到高于随机的准确率

**适用场景**：几乎所有 LLM 训练和推理场景；已成为主流框架（vLLM、TGI、SGLang）的底层加速基础。

**落地模型**：GPT-2、LLaMA、BERT 所有主流 LLM 的训练均受益；vLLM 的 paged attention 即基于 FlashAttention 思想。

---

## 三、多头架构类：Q/K/V Head 的组织方式

### 3.1 共性规律

多头架构类变体的核心洞察是：**Q/K/V head 的数量配置本身就是一组可调参数**。标准 MHA 中 H 个 query head 对应 H 个 key head 和 H 个 value head，三者数量永远相等。改变这个对称关系，会在"表达能力"和"推理效率"之间产生不同的折中。

### 3.2 Multi-Head Attention（MHA）

**论文**：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)（Vaswani et al., 2017）

**解决的问题**：单一 attention head 只能建模一种类型的 token 关系，多头机制使模型能在不同子空间同时学习不同的关联模式。

**创新点**：将 d_model 分成 H 个 d_k 维子空间，各自独立执行 attention 后 concatenate 再线性映射。这种设计让每个 head 能专注于捕捉不同类型的依赖（如句法、语义、位置相关）。

![Multi-Head Attention 架构图](/assets/attention-algorithms-overview/fig2-multi-head-attention.png)

*图 2：Multi-Head Attention 的核心结构。多个 attention head 并行独立计算各自子空间内的注意力，最后 concatenate 输出再经线性映射整合。<a href="https://arxiv.org/abs/1706.03762">[1]</a>*

**适用场景**：通用序列建模；是所有现代 LLM 的基础组件。

**落地模型**：原生 Transformer（BERT、T5）、GPT 系列、LLaMA 等。

### 3.3 Multi-Query Attention（MQA）

**论文**：[Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)（Shazeer, 2019）

**解决的问题**：增量推理（autoregressive decoding）时，每次生成新 token 都需要重新加载巨大的 K/V tensor，内存带宽成为主要瓶颈，导致 decode 速度极慢。

**创新点**：所有 Q head 共享同一组 key-value pair。K/V tensor 从 H×d_k 压缩到 1×d_k，大幅减少加载量。理论依据是"不同 head 学到的东西可以共享 K/V 表示，query 的多样性已足够提供表达能力"。

**实现效果**：decode 速度提升约 2-3 倍；质量略有下降但幅度可控。

**适用场景**：推理阶段对延迟敏感、batch size 较小的场景。

**落地模型**：PaLM（Google）在部分层使用；后续被更实用的 GQA 所替代。

### 3.4 Grouped-Query Attention（GQA）

**论文**：[GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)（Ainslie et al., 2023, EMNLP）

**解决的问题**：MQA 虽然快但质量损失明显；如果有现成的 MHA 模型，希望无需从头训练就能获得 MQA 的推理速度。

**创新点**：引入中间数量 G 组 KV heads（GQA），其中 G 小于 H 但大于 1，从而在 MHA 质量和 MQA 速度之间取得平衡。论文同时提出一种 uptraining 策略：从 MHA checkpoint 仅用 5% 原预训练计算量即可完成到 GQA 的转换。

**实现效果**：uptrained GQA 达到接近 MHA 的质量，同时推理速度与 MQA 相当。

**适用场景**：几乎所有现代 LLM 推理加速的首选方案。

**落地模型**：LLaMA 2/3、Mistral、DeepSeek 系列、Qwen 2、Command R+ 等主流模型均采用 GQA。

![GQA checkpoint 转换流程](/assets/attention-algorithms-overview/fig3-gqa-recycling.png)

*图 2：GQA 中将 MHA checkpoint 转换为 MQA/GQA 的流程。Key 和 Value 投影矩阵从所有 head 做 mean pooling 合入单一 head，实现低成本的架构迁移。<a href="https://arxiv.org/abs/2305.13245">[4]</a>*

![MHA / MQA / GQA 架构对比](/assets/attention-algorithms-overview/fig4-gqa-architecture.png)

*图 3：MHA、MQA 与 GQA 的架构对比。GQA 通过 KV heads 分组，在 MHA 的表达能力和 MQA 的推理速度之间取得平衡。<a href="https://arxiv.org/abs/2305.13245">[4]</a>*

### 3.5 Multi-Head Latent Attention（MLA）

**论文**：[DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)（DeepSeek-AI, 2024）

**解决的问题**：GQA 仍需要逐组存储 KV cache，LLM 在长上下文场景下的 KV cache 显存仍是瓶颈；DeepSeek-V2 希望通过更低精度的 latent 向量压缩 KV cache。

**创新点**：将 KV cache 压缩到低秩 latent 向量空间。具体地，MLA 对 key 和 value 均使用低秩联合压缩：
$$k^{LC}_i = W^{DK}k_i, \quad v^{LC}_i = W^{DV}v_i$$
推理时通过解码恢复。MLA 使得 KV cache 显著减小，同时通过 RoPE 位置编码保持位置信息。

**实现效果**：
- KV cache 减少 **93.3%**
- 训练成本节省 **42.5%**
- 生成吞吐量提升 **5.76 倍**
- 质量与 DeepSeek 67B 相当甚至更优

**适用场景**：超长上下文 LLM（128K）、MoE 架构的高效推理。

**落地模型**：DeepSeek-V2、DeepSeek-V2.5。

![DeepSeek-V2 整体架构图](/assets/attention-algorithms-overview/fig6-deepseekv2-arch.png)

*图 4：DeepSeek-V2 整体架构。MLA 通过低秩 Key-Value 联合压缩大幅减少 KV cache 显存占用，DeepSeekMoE 以稀疏计算降低训练成本。<a href="https://arxiv.org/abs/2405.04434">[5]</a>*

### 3.6 Interleaved Head Attention（IHA）

**论文**：[Interleaved Head Attention](https://arxiv.org/abs/2602.21371)（Duvvuri et al., 2026）

**解决的问题**：标准 MHA 中 H 个 head 产生 H 个独立 attention 矩阵，head 之间无信息交换，导致多步推理时无法有效聚合来自多个位置的证据。

**创新点**：通过 pseudo-heads 实现 cross-head mixing。具体地，将 P 个 pseudo-head（每 head 由所有 H 个原始 heads 的 Q/K/V 线性组合构成）引入 attention 过程。pseudo-head 之间交互可产生最多 P² 种 attention pattern，显著增加了表达多样性，同时参数量仅 O(H²P)。

![IHA pseudo-head 架构图](/assets/attention-algorithms-overview/fig8-iha-architecture.png)

*图 5：IHA 架构。Pseudo-head 通过原始 heads 的线性组合构建，head 之间交互产生更多 attention pattern，提升多步推理能力。<a href="https://arxiv.org/abs/2602.21371">[9]</a>*

**实现效果**：
- RULER 多键检索任务（4K-16K）：比全量 attention 提升 **10-20%**
- GSM8K：提升 **5.8%**；MATH-500（Majority Vote）：提升 **2.8%**

**适用场景**：复杂多步推理任务、对 chain-of-thought 质量有较高要求的场景。

**落地模型**：尚处于研究阶段，尚未见主流大模型采用。

---

## 四、各维度综合对比

| 算法 | 时间复杂度 | 空间复杂度 | 精度 | 主要创新 | 适用场景 | 代表模型 |
|------|-----------|-----------|------|---------|---------|---------|
| Softmax Attention | O(N²) | O(N²) | 精确 | 全量两两互注，scaled dot-product | 通用序列建模 | GPT, BERT, T5 |
| Cross-Attention | O(N·M) | O(N·M) | 精确 | 跨序列 attending | 多模态理解 | Flamingo, Gemini |
| Causal Attention | O(N²) | O(N²) | 精确 | causal mask，自回归约束 | 自回归生成 | GPT-2, LLaMA |
| Linear Attention | **O(N)** | O(N) | 近似 | 核函数 + 矩阵结合律 | 超长序列 | Performer, linear-transformers |
| Sparse Attention | O(N·w) / O(N√N) | 稀疏 | 精确或近似 | 选择性连接 | 长上下文 | Longformer, BigBird |
| FlashAttention | O(N²) | **O(N)** | **精确** | IO-aware tiling, 分块计算 | 训练加速，推理加速 | 几乎所有现代 LLM |
| MHA | O(N²)·H | O(N²)·H | 精确 | 多子空间并行学习 | 通用 | 原生 Transformer |
| MQA | O(N²) | **O(N)** | 轻微下降 | KV heads 共享 | 低延迟推理 | PaLM (部分层) |
| GQA | O(N²) | O(N·G) | 接近 MHA | KV heads 分组 | **实际落地最多** | LLaMA 2/3, Mistral, Qwen 2 |
| MLA | O(N²) | **O(N) 低秩压缩** | 接近 MHA | KV latent 压缩 | 超长上下文，MoE | DeepSeek-V2 |
| IHA | O(N²)·P | O(N²)·P | 持平或更好 | pseudo-head cross mixing | 多步推理 | 研究阶段 |

---

> **参考资料**
>
> [1] Vaswani et al. "[Attention Is All You Need](https://arxiv.org/abs/1706.03762)", NeurIPS 2017
>
> [2] Dao et al. "[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)", 2022
>
> [3] Shazeer. "[Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)", 2019
>
> [4] Ainslie et al. "[GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)", EMNLP 2023
>
> [5] DeepSeek-AI. "[DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model](https://arxiv.org/abs/2405.04434)", 2024
>
> [6] Katharopoulos et al. "[Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236)", ICML 2020
>
> [7] Child et al. "[Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509)", 2019
>
> [8] Zaheer et al. "[Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062)", NeurIPS 2020
>
> [9] Duvvuri et al. "[Interleaved Head Attention](https://arxiv.org/abs/2602.21371)", 2026
>
> [10] Barnfield et al. "[Multi-layer Cross-Attention is Provably Optimal for Multi-modal In-context Learning](https://arxiv.org/abs/2602.04872)", 2026
