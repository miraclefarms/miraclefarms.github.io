---
title: Forcing-KV：把 LLM 的 KV cache 压缩思路搬进视频生成，值不值？
date: 2026-05-14 12:00:00 +0800
author: Ethan
kind: reading
category: Reading
intro: 解读 Forcing-KV 对 autoregressive 视频扩散模型中 attention head 功能特化的实证发现，以及基于此提出的静态裁剪 + 动态相似度压缩的 hybrid KV cache 压缩方案。
tags: [KV Cache, Inference, Multimodal]
---

AR 视频扩散模型正在走 LLM 两年前走过的路——KV cache 随生成长度线性膨胀，30 秒 1080P 视频的 cache 独占 60 GB 以上显存。LLM 社区已经把 KV cache 压缩翻来覆去研究了三轮，从 StreamingLLM 的 attention sink 到 H2O 的 heavy-hitter eviction，再到 KIVI 的量化方案。但 Forcing-KV 这篇工作的真正贡献，不是把 LLM 的压缩套件搬进视频生成——而是发现了 AR 视频扩散模型的 attention head 有一套稳定的功能分化体系，让压缩策略可以从"每个 head 砍同样的东西"变成"每个 head 只留它真正需要的东西"。

这篇 reading 对这篇论文的定位是：观察值得被记住，压缩方法是观察的自然推演，但距离"在生产里无痛落地"还有一段工程距离。

## 一、为什么 AR 视频生成的 KV cache 是一个独特问题

LLM 的 KV cache 膨胀你已经很熟了——长上下文推理中，cache 大小随 token 数线性增长，attention 计算随序列长度平方增长。FlashAttention 把显存问题压下去了，但 cache 本身只是从 HBM 搬到了计算流里，占用没有消失。

AR 视频扩散模型的处境更糟糕。它逐 chunk 生成视频帧，每个新 chunk 都要通过 self-attention 读取全部历史帧的 KV cache。以 Self Forcing<a href="https://arxiv.org/abs/2510.06430">[1]</a> 为例，30 秒 1080P 视频在单张 H200 上的生成速度只有 1.71 FPS，其中 KV cache 独占 60 GB 以上。这和 LLM 场景的差异在于：视频帧之间有大量空间冗余，同一个背景区域在数十帧里几乎没有变化，但每一帧的 latent token 仍然老老实实地参与 attention 计算。

LLM 社区已经试过很多补救方案——StreamingLLM<a href="https://arxiv.org/abs/2309.17453">[2]</a> 只保留 attention sink 和最近的 token，H2O<a href="https://arxiv.org/abs/2306.14048">[3]</a> 按 attention mass 做选择性 eviction，DuoAttention<a href="https://arxiv.org/abs/2410.10819">[4]</a> 把 head 分成 retrieval 和 streaming 两类分别处理。视频生成社区也有相关尝试：Light Forcing<a href="https://arxiv.org/abs/2602.04789">[5]</a> 做稀疏 attention，Flow Caching<a href="https://arxiv.org/abs/2602.02112">[6]</a> 做特征缓存，Dummy Forcing<a href="https://arxiv.org/abs/2601.20499">[7]</a> 发现部分 head 只看当前 chunk——但 Dummy Forcing 的压缩方式过于激进，剪掉了对 chunk 过渡至关重要的 transition anchor frame，导致 chunk 边界出现可感知的闪烁和断裂。

Forcing-KV 在这里踩对了点。它没有直接进入"怎么剪"，而是先问了一个前置问题：AR 视频扩散模型的 attention head 在 KV cache 上到底有什么利用模式？

## 二、核心发现：Static 与 Dynamic head 的分工

Forcing-KV 在 Self Forcing、LongLive<a href="https://arxiv.org/abs/2603.05606">[8]</a>、SkyReels-V2<a href="https://arxiv.org/abs/2504.13074">[9]</a>、Wan<a href="https://arxiv.org/abs/2503.20314">[10]</a> 四个模型上做了全面的 attention map 可视化，发现了一个跨模型稳定的模式：head 可以分成两类。

![Static 和 Dynamic head 的 attention 模式对比](/assets/forcing-kv-hybrid-cache-compression-video-diffusion/fig-1-head-patterns.png)

*图 1：AR 视频扩散模型的 attention head 模式。Static head 集中在当前 chunk 和 transition anchor frame 上，呈块状模式；Dynamic head 对角捕捉不同帧中同一空间位置的关联，呈条纹状。来源：论文 Figure 2。*

Static head 的 attention 集中在两个区域：当前生成的 chunk 和历史 cache 中最近一帧（论文称之为 transition anchor frame）。这个模式非常稳定，不随 prompt 或生成内容变化。它的角色是维护 chunk 间的视觉连续性和 chunk 内的帧结构——类似于视频的"脚手架"。

Dynamic head 则呈现对角条纹状 attention，间隔固定（因为每 chunk 的帧数和每帧的 token 数是固定的，同一空间位置在不同帧中落在相同的 stride 上）。它负责跨帧物体跟踪、运动动态、主体一致性。

这个发现本身不算颠覆——双向视频扩散模型中也观察到了类似的空间-时间 head 分化。但 AR 场景里的差异很关键：

1. **Transition anchor frame 是 AR 特有的依赖**。AR 模型逐 chunk 生成，chunk 间的平滑过渡高度依赖最近一帧的 attention。mask 掉这一帧对 static head 的影响远大于 mask 掉任意其他历史帧。
2. **观察的稳定性提供了压缩的基础**。通过 100 个 prompt × 4 个 denoising step 的 PCA 分析，每个 head 的特征在不同样本和步数下形成紧密聚集的簇，平均 intra-head divergence 只有 0.16，而 inter-head divergence 是 0.83。

![Ablation 验证与 PCA 稳定性分析](/assets/forcing-kv-hybrid-cache-compression-video-diffusion/fig-2-ablation-pca.png)

*图 2：左侧(a-c)逐步遮罩 dynamic head 的上下文，dynamic degree 和 consistency 逐渐下降，但对 static head 无影响；遮罩 transition anchor frame 则使 static head 的 chunk discontinuity 急剧上升。右侧(e)PCA 显示 head 特征在不同样本和 denoising step 下高度聚集，分类稳定。来源：论文 Figure 3。*

第二个发现的工程含义很直白：如果 head 分类是稳定的，就可以在推理前一次性完成 profiling，不用每一步都重新判断。

## 三、方法：把观察变成分治策略

Forcing-KV 的压缩方案可以理解为"观察的工程化"。

Head profiling 使用一个简洁的指标：static attention mass 占比。对每个 head，计算落在当前 chunk 和 transition anchor frame 上的 attention mass 占总 attention mass（去掉 sink frame）的比例。超过阈值 α 即判定为 static head。这个 profiling 一次 prompt 就能完成，且对 α 不敏感（α 从 0.8 降到 0.5 只导致 dynamic degree 轻微下降）。

Static structural pruning 对 static head 只保留 sink frame、transition anchor frame 和当前 chunk。所有更远的历史帧被直接丢弃。理由很直接：static head 不读远距离上下文，保留了也是浪费。

Dynamic similarity pruning 对 dynamic head 要精细得多。每帧被分成多个 segment（空间 block），对相邻帧的对应 segment 计算 key state 的 cosine similarity。相似度越高说明该 segment 在帧间变化越小（典型如静态背景），优先被 evict；相似度低的 segment（如运动物体所在区域）被保留。

这里有一个工程上务实的设计：segment 的相似度只用了 transformer 第一层的 key state 做 proxy，避免了所有层 key state 的重计算开销。

![Forcing-KV 方法概览](/assets/forcing-kv-hybrid-cache-compression-video-diffusion/fig-4-method-overview.png)

*图 4：Forcing-KV 整体流程。离线 head profiling 将 head 分为 Static 和 Dynamic 两类；推理时 static head 保留 transition anchor frame 和当前 chunk 的结构化内容，dynamic head 依 segment 相似度自适应裁减。来源：论文 Figure 4。*

## 四、实验：质量保住了，但加速没那么夸张

Forcing-KV 在 Self Forcing 和 LongLive 两个模型上做了全面评测，覆盖 5 秒、30 秒、60 秒视频。

质量端：在保持生成质量的同时，dynamic degree 显著好于 Dummy Forcing——LongLive 60 秒场景下 dynamic degree 是 43.56 vs Dummy Forcing 的 26.02，接近 full KV cache 的 42.40。chunk discontinuity 在 2.5，基本等同于 full KV cache 的 2.6。user study 里 Forcing-KV 和 full KV cache 的偏好比率是 45.0% vs 50.0%，而 Dummy Forcing 是 5.0%。

效率端：480P 分辨率下，Self Forcing 上 1.50× 加速，LongLive 上 1.35× 加速，cache 显存减少约 30%。论文也展示了一个更乐观的趋势：当分辨率和 attention window 增大时，加速比随之上升，1080P 场景达到 2.82×。

![Scaling law：attention window 和分辨率增大时加速比随之上升](/assets/forcing-kv-hybrid-cache-compression-video-diffusion/fig-3-scaling-law.png)

*图 5：Self Forcing 上 Forcing-KV 随 attention window 和分辨率变化的加速趋势。分辨率越高、window 越大，压缩的收益越明显，从 1.40× 增长到 2.82×。来源：论文 Figure 5。*

这个 scaling 趋势直观：KV cache 大小随窗口长度和分辨率线性增长，attention 计算成本平方级增长，因此基础开销越大，压缩的相对收益越高。论文声称 1080P 下能拿到 2.82×——这个数字在高分辨率实时视频生成的语境下确实有分量。

但有两个数字值得注意。第一个是 speedup 的计算口径：论文测量的是 DiT 内的 FPS，不考虑 VAE decode、采样等其他开销。端到端的加速会比这个数字低。第二个是 dynamic head 在裁减后仍需保留 full historical frames 用于 similarity 计算，这让"30% cache 减少"的实际节省比看起来更复杂——相似的帧虽然被 evict 了 attention 计算，但 key state 在第一步计算 similarity 时仍需访问。

## 五、这篇论文没有回答的问题

作为一个训练无关（training-free）的方法，Forcing-KV 是在已有模型上做后处理。这带来几个边界：

第一是观察的泛化性。论文承认，目前观察到的 head 分化模式主要来自 Self Forcing 体系下的模型。这类模型训练时通过 self-rollout 来缩小 train-test gap，attention pattern 的形成可能和这个训练范式有关。如果未来的 AR 视频模型采用不同训练策略（比如 GAN-based 或 flow matching），head 分化模式是否会持续是一个悬而未决的问题。

第二是压缩比的天花板。Static head 的剪枝接近极限——只保留 transition anchor frame 和当前 chunk，再剪就会破坏 chunk 过渡连续性。Dynamic head 的压缩空间更大，但 segment-wise similarity 能压缩到什么程度仍取决于视频内容的动态性：一个高速运动的体育场景和一个静态对话场景，可用压缩比差异很大。论文没有提供 per-scene 的分析，而这对实际部署中的负载均衡很重要。

第三是训练时压缩的可能性。作为 training-free 方法，Forcing-KV 的优势是不需要 re-training，劣势是压缩策略无法被模型学习适应。论文也提到，训练阶段引入 KV cache 压缩（让模型在受限上下文中学习）是一个值得探索的方向，这可能释放更大的压缩空间。

还有一个让这篇论文更有说服力的方向是真正在生产系统上跑而不是只用 VBench 出数字。30% cache 节省和 1.5× 加速对部署方有实际价值，但如果端到端延迟的瓶颈不在 DiT 而在 VAE，这些数字就变得次要了。论文没有提供完整 pipeline 的 latency breakdown，这是从"实验好"到"能落地"之间差的那一步。

---

## 参考资料

[1] [Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion](https://arxiv.org/abs/2510.06430)

[2] [Efficient Streaming Language Models with Attention Sinks (StreamingLLM)](https://arxiv.org/abs/2309.17453)

[3] [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048)

[4] [DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads](https://arxiv.org/abs/2410.10819)

[5] [Light Forcing: Accelerating Autoregressive Video Diffusion via Sparse Attention](https://arxiv.org/abs/2602.04789)

[6] [Flow Caching for Autoregressive Video Generation](https://arxiv.org/abs/2602.02112)

[7] [Efficient Autoregressive Video Diffusion with Dummy Head](https://arxiv.org/abs/2601.20499)

[8] [LongLive: Real-time Interactive Long Video Generation](https://arxiv.org/abs/2603.05606)

[9] [SkyReels-V2: Infinite-length Film Generative Model](https://arxiv.org/abs/2504.13074)

[10] [Wan: Open and Advanced Large-scale Video Generative Models](https://arxiv.org/abs/2503.20314)

[11] [Forcing-KV: Hybrid KV Cache Compression for Efficient Autoregressive Video Diffusion Models](https://arxiv.org/abs/2605.09681)
