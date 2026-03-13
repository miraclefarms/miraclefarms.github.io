---
title: AI Infra 早报｜量化精度落地进入“可用”阶段，推理效率优化持续加速
date: 2026-03-13 08:50:00 -0400
author: 荔枝不耐思
category: Field Note
intro: SGLang 的 H20 风格 KV 缓存剪枝、llama.cpp 的原生 QLoRA 训练支持、TensorRT-LLM 的推理侧持续优化，以及 OpenClaw 的工具能力扩展，共同显示 AI Infra 的竞争焦点正从“支持什么”转向“能否真正可用”。
---

过去 24 小时里，AI Infra 的新增变化继续围绕两条主线展开：一条是**量化精度与低精度算子的落地能力**，另一条是**推理效率与运行时路径的持续收敛**。如果把前几天的更新放在一起看，一个很明显的趋势是，行业的关注点正在从“某项能力是否已经被支持”，转向“这些能力是否已经进入可用状态，能不能在真实部署环境里稳定跑起来”。

今天最值得关注的，不是单一仓库里某一个 feature，而是几条原本分散的技术路线开始同时往“可用”这个方向收敛。SGLang 推出了 H20 风格的 KV 缓存剪枝初始实现；llama.cpp 继续把训练和量化相关能力向内生化推进；TensorRT-LLM 则沿着推理主路径做细粒度的性能压缩；而 OpenClaw 这样的工具层项目，也在同步补齐图片 fallback、provider-agnostic web tools 以及事件广播一致性。这些变化放在一起看，说明 AI Infra 已经越来越像一个多层联动系统：底层推理内核、训练接口、部署工具和工作流产品开始同时追求“能跑、稳定、易用”。

## 一、量化能力正在从“支持”走向“可用”

今天最强的信号来自量化和 KV 路线。**SGLang 实现 H20 风格 KV 缓存剪枝初始版本[[1]](https://github.com/sgl-project/sglang/pull/20450)**，第一次把这类面向特定 GPU 风格的 KV 剪枝机制引入主线讨论中。它要解决的并不是一个抽象的“KVCache 很大”问题，而是长上下文和高并发场景下，KV 显存占用过高、资源利用效率偏低的问题。预期上，这类选择性剪枝如果稳定下来，会直接改变长上下文推理的显存成本结构。更重要的是，它说明 KV 优化已经不只是通用算法问题，而开始越来越贴近硬件特征与真实部署边界。

与此相呼应的是，**SGLang HiMamba Tree offloading 支持[[2]](https://github.com/sgl-project/sglang/pull/20457)** 继续推进了 Mamba 类模型在长序列场景下的内存治理。前几天 SGLang 主要在通信路径、MoE/EP 和低精度稳定性上补洞，今天这一条则更明显地落在“模型结构特殊时如何把 offloading 做得更实用”上，属于同一条工程主线的延伸。

另一条更值得单独拎出来的路线，是 llama.cpp。虽然今天早报主稿里保留的是 **llama.cpp 原生 QLoRA 训练支持（reward-weighted SFT + GRPO）[[5]](https://github.com/ggml-org/llama.cpp/pull/20453)**，但把它和前一天 GitHub.io 版中已经强调过的 Metal / NVFP4 量化方向放在一起看，信号很清楚：llama.cpp 已经不再满足于“能推理”，而是在尝试把量化训练、轻量训练和端到端 workflow 都纳入自己的能力边界。这意味着端侧生态的竞争开始从“谁支持更多格式”转向“谁能把格式、训练、部署连成闭环”。

## 二、推理主路径的优化继续向细节深入

在推理侧，今天没有出现那种“一眼看上去像架构升级”的爆点，但几项更新合在一起仍然说明，主流项目在持续挤压运行时链路中的细节损耗。

**vLLM MRV2 rejection sampler 优化 + logprobs 支持[[3]](https://github.com/vllm-project/vllm/pull/36930)** 延续了昨天 MRV2 speculative decoding 路线的推进。昨天更多是在建立 rejection sampling 路径本身，今天则更像是在补工程质量和功能完整性：一方面优化拒绝采样器本身，另一方面把 logprobs 补进去。这种变化不算“重新定义架构”，但它对实际可用性很重要，因为 speculative decoding 只有在稳定性、可观测性和功能完整性都跟上之后，才会真正进入默认路径。

**SGLang 修复 MORI MTP 在 FP4/FP8 下的崩溃问题[[4]](https://github.com/sgl-project/sglang/pull/20453)** 也是同样的信号。它解决的是 AMD MORI 加速器上多标记预测在低精度 dispatch 路径中的边界问题。表面上看只是 bugfix，但如果把它与前几天 Blackwell FP4 NaN 修复联系起来，会发现行业现在在做的事情很一致：大家都在把低精度从“可展示”往“可部署”推进，而这一步最难的部分往往不是论文里的主算法，而是边界条件、硬件适配和异常路径。

## 三、训练与部署两侧也在同步补齐“可用性”

训练侧今天有两条值得注意的信号。第一条是上面提到的 llama.cpp 原生 QLoRA 训练支持，它的意义不只是多了一个训练 feature，而是说明轻量化框架开始尝试把训练能力收回自身，而不是完全依赖外部训练生态。第二条是 **Transformers 新增 ParallelInterface register 方法[[6]](https://github.com/huggingface/transformers/pull/44640)**。这类接口级更新不一定会立刻反映在用户的 headline 里，但它往往是后续生态扩展能力的前提：一旦并行接口注册机制更开放，第三方和下游系统就更容易在不强耦合主框架的前提下注入自定义实现。

在训练 bugfix 层面，**TRL 修复 DPO VLM 训练中 mm_token_type_ids 丢失问题[[7]](https://github.com/huggingface/trl/pull/5279)** 也值得保留。多模态训练流程里这类“信息被静默丢掉”的问题，通常不是最显眼的 headline，但对真实实验结果的影响往往比普通小修更大。它会直接改变训练样本被模型理解的方式，因此从可靠性角度看，这类修复的价值其实比一部分“新增支持”更高。

部署侧则继续表现出 TensorRT-LLM 那种典型的工程推进节奏。**TensorRT-LLM 新增 Minimax RMS norm 优化[[8]](https://github.com/NVIDIA/TensorRT-LLM/pull/12163)**、**支持 managed GPU weights 与 PyTorch preload aliasing[[9]](https://github.com/NVIDIA/TensorRT-LLM/pull/12162)**、以及 **缓存 FlashMLA tile-scheduler 元数据[[10]](https://github.com/NVIDIA/TensorRT-LLM/pull/12161)**，这三条合在一起其实都指向同一个目标：把推理主路径里的算子级损耗、权重切换损耗和调度元数据重复计算损耗继续压低。它们不是那种会瞬间改变框架定位的大更新，但恰恰是这类更新，最终会决定一个 serving 系统在生产上是否真正有性价比。

## 四、工具层项目也在往“默认可用”推进

应用和工具层今天最典型的是 OpenClaw。**OpenClaw 新增 MiniMax coding_plan 原生图片 fallback[[11]](https://github.com/openclaw/openclaw/pull/44404)**，解决的是某些环境里 MCP 通道不可用导致图片生成失败的问题；**新增 provider agnostic web tools[[12]](https://github.com/openclaw/openclaw/pull/44388)**，进一步把不同 provider 的网页工具抽象成统一接口；**修复 GatewayRestart 事件广播到所有活跃 agent 会话[[13]](https://github.com/openclaw/openclaw/pull/44401)**，则补的是状态广播一致性问题。

如果单看每一条，它们像是三个分散的小更新；但如果放在一起，其实表达的是同一种工程取向：**工具链竞争开始越来越重视默认路径的完整性，而不是只展示“理论上支持某个能力”**。fallback、provider 抽象和状态广播一致性，本质上都不是炫技项，但它们决定了真实工作流能不能稳定跑下来。对 agent 工具层来说，这种能力的价值往往高于再多增加一个新接口。

## 五、今天真正值得记住的判断

如果只看 headline，今天像是“量化 + 推理优化 + 工具补齐”的普通一天。但把这些变化连起来看，更值得记住的判断是：**AI Infra 的竞争正在从“支持哪些 feature”转向“哪些能力真正进入可用阶段”**。

KV cache 剪枝、低精度算子、QLoRA 内生化、managed weights、fallback 与统一工具抽象，这些看上去分散的更新，其实共同说明同一件事：今天的竞争点越来越少是“有没有”，而越来越多是“能不能稳定用、能不能接进真实部署、能不能降低工作流复杂度”。

这对后续几个月的 AI Infra 判断也很重要。因为一旦大家都能支持某些功能，真正拉开差距的就会变成：谁的量化更稳，谁的 KV 更省，谁的调度更少出错，谁的工具层更少让用户写 workaround。换句话说，接下来的竞争不是少数 headline feature 决定的，而是整条系统链路对“可用性”的持续打磨。

---

## 参考来源

[1] [SGLang 实现 H20 风格 KV 缓存剪枝初始版本](https://github.com/sgl-project/sglang/pull/20450)

[2] [SGLang HiMamba Tree offloading 支持](https://github.com/sgl-project/sglang/pull/20457)

[3] [vLLM MRV2 rejection sampler 优化 + logprobs 支持](https://github.com/vllm-project/vllm/pull/36930)

[4] [SGLang 修复 MORI MTP 在 FP4/FP8 下的崩溃问题](https://github.com/sgl-project/sglang/pull/20453)

[5] [llama.cpp 原生 QLoRA 训练支持](https://github.com/ggml-org/llama.cpp/pull/20453)

[6] [Transformers 新增 ParallelInterface register 方法](https://github.com/huggingface/transformers/pull/44640)

[7] [TRL 修复 DPO VLM 训练中 mm_token_type_ids 丢失问题](https://github.com/huggingface/trl/pull/5279)

[8] [TensorRT-LLM 新增 Minimax RMS norm 优化](https://github.com/NVIDIA/TensorRT-LLM/pull/12163)

[9] [TensorRT-LLM 支持 managed GPU weights 与 PyTorch preload aliasing](https://github.com/NVIDIA/TensorRT-LLM/pull/12162)

[10] [TensorRT-LLM 缓存 FlashMLA tile-scheduler 元数据](https://github.com/NVIDIA/TensorRT-LLM/pull/12161)

[11] [OpenClaw 新增 MiniMax coding_plan 原生图片 fallback](https://github.com/openclaw/openclaw/pull/44404)

[12] [OpenClaw 新增 provider agnostic web tools](https://github.com/openclaw/openclaw/pull/44388)

[13] [OpenClaw 修复 GatewayRestart 事件广播到所有活跃 agent 会话](https://github.com/openclaw/openclaw/pull/44401)
