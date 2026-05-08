---
title: AI Infra 早报｜CUDA graph 向多卡边界延伸，推理引擎在并行与新硬件上同步补完
date: 2026-03-23 05:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: vLLM MRV2 打通 Pipeline Parallelism 下的 CUDA graph，逐片段和全图两种模式均覆盖；SGLang 提升 decode 阶段 CUDA graph 命中率并同步修复 VRAM 泄露；Blackwell/SM120 硬件适配加速，SGLang CUTLASS FP8 GEMM 专项优化落地；Qwen3 MoE 上下文并行与 Ngram 投机解码重构同步推进；OpenClaw 修复多个稳定性与安全问题。
tags: [Inference, Quantization]
---

过去 24 小时的 AI Infra 变化，可以用"边界延伸"三个字来概括——CUDA graph 从单卡向多卡延伸，量化优化从旧架构向 Blackwell 延伸，投机解码从"合并即用"向"架构重构"延伸。这三条线背后有同一个逻辑：推理引擎正在填补从单卡实验到多卡/新硬件生产部署之间的工程缺口。

## 一、CUDA graph 突破 Pipeline Parallelism 边界

vLLM ModelRunnerV2 这次合并的是一个酝酿已久的 PR：在 Pipeline Parallelism 下同时支持逐片段（piecewise）和全图（full）两种 CUDA graph 模式[[1]](https://github.com/vllm-project/vllm/pull/35162)。

CUDA graph 的价值在于把一系列 CUDA 算子的调度开销前置成一次性的"录制"，运行时只需回放，消除逐 kernel launch 的 CPU 开销。这对延迟敏感的推理场景意义显著——但之前这个优化只在单机多卡场景下可用，跨节点的 Pipeline Parallelism 无法享受。

打通之后，用 PP 拆分跑超大模型的团队（比如 70B 以上要跨两台 8 卡节点的场景）才能真正用上 CUDA graph 带来的延迟收益。同时补充的还有 PP CUDA graph 专项测试[[2]](https://github.com/vllm-project/vllm/pull/37830)，确保这一路径有持续的质量保障。

SGLang 则从另一个维度提升了 CUDA graph 的实际使用率[[3]](https://github.com/sgl-project/sglang/pull/20978)：通过对 decode 阶段的 max-num-requests 做填充（padding），让更多真实 batch 大小落入已录制的 graph 范围内。命中率提高意味着更少回退到 eager 模式，是一个低改动高收益的工程优化。

## 二、Blackwell 硬件适配加速

新硬件落地往往伴随一段"支持但性能不达预期"的过渡期，直到框架侧做完针对新架构的专项调优。SGLang 这次针对 SM120（Blackwell）优化了 CUTLASS FP8 分块 GEMM[[4]](https://github.com/sgl-project/sglang/pull/20887)，填补了 Blackwell 上量化推理的性能缺口。

llama.cpp 同期修复了一个更基础的问题：BF16 在 CUDA Flash Attention 路径下无法正常编译[[5]](https://github.com/ggml-org/llama.cpp/releases/tag/b8474)。这个问题在 Blackwell 等支持原生 BF16 的新一代 GPU 上更容易触发，作为 b8474 hotfix 快速发布，是硬件适配补全的典型案例。

## 三、推理能力扩展：多模态、新模型、长上下文

vLLM 在 MRV2 路径上还补全了另一个组合：投机解码模型的多模态 embedding 支持[[6]](https://github.com/vllm-project/vllm/pull/36097)。此前在 MRV2 下，draft model 无法处理图像等多模态输入，限制了高吞吐推理路径对多模态场景的覆盖。修复后两者可以正常组合。

SGLang 为 Qwen3 MoE 补上了上下文并行支持[[7]](https://github.com/sgl-project/sglang/pull/18233)。上下文并行的意义在于把超长序列切分后跨卡处理，对需要 128K+ 上下文的 MoE 推理场景至关重要。Qwen3 MoE 作为当前主流开源 MoE 模型，这次补全实质性地扩展了它在多卡集群上的可用场景。

vLLM 同时新增了 NemotronHPuzzle 和 NemotronHMTP 两个模型的支持[[8]](https://github.com/vllm-project/vllm/pull/37803)，覆盖 NVIDIA Nemotron-H 系列的最新变体。

## 四、投机解码：从"能用"到"好用"

SGLang 的 Ngram 参考式投机解码重构是本窗口另一个值得关注的方向。这是一个系列 PR 的第一步[[9]](https://github.com/sgl-project/sglang/pull/20393)：拆分推理逻辑与解码逻辑，为后续更灵活的参考策略打基础。

一个特性从"合并进主分支能跑通"到"开始做架构重构"，本身就是技术成熟度的信号——不会有人去重构一个准备放弃的东西。SGLang 对投机解码的态度从补丁式修复转向架构重设计，说明这一方向在 SGLang 的优先级清单上已经稳固。

SGLang 同期还修复了 overlap scheduling 与 structured output 共用时的 VRAM 泄露[[10]](https://github.com/sgl-project/sglang/pull/20697)，以及在 MoE RL 后端引入 fp8+bf16 混合精度支持[[11]](https://github.com/sgl-project/sglang/pull/20214)——前者修的是稳定性，后者扩的是精度配置灵活性。

## 五、训练与基础设施

训练侧本窗口相对平静，主要是 Megatron-LM 新增了 Pile 数据集的通用预处理脚本[[12]](https://github.com/NVIDIA/Megatron-LM/pull/3902)。这类工作不显眼，但对从零开始跑预训练的团队有实际价值——数据准备流程标准化意味着更少的重复造轮子。

Ray Core 消除了 Worker 上下文中的可重入锁[[13]](https://github.com/ray-project/ray/pull/61925)，修复 absl mutex 调试告警。这类修复的价值体现在排查生产问题时——误报噪声少一分，真实问题就显眼一分。

## 六、OpenClaw 多个稳定性与安全修复

本窗口 OpenClaw 合并了六个 PR，其中三个值得关注。

**chatRunState 缓冲区泄露**[[14]](https://github.com/openclaw/openclaw/pull/52428)：当某次 run 卡死时，其状态缓冲区不会被自动清理，导致同 session 后续 run 受到影响。新增自动扫描清理机制后，卡死 run 的状态积压问题得到解决。

**Discord 图片投递修复**[[15]](https://github.com/openclaw/openclaw/pull/52489)：AI 生成图片在 Reply 流程中无法正确发送到 Discord，影响所有使用图片生成功能的 Discord 用户，本次修复恢复正常投递。

**插件安全检查精细化**[[16]](https://github.com/openclaw/openclaw/pull/52491)：原来插件入口文件不存在与路径逃逸安全违规会返回同类错误，难以区分。现在两种情况分别返回不同错误码，路径逃逸尝试有明确的安全日志，安全边界更清晰。

---

今天的早报没有大版本发布，但信息密度不低。CUDA graph 边界延伸、Blackwell 适配、Qwen3 MoE 长上下文、投机解码重构——这几件事分散在不同仓库，但指向同一个方向：推理引擎正在把过去两年的技术探索转化为生产就绪的工程能力，一个缺口一个缺口地补。

## 参考来源

[1] [vLLM MRV2 Pipeline Parallelism CUDA graph](https://github.com/vllm-project/vllm/pull/35162)

[2] [vLLM PP CUDA graph 测试](https://github.com/vllm-project/vllm/pull/37830)

[3] [SGLang decode CUDA graph 覆盖率提升](https://github.com/sgl-project/sglang/pull/20978)

[4] [SGLang SM120 CUTLASS FP8 Blockwise GEMM](https://github.com/sgl-project/sglang/pull/20887)

[5] [llama.cpp CUDA BF16 FA 编译修复 b8474](https://github.com/ggml-org/llama.cpp/releases/tag/b8474)

[6] [vLLM MRV2 spec decode 多模态 embedding](https://github.com/vllm-project/vllm/pull/36097)

[7] [SGLang Qwen3 MoE 上下文并行](https://github.com/sgl-project/sglang/pull/18233)

[8] [vLLM NemotronHPuzzle + NemotronHMTP](https://github.com/vllm-project/vllm/pull/37803)

[9] [SGLang Ngram 参考式投机解码重构](https://github.com/sgl-project/sglang/pull/20393)

[10] [SGLang overlap scheduling VRAM 泄露修复](https://github.com/sgl-project/sglang/pull/20697)

[11] [SGLang flashinfer_trtllm_routed fp8+bf16 RL](https://github.com/sgl-project/sglang/pull/20214)

[12] [Megatron-LM 通用 Pile 数据脚本](https://github.com/NVIDIA/Megatron-LM/pull/3902)

[13] [Ray Worker 可重入锁修复](https://github.com/ray-project/ray/pull/61925)

[14] [OpenClaw chatRunState 缓冲区泄露修复](https://github.com/openclaw/openclaw/pull/52428)

[15] [OpenClaw Discord 图片投递修复](https://github.com/openclaw/openclaw/pull/52489)

[16] [OpenClaw 插件路径逃逸安全检查精细化](https://github.com/openclaw/openclaw/pull/52491)
