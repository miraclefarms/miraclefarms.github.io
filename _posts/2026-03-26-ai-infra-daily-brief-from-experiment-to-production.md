---
title: AI Infra 早报｜从实验到生产，LLM 服务路径全面进入工程化验收期
date: 2026-03-26 05:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: Ray Serve LLM API 正式升级 Beta，结束在实验性状态的漫长等待；LMCache 24 小时内集中合并三个 CLI 子命令，运维接口走向体系化；vLLM 集成 FlashInfer NVFP4 CuteDSL MoE 内核，llama.cpp 正式注册 Qwen3 架构；OpenClaw v2026.3.24 正式版完成 Teams 官方 SDK 迁移，并带来 OpenAI 兼容接口与集中安全加固。
tags: [Inference, Quantization, MoE]
---

今天有一条主线贯穿各个项目：从实验到正式。不是新架构的发布，不是论文复现，而是已有的东西被系统性地推到可以承诺的状态。

Ray Serve 的 LLM API 升级为 Beta，意味着官方认为这条路已经稳定到可以向用户作出 API 兼容性承诺。LMCache 在同一个 24 小时窗口合并了三个 CLI 子命令，把零散的运维工具串成了一个体系。OpenClaw 的 v2026.3.24 正式版把 Teams 官方 SDK 迁移、OpenAI 接口兼容和安全加固打包在一起交付，体量不小。量化侧，vLLM 集成了 FlashInfer 的 NVFP4 CuteDSL MoE 内核，llama.cpp 完成 Qwen3 架构注册——两件事都在继续把"新模型 + 新量化"的工程路径铺实。

## 一、Ray Serve LLM API：Beta 意味着什么

Ray 今天把 Serve LLM API 提升到了 Beta [[1]](https://github.com/ray-project/ray/pull/62054)，同时把内部 LLM 依赖升级到 vLLM v0.18.0 [[2]](https://github.com/ray-project/ray/pull/61952)。

Beta 这个标志在 Ray 的发布策略里有明确含义：API 形态基本稳定、向后兼容性有承诺、官方会积极维护。对于基于 Ray Serve 搭建 LLM 推理服务的团队，这意味着可以减少对 API 变动的防御性包装，以更有信心的方式构建上层逻辑。

Ray 作为 LLM 服务基础设施的定位本来就和 vLLM、SGLang 不在同一层次——它提供的是调度、路由、弹性伸缩，而不是内核级的推理优化。Beta 标志是在宣布：这一层的工程化已经到位。

## 二、vLLM 与 llama.cpp：Blackwell 量化与 Qwen3 本地化

vLLM 今天集成了 FlashInfer 的 NVFP4 CuteDSL MoE 内核 [[3]](https://github.com/vllm-project/vllm/pull/38050)。过去 vLLM 的 NVFP4 MoE 路径依赖通用内核，在 Blackwell 上的吞吐还有大量空间没有释放。FlashInfer 的 CuteDSL 实现直接操作 NVIDIA CuTe 的 tile 抽象，可以更精准地利用 Blackwell 的 tensor core 和内存层次。这是 NVFP4 MoE 推理在 vLLM 里走向高性能路径的关键一步。同期，针对 Blackwell 新 SM 子型 SM120 的 CUTLASS blockwise FP8 GEMM 也完成了优化 [[4]](https://github.com/vllm-project/vllm/pull/37970)，FP8 量化在 Blackwell 上的性能基线又往上推了一格。

llama.cpp 今天注册了 Qwen3 模型架构 [[5]](https://github.com/ggml-org/llama.cpp/pull/20967)，意味着 Qwen3 系列可以通过标准流程转换为 GGUF 并在 llama.cpp 下本地推理。这件事听起来低调，但对应的影响范围不小：llama.cpp 生态覆盖了大量个人用户、边缘设备和需要离线运行的场景，Qwen3 的 GGUF 支持会带动一批量化版本和适配工作相继落地。

## 三、LMCache CLI 三连：运维接口的体系化

LMCache 在这个窗口集中合并了三个 CLI 子命令：`lmcache kvcache` [[6]](https://github.com/LMCache/LMCache/pull/2827)、`lmcache server` [[7]](https://github.com/LMCache/LMCache/pull/2836)、`lmcache query` [[8]](https://github.com/LMCache/LMCache/pull/2846)。

这三个子命令分别覆盖 KV cache 管理与状态查询、服务启动与管理、engine 状态查询。单独看每一个都不是特别大的功能，但在同一个 24 小时窗口集中合并，意味着 LMCache 的 CLI 接口正在从"有就行"走向"规划好的体系"。对运维人员来说，有了这套 CLI，管理 KV cache 服务的生命周期不再需要自己写脚本绕路。

同期，PD 分离模式下的 pin count 平衡 bug 也得到了修复 [[9]](https://github.com/LMCache/LMCache/pull/2786)。这个 bug 在长期运行中会导致内存泄漏或缓存命中率异常，是实际部署中会遇到的那类问题。

## 四、Mooncake 的缓存性能与 Megatron 的异构并行

Mooncake 今天合并了两个方向的改动。一是 P2P 代理缓冲区 kP2PBufferSize 扩容 [[10]](https://github.com/kvcache-ai/Mooncake/pull/1740)，此前偏小的默认值在大规模 KV cache 传输时成为性能瓶颈，扩容后 PD 分离场景的传输吞吐可以达到理论上限。二是新增对象硬锁定（hard pin）机制 [[11]](https://github.com/kvcache-ai/Mooncake/pull/1728)，允许把关键的 KV cache 对象标记为驱逐保护，在内存压力下不会被 eviction 策略清除。

Megatron-LM 侧，MiMo 异构并行的工程化在继续推进。非 colocated 场景（prefill 和 decode 不在同一组 GPU 上）现在支持分布式 checkpoint [[12]](https://github.com/NVIDIA/Megatron-LM/pull/4020)，训练中断可以恢复。同时修复了 Bridge Communicator 在非对称 DP 场景下的 2D 张量通信错误 [[13]](https://github.com/NVIDIA/Megatron-LM/pull/4021)，异构集群（不同规格 GPU 混部）的训练正确性得到保障。

## 五、TRT-LLM PDL、SGLang 扩散内核与 DeepSpeed AutoTP

TRT-LLM 的 CuTE DSL top-k 内核新增了 PDL 依赖启动支持 [[14]](https://github.com/NVIDIA/TensorRT-LLM/pull/12506)。PDL 允许 CUDA 内核在 GPU 上直接触发依赖内核，消除 MoE 路由排序路径上往返 CPU 的调度延迟，对 H100/B200 上的 MoE 推理有实质性的影响。

SGLang 的扩散模型路径今天引入了 AKO4ALL 内核优化框架 [[15]](https://github.com/sgl-project/sglang/pull/21323)，覆盖多类扩散模型的算子加速，不再是单点 patch；同期 Qwen 扩散模型的 Triton modulation kernels 也完成提速 [[16]](https://github.com/sgl-project/sglang/pull/21318)。

DeepSpeed AutoTP 新增了对 HuggingFace tp_plan 接口的支持 [[17]](https://github.com/deepspeedai/DeepSpeed/pull/7901)。这个变化看起来低调，但对实际迁移成本影响不小——HuggingFace 模型现在可以直接携带自身的 tp_plan 接入 DeepSpeed AutoTP，不需要再做一层手动适配。

## 六、OpenClaw v2026.3.24：打包交付的一次大版本

OpenClaw 今天发布了 v2026.3.24 正式版 [[18]](https://github.com/openclaw/openclaw/releases/tag/v2026.3.24)，是本窗口体量最大的交付。

Teams 频道迁移至官方 Teams SDK，获得了流式回复、welcome cards、feedback 卡片、typing indicators 和原生 AI labeling，对话体验与 Teams 平台规范对齐。Gateway 层新增 `/v1/models` 和 `/v1/embeddings` 端点，支持显式 model override 透传，打通与 RAG 框架和 OpenAI 兼容客户端的对接——这是把 OpenClaw 作为 API 中间层使用时的关键能力。Skills Control UI 也做了重构，新增状态过滤标签和一键安装 recipe，7 个内置 skill 支持依赖自动安装。

安全侧的集中度比较高。媒体解析层新增路径穿越和主目录访问防护 [[19]](https://github.com/openclaw/openclaw/pull/54642)，启动前过滤不可信 CWD `.env` 条目防止工作目录注入 [[20]](https://github.com/openclaw/openclaw/pull/54631)，低权限浏览器请求面的 profile 重置也被阻断 [[21]](https://github.com/openclaw/openclaw/pull/54618)。这些修复集中在一个版本里交付，是系统从内部使用走向更广泛部署前通常会做的安全收口。此外，Minimax M2.7 图像生成 provider 也在同版本加入 [[22]](https://github.com/openclaw/openclaw/pull/54487)，图像生成能力扩展至国产大模型。

---

今天没有颠覆性的架构出现，但这种"稳定推进"的日子在 AI Infra 里往往比大版本发布更重要。值得盯住的信号：llama.cpp 的 Qwen3 注册之后，GGUF 量化版本和各种适配 PR 会接连涌来；LMCache 的 CLI 体系化完成后，下一步是更完整的可观测性；Ray Serve LLM Beta 落地后，围绕它的生态集成会加速。

## 参考来源

[1] [Ray Serve LLM API Beta](https://github.com/ray-project/ray/pull/62054)

[2] [Ray LLM 升级 vLLM v0.18.0](https://github.com/ray-project/ray/pull/61952)

[3] [vLLM FlashInfer NVFP4 CuteDSL MoE 内核集成](https://github.com/vllm-project/vllm/pull/38050)

[4] [vLLM SM120 CUTLASS FP8 GEMM 优化](https://github.com/vllm-project/vllm/pull/37970)

[5] [llama.cpp 注册 Qwen3 架构](https://github.com/ggml-org/llama.cpp/pull/20967)

[6] [LMCache lmcache kvcache 子命令](https://github.com/LMCache/LMCache/pull/2827)

[7] [LMCache lmcache server 子命令](https://github.com/LMCache/LMCache/pull/2836)

[8] [LMCache lmcache query 子命令](https://github.com/LMCache/LMCache/pull/2846)

[9] [LMCache PD 分离 pin count 修复](https://github.com/LMCache/LMCache/pull/2786)

[10] [Mooncake P2P 缓冲区扩容](https://github.com/kvcache-ai/Mooncake/pull/1740)

[11] [Mooncake 对象硬锁定机制](https://github.com/kvcache-ai/Mooncake/pull/1728)

[12] [Megatron-LM 非 colocated MiMo 分布式 checkpoint](https://github.com/NVIDIA/Megatron-LM/pull/4020)

[13] [Megatron-LM 非对称 DP 2D 张量通信修复](https://github.com/NVIDIA/Megatron-LM/pull/4021)

[14] [TRT-LLM CuTE DSL top-k PDL 支持](https://github.com/NVIDIA/TensorRT-LLM/pull/12506)

[15] [SGLang 扩散模型 AKO4ALL 内核优化](https://github.com/sgl-project/sglang/pull/21323)

[16] [SGLang Qwen 扩散 Triton modulation 加速](https://github.com/sgl-project/sglang/pull/21318)

[17] [DeepSpeed AutoTP HuggingFace tp_plan 支持](https://github.com/deepspeedai/DeepSpeed/pull/7901)

[18] [OpenClaw v2026.3.24 正式版](https://github.com/openclaw/openclaw/releases/tag/v2026.3.24)

[19] [OpenClaw 媒体解析层路径穿越防护](https://github.com/openclaw/openclaw/pull/54642)

[20] [OpenClaw 启动前过滤不可信 .env](https://github.com/openclaw/openclaw/pull/54631)

[21] [OpenClaw 低权限面 profile 重置阻断](https://github.com/openclaw/openclaw/pull/54618)

[22] [OpenClaw Minimax 图像生成 provider](https://github.com/openclaw/openclaw/pull/54487)
