---
title: AI Infra 早报｜扩散模型量化与 MoE LoRA 破题，KV cache 存储选项全面扩张
date: 2026-03-25 05:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: SGLang 为 Flux.2 带来 NVFP4 量化推理，TRT-LLM 同日支持 Qwen3.5 NVFP4，量化格式从 LLM 向扩散模型蔓延；SGLang MoE LoRA 支持 TP，打通 MoE 微调到部署全路径；LMCache 一天内合并 Valkey 集群/TLS 和原生文件系统两个连接器；TRL 集中重构 SFT/DPO 截断逻辑；llama.cpp 支持原生 HF 缓存目录；TRT-LLM log probs 归一化行为变更需关注。
tags: [Quantization, MoE, Multimodal]
---

今天有两条主线值得单独梳理，背后各自藏着一个方向性信号。

**第一条：NVFP4 量化正在从 LLM 推理蔓延到扩散模型。** SGLang 为 Flux.2 带来了 NVFP4 量化[[1]](https://github.com/sgl-project/sglang/pull/20137)，TRT-LLM 同日为 Qwen3.5 添加了 NVFP4 支持[[2]](https://github.com/NVIDIA/TensorRT-LLM/pull/12302)。NVFP4 是 Blackwell GPU 的原生 4-bit 浮点量化格式，此前主要在 LLM 推理加速中出现。Flux.2 获得支持意味着图像生成工作负载也开始纳入新一代量化格式的覆盖范围——这不是偶然，而是框架侧在 Blackwell 上系统性推进量化支持的节奏信号。

**第二条：LMCache 的存储连接器生态正在快速铺开。** 加上昨天的 Device-DAX，今天又上线了 Valkey 集群/TLS/GLIDE 优化[[25]](https://github.com/LMCache/LMCache/pull/2790)和原生文件系统连接器[[26]](https://github.com/LMCache/LMCache/pull/2779)——三天三个不同的存储后端。这个节奏说明 KV cache 的存储层正在从"够用"走向"覆盖各种生产基础设施"的竞争阶段。

## 一、推理侧：多硬件、多精度、更多模型

**SGLang 这次补上了 MoE LoRA 的 Tensor Parallelism 支持**[[3]](https://github.com/sgl-project/sglang/pull/14105)，意义不小。LoRA 微调 MoE 模型此前在 SGLang 中缺失，加上 TP 意味着微调好的 MoE LoRA 适配器可以在多卡部署中直接使用。对需要对 DeepSeek、Qwen3 等 MoE 模型做定制化的团队，这打通了从"实验室微调"到"生产多卡推理"的完整链路。

AMD 平台的支持继续推进：DeepSeek V3.2 在 MI355/MI300 上获得了基于 Tilelang 的稀疏前向算子[[4]](https://github.com/sgl-project/sglang/pull/19945)，专为 AMD 架构优化的内核路径。SGLang 在 AMD 上的 DS V3.2 性能从此不再只是"能跑"。

**vLLM 这边是一批工程质量提升的 PR**，集中在几个方向：

- **NFS 场景**[[6]](https://github.com/vllm-project/vllm/pull/37673)：自动开启 NFS 预取并通过 RAM guard 防止 OOM，解决了从网络文件系统加载大模型权重慢的实际痛点；
- **部署便利性**[[8]](https://github.com/vllm-project/vllm/pull/37233)：flashinfer-cubin 成为默认依赖，最优推理内核开箱即用；
- **torch.compile 性能**[[10]](https://github.com/vllm-project/vllm/pull/37485)：inductor 运行时断言默认关闭，消除 serving 场景下的断言开销；
- **MRV2 + DS V3.2 修复**[[9]](https://github.com/vllm-project/vllm/pull/38030)：DS V3.2 在 MRV2 推理路径下的 bug 修复，打通高性能路径；
- **DBO 通用化**[[7]](https://github.com/vllm-project/vllm/pull/37926)：动态批次优化不再只对特定模型生效。

FlexAttention 获得自定义 mask mod 支持[[11]](https://github.com/vllm-project/vllm/pull/37692)，为滑动窗口注意力、文档级掩码等非标准注意力模式提供了灵活的编程接口。

**llama.cpp 这次有一个对普通用户影响最大的进展**：原生 HF 缓存目录支持[[12]](https://github.com/ggml-org/llama.cpp/releases/tag/b8498)。过去从 HF 下载的模型要用 llama.cpp 必须先转换格式，现在可以直接从 `~/.cache/huggingface/hub` 读取，彻底消除了这个使用摩擦。同期，pipeline parallelism 的 ggml-backend 图复用重新启用[[13]](https://github.com/ggml-org/llama.cpp/releases/tag/b8507)，Apple Silicon 上的 Metal Flash Attention 扩展至 HSK/HSV=512[[14]](https://github.com/ggml-org/llama.cpp/releases/tag/b8500)。

TRT-LLM 除了 Qwen3.5 NVFP4 和 MoE 动态 SMEM 路由之外，**有一个需要特别注意的 breaking change**：log probs 默认不再做序列长度归一化[[16]](https://github.com/NVIDIA/TensorRT-LLM/pull/12366)。如果你的评估流程依赖 TRT-LLM 的 log prob 输出做 perplexity 计算或模型选择，升级前需要检查并调整。

## 二、训练侧：TRL 的正确性集中修复

TRL 在一天内合并了四个相关 PR，核心是 **SFT/DPO 数据处理截断时机的统一修正**[[17]](https://github.com/huggingface/trl/pull/5363)[[18]](https://github.com/huggingface/trl/pull/5359)[[19]](https://github.com/huggingface/trl/pull/5350)[[20]](https://github.com/huggingface/trl/pull/5362)。

问题的本质是：原来的实现在 collation（padding）之后再做截断，这意味着 padding token 可能占据了本应是真实内容的位置，导致 tokenizer 边界不对齐。四个 PR 统一改为"先截断到目标长度，再做 padding"，同时简化了 SFT tokenization 和 DPO DataCollatorForPreference 的代码路径。

这类 bug 的特点是"静默影响训练结果"——训练不会崩溃，但数据处理的轻微错误会在损失函数层面持续产生偏差。如果你在用 TRL 的 SFTTrainer 或 DPOTrainer，建议确认使用的是最新版本。

Megatron-LM 方面，**vLLM fakequant 格式导出支持**[[21]](https://github.com/NVIDIA/Megatron-LM/pull/3050)打通了一个长期缺失的链路：用 Megatron-LM 训练的模型可以直接导出为 vLLM 兼容的量化格式，无需中间转换步骤。对同时使用 Megatron-LM 做训练、vLLM 做部署的团队，这简化了量化部署工作流。

MoE 层的 Protocol 接口迁移[[22]](https://github.com/NVIDIA/Megatron-LM/pull/3426)和 MIMO 模型多模块异构并行支持[[23]](https://github.com/NVIDIA/Megatron-LM/pull/3211)则是 Megatron-LM 架构层面的演进，提升了 MoE 和多模态模型的并行训练灵活性。

## 三、KV cache 存储：一天两个新连接器

**Mooncake 的 HA 后端昨天还只是接口抽象，今天落了第一个实现**：Redis 领导者选举后端[[24]](https://github.com/kvcache-ai/Mooncake/pull/1722)，并附带完整的 HA 回归测试套件。有了具体实现和测试，HA 存储才算从"设计"走向"可用"。

**LMCache 一天两个连接器**：Valkey（Redis 开源分支）连接器获得集群模式、TLS 加密和 GLIDE 客户端优化[[25]](https://github.com/LMCache/LMCache/pull/2790)——这三个特性加在一起基本覆盖了企业级 Redis 部署的核心需求（水平扩展、合规加密、高性能客户端）；原生文件系统连接器[[26]](https://github.com/LMCache/LMCache/pull/2779)则提供了零依赖的最简持久化选项。

三天内新增三个存储后端（Device-DAX、Valkey、本地 FS），说明 LMCache 的存储层设计在系统性地覆盖不同的生产环境：高速持久内存、企业级分布式缓存、普通文件系统，各有其适用场景。

## 四、OpenClaw：OpenAI 兼容与安全加固

OpenClaw gateway 补全了两个 OpenAI 兼容端点[[27]](https://github.com/openclaw/openclaw/pull/53992)：`/v1/models`（返回可用模型列表）和 `/v1/embeddings`（文本向量化）。使用标准 OpenAI SDK 的应用现在可以无改动地切换到 OpenClaw gateway，API 兼容层的完整度提升。

安全侧有两个修复：沙箱媒体路径绕过漏洞关闭[[28]](https://github.com/openclaw/openclaw/pull/54034)（mediaUrl/fileUrl 别名不再可以逃逸沙箱边界），以及 `/allowlist` 等内部命令强制要求 operator.admin 权限[[29]](https://github.com/openclaw/openclaw/pull/54097)。前者防的是文件系统越权访问，后者防的是配置篡改——都是生产环境安全加固的必要动作。

---

今天的内容从表面看是"各处小步"，但有几个值得关注的方向性信号：NVFP4 开始覆盖扩散模型、MoE LoRA 补全了多卡部署缺口、LMCache 的存储连接器生态在快速铺开、TRL 集中修了一批静默影响训练正确性的 bug。这些单独看都不算轰动，但拼在一起，是 AI Infra 工具链走向"生产级完备"的稳定节奏。

## 参考来源

[1] [SGLang Flux.2 NVFP4 量化](https://github.com/sgl-project/sglang/pull/20137)

[2] [TRT-LLM Qwen3.5 NVFP4 支持](https://github.com/NVIDIA/TensorRT-LLM/pull/12302)

[3] [SGLang MoE LoRA + TP 支持](https://github.com/sgl-project/sglang/pull/14105)

[4] [SGLang AMD DS V3.2 Tilelang 稀疏算子](https://github.com/sgl-project/sglang/pull/19945)

[5] [SGLang SM90+ MoE 路由 FlashInfer tinygemm](https://github.com/sgl-project/sglang/pull/20755)

[6] [vLLM NFS 自动预取 + RAM guard](https://github.com/vllm-project/vllm/pull/37673)

[7] [vLLM DBO 通用化](https://github.com/vllm-project/vllm/pull/37926)

[8] [vLLM flashinfer-cubin 默认依赖](https://github.com/vllm-project/vllm/pull/37233)

[9] [vLLM MRV2 DS V3.2 修复](https://github.com/vllm-project/vllm/pull/38030)

[10] [vLLM inductor 运行时断言默认关闭](https://github.com/vllm-project/vllm/pull/37485)

[11] [vLLM FlexAttention 自定义 mask mod](https://github.com/vllm-project/vllm/pull/37692)

[12] [llama.cpp HF 缓存原生支持 b8498](https://github.com/ggml-org/llama.cpp/releases/tag/b8498)

[13] [llama.cpp PP 图复用重启 b8507](https://github.com/ggml-org/llama.cpp/releases/tag/b8507)

[14] [llama.cpp Metal FA HSK/HSV=512 b8500](https://github.com/ggml-org/llama.cpp/releases/tag/b8500)

[15] [TRT-LLM MoE 动态 SMEM 块路由](https://github.com/NVIDIA/TensorRT-LLM/pull/12456)

[16] [TRT-LLM log probs 默认不归一化（breaking）](https://github.com/NVIDIA/TensorRT-LLM/pull/12366)

[17] [TRL SFT tokenization 简化](https://github.com/huggingface/trl/pull/5363)

[18] [TRL SFT 移除 post-collation 截断](https://github.com/huggingface/trl/pull/5359)

[19] [TRL DPO 移除 post-collation 截断](https://github.com/huggingface/trl/pull/5350)

[20] [TRL DPO DataCollatorForPreference 简化](https://github.com/huggingface/trl/pull/5362)

[21] [Megatron-LM vLLM fakequant 导出](https://github.com/NVIDIA/Megatron-LM/pull/3050)

[22] [Megatron-LM MoE 层 Protocol 接口迁移](https://github.com/NVIDIA/Megatron-LM/pull/3426)

[23] [Megatron-LM MIMO 多模块异构并行](https://github.com/NVIDIA/Megatron-LM/pull/3211)

[24] [Mooncake Redis HA 领导者选举后端](https://github.com/kvcache-ai/Mooncake/pull/1722)

[25] [LMCache Valkey 集群/TLS/GLIDE 优化](https://github.com/LMCache/LMCache/pull/2790)

[26] [LMCache 原生文件系统连接器](https://github.com/LMCache/LMCache/pull/2779)

[27] [OpenClaw gateway OpenAI 兼容端点](https://github.com/openclaw/openclaw/pull/53992)

[28] [OpenClaw 沙箱媒体路径绕过修复](https://github.com/openclaw/openclaw/pull/54034)

[29] [OpenClaw /allowlist 管理员权限要求](https://github.com/openclaw/openclaw/pull/54097)
