---
title: AI Infra 早报｜推理框架夯实基础，Gemma 4 多模态浪潮启动
date: 2026-04-04 05:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: 过去 72 小时，AI 推理框架加速迭代，Google Gemma 4 多模态能力在 vLLM、llama.cpp、TensorRT-LLM 多处同步落地。MoE 路由、缓存管理、跨芯片平台生态适配成为核心工程重点。分布式缓存与推理框架紧耦合，从端侧到云端的完整链路日趋成熟。
tags: [Inference, Multimodal, MoE, Quantization]
---

## 推理框架的多模态新时代

过去的 72 小时里，AI 基础设施社区经历了一次深度的、系统性的升级。与其说是"大模型的新发布"，不如说是"推理框架们争相适配新时代需求"的基础设施竞争。

首先映入眼帘的是 **Google Gemma 4 的多模态支持**。这不仅仅是一个模型的发布，而是被 vLLM、llama.cpp、TensorRT-LLM 等关键框架几乎同步地纳入支持范围。Gemma 4 带来了三个关键特性：MoE（混合专家）路由、多模态输入处理（视觉 + 音频框架），以及完整的 Function Calling 工具链。

这意味着什么？意味着多模态推理能力已经从"某些框架的选装项"升级成了"新一代模型的标配需求"。[[1]](https://github.com/vllm-project/vllm/pull/38826) [[2]](https://github.com/ggml-org/llama.cpp/pull/21309) 的并行推进，标志着生态已经形成了共识。

## 推理引擎的深度迭代

**vLLM** 在这 72 小时里合并了 25+ 条关键 PR，反映出框架在自身完善上的节奏。除了 Gemma 4 支持，更值得关注的是：

- **Renormalize 路由的重启** [[4]](https://github.com/vllm-project/vllm/pull/38859)：这个修复解决了 TRT-LLM 的 MoE 专家路由在某些场景下的崩溃问题，说明 MoE 在实际部署中仍有尖锐的工程挑战。
  
- **NVFP4 + MTP（多头注意力）的 Qwen3.5 适配** [[5]](https://github.com/vllm-project/vllm/pull/38832)：量化策略与新模型架构的结合变得越来越细致，框架需要面对越来越多的"特殊情况"。

- **Intel GPU 与 AMD 芯片的 CI 强化**：三个连续的 Intel XPU CI skip PR [[6]](https://github.com/vllm-project/vllm/pull/38904) [[7]](https://github.com/vllm-project/vllm/pull/38899) 反映出这些新硬件平台的生态仍在爬坡期。

**SGLang** 则专注于高性能内核的完善：[[8]](https://github.com/sgl-project/sglang/pull/22065)

- **HiSparse 稀疏路由优化**：针对 DSA（Dynamic Sparse Attention）模型的专项优化，说明稀疏计算已经从理论走向实践。
  
- **FlashAttention 3/4 的懒加载导入**：冷启动开销的优化体现了框架对用户体验的关注。

- **Kernel Release Workflow 的连续修复**：这一系列修复表明，高性能内核的发布流程仍然是基础设施的关键瓶颈。

**TensorRT-LLM** 采用了双轮并进的策略：[[13]](https://github.com/NVIDIA/TensorRT-LLM/pulls?q=is:pr+merged:2026-04-01..2026-04-04)

**DSA（Dynamic Sparse Attention）路线**包括：
- KV 缓存估算的修复 [[14]](https://github.com/NVIDIA/TensorRT-LLM/pull/12683)，确保 draft model 的内存配额准确
- Context Chunking 的 Token 计数校准 [[15]](https://github.com/NVIDIA/TensorRT-LLM/pull/12682)
- Decode 路径的多流并执行优化 [[16]](https://github.com/NVIDIA/TensorRT-LLM/pull/12691)

**MLA（Multi-Head Latent Attention）路线**则通过将 MLA 生成支持集成到 TrtllmGen 注意力后端 [[17]](https://github.com/NVIDIA/TensorRT-LLM/pull/12606)，让 Deepseek R1/V3 等 MLA 架构模型在 TRTLLM 上获得了一级公民地位。

**llama.cpp** 的故事是"从通用到高效的进阶"：[[19]](https://github.com/ggml-org/llama.cpp/pull/21309)

从 Gemma 4 支持、WebGPU 缓冲区管理重构 [[20]](https://github.com/ggml-org/llama.cpp/pull/21278)，到 SYCL 5GB KV 缓存挂起修复 [[21]](https://github.com/ggml-org/llama.cpp/pull/21283)，llama.cpp 正在从"能跑"升级到"能高效跑"。多硬件平台的统一已经不再是梦想。

## 训练框架的系统性完善

**Megatron-LM** 在这个时间窗口里完成了几项重要的框架级改进：[[22]](https://github.com/NVIDIA/Megatron-LM/pulls?q=is:pr+merged:2026-04-01..2026-04-04)

- **Emerging Optimizer 框架的整合** [[23]](https://github.com/NVIDIA/Megatron-LM/pull/4119)：这意味着不同的参数高效算法现在可以以插件的形式接入，训练框架正向真正的通用化演进。
  
- **Hybrid FSDP 的完善** [[24]](https://github.com/NVIDIA/Megatron-LM/pull/4105) [[25]](https://github.com/NVIDIA/Megatron-LM/pull/4071) [[26]](https://github.com/NVIDIA/Megatron-LM/pull/4054)：MXFP8 转置 buffer 的双倍缓存问题、MoE 动态路由的 padding 跳过、梯度规约的双缓冲不足——这些修复显示分布式训练的可靠性在逐层完善。

**DeepSpeed** 的贡献虽然数量较少，但质量突出：[[27]](https://github.com/deepspeedai/DeepSpeed/pull/7948)

ZeRO 阶段与自动梯度的不兼容问题在通过 Flat buffer 的 Detach 操作得到解决 [[28]](https://github.com/deepspeedai/DeepSpeed/pull/7948)。这类修复虽然看似细微，但直接关系到大规模分布式训练的稳定性。

## 缓存策略的生态成熟

如果说推理引擎在优化"如何快速计算"，那么分布式缓存系统在解决"如何有效管理"。

**Mooncake** 作为 KV 缓存专用系统，在这个时间窗口里完成了从"纯存储"向"多语言生态"的演进：

- **DRAM-CXL-SSD 多协议支持** [[31]](https://github.com/kvcache-ai/Mooncake/pull/1818)：这不仅仅是一个协议增强，而是承认了异构存储层次（内存→CXL→SSD）在高吞吐场景中的必要性。
  
- **Lock-Free P2P 路由缓存** [[32]](https://github.com/kvcache-ai/Mooncake/pull/1793)：无锁数据结构的引入意味着 P2P 通信瓶颈开始被认真对待。
  
- **Go 语言绑定 + C API** [[33]](https://github.com/kvcache-ai/Mooncake/pull/1764) [[33b]](https://github.com/kvcache-ai/Mooncake/pull/1763)：多语言支持预示着 Mooncake 正在从"推理框架的附属品"演变为独立的基础设施服务。

**LMCache** 则从另一个角度补全生态：

- **L2 Native 后端的驱逐支持** [[34]](https://github.com/LMCache/LMCache/pull/2939)：这让 LMCache 能够应对内存压力下的主动驱逐，而不仅仅是被动的缓存替换。
  
- **Mooncake 适配器的完成** [[35]](https://github.com/LMCache/LMCache/pull/2911)：这意味着推理框架、独立缓存层、分布式存储系统之间的协议正在标准化。

完整的链路已经形成：**端侧推理（llama.cpp）→ 单卡推理（vLLM）→ 多卡推理（SGLang）→ 分布式缓存（Mooncake）**。这不仅仅是技术栈的堆砌，而是每一层都在为上一层的效率而优化。

## 应用层的同步升级

**Hugging Face TRL** 的更新虽然看起来是细节，但反映了训练生态的逐步规范化：

- **KTO 与 DPO 的对齐** [[41]](https://github.com/huggingface/trl/pull/5447)：RLHF 算法的标准化程度在上升，不同框架之间的兼容性也在增强。
  
- **vLLM 最低版本提升至 0.11.0** [[42]](https://github.com/huggingface/trl/pull/5443)：这个版本约束的硬化说明，推理框架与训练框架的耦合变得更紧密，生态成熟度的标志。

**Ray** 的可观测性升级也值得关注：

- **Serve 自动扩展延迟修复** [[45]](https://github.com/ray-project/ray/pull/62331)：从相对时间到真实时间，这个看似简单的改变背后是对生产部署真实需求的深入理解。
  
- **意外工作进程失败指标 + 仪表板面板** [[48]](https://github.com/ray-project/ray/pull/62297)：可观测性与鲁棒性的同步升级正在成为常态。

**OpenClaw** 在这个时间窗口里推进了安全性与多协议兼容性的双重建设：

- **Discord 代理与捆绑通道的安全加固** [[50]](https://github.com/openclaw/openclaw/pull/60455)
  
- **Mistral、Moonshot、Responses 等多协议的传输层 compat 统一** [[51]](https://github.com/openclaw/openclaw/pull/60405) [[52]](https://github.com/openclaw/openclaw/pull/60411)

这表明 Agent 框架已经从"某一个提供商的工具"演变成了"真正的跨平台基础设施"。

## 结语

这 72 小时的演进不是"大模型新发布秀"，而是"基础设施框架在为新时代做准备"的缩影。Gemma 4 催化了多模态推理的同步需求；MoE 路由、KV 缓存管理、量化策略从实验性特征演变成了生产标配；跨芯片平台的工程工作已经成为日常。

AI infra 不再是被动地追赶模型演进，而是与模型创新形成了**同步共舞**的新时代。推理框架在优化，训练框架在完善，缓存系统在演进，应用生态在扩展——这不是巧合，这是一个自洽的、有方向的系统性升级。

[[1] vLLM Gemma 4 支持](https://github.com/vllm-project/vllm/pull/38826)

[[2] llama.cpp Gemma 4 支持](https://github.com/ggml-org/llama.cpp/pull/21309)

[[4] vLLM Renormalize 路由修复](https://github.com/vllm-project/vllm/pull/38859)

[[5] vLLM NVFP4+MTP Qwen3.5 适配](https://github.com/vllm-project/vllm/pull/38832)

[[6] vLLM Intel GPU CI skip #38904](https://github.com/vllm-project/vllm/pull/38904)

[[7] vLLM Intel GPU CI skip #38899](https://github.com/vllm-project/vllm/pull/38899)

[[8] SGLang HiSparse 与 FA3/4 优化](https://github.com/sgl-project/sglang/pull/22065)

[[13] TensorRT-LLM 72h PR 集合](https://github.com/NVIDIA/TensorRT-LLM/pulls?q=is:pr+merged:2026-04-01..2026-04-04)

[[14] TensorRT-LLM KV 缓存估算修复](https://github.com/NVIDIA/TensorRT-LLM/pull/12683)

[[15] TensorRT-LLM Context Chunking Token 计数](https://github.com/NVIDIA/TensorRT-LLM/pull/12682)

[[16] TensorRT-LLM Decode 多流优化](https://github.com/NVIDIA/TensorRT-LLM/pull/12691)

[[17] TensorRT-LLM MLA 生成支持](https://github.com/NVIDIA/TensorRT-LLM/pull/12606)

[[19] llama.cpp 72h PR 集合](https://github.com/ggml-org/llama.cpp/pull/21309)

[[20] llama.cpp WebGPU 缓冲区重构](https://github.com/ggml-org/llama.cpp/pull/21278)

[[21] llama.cpp SYCL 5GB KV 缓存修复](https://github.com/ggml-org/llama.cpp/pull/21283)

[[22] Megatron-LM 72h PR 集合](https://github.com/NVIDIA/Megatron-LM/pulls?q=is:pr+merged:2026-04-01..2026-04-04)

[[23] Megatron-LM Emerging Optimizer 框架](https://github.com/NVIDIA/Megatron-LM/pull/4119)

[[24] Megatron-LM MXFP8 转置 buffer FSDP](https://github.com/NVIDIA/Megatron-LM/pull/4105)

[[25] Megatron-LM MoE 动态路由](https://github.com/NVIDIA/Megatron-LM/pull/4071)

[[26] Megatron-LM FSDP 梯度规约双缓冲](https://github.com/NVIDIA/Megatron-LM/pull/4054)

[[27] DeepSpeed 72h PR 集合](https://github.com/deepspeedai/DeepSpeed/pull/7948)

[[28] DeepSpeed ZeRO Detach 修复](https://github.com/deepspeedai/DeepSpeed/pull/7948)

[[31] Mooncake DRAM-CXL-SSD 多协议](https://github.com/kvcache-ai/Mooncake/pull/1818)

[[32] Mooncake Lock-Free P2P 路由](https://github.com/kvcache-ai/Mooncake/pull/1793)

[[33] Mooncake Go 语言绑定](https://github.com/kvcache-ai/Mooncake/pull/1764)

[[33b] Mooncake C API](https://github.com/kvcache-ai/Mooncake/pull/1763)

[[34] LMCache L2 Native 驱逐](https://github.com/LMCache/LMCache/pull/2939)

[[35] LMCache Mooncake 适配](https://github.com/LMCache/LMCache/pull/2911)

[[41] TRL KTO DPO 对齐](https://github.com/huggingface/trl/pull/5447)

[[42] TRL vLLM 版本约束](https://github.com/huggingface/trl/pull/5443)

[[45] Ray Serve 自动扩展延迟](https://github.com/ray-project/ray/pull/62331)

[[48] Ray 意外进程失败指标](https://github.com/ray-project/ray/pull/62297)

[[50] OpenClaw Discord 安全加固](https://github.com/openclaw/openclaw/pull/60455)

[[51] OpenClaw Mistral compat](https://github.com/openclaw/openclaw/pull/60405)

[[52] OpenClaw Moonshot compat](https://github.com/openclaw/openclaw/pull/60411)
