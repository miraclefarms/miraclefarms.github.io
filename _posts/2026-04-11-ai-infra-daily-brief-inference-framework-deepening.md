---
title: AI Infra 早报｜推理框架多维迭代深化，生产部署侧持续完善
date: 2026-04-11 05:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: 过去 24 小时，AI Infra 各项目持续迭代深化。vLLM 完成 Gemma4 Eagle3 与 EXAONE-4.5 支持，SGLang 扩展 MoE 显存优化与 Diffusion FP8，llama.cpp 密集发布 7+ 版本，TRL 扩展多模型 Tool Calling，OpenClaw 引入严格 Agent 执行契约。本窗口以 A/B 级迭代为主，无 S 级重大发布。
tags: [Inference, Multimodal, Quantization]
---

过去 24 小时，AI Infra 各项目在推理优化、训练支持、生产部署三个维度同步推进。vLLM 完成了 Gemma4 Eagle3 支持与 EXAONE-4.5 模型接入，SGLang 在 MoE 显存优化与 Diffusion FP8 支持上取得进展，llama.cpp 密集发布 7+ 个版本完善多后端能力，TRL 则扩展了 Qwen3-VL 与 GLM-4-MoE 的 Tool Calling 训练支持。应用侧，OpenClaw 引入了严格的 Agent 执行契约，这是一个值得关注的方向信号。

## 一、vLLM：模型生态持续扩展

vLLM 今天最值得关注的是模型支持的扩展。**Gemma4 Eagle3 支持**[[1]](https://github.com/vllm-project/vllm/pull/39450) 完成了 qwen3-vl 的 EAGLE3 推理支持，这是多模态模型生态的重要补充。与此同时，**EXAONE-4.5**[[2]](https://github.com/vllm-project/vllm/pull/39388) 模型接入标志着 LG 的新型大模型进入 vLLM 支持列表。

在量化层面，**GGUF 非标准量化类型支持**[[3]](https://github.com/vllm-project/vllm/pull/39471) 增加了对 UD-IQ1_S 等非标准量化格式的 prefix 支持，这对使用自定义量化方案的用户是实际需求。多个 bugfix 也值得关注：量化 KV Cache crash 修复[[4]](https://github.com/vllm-project/vllm/pull/39444) 解决了 extract_hidden_states 在 quantized KV cache dtype 下的崩溃问题；显式未量化 kv cache dtype 精度修复[[5]](https://github.com/vllm-project/vllm/pull/38922) 修正了之前损坏的支持；kv_cache_dtype_skip_layers 导致的 FlashInfer crash 修复[[6]](https://github.com/vllm-project/vllm/pull/39002) 则解决了特定配置下的稳定性问题。

## 二、SGLang：MoE 优化与 Diffusion FP8 并行推进

SGLang 今天在两个方向上同时取得进展：MoE 显存优化和 Diffusion FP8 支持。

**MoE 并行组显存优化**[[7]](https://github.com/sgl-project/sglang/pull/22515) 通过减少 MoE parallel groups 的 GPU 显存占用，为大规模 MoE 模型部署提供了更充裕的显存空间。这一优化对于使用 DeepSeek 等 MoE 架构的团队尤为实际。

**Diffusion ModelOpt FP8 支持**[[8]](https://github.com/sgl-project/sglang/pull/22365) 增加了 Flux1、Flux2 和 Wan2.2 的 FP8 推理支持，使得 diffusion 模型也能享受量化推理的效率提升。这是 SGLang 在多模态生成领域的持续深耕。

此外，Qwen3 Next MTP 的 NCCL AllGather hang 问题修复[[9]](https://github.com/sgl-project/sglang/pull/22458) 解决了多节点训练的关键瓶颈；GDN 非连续 Tensor 支持[[10]](https://github.com/sgl-project/sglang/pull/22312) 则修正了 Qwen3.5-27B 的精度回归问题；**Kimi K25 EPD 支持**[[11]](https://github.com/sgl-project/sglang/pull/22269) 标志着 SGLang 在 Expert Parallel Deployment 方向的支持范围进一步扩大。

## 三、llama.cpp：密集版本迭代与多后端完善

llama.cpp 今天发布了 7+ 个版本，在多个后端上同步推进。

**Q1_0 Vulkan 支持**[[12]](https://github.com/ggml-org/llama.cpp/pull/21539) 完善了 1-bit 量化在 Vulkan 后端的覆盖，使得低精度量化在 AMD/Intel GPU 上更加可用。**Gemma4 Reasoning Budget Sampler**[[13]](https://github.com/ggml-org/llama.cpp/pull/21697) 启用了 thinking budget 采样器，让 Gemma4 的推理能力可以得到更精细的控制。

**WebGPU Intel 非方阵支持**[[14]](https://github.com/ggml-org/llama.cpp/pull/21669) 增加了 Intel GPU 非方形子矩阵配置的支持，这对于使用集成显卡的用户是实际改进。最值得关注的是 **GGML 后端无关 Tensor Parallelism 实验支持**[[15]](https://github.com/ggml-org/llama.cpp/pull/21380)，这是 llama.cpp 在异构分布式推理方向的重要探索，未来可能实现在不同后端上使用统一的 TP 策略。

## 四、TRL：多模型 Tool Calling 训练支持扩展

TRL 今天继续扩展 Tool Calling 训练支持。

**Qwen3-VL Tool Calling**[[17]](https://github.com/huggingface/trl/pull/5469) 和 **GLM-4-MoE Tool Calling**[[18]](https://github.com/huggingface/trl/pull/5463) 的支持加入，使得这两个主流国产模型可以进入 TRL 的 tool calling 训练流程。这对于构建具备工具调用能力的国产模型至关重要。

**Llama 3 generation markers**[[19]](https://github.com/huggingface/trl/pull/5493) 新增了带 generation markers 的训练模板，支持更精细的生成控制。**DistillationTrainer trackio 支持**[[20]](https://github.com/huggingface/trl/pull/5501) 则为蒸馏训练提供了 IO 追踪能力，便于分析训练瓶颈。

## 五、TensorRT-LLM 与 Mooncake：生产部署侧持续完善

TensorRT-LLM 今天有多项部署相关的改进。**FP8 LoRA weight 文件加载支持**[[24]](https://github.com/NVIDIA/TensorRT-LLM/pull/12848) 使得 FP8 格式的 LoRA 权重可以直接加载，降低了量化部署的复杂度。**KV Cache 统计监控增强**[[25]](https://github.com/NVIDIA/TensorRT-LLM/pull/12413) 提供了更精细的 kv cache 观测能力，这对于调试大规模部署问题非常有帮助。

Disagg KV Cache 传输 crash 修复[[26]](https://github.com/NVIDIA/TensorRT-LLM/pull/12909) 和 LoRA Overallocation 部分修复[[27]](https://github.com/NVIDIA/TensorRT-LLM/pull/12817) 则是生产稳定性的重要保障。

Mooncake 的 **Grouped Scatter RDMA Reads 支持**[[28]](https://github.com/kvcache-ai/Mooncake/pull/1717) 新增了 get_into_ranges 方法，为 RDMA 高效读取提供了更灵活的接口，这对于使用高速网络的生产部署是实际优化。

Ray 新增的 **ClockInterface 共享测试工具**[[29]](https://github.com/ray-project/ray/pull/62476) 提供了统一的时钟接口和 fake clock，增强了测试基础设施的可靠性。

## 六、OpenClaw：Agent 执行契约与安全加固

OpenClaw 今天最值得关注的是 **strict-agentic 执行契约**[[31]](https://github.com/openclaw/openclaw/pull/64241)，这代表了 OpenClaw 在 agent 行为规范化方向的重要尝试。新增的 strict-agentic 执行契约修订了 update_plan 语义，使得 agent 的规划执行更加可预期。

安全方面，**浏览器导航安全加固**[[33]](https://github.com/openclaw/openclaw/pull/64367) 收紧了严格浏览器主机名导航防护，**Agent Hook 信任处理规范化**[[35]](https://github.com/openclaw/openclaw/pull/64372) 则规范化了系统事件的信任处理逻辑。这些都是持续的安全加固工作。

功能层面，Gateway Dreaming 启动修复[[32]](https://github.com/openclaw/openclaw/pull/64258) 解决了启动时的 reconcile 问题，Matrix ACP 线程绑定修复[[34]](https://github.com/openclaw/openclaw/pull/64343) 保留了 thread binding targets，Telegram 媒体发送路由修复[[36]](https://github.com/openclaw/openclaw/pull/64492) 和 Feishu /btw 路由修复[[37]](https://github.com/openclaw/openclaw/pull/64324) 则完善了多渠道接入的稳定性。

## 结论

今天值得关注的判断是：**推理框架正在从"功能支持"向"深度优化"演进**。vLLM 密集的模型接入与量化 bugfix、SGLang 在 MoE 显存优化与 Diffusion FP8 的双线推进、以及 llama.cpp 后端无关 TP 的实验探索，都在指向同一个方向——基础能力已经初步就绪，现在的核心是把它做深、做精。

TRL 在 Tool Calling 训练支持上的持续扩展，则反映了 agent 能力构建的下一个重点：让模型不仅能"说话"，还能"做事"。

---

## 参考来源

[1] [vLLM Gemma4 Eagle3 支持](https://github.com/vllm-project/vllm/pull/39450)

[2] [vLLM EXAONE-4.5 模型接入](https://github.com/vllm-project/vllm/pull/39388)

[3] [vLLM GGUF 非标准量化类型支持](https://github.com/vllm-project/vllm/pull/39471)

[4] [vLLM 量化 KV Cache crash 修复](https://github.com/vllm-project/vllm/pull/39444)

[5] [vLLM 显式未量化 kv cache dtype 修复](https://github.com/vllm-project/vllm/pull/38922)

[6] [vLLM FlashInfer crash 修复](https://github.com/vllm-project/vllm/pull/39002)

[7] [SGLang MoE 并行组显存优化](https://github.com/sgl-project/sglang/pull/22515)

[8] [SGLang Diffusion ModelOpt FP8 支持](https://github.com/sgl-project/sglang/pull/22365)

[9] [SGLang Qwen3 Next MTP NCCL 修复](https://github.com/sgl-project/sglang/pull/22458)

[10] [SGLang GDN 非连续 Tensor 支持](https://github.com/sgl-project/sglang/pull/22312)

[11] [SGLang Kimi K25 EPD 支持](https://github.com/sgl-project/sglang/pull/22269)

[12] [llama.cpp Q1_0 Vulkan 支持](https://github.com/ggml-org/llama.cpp/pull/21539)

[13] [llama.cpp Gemma4 Reasoning Budget Sampler](https://github.com/ggml-org/llama.cpp/pull/21697)

[14] [llama.cpp WebGPU Intel 非方阵支持](https://github.com/ggml-org/llama.cpp/pull/21669)

[15] [llama.cpp 后端无关 Tensor Parallelism](https://github.com/ggml-org/llama.cpp/pull/21380)

[16] [llama.cpp 下载进度回调接口](https://github.com/ggml-org/llama.cpp/pull/21735)

[17] [TRL Qwen3-VL Tool Calling 支持](https://github.com/huggingface/trl/pull/5469)

[18] [TRL GLM-4-MoE Tool Calling 支持](https://github.com/huggingface/trl/pull/5463)

[19] [TRL Llama 3 generation markers](https://github.com/huggingface/trl/pull/5493)

[20] [TRL DistillationTrainer trackio 支持](https://github.com/huggingface/trl/pull/5501)

[21] [Megatron-LM DiT conditions_embeddings 支持](https://github.com/NVIDIA/Megatron-LM/pull/4134)

[22] [Megatron-LM MXFP8 FP8 DPA 支持](https://github.com/NVIDIA/Megatron-LM/pull/4066)

[23] [Megatron-LM CUDA Graph + Activation Offloading](https://github.com/NVIDIA/Megatron-LM/pull/4190)

[24] [TensorRT-LLM FP8 LoRA weight 文件加载支持](https://github.com/NVIDIA/TensorRT-LLM/pull/12848)

[25] [TensorRT-LLM KV Cache 统计监控增强](https://github.com/NVIDIA/TensorRT-LLM/pull/12413)

[26] [TensorRT-LLM Disagg KV Cache 传输 crash 修复](https://github.com/NVIDIA/TensorRT-LLM/pull/12909)

[27] [TensorRT-LLM LoRA Overallocation 部分修复](https://github.com/NVIDIA/TensorRT-LLM/pull/12817)

[28] [Mooncake Grouped Scatter RDMA Reads 支持](https://github.com/kvcache-ai/Mooncake/pull/1717)

[29] [Ray ClockInterface 共享测试工具](https://github.com/ray-project/ray/pull/62476)

[30] [Ray S3 Tokenizer 修复](https://github.com/ray-project/ray/pull/62121)

[31] [OpenClaw 严格 Agent 执行契约](https://github.com/openclaw/openclaw/pull/64241)

[32] [OpenClaw Gateway Dreaming 启动修复](https://github.com/openclaw/openclaw/pull/64258)

[33] [OpenClaw 浏览器导航安全加固](https://github.com/openclaw/openclaw/pull/64367)

[34] [OpenClaw Matrix ACP 线程绑定修复](https://github.com/openclaw/openclaw/pull/64343)

[35] [OpenClaw Agent Hook 信任处理规范化](https://github.com/openclaw/openclaw/pull/64372)

[36] [OpenClaw Telegram 媒体发送路由修复](https://github.com/openclaw/openclaw/pull/64492)

[37] [OpenClaw Feishu /btw 路由修复](https://github.com/openclaw/openclaw/pull/64324)

[38] [OpenClaw Skill taskflow frontmatter 修复](https://github.com/openclaw/openclaw/pull/64469)
