---
title: AI Infra 早报｜推理框架密集迭代，多模态与分布式缓存持续深化
date: 2026-04-07 05:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: 过去 24 小时，AI Infra 呈现三大核心信号：推理框架从"功能可用"转向"默认最优"——SGLang v0.5.10 将 Piecewise CUDA Graph 设为默认、llama.cpp 落地 Q1_0 1-bit 量化；多模态支持从选装走向标配——vLLM 完善 Gemma4 Fast Prefill、llama.cpp 新增 HunyuanOCR 支持；分布式缓存层与推理引擎紧耦合，Mooncake 修复 TE endpoint bug、LMCache CI 完善。整体呈现"推理即基础设施"的成熟特征。
tags: [Inference, Multimodal, KV Cache, Quantization]
---

过去 24 小时，AI Infra 最值得关注的变化是**推理框架从"功能可用"阶段加速进入"默认行为最优"阶段**。SGLang v0.5.10 将 Piecewise CUDA Graph 设为默认执行模式，llama.cpp 落地 Q1_0 1-bit 量化，vLLM 在多条产品线上并行迭代。这些变化的共同指向是：推理引擎不再是"能跑"的代名词，而是"跑得最优"的基础设施。

## 一、SGLang v0.5.10：Piecewise CUDA Graph 默认为生产ready

SGLang v0.5.10 最核心的发布要点是 **Piecewise CUDA Graph 设为默认执行模式**[[1]](https://github.com/sgl-project/sglang/releases/tag/v0.5.10)。

在此之前，具有复杂控制流 patterns 的模型需要用户手动配置才能获得 CUDA Graph 的性能收益。现在的变化意味着：开箱即可获得内存开销降低、吞吐提升的收益，这对生产部署来说是显著的体验提升。

同期发布的还有 **Elastic EP for Partial Failure Tolerance**[[2]](https://lmsys.org/blog/2026-03-25-eep-partial-failure-tolerance/)，支持 DeepSeek MoE 部署时单卡故障不中断服务——当一张 GPU 失败时，系统自动 redistributes expert weights 并继续服务，无需 full restart。对大规模 MoE 部署来说，这是关键的生产保障。

**GPU Staging Buffer for PD Disaggregation**[[3]](https://github.com/sgl-project/sglang/pull/19890) 则将 scattered head slices gather 成 contiguous memory 进行 bulk RDMA transfer，在 Qwen3.5 模型的 Prefill TP4+Decode DEP4 配置下，大规模并发的 TPS/Gpu 提升了约 5 倍。

此外，**HiSparse 稀疏注意力后端**[[4]](https://github.com/sgl-project/sglang/pull/20343) 面向长上下文推理场景，通过 sparsity-aware attention 降低计算成本。

## 二、llama.cpp：1-bit 量化与多模态并进

llama.cpp 在过去 24 小时发布了多个版本，核心亮点是 **Q1_0 1-bit 量化支持（CPU）**[[5]](https://github.com/ggml-org/llama.cpp/pull/21273)。

1-bit 量化是目前最极致的压缩方案之一，在 CPU 上实现意味着可以在消费级硬件上运行大模型，这对端侧推理场景意义重大。

同步推出的还有 **HunyuanOCR 视觉模型支持**[[6]](https://github.com/ggml-org/llama.cpp/releases/tag/b8670)，新增持文字和视觉模型、perceiver-based projector、独立的 chat template 等。**优化的 flash_attn_stream_k_fixup CUDA 内核**[[7]](https://github.com/ggml-org/llama.cpp/pull/21159) 则针对 nblocks_stream_k 是 ntiles_dst 倍数的场景进行了专项优化。

**Gemma4 BPE 分词器的字节 token 处理**[[8]](https://github.com/ggml-org/llama.cpp/pull/21488) 是另一个值得关注的支持——Gemma4 在 llama.cpp 上的支持从 MoE 路由扩展到了分词器层面。

## 三、vLLM：多产品线并行迭代

vLLM 在过去 24h 合并了 20+ 条 PRs，覆盖多个产品线：

**DeepSeek V3.2 挂起问题修复**[[9]](https://github.com/vllm-project/vllm/pull/39098) 通过设置 `skip_attn=False` 解决了 V3.2 挂起问题，这是 DeepSeek 系列模型在 vLLM 上稳定运行的关键 bugfix。

**Gemma4 Fast Prefill 优化**[[10]](https://github.com/vllm-project/vllm/pull/38879) 提升了多模态模型的首 token 速度。**Mistral Grammar Factory 支持**[[11]](https://github.com/vllm-project/vllm/pull/38150) 增强了结构化输出能力。**MLA FA4 重新启用**[[12]](https://github.com/vllm-project/vllm/pull/38819) 将 FlashAttention4 回归 MLA prefill 默认后端。

此外还新增了多款新模型支持：Param2MoE、Eagle3 speculative decoding、MiniMax-M2 等[[13]](https://github.com/vllm-project/vllm/pull/37512)。

## 四、分布式缓存层：Mooncake + LMCache 紧耦合推理框架

缓存层的 bugfix 直接影响推理服务的稳定性。

**Mooncake 修复 Store TE endpoint 填充错误**[[14]](https://github.com/kvcache-ai/Mooncake/pull/1827) 确保 Torch Extend endpoint 正确路由。**优雅 shutdown 实现**[[15]](https://github.com/kvcache-ai/Mooncake/pull/1795) 和 CPU-only 测试回归 CI 提升了进程的管控能力。

**LMCache CI 目标调整**[[16]](https://github.com/LMCache/LMCache/pull/2958) 适配 K3 nightly 综合测试。

这些变化的信号是：缓存层与推理框架的紧耦合已经成为 standard practice，KV Cache 的正确性是生产部署的生命线。

## 五、OpenClaw v2026.4.5：视频/音乐生成 + ComfyUI 工作流

**OpenClaw v2026.4.5**[[17]](https://github.com/openclaw/openclaw/releases/tag/v2026.4.5) 是今天应用层最重磅的更新：

- **内置 video_generate 工具**[[18]](https://github.com/openclaw/openclaw/releases/tag/v2026.4.5)：agents 可直接创建视频并返回生成的媒体
- **内置 music_generate 工具**[[19]](https://github.com/openclaw/openclaw/releases/tag/v2026.4.5)：支持 Google Lyria、 MiniMax、Comfy 工作流
- **ComfyUI 集成**[[20]](https://github.com/openclaw/openclaw/releases/tag/v2026.4.5)：本地/云端 Comfy 工作流支持图片/视频/音乐生成
- **Mantle 支持**[[21]](https://github.com/openclaw/openclaw/releases/tag/v2026.4.5)：Amazon Bedrock 上托管模型的简化配置

同期还包含多项安全修复：移除遗留的 danger config aliases、Discord 代理安全加固[[22]](https://github.com/openclaw/openclaw/pull/62078)、危险构建工具环境变量屏蔽[[23]](https://github.com/openclaw/openclaw/pull/62079)、ReDoS 防护[[24]](https://github.com/openclaw/openclaw/pull/61903) 等。

## 结论

过去 24 小时最核心的判断是：**推理框架正在从"功能可用"转向"默认最优"阶段**。SGLang v0.5.10 将 Piecewise CUDA Graph 设为默认、llama.cpp 落地 Q1_0 1-bit 量化、vLLM 并行迭代多条产品线——这些变化的共同指向是：推理引擎不再需要用户做复杂的配置选择，最优路径已经成为开箱默认行为。这标志着 AI Infra 在推理侧已经进入了"基础设施即默认"的成熟阶段。

---

## 参考来源

[1] [SGLang v0.5.10 Release](https://github.com/sgl-project/sglang/releases/tag/v0.5.10)

[2] [Elastic EP for Partial Failure Tolerance](https://lmsys.org/blog/2026-03-25-eep-partial-failure-tolerance/)

[3] [GPU Staging Buffer for PD Disaggregation](https://github.com/sgl-project/sglang/pull/19890)

[4] [HiSparse for Sparse Attention](https://github.com/sgl-project/sglang/pull/20343)

[5] [ggml: add Q1_0 1-bit quantization support (CPU)](https://github.com/ggml-org/llama.cpp/pull/21273)

[6] [model: add HunyuanOCR support](https://github.com/ggml-org/llama.cpp/releases/tag/b8670)

[7] [CUDA: Write an optimized flash_attn_stream_k_fixup kernel](https://github.com/ggml-org/llama.cpp/pull/21159)

[8] [vocab: add byte token handling to BPE detokenizer for Gemma4](https://github.com/ggml-org/llama.cpp/pull/21488)

[9] [Fix hanging issue with DeepSeek V3.2](https://github.com/vllm-project/vllm/pull/39098)

[10] [Gemma4 Enable Fast Prefill Optimization](https://github.com/vllm-project/vllm/pull/38879)

[11] [Mistral Grammar Support Grammar Factory](https://github.com/vllm-project/vllm/pull/38150)

[12] [Re-enable FA4 as default MLA prefill backend](https://github.com/vllm-project/vllm/pull/38819)

[13] [MiniMax-M2: add Eagle3 speculative decoding support](https://github.com/vllm-project/vllm/pull/37512)

[14] [Fixed the problem of filling in the wrong TE endpoint](https://github.com/kvcache-ai/Mooncake/pull/1827)

[15] [Implement graceful shutdown and reland CPU-only tests to CI](https://github.com/kvcache-ai/Mooncake/pull/1795)

[16] [Change dst for K3 nightly comprehensive results](https://github.com/LMCache/LMCache/pull/2958)

[17] [openclaw 2026.4.5 Release](https://github.com/openclaw/openclaw/releases/tag/v2026.4.5)

[18] [Agents/video generation: add the built-in video_generate tool](https://github.com/openclaw/openclaw/releases/tag/v2026.4.5)

[19] [Tools/music generation: add the built-in music_generate tool](https://github.com/openclaw/openclaw/releases/tag/v2026.4.5)

[20] [Providers/ComfyUI: add bundled workflow media plugin](https://github.com/openclaw/openclaw/releases/tag/v2026.4.5)

[21] [Providers/Amazon Bedrock: add bundled Mantle support](https://github.com/openclaw/openclaw/releases/tag/v2026.4.5)

[22] [fix: this is a real approval boundary bypass](https://github.com/openclaw/openclaw/pull/62078)

[23] [fix: multiple dangerous build tool environment variables](https://github.com/openclaw/openclaw/pull/62079)

[24] [fix(agents): prevent ReDoS in interpreter heuristic regexes](https://github.com/openclaw/openclaw/pull/61903)