---
title: AI Infra 早报｜TRT-LLM 全力冲刺 DeepSeek 推理产品线，Mooncake 引入语义级 KV 缓存管理
date: 2026-05-09 08:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: TRT-LLM 48 小时内连合 6 个 DeepSeek PR 覆盖全系列推理优化；Mooncake 首次在生产级 KV cache 系统实现 keyword-level 条件记忆；SGLang decode 热路径去阻塞 + AMD diffusion 全栈推进；推理部署生态加速多硬件多平台。
tags: [TRT-LLM, SGLang, Mooncake, KV-Cache, MoE, AMD]
---

![题图](/assets/2026-05-09-ai-infra-daily-brief/cover.png)


推理框架的竞争格局正在发生一次质变：从"谁先支持新模型架构"升级为"谁能在特定模型系列上跑得更快更稳"。TRT-LLM 在过去 48 小时内密集合入 6 个 DeepSeek 相关 PR，从 R1 的多流 MLA/MoE、V4 的 rotate activation gating，到 1024-expert 路由和 MegaMoE DeepGEMM kernel，**把 DeepSeek 全系列（R1/V3/V4）的推理性能当作一条独立产品线来推进**。与此同时，Mooncake 合入基于 DeepSeek 论文的 Engram 条件记忆支持，KV cache 管理从"全量保留或全量淘汰"进化到"按语义价值选择性保留"——这是一个值得单独拎出来讨论的范式切换。

## 一、TRT-LLM：DeepSeek 推理优化进入产品线化阶段

6 个 PR 的覆盖面本身就说明问题。AutoDeploy 为 DeepSeek-R1 启用了 multi-stream MLA/MoE 和 shared expert overlap[[1]](https://github.com/NVIDIA/TensorRT-LLM/pull/12946)，直接瞄准 decode 阶段的 throughput 瓶颈。DeepSeek V4 的 rotate activation gating 被条件化到 `HAS_FAST_HADAMARD` flag[[2]](https://github.com/NVIDIA/TensorRT-LLM/pull/13889)，说明 TRT-LLM 团队在针对 V4 compressor 的后处理做精细化的硬件能力适配，而非一刀切。expert 路由扩展到 1024 个 expert[[3]](https://github.com/NVIDIA/TensorRT-LLM/pull/13186)，配合 MegaMoEDeepGemmFusedMoE 后端 wrapping 了 DeepGEMM 的 fp8_fp4_mega_moe kernel[[4]](https://github.com/NVIDIA/TensorRT-LLM/pull/13384)——这两者组合在一起，指向的是超大 expert 池场景下的计算效率优化。

在 disagg serving 方向，KV reuse transceiver v2 实现了 KV block 的跨节点复用[[5]](https://github.com/NVIDIA/TensorRT-LLM/pull/13115)，而 MoE autotune 通过环境变量可配置化[[6]](https://github.com/NVIDIA/TensorRT-LLM/pull/13667)，降低了调优门槛。这组 PR 的共同特征是：每一个都在解决 DeepSeek 推理部署中的具体瓶颈，而且**从 MLA attention 到 MoE kernel 到 disagg 通信链路形成了完整覆盖**。NVIDIA 在推理市场的策略已经很清晰——以 TRT-LLM 为载体，逐个模型系列做深度垂直优化。

## 二、SGLang：decode 热路径去阻塞与 AMD diffusion 全栈优化

SGLang 的这批更新有两条明确主线。

第一条是 decode 延迟的极致压缩。阻塞式 H2D copy 被 removal[[7]](https://github.com/sgl-project/sglang/pull/24627)，JIT custom allreduce 默认开启[[8]](https://github.com/sgl-project/sglang/pull/24363)以避免 allreduce 的 worst-case 延迟，FP8 KV 路径切换到原生实现[[9]](https://github.com/sgl-project/sglang/pull/24129)并去掉了不必要的 bf16 assert[[10]](https://github.com/sgl-project/sglang/pull/24686)。这四个改动叠加起来，对 decode 阶段的延迟改善是结构性的——不是某个 kernel 快了 5%，而是**把 decode 路径上每个会 stall 的环节逐一消除了**。

第二条是 AMD ROCm diffusion 的全栈推进。FP8 MLA attention kernel 替换了 per-tensor flash attention[[11]](https://github.com/sgl-project/sglang/pull/20319)，Conv3D 通过 temporal unfolding 数学等价变换为 Conv2D[[12]](https://github.com/sgl-project/sglang/pull/22971)，RMSNorm 从 naive triton 实现替换为 aiter 实现、单 kernel 从 430us 降到 290us[[13]](https://github.com/sgl-project/sglang/pull/24360)。SGLang 正在把 AMD diffusion 推到和 CUDA 同等优先级，而不仅仅是"能用"。

## 三、Mooncake：Engram 条件记忆与多硬件扩张

Mooncake 合入的 Engram 支持来自 DeepSeek 的论文"Conditional Memory via Scalable Keyword-Level KV Cache Compression"[[14]](https://github.com/kvcache-ai/Mooncake/pull/1483)。**这是首次在生产级 KV cache 系统中实现 keyword-level 的条件记忆**——系统可以根据 KV block 的语义价值决定是否保留，而非简单地按 LRU 或 TTL 淘汰。对于一个服务于 prefix caching 和 disagg serving 的存储层来说，这意味着跨请求的 KV 复用率有望显著提升，因为被保留下来的恰好是最有价值的语义 anchor。

硬件生态方面，AMD CDNA4 (ROCm/HIP) 平台支持已合入[[15]](https://github.com/kvcache-ai/Mooncake/pull/2021)，P2P proxy 重构为 credit-based flow control[[16]](https://github.com/kvcache-ai/Mooncake/pull/1971)，两轮 RDMA use-after-free 修复[[17]](https://github.com/kvcache-ai/Mooncake/pull/2047)[[18]](https://github.com/kvcache-ai/Mooncake/pull/1903)说明 Mooncake 正在多硬件、高并发场景下快速积累稳定性。Mooncake 正在从 NVIDIA-only 转向真正的多硬件 KV cache 中间件。

## 四、推理部署生态加速多元化

同一天内有四个独立项目各自突破了非 NVIDIA / 非 x86 的部署路径，这不是巧合。NIXL 将 Intel XPU 设备识别为 VRAM[[19]](https://github.com/ai-dynamo/nixl/pull/1534)，LMCache 加入了 Azure Blob NIXL 后端[[20]](https://github.com/LMCache/LMCache/pull/3160)和 AMD GPU operator 支持[[21]](https://github.com/LMCache/LMCache/pull/3211)，llama.cpp 支持 Vertex AI 兼容 API[[22]](https://github.com/ggml-org/llama.cpp/pull/22545)，tokenspeed 为 MI300/MI350 硬编码了 LDS size 并探测 xGMI 拓扑[[23]](https://github.com/lightseekorg/tokenspeed/pull/25)。

推理栈的硬件和云平台选择正在从事实上的单一标准变成多选项。**这对用户侧的意义是：选定推理框架后，切换底层硬件或云平台的迁移成本正在被这些中间层逐步消化。**

## 五、Megatron-LM 遗留代码清理与训练可观测性

Megatron 连续删除了 legacy transformer[[24]](https://github.com/NVIDIA/Megatron-LM/pull/4207) 和 legacy GPT 代码[[25]](https://github.com/NVIDIA/Megatron-LM/pull/4322)，同时新增 GPU sniff test 检测硬件 straggler[[26]](https://github.com/NVIDIA/Megatron-LM/pull/4662)、optimizer CG 内存池共享[[27]](https://github.com/NVIDIA/Megatron-LM/pull/4521)、fine-grained offload 节流[[28]](https://github.com/NVIDIA/Megatron-LM/pull/4690)。清理和可观测性同步推进，配合 26.04-alpha.rc2 发布[[29]](https://github.com/NVIDIA/Megatron-LM/releases/tag/26.04-alpha.rc2)，**说明 Megatron 在为下一个大规模训练周期做架构准备——砍掉历史包袱的同时加固稳定性基础设施。**

补充值得一提的是 Ray Serve 为 LLM 应用启用了 direct streaming[[30]](https://github.com/ray-project/ray/pull/63167)，绕过 Serve proxy 直连后端 ASGI，对长文本流式场景的延迟改善应该很明显。trl 修复了一个 5 GB+ 的 CUDA 显存泄漏[[31]](https://github.com/huggingface/trl/pull/5700)并新增了 MFU 计算辅助函数[[32]](https://github.com/huggingface/trl/pull/5698)。llama.cpp 合入 MiMo V2.5 支持[[33]](https://github.com/ggml-org/llama.cpp/pull/22493)、CUDA snake activation 5-op 融合[[34]](https://github.com/ggml-org/llama.cpp/pull/22667)和 batch out_prod cublas 优化[[35]](https://github.com/ggml-org/llama.cpp/pull/22651)，b9080 release 支持 Gemma4_26B_A4B_NVFP4[[36]](https://github.com/ggml-org/llama.cpp/releases/tag/b9080)。DeepSpeed 发布 v0.19.0[[37]](https://github.com/deepspeedai/DeepSpeed/releases/tag/v0.19.0)。

## 六、今天真正值得记住的判断

TRT-LLM 用 48 小时 6 个 PR 的节奏表明，推理框架的竞争已经从"谁先支持新架构"进入了"谁的 DeepSeek 路线跑得更快更稳"的新阶段。而 Mooncake 的 Engram 条件记忆指向了一个更深层的趋势：**KV cache 管理正在从机械的淘汰策略转向基于语义价值的智能保留**——这条路线如果走通，对 disagg serving 的跨请求 KV 复用率会有本质提升。

---

## 参考来源

[1] [AutoDeploy: Optimize DeepSeek-R1 model performance](https://github.com/NVIDIA/TensorRT-LLM/pull/12946)

[2] [Gate DeepSeek V4 rotate activation on HAS_FAST_HADAMARD](https://github.com/NVIDIA/TensorRT-LLM/pull/13889)

[3] [Update deepseek routing — expand to 1024 experts](https://github.com/NVIDIA/TensorRT-LLM/pull/13186)

[4] [Add MegaMoEDeepGemmFusedMoE backend wrapping DeepGEMM](https://github.com/NVIDIA/TensorRT-LLM/pull/13384)

[5] [Introduce KV reuse in transceiver v2](https://github.com/NVIDIA/TensorRT-LLM/pull/13115)

[6] [Improve TRTLLM MoE autotune in DEP](https://github.com/NVIDIA/TensorRT-LLM/pull/13667)

[7] [logits: remove blocking H2D copy](https://github.com/sgl-project/sglang/pull/24627)

[8] [Turn on JIT custom AR implementation by default](https://github.com/sgl-project/sglang/pull/24363)

[9] [fix(aiter): drop FP8 KV upcast; use native FP8 path](https://github.com/sgl-project/sglang/pull/24129)

[10] [Remove unnecessary bf16 assert in rotate_activation](https://github.com/sgl-project/sglang/pull/24686)

[11] [Support fp8 MLA for diffusion model (AMD)](https://github.com/sgl-project/sglang/pull/20319)

[12] [Temporal-unfolded batched Conv2D for ROCm VAE decode](https://github.com/sgl-project/sglang/pull/22971)

[13] [Replace naive triton RMSNorm with aiter RMSNorm for diffusion](https://github.com/sgl-project/sglang/pull/24360)

[14] [Support Engram — conditional memory](https://github.com/kvcache-ai/Mooncake/pull/1483)

[15] [Add AMD CDNA4 (ROCm/HIP) platform support](https://github.com/kvcache-ai/Mooncake/pull/2021)

[16] [Refactor P2PProxy with credit-based flow control](https://github.com/kvcache-ai/Mooncake/pull/1971)

[17] [Add reference counting to RdmaTask to prevent UAF](https://github.com/kvcache-ai/Mooncake/pull/2047)

[18] [Fix use-after-free crash in ibv_post_send](https://github.com/kvcache-ai/Mooncake/pull/1903)

[19] [Recognize Intel XPU devices as VRAM](https://github.com/ai-dynamo/nixl/pull/1534)

[20] [Add support for AZURE_BLOB NIXL backend](https://github.com/LMCache/LMCache/pull/3160)

[21] [Add gpuVendor field to support AMD GPUs in operator](https://github.com/LMCache/LMCache/pull/3211)

[22] [server: support Vertex AI compatible API](https://github.com/ggml-org/llama.cpp/pull/22545)

[23] [Hardcode MI300/MI350 LDS size; probe xGMI topology](https://github.com/lightseekorg/tokenspeed/pull/25)

[24] [Remove legacy transformer and modules](https://github.com/NVIDIA/Megatron-LM/pull/4207)

[25] [Remove legacy GPT code](https://github.com/NVIDIA/Megatron-LM/pull/4322)

[26] [Add periodic GPU sniff tests to detect hardware stragglers](https://github.com/NVIDIA/Megatron-LM/pull/4662)

[27] [Allow optimizer CG to share the same pool as full-iter CG](https://github.com/NVIDIA/Megatron-LM/pull/4521)

[28] [Add a knob to throttle max allowed inflight offload](https://github.com/NVIDIA/Megatron-LM/pull/4690)

[29] [Megatron-LM 26.04-alpha.rc2 release](https://github.com/NVIDIA/Megatron-LM/releases/tag/26.04-alpha.rc2)

[30] [Enable direct streaming for Ray Serve LLM apps](https://github.com/ray-project/ray/pull/63167)

[31] [Fix 5 GB+ CUDA memory leak in activation offloading](https://github.com/huggingface/trl/pull/5700)

[32] [Add MFU helpers](https://github.com/huggingface/trl/pull/5698)

[33] [Add MiMo V2.5 model support](https://github.com/ggml-org/llama.cpp/pull/22493)

[34] [Fuse snake activation in CUDA](https://github.com/ggml-org/llama.cpp/pull/22667)

[35] [Batch out_prod inner loop with cublasSgemmStridedBatched](https://github.com/ggml-org/llama.cpp/pull/22651)

[36] [llama.cpp b9080 release: Gemma4_26B_A4B_NVFP4](https://github.com/ggml-org/llama.cpp/releases/tag/b9080)

[37] [DeepSpeed v0.19.0 release](https://github.com/deepspeedai/DeepSpeed/releases/tag/v0.19.0)