---
title: AI Infra 早报｜推理框架从通用优化转向逐模型精调，DSv4 推理栈进入工程阶段
date: 2026-05-06 08:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: 通用 kernel 优化空间收窄，SGLang 一天为四条模型路径推送专用优化；Megatron DSv4 HybridModel 推理栈从实验转入工程。
---

推理框架的竞争焦点正在发生一次静默的迁移。过去两年的主旋律是通用 attention kernel 的比拼——谁的 FlashAttention 实现更快、谁的 KV cache 调度更高效。但今天这批更新释放了一个明确的信号：**通用优化的边际收益已经显著收窄，框架之间的胜负手正在转向"对具体模型架构的极致适配"**。SGLang 在一天之内为 HiSparse、Gemma4 VLM、Diffusion/Hunyuan3D 四条路径同时推送专用 kernel；TRT-LLM 为 Qwen3.5 的 GDN 层做了三个 Triton kernel 融合；llama.cpp 用 Fast Walsh-Hadamard Transform 把 KV rotation 复杂度从 O(N²) 降到了 O(N log N)。与此同时，Megatron 在三天内为 DSv4 HybridModel 推理栈集中落地了 CSA/HCA、shared expert overlap、grouped gemm MoE 等一系列关键组件——DeepSeek V4 混合架构的推理路径已经从实验阶段转入工程阶段。

## 一、SGLang：一天四条模型路径的专用优化

SGLang 今天的更新密度本身就说明问题——不再是"一个通用改进惠及所有模型"，而是针对不同模型架构分别做 kernel 级适配。

HiSparse 方面，#23013 让 HiSparse 支持 FP8 KV cache，做法是路由到 flashmla_kv backend 来绕过 flashmla_sparse 不接受 FP8 输入的限制[[1]](https://github.com/sgl-project/sglang/pull/23013)。此前 HiSparse 只能用 BF16 KV，FP8 的引入直接降低了显存占用。#23391 则为滑动窗口注意力增加了 SWA HiCache 加速路径，统一了 radix cache[[2]](https://github.com/sgl-project/sglang/pull/23391)。

Gemma4 VLM 的优化集中在 #24048，融合了 PCG、fused RMSNorm、residual add 和 scalar 操作[[3]](https://github.com/sgl-project/sglang/pull/24048)。这类融合在 VLM 推理中特别有价值，因为视觉编码器和语言模型之间的 norm 层和投影层通常碎片化严重。

Diffusion 路径的更新更为密集：#24287 为 Hunyuan3D shape denoising 融合了 Flux-style norm/modulation 和 QK norm[[4]](https://github.com/sgl-project/sglang/pull/24287)；#24431 修复了 FSDP sharding 并暴露 Qwen-Image transformer 的分片规则[[5]](https://github.com/sgl-project/sglang/pull/24431)；#24332 让 Diffusion CFG-parallel 的 all_reduce/broadcast 容忍非连续张量，修复了 JoyAI auto CFG 并行[[6]](https://github.com/sgl-project/sglang/pull/24332)。

此外，#24424 为 AMD 平台的 DeepSeek V4 compressor 做了 element-wise kernel fusion[[7]](https://github.com/sgl-project/sglang/pull/24424)，#16859 在 RL 路径中为 DeepEP a2a 支持 `--enable-return-routed-experts`，让每个 attn-TP rank 只看到自己的 expert 子集[[8]](https://github.com/sgl-project/sglang/pull/16859)。

## 二、TRT-LLM：decode 路径的协调冲刺

TRT-LLM 这批更新呈现出一个明显的协调特征——围绕 decode 阶段的延迟和吞吐做定向优化。

最直接的改动是 #13505，消除了约 6 秒的 FMHA JIT 重编译。做法是在 CUDA graph warmup 阶段对齐 kernel 选择，避免 eager generation 时触发 JIT 重编译，同时 drop cubin 节省显存[[9]](https://github.com/NVIDIA/TensorRT-LLM/pull/13505)。6 秒在延迟敏感场景下是一个不可忽视的开销。#13748 优化了 beam-search 从 prefill 切换到 decode 的开销，直接影响 TTFT[[10]](https://github.com/NVIDIA/TensorRT-LLM/pull/13748)。#13012 则降低了 AutoDeploy decode 调度循环中的 Python/C++ API 调用开销，用 `get_last_tokens` 替代逐 token 调用[[11]](https://github.com/NVIDIA/TensorRT-LLM/pull/13012)。

在 prefill 侧，#13677 实现了 AutoDeploy MLA chunked prefill loop，在有 cache hit 的场景下直接从 KV cache 读回前缀而非重算[[12]](https://github.com/NVIDIA/TensorRT-LLM/pull/13677)。#13410 增加了 context multiCtaKv sparse fmha 支持[[13]](https://github.com/NVIDIA/TensorRT-LLM/pull/13410)。

模型适配方面，#12966 为 Qwen3.5 GDN 层做了三个 Triton kernel 融合——sigmoid 融入 gating kernel、split QKV 融合、split transpose 融合[[14]](https://github.com/NVIDIA/TensorRT-LLM/pull/12966)。#13247 让 AutoDrop 对接 Llama 4、Qwen3 Next、Qwen3 MoE[[15]](https://github.com/NVIDIA/TensorRT-LLM/pull/13247)。

## 三、Megatron：DSv4 HybridModel 推理栈密集落地

Megatron 在三天内为 DSv4 HybridModel 推理栈推送了大量组件，节奏之密集说明这条路径已经过了原型验证阶段，正在工程化。

CSA/HCA 原型已经加入 HybridModel（#4569），目前支持 CP=1、TP=1[[16]](https://github.com/NVIDIA/Megatron-LM/pull/4569)。mHC 支持基于 HybridModel/HybridStack 构建（#4469）[[17]](https://github.com/NVIDIA/Megatron-LM/pull/4469)。Shared expert overlap 通过 allgatherv 在推理中隐藏通信开销（#4570）[[18]](https://github.com/NVIDIA/Megatron-LM/pull/4570)。#4566 移植了 vLLM 的 grouped gemm kernel 到推理优化 MoE backend[[19]](https://github.com/NVIDIA/Megatron-LM/pull/4566)。

一个值得关注的架构决策是 #4306——推理上下文簿记从 GPU 迁到 pinned CPU，引入 ContextGPUView 作为 forward-pass 的唯一 GPU 接口[[20]](https://github.com/NVIDIA/Megatron-LM/pull/4306)。这种设计减少了 GPU 上的簿记开销，但需要确保 CPU-GPU 同步不会引入新的瓶颈。

集成测试也在同步推进：#4596 覆盖了 CSA/HCA attention + hash MoE routing + clamped SwiGLU[[21]](https://github.com/NVIDIA/Megatron-LM/pull/4596)；#4482 实现了 GPT_model 与 Hybrid_model 之间的 checkpoint 转换[[22]](https://github.com/NVIDIA/Megatron-LM/pull/4482)。安全层面，#4612 将 prefix caching 哈希从 polynomial rolling hash 替换为 SHA-256[[23]](https://github.com/NVIDIA/Megatron-LM/pull/4612)。

## 四、llama.cpp：基础算子的数学改进

llama.cpp 今天的两个核心改动都是数学层面的优化，而非工程层面的调参。

#22631 用 Fast Walsh-Hadamard Transform 替代矩阵乘法做 KV cache rotation，将复杂度从 O(N²) 降到 O(N log N)[[24]](https://github.com/ggml-org/llama.cpp/pull/22631)。这是一个量级级别的改进，尤其对长上下文场景影响显著。#22423 在 CPU 后端融合了 RMS_NORM + MUL，单 pass 计算消除了中间张量[[25]](https://github.com/ggml-org/llama.cpp/pull/22423)。

模型扩展方面，#22101 支持 IBM granite-4.0-1b-speech（Conformer encoder + QFormer projector）[[26]](https://github.com/ggml-org/llama.cpp/pull/22101)。#21848 实现了 `/models?reload=1` 热加载 API[[27]](https://github.com/ggml-org/llama.cpp/pull/21848)。#22679 将推理状态保存到 device buffer，消除 D2H 拷贝开销[[28]](https://github.com/ggml-org/llama.cpp/pull/22679)。

## 五、Mooncake 传输层从"能跑"到"不挂"

Mooncake 连续修复了三个多节点真实部署中的 bug：#1959 修复了 RDMA transport 连接设置中的循环死锁[[29]](https://github.com/kvcache-ai/Mooncake/pull/1959)；#2035 修复了 dmabuf 内存注册必须使用 allocation base addr 的边界要求[[30]](https://github.com/kvcache-ai/Mooncake/pull/2035)；#2034 修复了 dmabuf 注册前未初始化 CUDA primary context 的问题[[31]](https://github.com/kvcache-ai/Mooncake/pull/2034)。#2028 统一了 nvlink 和 ubshmem allocator 管线[[32]](https://github.com/kvcache-ai/Mooncake/pull/2028)。传输层正在经历从功能实现到生产健壮性的关键过渡。

TRL 这边，#5690 更新了 GRPO Liger kernel，加入 delta two-sided clipping、vespo 和 KL bias correction[[33]](https://github.com/huggingface/trl/pull/5690)。#5696 引入 OpenReward Standard 环境适配器[[34]](https://github.com/huggingface/trl/pull/5696)。#5406 实现了 length-normalized DPO loss[[35]](https://github.com/huggingface/trl/pull/5406)。RL 训练栈的工程成熟度在稳步提升。

## 今天真正值得记住的判断

推理框架的竞争正在从"谁的通用 attention 更快"转向"谁对更多模型做到了极致适配"。这背后是一个朴素的经济学：通用 kernel 的优化空间已经被 FlashAttention、FlashInfer、FA3 等几轮迭代榨得差不多了，剩下的收益集中在模型架构的边角——每一个 VLM 的 norm 融合、每一个 MoE 的 expert overlap、每一个 sparse attention 的 FP8 路径，都是 1-3% 的增量，但加起来决定了一个框架能否在 benchmark 中胜出。与此同时，**Megatron 对 DSv4 推理栈的工程化推进速度暗示着，下一个阶段的竞争重心可能从 serving 框架上移到训练-推理全栈的纵向整合**。

---

## 参考来源

[1] [SGLang #23013 — HiSparse FP8 KV cache](https://github.com/sgl-project/sglang/pull/23013)

[2] [SGLang #23391 — SWA HiCache unified radix cache](https://github.com/sgl-project/sglang/pull/23391)

[3] [SGLang #24048 — Gemma4 VLM PCG + fused RMSNorm](https://github.com/sgl-project/sglang/pull/24048)

[4] [SGLang #24287 — Hunyuan3D shape denoising fusion](https://github.com/sgl-project/sglang/pull/24287)

[5] [SGLang #24431 — Diffusion FSDP sharding fix](https://github.com/sgl-project/sglang/pull/24431)

[6] [SGLang #24332 — Diffusion CFG-parallel non-contiguous tensor fix](https://github.com/sgl-project/sglang/pull/24332)

[7] [SGLang #24424 — AMD DeepSeek V4 compressor kernel fusion](https://github.com/sgl-project/sglang/pull/24424)

[8] [SGLang #16859 — DeepEP a2a enable-return-routed-experts](https://github.com/sgl-project/sglang/pull/16859)

[9] [TRT-LLM #13505 — Eliminate ~6s FMHA JIT recompilation](https://github.com/NVIDIA/TensorRT-LLM/pull/13505)

[10] [TRT-LLM #13748 — Optimize beam-search prefill→decode switch](https://github.com/NVIDIA/TensorRT-LLM/pull/13748)

[11] [TRT-LLM #13012 — AutoDeploy reduce decode dispatch overhead](https://github.com/NVIDIA/TensorRT-LLM/pull/13012)

[12] [TRT-LLM #13677 — AutoDeploy MLA chunked prefill loop](https://github.com/NVIDIA/TensorRT-LLM/pull/13677)

[13] [TRT-LLM #13410 — context multiCtaKv sparse fmha](https://github.com/NVIDIA/TensorRT-LLM/pull/13410)

[14] [TRT-LLM #12966 — Qwen3.5 GDN Triton kernel fusion](https://github.com/NVIDIA/TensorRT-LLM/pull/12966)

[15] [TRT-LLM #13247 — AutoDrop Llama 4 / Qwen3 Next / Qwen3 MoE](https://github.com/NVIDIA/TensorRT-LLM/pull/13247)

[16] [Megatron #4569 — CSA/HCA prototype for HybridModel](https://github.com/NVIDIA/Megatron-LM/pull/4569)

[17] [Megatron #4469 — mHC support HybridModel on DSv4](https://github.com/NVIDIA/Megatron-LM/pull/4469)

[18] [Megatron #4570 — Shared expert overlap with allgatherv](https://github.com/NVIDIA/Megatron-LM/pull/4570)

[19] [Megatron #4566 — Grouped gemm MoE kernel from vLLM](https://github.com/NVIDIA/Megatron-LM/pull/4566)

[20] [Megatron #4306 — ContextGPUView inference context on CPU](https://github.com/NVIDIA/Megatron-LM/pull/4306)

[21] [Megatron #4596 — DSv4 hybrid transformer integration test](https://github.com/NVIDIA/Megatron-LM/pull/4596)

[22] [Megatron #4482 — GPT↔Hybrid checkpoint conversion](https://github.com/NVIDIA/Megatron-LM/pull/4482)

[23] [Megatron #4612 — Prefix caching SHA-256](https://github.com/NVIDIA/Megatron-LM/pull/4612)

[24] [llama.cpp #22631 — FWHT for KV cache rotation](https://github.com/ggml-org/llama.cpp/pull/22631)

[25] [llama.cpp #22423 — CPU RMS_NORM + MUL fusion](https://github.com/ggml-org/llama.cpp/pull/22423)

[26] [llama.cpp #22101 — IBM granite-4.0-1b-speech support](https://github.com/ggml-org/llama.cpp/pull/22101)

[27] [llama.cpp #21848 — /models?reload=1 hot reload API](https://github.com/ggml-org/llama.cpp/pull/21848)

[28] [llama.cpp #22679 — Inference state on device buffer](https://github.com/ggml-org/llama.cpp/pull/22679)

[29] [Mooncake #1959 — Fix RDMA connection setup circular deadlock](https://github.com/kvcache-ai/Mooncake/pull/1959)

[30] [Mooncake #2035 — dmabuf allocation base addr fix](https://github.com/kvcache-ai/Mooncake/pull/2035)

[31] [Mooncake #2034 — Initialize CUDA primary context before dmabuf](https://github.com/kvcache-ai/Mooncake/pull/2034)

[32] [Mooncake #2028 — Unify nvlink/ubshmem allocator pipeline](https://github.com/kvcache-ai/Mooncake/pull/2028)

[33] [TRL #5690 — GRPO Liger kernel delta/vespo/KL bias correction](https://github.com/huggingface/trl/pull/5690)

[34] [TRL #5696 — OpenReward Standard environment adapter](https://github.com/huggingface/trl/pull/5696)

[35] [TRL #5406 — Length-normalized DPO loss](https://github.com/huggingface/trl/pull/5406)