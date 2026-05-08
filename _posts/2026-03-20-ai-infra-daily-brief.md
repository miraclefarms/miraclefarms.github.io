---
title: AI Infra 早报｜投机采样进驻引擎主链路，KV 内存管理跨越抽象门槛
date: 2026-03-20 05:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: vLLM ModelRunnerV2 补全投机解码 rejection sampler 全路径，TensorRT-LLM 将投机解码接入 TrtllmGen 高性能后端；KV 内存管理多线升级——TRT-LLM 约束式分区、Mooncake 连接器抽象、LMCache 非连续分配；训练侧 TRL 引入实验性 DPPO，Megatron-LM 实现 RL forced lag 机制。
tags: [Speculative Decoding, KV Cache]
---

过去 24 小时，AI Infra 有两条主线同步推进。一是**投机采样在主流推理引擎中走向工程标配**：两大引擎在同一窗口各自补完了关键缺口，不再是实验特性。二是**KV 内存管理的抽象层正在系统性重建**：TRT-LLM、Mooncake、LMCache 三个项目几乎在同一天各自推进了 KV 内存的架构升级，方向都是"从一块显存变成有接口的基础设施组件"。

## 一、投机采样走向主链路：两大引擎同步补完

**vLLM 这次补的是 ModelRunnerV2 下 rejection sampler 的两个缺口**：greedy 路径不完整[[1]](https://github.com/vllm-project/vllm/pull/37238)，以及 logprobs 缺失[[2]](https://github.com/vllm-project/vllm/pull/37237)。这两个问题加在一起意味着投机解码在 MRV2 下不能完整覆盖主要用例——某些场景能用，某些场景不能用，工程上等于"不稳定支持"。补完之后，rejection sampler 在 MRV2 路径下才算真正进入主链路。

**TensorRT-LLM 的进展发生在另一个层面**：将投机解码接入 TrtllmGen 注意力后端[[3]](https://github.com/NVIDIA/TensorRT-LLM/pull/12267)。TrtllmGen 是 TRT-LLM 为新架构设计的高性能注意力路径，此前投机解码绕过了这一层，意味着开启投机解码就必须放弃 TrtllmGen 的性能收益。打通之后，两者可以同时开启。

两个项目在同一时间窗口各自做了这件事，不是巧合——这是一个技术周期到了某个成熟度节点后的集体信号。

## 二、KV 内存管理的三层升级

这一天 KV 内存管理方向的更新尤其密集，而且每个项目升级的是不同的层次。

**TensorRT-LLM 做的是策略层**：为 KVCacheManagerV2 引入约束式内存分区[[14]](https://github.com/NVIDIA/TensorRT-LLM/pull/12212)，允许按策略把 KV cache 显存划成独立区域。这直接服务于多租户场景：不同请求或租户的 KV 内存互相隔离，SLA 保障才有物理基础。

**Mooncake 做的是存储后端抽象**：为 Store 层统一引入 Connector 接口[[15]](https://github.com/kvcache-ai/Mooncake/pull/1707)，OSS、HuggingFace Hub 和 Redis 都可以作为 KV cache 的持久化或分发后端接入。此前 Mooncake 的存储后端是特定绑定的，接口化之后才真正支持多云和混合存储场景。

**LMCache 做的是分配器层**：MemoryAllocator 新增非连续内存块分配支持[[16]](https://github.com/LMCache/LMCache/pull/2767)，允许在物理显存碎片化时仍然有效利用可用内存；同步推进 NIXL OBJ 插件的 VRAM_SEG 支持[[17]](https://github.com/LMCache/LMCache/pull/2640)，完善与 NIXL 传输层的集成。此外还将全局 transfer_lock 拆分为 per-device 粒度[[18]](https://github.com/LMCache/LMCache/pull/2816)，消除多设备并发时的死锁问题。

三个项目分别在策略、后端和分配器三个维度各自升级，方向一致但不重叠——KV cache 正在从"一块共享显存"变成一套有层次、有接口、可扩展的基础设施。

## 三、推理侧的其他进展

**vLLM 默认开启 Triton autotuning 磁盘缓存[[4]](https://github.com/vllm-project/vllm/pull/37188)**，对频繁重启的容器化部署场景意义明显——此前每次启动都要重新调优，冷启动开销不可忽视；默认化之后这个问题系统性消失。

**SGLang 这天落地了两个内核更新**：CuteDSL mm_fp4 后端[[5]](https://github.com/sgl-project/sglang/pull/18801) 扩充 FP4 量化选项，Expert Specialization Grouped GEMM[[6]](https://github.com/sgl-project/sglang/pull/15471) 为 MoE 专家专化场景补上专用内核（sgl-kernel 系列第 6/7 步）。同时修复了 DeepSeek-R1 在数据并行模式下触发 GPU-fault 的问题[[7]](https://github.com/sgl-project/sglang/pull/20841)，这个 bug 直接影响 DP 模式的大规模生产部署。

**llama.cpp 在端侧两条线同步推进**：Hexagon HMX NPU 后端新增矩阵加速算子[[8]](https://github.com/ggml-org/llama.cpp/pull/20693)，将移动端 NPU 覆盖进一步扩宽；WebGPU 侧同步补全 Qwen3.5 所需算子[[9]](https://github.com/ggml-org/llama.cpp/pull/20687)，端侧推理的硬件覆盖宽度持续延伸。

## 四、训练侧：后训练算法库在扩容

**TRL 合并了实验性 DPPO（Divergence Proximal Policy Optimization）[[10]](https://github.com/huggingface/trl/pull/5117)**。DPPO 在 PPO 基础上引入散度约束，旨在控制策略更新幅度、提升训练稳定性。以实验性标记合并，说明 TRL 有意系统性地扩宽 RL 算法覆盖——不只是 PPO/GRPO，而是把更多研究成果变成工程基线。同期修复了 GRPO tool-calling 循环中的 re-tokenization bug[[11]](https://github.com/huggingface/trl/pull/5242)，原因是字符串拼接替代了 token ID 直接拼接，导致 tokenizer 边界对齐错误。

**Megatron-LM 完成了 RL forced lag（强制滞后）机制[[13]](https://github.com/NVIDIA/Megatron-LM/pull/3517)**，并新增新兴优化器支持[[12]](https://github.com/NVIDIA/Megatron-LM/pull/3907)。forced lag 通过显式控制策略更新与价值函数更新之间的时序滞后，规避梯度估计偏差，是 RL 训练走向生产级可靠性的重要补全。

## 五、应用侧

**llama.cpp server** 完成了两项稳定性工作：统一采样参数默认值的 source of truth[[19]](https://github.com/ggml-org/llama.cpp/pull/20558)（此前 server 与底层 common 库各自定义导致行为不一致），以及修复 router 模式下的死锁问题[[20]](https://github.com/ggml-org/llama.cpp/pull/20763)。oaicompat 响应新增 cached_tokens 字段[[21]](https://github.com/ggml-org/llama.cpp/pull/19361)，使前端可以观测提示缓存命中情况，对成本优化有实用价值。

**OpenClaw CLI 新增失效凭据自动清理[[22]](https://github.com/openclaw/openclaw/pull/50639)**，切换模式时自动剔除不活跃的 gateway auth credentials，减少手动干预并降低泄露面。

---

## 参考来源

[1] [vLLM MRV2 rejection sampler greedy 路径补全](https://github.com/vllm-project/vllm/pull/37238)

[2] [vLLM MRV2 rejection sampler logprobs 支持](https://github.com/vllm-project/vllm/pull/37237)

[3] [TensorRT-LLM TrtllmGen 后端投机解码](https://github.com/NVIDIA/TensorRT-LLM/pull/12267)

[4] [vLLM Triton autotuning 磁盘缓存默认开启](https://github.com/vllm-project/vllm/pull/37188)

[5] [SGLang CuteDSL mm_fp4 后端](https://github.com/sgl-project/sglang/pull/18801)

[6] [SGLang Expert Specialization Grouped GEMM](https://github.com/sgl-project/sglang/pull/15471)

[7] [SGLang DeepSeek-R1 DP GPU-fault 修复](https://github.com/sgl-project/sglang/pull/20841)

[8] [llama.cpp Hexagon HMX NPU 后端](https://github.com/ggml-org/llama.cpp/pull/20693)

[9] [llama.cpp WebGPU Qwen3.5 算子支持](https://github.com/ggml-org/llama.cpp/pull/20687)

[10] [TRL Divergence PPO 实验性实现](https://github.com/huggingface/trl/pull/5117)

[11] [TRL GRPO tool-calling re-tokenization 修复](https://github.com/huggingface/trl/pull/5242)

[12] [Megatron-LM 新增新兴优化器](https://github.com/NVIDIA/Megatron-LM/pull/3907)

[13] [Megatron-LM RL forced lag 实现](https://github.com/NVIDIA/Megatron-LM/pull/3517)

[14] [TensorRT-LLM KVCacheManagerV2 约束式内存分区](https://github.com/NVIDIA/TensorRT-LLM/pull/12212)

[15] [Mooncake Store 连接器抽象（OSS/HF/Redis）](https://github.com/kvcache-ai/Mooncake/pull/1707)

[16] [LMCache 非连续内存分配支持](https://github.com/LMCache/LMCache/pull/2767)

[17] [LMCache NIXL OBJ VRAM_SEG 支持](https://github.com/LMCache/LMCache/pull/2640)

[18] [LMCache per-device transfer_lock 死锁修复](https://github.com/LMCache/LMCache/pull/2816)

[19] [llama.cpp server 采样参数 source of truth 统一](https://github.com/ggml-org/llama.cpp/pull/20558)

[20] [llama.cpp server router 死锁修复](https://github.com/ggml-org/llama.cpp/pull/20763)

[21] [llama.cpp oaicompat cached_tokens 字段](https://github.com/ggml-org/llama.cpp/pull/19361)

[22] [OpenClaw CLI 失效凭据自动清理](https://github.com/openclaw/openclaw/pull/50639)
