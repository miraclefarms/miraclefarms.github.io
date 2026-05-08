---
title: AI Infra 早报｜推理框架向 MoE 大模型全面调优，KV 缓存链路多点收敛
date: 2026-03-19 05:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: vLLM 针对 Qwen3.5 H200 MoE triton 配置实现 9.9% E2E 吞吐提升；TensorRT-LLM fused allreduce+RMSNorm 上线，KV cache 传输超时改为 60s（breaking）；SGLang 修复 DP attention overlap 跨 rank 不一致与 MiniMax M2 KV scale 加载错误；TRL 修复 DPO collator 截断顺序并支持 reward 自定义日志；Mooncake 推进 TENT 集成；OpenClaw 封堵 host exec sandbox env injection 漏洞并默认升级 MiniMax M2.7。
tags: [MoE, KV Cache, Speculative Decoding]
---

过去 24 小时，推理侧呈现两条清晰主线。一是 MoE 模型的推理调优走向精细化——vLLM 针对 Qwen3.5 在 H200 上的 MoE triton 配置实现 9.9% E2E 吞吐提升，TensorRT-LLM 落地 fused allreduce+RMSNorm 算子。二是 KV 缓存链路在多个层面同步收敛——vLLM KV offload 系列持续推进、SGLang 修复 MiniMax M2 KV scale 加载、TensorRT-LLM 将 KV cache 传输超时默认值改为 60s（breaking change）、Mooncake 推进与 TENT 的 PG 集成。训练侧则以 DPO 数据处理修复和 reward 可观测性增强为主。

## 推理侧

**vLLM 针对 Qwen3.5 H200 MoE 调优 triton 配置，E2E 吞吐提升 9.9%[[1]](https://github.com/vllm-project/vllm/pull/37340)** — 通过手工调优 MoE triton 内核配置填补 H200 上 Qwen3.5 的默认配置缺口。MoE 推理效率高度依赖内核配置与硬件特性的精确匹配，此次调优可直接用于生产部署。

**vLLM 修复 EP/DP 异步调度中 device 0 额外 CUDA context 问题[[2]](https://github.com/vllm-project/vllm/pull/37449)** — Expert Parallel + Data Parallel 组合下异步调度会在 device 0 产生多余 CUDA context，导致显存占用异常。修复后 EP/DP 部署的显存行为恢复正常。

**TensorRT-LLM fused allreduce+RMSNorm 算子上线[[3]](https://github.com/NVIDIA/TensorRT-LLM/pull/12201)** — 两步操作融合为单一算子，支持可选 residual，减少显存带宽压力，降低推理延迟。

**TensorRT-LLM KV cache 传输超时默认值改为 60s（breaking change）[[4]](https://github.com/NVIDIA/TensorRT-LLM/pull/12249)** — 旧默认值过低导致长上下文高负载场景频繁超时失败。升级时需注意：原依赖旧超时值的逻辑需重新评估。

**SGLang 修复 DP attention overlap 跨 rank 决策不一致[[5]](https://github.com/sgl-project/sglang/pull/20853)** — DP ranks 对 overlap 禁用的判断不一致会导致结果不确定。修复后所有 rank 在相同条件下行为一致。

**SGLang 修复 MiniMax M2 模型 KV cache scale 加载错误[[6]](https://github.com/sgl-project/sglang/pull/20870)** — scale 加载路径 bug 导致 MiniMax M2 量化推理结果异常，修复后可正常使用。

**vLLM KV offload+HMA 系列第 6/N 步[[7]](https://github.com/vllm-project/vllm/pull/37405)** — 拆分 offloading_connector.py，持续将 KV offload 能力模块化。

## 训练侧

**TRL DPOTrainer collator 修复：先截断再 padding[[8]](https://github.com/huggingface/trl/pull/5305)** — 原实现在 padding 后截断，序列对齐方式错误，可能导致 DPO 训练数据静默损坏。这是影响训练正确性的基础性修复。

**TRL reward function 支持记录额外列和标量指标[[9]](https://github.com/huggingface/trl/pull/5233)** — 允许 reward function 在返回分数的同时记录自定义列和标量，使 RLHF 训练过程中 reward 的内部状态可观测、可调试。

**Megatron-LM 修复 A2A Overlap 中 MoE 层 recompute assertion 缺失[[10]](https://github.com/NVIDIA/Megatron-LM/pull/3916)** — All-to-All 通信与计算 overlap 模式下，MoE gradient checkpointing 缺少必要校验，存在静默错误风险。

**Megatron-LM eviction 逻辑热修复[[11]](https://github.com/NVIDIA/Megatron-LM/pull/3914)** — 针对内存管理中 eviction 策略 bug 的高优热修复。

## 生产部署侧

**Mooncake 推进与 TENT 的 PG 集成[[12]](https://github.com/kvcache-ai/Mooncake/pull/1676)** — 初始化 Mooncake Process Group 与 TENT 分布式推理框架的集成路径，同步修复 P2P 内存区域本地注册问题[[13]](https://github.com/kvcache-ai/Mooncake/pull/1690)，将 Mooncake KV cache 传输能力引入大规模 TENT 场景。

**Megatron-LM 新增 GB200 并行集成测试[[14]](https://github.com/NVIDIA/Megatron-LM/pull/3901)** — CI 中增加针对 NVIDIA GB200 的并行集成测试轨道，为 GB200 正式支持铺路。

## 应用侧

**OpenClaw 修复 host exec sandbox env injection 安全漏洞[[15]](https://github.com/openclaw/openclaw/pull/49702)** — 封堵 build-tool 和 glibc 环境变量注入向量，防止恶意代码通过环境变量操控沙箱行为。

**OpenClaw 默认模型升级至 MiniMax M2.7[[16]](https://github.com/openclaw/openclaw/pull/49691)** — 新增 M2.7 系列并将默认模型从 M2.5 更新至 M2.7，无需手动配置即可使用最新版本。

---

## 参考文献

[1] [vLLM Qwen3.5 H200 MoE triton 调优 +9.9%](https://github.com/vllm-project/vllm/pull/37340)

[2] [vLLM EP/DP device 0 cuda context 修复](https://github.com/vllm-project/vllm/pull/37449)

[3] [TensorRT-LLM fused allreduce+RMSNorm](https://github.com/NVIDIA/TensorRT-LLM/pull/12201)

[4] [TensorRT-LLM KV cache 传输超时改为 60s](https://github.com/NVIDIA/TensorRT-LLM/pull/12249)

[5] [SGLang DP attention overlap 跨 rank 一致性修复](https://github.com/sgl-project/sglang/pull/20853)

[6] [SGLang MiniMax M2 KV cache scale 加载修复](https://github.com/sgl-project/sglang/pull/20870)

[7] [vLLM KV offload+HMA 第 6 步](https://github.com/vllm-project/vllm/pull/37405)

[8] [TRL DPOTrainer 先截断再 padding 修复](https://github.com/huggingface/trl/pull/5305)

[9] [TRL reward function 额外列与标量记录](https://github.com/huggingface/trl/pull/5233)

[10] [Megatron-LM A2A Overlap MoE recompute assertion 修复](https://github.com/NVIDIA/Megatron-LM/pull/3916)

[11] [Megatron-LM eviction 热修复](https://github.com/NVIDIA/Megatron-LM/pull/3914)

[12] [Mooncake PG 与 TENT 集成](https://github.com/kvcache-ai/Mooncake/pull/1676)

[13] [Mooncake P2P 本地内存强制注册](https://github.com/kvcache-ai/Mooncake/pull/1690)

[14] [Megatron-LM GB200 并行集成测试](https://github.com/NVIDIA/Megatron-LM/pull/3901)

[15] [OpenClaw host exec sandbox env injection 安全修复](https://github.com/openclaw/openclaw/pull/49702)

[16] [OpenClaw 默认模型升级至 MiniMax M2.7](https://github.com/openclaw/openclaw/pull/49691)
