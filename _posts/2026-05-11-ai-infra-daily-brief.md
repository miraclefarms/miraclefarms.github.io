---
title: AI Infra 早报｜DeepSeek V4 推理全栈量产就绪，三大框架同日收口
date: 2026-05-11 08:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: vLLM、SGLang、TRT-LLM 同日合入 DSV4 量产关键能力，PP、NVFP4 AsyncTP、disaggregation 修复、专用 attention kernel 同步就位。TokenSpeed 一周 30+ PR 从概念验证加速跃过。
tags: [DeepSeek, MoE, KV-Cache, TokenSpeed, Heterogeneous-Hardware]
---

![题图](/assets/2026-05-11-ai-infra-daily-brief/cover.jpg)


DeepSeek V4 在本窗口正式跨过了“能投产”的门槛。vLLM、SGLang、TRT-LLM 在同一天窗内各自合入了生产部署所需的收口能力——Pipeline Parallelism、FP4 量化全链路、disaggregation 修复、专用 sparse attention kernel，几乎同步就位。一个开源推理生态在单一模型上集体在同一个窗口收口，这种同步性本身就是信号。

## 一、DeepSeek V4：PP、FP4、disaggregation 同日就位

vLLM 给 DSV4 补齐了 Pipeline Parallelism 支持[[1]](https://github.com/vllm-project/vllm/pull/41694)，这是大规模部署并行策略的最后一块拼图——在此之前 DSV4 在 vLLM 上缺少 PP，节点间扩展方式受限。同时 NVFP4 all-gather + GEMM fusion 接入了 AsyncTP[[2]](https://github.com/vllm-project/vllm/pull/41882)，打通了 FP4 量化 + 序列并行 + 异步 TP 的完整推理管线。v0.20.2 patch release 专门修了 DSV4 sparse attention 的 bug[[3]](https://github.com/vllm-project/vllm/releases/tag/v0.20.2)，已经到了修边角的地步。

SGLang 侧的统一 dispatch 路径[[4]](https://github.com/sgl-project/sglang/pull/24888)消除了独立 `state_type="dsv4"` 判别器带来的路径分歧；mooncake disaggregation 场景下 NIXL 传输缺失分支的修复[[5]](https://github.com/sgl-project/sglang/pull/24878)则补上了分散式推理的关键链路。

TRT-LLM 的进展分量更重——`o_a_proj` 保持原生 FP8 格式不动，同时移植了 vLLM 的 `fused_inv_rope_fp8_quant`[[6]](https://github.com/NVIDIA/TensorRT-LLM/pull/13938)，**首次在 TRT-LLM 侧打通了 DSV4 的 FP8 全链路**。dynamic sparse attention 专用 kernel[[7]](https://github.com/NVIDIA/TensorRT-LLM/pull/13652)和 disaggregation CI 覆盖 b200/b300[[8]](https://github.com/NVIDIA/TensorRT-LLM/pull/13874)同批合入，Blackwell 平台的正式验证流程已经启动。

## 二、TokenSpeed：一周 30+ PR，从概念验证到可评测

TokenSpeed 本周的密集合入值得单独一写。一周 30+ PR 覆盖了 DSV4-Flash 性能路径[[9]](https://github.com/lightseekorg/tokenspeed/pull/30)、Qwen3.5 runtime prepare 开销优化[[10]](https://github.com/lightseekorg/tokenspeed/pull/32)、Mamba 前缀缓存[[11]](https://github.com/lightseekorg/tokenspeed/pull/15)、O(k log N) 驱逐算法[[12]](https://github.com/lightseekorg/tokenspeed/pull/18)和 AMD ROCm 支持[[13]](https://github.com/lightseekorg/tokenspeed/pull/36)——**一个新推理引擎在不到一周内完成了从“概念验证”到“可评测”的跳跃**。

Mamba 前缀缓存的合入尤其值得注意：hybrid linear attention 的 scheduler-side slot 管理、COW restore、PD 模式支持，说明 TokenSpeed 不是在做简单的模型适配，而是在缓存策略层面有自己的设计路径。O(k log N) 驱逐算法用持久 LRU 集合替代每次 O(N log N) 重建堆，将驱逐操作从瓶颈变为常数级——工程上干净的选择。

## 三、MoE 推理效率：kernel 走向编译器，硬件走向多元

Gemma4、Kimi K2.5、Nemotron-H 等新 MoE 模型引爆了一轮推理 kernel 优化潮。SGLang 把 Gemma4 Attention RMSNorm 的三次独立 kernel launch 融合为一次[[14]](https://github.com/sgl-project/sglang/pull/24696)，DeepGemm `tf32_hc_prenorm_gemm` 融入 big_fuse[[15]](https://github.com/sgl-project/sglang/pull/24775)。TRT-LLM 则用 CUTEDSL 为 Nemotron-H 引入编译器驱动的 activation fusion[[16]](https://github.com/NVIDIA/TensorRT-LLM/pull/12884)，融合不再是一条手工 unroll 的循环——编译器开始替代手工 kernel 调优。

异构硬件侧同步推进。Mooncake 新增 maca_transport 为 Metax MACA C500 打通 intra-node P2P 传输层[[17]](https://github.com/kvcache-ai/Mooncake/pull/2059)，LMCache 的 ROCm Triton block-sparse attention backend[[18]](https://github.com/LMCache/LMCache/pull/3092)让 KV cache 复用能在 AMD GPU 上跑通。非 CUDA 路线的支持正在从“能编译”走向“能高性能运行”。

## 四、KV 缓存：从存储问题升级为检索问题

Mooncake 的 Engram 支持[[19]](https://github.com/kvcache-ai/Mooncake/pull/1483)基于 DeepSeek “Conditional Memory” 论文，在 KV cache 层面实现 keyword-level 语义检索复用。**KV 缓存不再只管内存够不够，而是管什么值得留在缓存里**——从分配管理走向内容感知的语义索引。磁盘副本回读的全链路修复[[20]](https://github.com/kvcache-ai/Mooncake/pull/2004)让此前持续返回 INVALID_PARAMS 的磁盘路径重新可用，这是一个低调但关键的可靠性补丁。

## 五、今天真正值得记住的判断

- **DSV4 从“能跑”进入“能投产”阶段**：三大框架同日收口不是巧合，是开源推理生态对 DSV4 生产部署的集体信号。PP、FP4、FP8 全链路、disaggregation 修复同步就位，意味着部署方不再需要“凑合着用”。
- **TokenSpeed 正加速走出边缘**：一周 30+ PR 覆盖 DSV4、Qwen3.5、Kimi K2.5 三条模型线加 ROCm 支持，从“新项目”变成“对比基准”可能只需要再几个月。
- **KV 缓存方向性升级**：Engram 语义缓存和 O(k log N) 驱逐是两个独立但指向一致的方向——缓存管理的核心问题从容量变成价值判断。

---

## 参考来源

[1] [vLLM #41694: DSV4 Pipeline Parallelism 支持](https://github.com/vllm-project/vllm/pull/41694)

[2] [vLLM #41882: NVFP4 all-gather + GEMM fusion 接入 AsyncTP](https://github.com/vllm-project/vllm/pull/41882)

[3] [vLLM v0.20.2 release: 修 DSV4 sparse attention 等 bug](https://github.com/vllm-project/vllm/releases/tag/v0.20.2)

[4] [SGLang #24888: 统一 DSV4 dispatch 路径](https://github.com/sgl-project/sglang/pull/24888)

[5] [SGLang #24878: 修复 DSV4 mooncake disaggregation NIXL 传输](https://github.com/sgl-project/sglang/pull/24878)

[6] [TRT-LLM #13938: DSV4 FP8 o_a_proj + fused_inv_rope_fp8_quant](https://github.com/NVIDIA/TensorRT-LLM/pull/13938)

[7] [TRT-LLM #13652: DSV4 MLA dynamic sparse attention kernel](https://github.com/NVIDIA/TensorRT-LLM/pull/13652)

[8] [TRT-LLM #13874: DSV4 disaggregation CI 覆盖 b200/b300](https://github.com/NVIDIA/TensorRT-LLM/pull/13874)

[9] [TokenSpeed #30: DSV4-Flash 性能路径](https://github.com/lightseekorg/tokenspeed/pull/30)

[10] [TokenSpeed #32: Qwen3.5 runtime prepare 开销优化](https://github.com/lightseekorg/tokenspeed/pull/32)

[11] [TokenSpeed #15: Mamba 前缀缓存](https://github.com/lightseekorg/tokenspeed/pull/15)

[12] [TokenSpeed #18: O(k log N) 驱逐算法](https://github.com/lightseekorg/tokenspeed/pull/18)

[13] [TokenSpeed #36: AMD ROCm MI355 eval CI 支持](https://github.com/lightseekorg/tokenspeed/pull/36)

[14] [SGLang #24696: Gemma4 Attention RMSNorm kernel 融合](https://github.com/sgl-project/sglang/pull/24696)

[15] [SGLang #24775: MHC pipeline DeepGemm fusion 优化](https://github.com/sgl-project/sglang/pull/24775)

[16] [TRT-LLM #12884: CUTEDSL MoE backend 编译器驱动 activation fusion](https://github.com/NVIDIA/TensorRT-LLM/pull/12884)

[17] [Mooncake #2059: maca_transport Metax MACA C500 intra-node P2P](https://github.com/kvcache-ai/Mooncake/pull/2059)

[18] [LMCache #3092: ROCm Triton block-sparse attention backend](https://github.com/LMCache/LMCache/pull/3092)

[19] [Mooncake #1483: Engram 语义缓存支持](https://github.com/kvcache-ai/Mooncake/pull/1483)

[20] [Mooncake #2004: GPU KV cache 磁盘副本回读修复](https://github.com/kvcache-ai/Mooncake/pull/2004)