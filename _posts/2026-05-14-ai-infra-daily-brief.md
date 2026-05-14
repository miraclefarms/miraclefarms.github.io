---
title: AI Infra 早报｜量化配置从枚举走向可组合规格，NVFP4 跨框架同日就位
date: 2026-05-14 08:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: vLLM 用 QuantKey/QuantSpec 重写量化配置架构，SGLang 与 TRT-LLM 同日合入 NVFP4 支持——量化从逐格式硬编码适配转向可组合、可 per-layer 覆盖的规格系统，NVFP4 成为首个跨框架量产验证的量化格式。
tags: [Quantization, KV Cache, Speculative Decoding, Attention, MoE]
---

![题图](/assets/2026-05-14-ai-infra-daily-brief/cover.png)


vLLM 把量化配置从枚举硬编码改成了 `QuantSpec(weight, activation)` 的可组合规格[[1]](https://github.com/vllm-project/vllm/pull/41566)，同一天 SGLang 接入 MXFP4 MoE[[5]](https://github.com/sgl-project/sglang/pull/24816)，TRT-LLM 正式支持 DSV4 NVFP4[[8]](https://github.com/NVIDIA/TensorRT-LLM/pull/14026)。量化不再是"某个模型碰巧支持某个格式"的碎片化状态，而是一条可组合、可逐层覆盖的基础设施管线——NVFP4 作为 DeepSeek V4 / GPT-OSS 的主力量化格式，今天跨三大框架同时就位，是这个管线第一次量产验证。

## 一、QuantKey/QuantSpec：量化配置的结构性重写

vLLM #41566 引入 `QuantKey` 和 `QuantSpec`，用 `QuantSpec(weight, activation)` 替代原来的 `OnlineQuantScheme` 枚举。新架构允许按 layer name 精确寻址并独立覆盖 activation 量化方式——**这意味着"这个模型的第 37 层用 FP8 激活、其余层用 INT8"这种需求从手动 hack 变成了配置声明**[[1]](https://github.com/vllm-project/vllm/pull/41566)。同窗口 vLLM 还合入了 Quark NVFP4 checkpoint 加载[[2]](https://github.com/vllm-project/vllm/pull/35859)和 NVFP4 KV 在 sliding window 下的 page size 修复[[3]](https://github.com/vllm-project/vllm/pull/42464)，让 GPT-OSS 在 NVFP4 全链路上可正确运行。SGLang 修复了 NVFP4 权重处理后 src scale 被释放导致热重载崩溃的问题[[7]](https://github.com/sgl-project/sglang/pull/25190)——当量化变成 per-layer 规格组合，权重生命周期管理也要跟上。

## 二、KV Offload 从"通不通"转到"怎么管"

昨天日报记录了 KV offload 跨框架打通传输路径；今天的焦点挪到了搬完之后的管理。vLLM 合入多级 KV offload 框架，在单级 CPU offload 上扩展链式二级 tier[[10]](https://github.com/vllm-project/vllm/pull/40020)，并为 OffloadingManager 注入 per-request 身份追踪[[11]](https://github.com/vllm-project/vllm/pull/42507)。LMCache 把 PD 后端的 `batched_submit_put_task` 改为 fire-and-forget enqueue，worker 线程不再阻塞等待远程 alloc + RDMA write[[12]](https://github.com/LMCache/LMCache/pull/3038)；同时用 `batched_contains()` 替代逐 key 串行查询[[13]](https://github.com/LMCache/LMCache/pull/2966)。Mooncake 引入 `ObjectDataType` 枚举让元数据层知道每块存的是什么[[14]](https://github.com/kvcache-ai/Mooncake/pull/1719)，并扩展 DSA 式分配策略基准[[15]](https://github.com/kvcache-ai/Mooncake/pull/2080)。**分层存储、类型感知、非阻塞传输——KV offload 的核心竞争力已经从"能不能搬"变成"搬完怎么管"。**

## 三、推测解码：自定义提案者打开实验空间

vLLM 合入自定义 callable proposer 后端[[17]](https://github.com/vllm-project/vllm/pull/39487)，研究者可以直接传一个 Python callable 当 drafter，不用再加载完整 HF 模型——**这是推测解码从"厂商预置方案"走向"研究者可快速实验新策略"的分界点**。配套地，hidden-state 提取开始支持 Qwen3.5 等混合注意力模型[[18]](https://github.com/vllm-project/vllm/pull/39949)，verifier 也移除了硬编码模型白名单检查改为能力检测[[19]](https://github.com/vllm-project/vllm/pull/42536)。EAGLE3 侧，TokenSpeed 形成了 embed+norm 融合[[21]](https://github.com/lightseekorg/tokenspeed/pull/78)→ AR-Norm + FP8 decode 融合[[22]](https://github.com/lightseekorg/tokenspeed/pull/124)的内核链，lm_head 从 94μs 继续下压[[23]](https://github.com/lightseekorg/tokenspeed/pull/126)——drafter 循环里每微秒都在被榨干。

## 四、MLA 内核竞速，DeepSeek 架构专属优化白热化

SGLang 正式集成 tokenspeed_mla 作为 prefill/decode 内核，覆盖 fp8 KV cache 和 Blackwell[[26]](https://github.com/sgl-project/sglang/pull/24925)；vLLM 在 ROCm 侧接入 aiter mhc 内核[[29]](https://github.com/vllm-project/vllm/pull/41946)。SGLang 补齐 MLA LoRA 的 q_b_proj / kv_b_proj 支持[[27]](https://github.com/sgl-project/sglang/pull/25001)，修复 AMD 双重 RoPE 旋转[[28]](https://github.com/sgl-project/sglang/pull/24148)。一个注意力架构同时被 NVIDIA、AMD、FlashInfer 三条内核路径并行优化——**DeepSeek MLA 已经成为跨框架、跨硬件的内核竞争焦点**。

## 五、TokenSpeed 从概念验证到一行命令可用

`ts serve` 合入 SMG gateway + gRPC 引擎，一行命令启动 OpenAI 兼容推理服务[[33]](https://github.com/lightseekorg/tokenspeed/pull/97)。CLI 体验全面补齐：默认端口恢复 8000[[34]](https://github.com/lightseekorg/tokenspeed/pull/114)、position model 参数[[37]](https://github.com/lightseekorg/tokenspeed/pull/128)、启动 banner[[36]](https://github.com/lightseekorg/tokenspeed/pull/127)。Kimi K2.5 专属优化链（lm_head → embed+norm → AR-Norm + FP8 decode）成型，mamba prefix cache 消除了 snapshot 中的 `cudaStreamSynchronize`[[39]](https://github.com/lightseekorg/tokenspeed/pull/77)。一周前还是 30+ PR 的概念验证，今天已经是一个 `pip install && ts serve model` 就能跑起来的推理服务。

## 今天真正值得记住的判断

量化配置的架构性重构是今天最本质的变化：vLLM 的 QuantKey/QuantSpec 把量化从"枚举+硬编码覆盖"变成"可组合规格"，NVFP4 跨三大框架同日就位是这个规格系统的首次量产验证。当量化变成可声明的、可逐层组合的基础设施，新格式和新模型的适配周期会被大幅压缩——这才是今天 9 个量化相关 PR 背后的结构性信号。

---

## 参考来源

[1] [vLLM #41566: Rework quantization_config to use QuantKey](https://github.com/vllm-project/vllm/pull/41566)

[2] [vLLM #35859: Support loading Quark NVFP4 checkpoints](https://github.com/vllm-project/vllm/pull/35859)

[3] [vLLM #42464: Patch SlidingWindowSpec for nvfp4 kv](https://github.com/vllm-project/vllm/pull/42464)

[5] [SGLang #24816: Add FlashInfer SM90 MXFP4 MoE backend](https://github.com/sgl-project/sglang/pull/24816)

[7] [SGLang #25190: Fix nvfp4 hot-reload crash](https://github.com/sgl-project/sglang/pull/25190)

[8] [TRT-LLM #14026: Support NVFP4 dsv4](https://github.com/NVIDIA/TensorRT-LLM/pull/14026)

[10] [vLLM #40020: Add multi-tier KV cache offloading framework](https://github.com/vllm-project/vllm/pull/40020)

[11] [vLLM #42507: Add req_id to ReqContext for per-request tracking](https://github.com/vllm-project/vllm/pull/42507)

[12] [LMCache #3038: Fully async PD backend](https://github.com/LMCache/LMCache/pull/3038)

[13] [LMCache #2966: Add batched_contains() for NixlDynamicStorageBackend](https://github.com/LMCache/LMCache/pull/2966)

[14] [Mooncake #1719: Add ObjectDataType enum for type-aware metadata](https://github.com/kvcache-ai/Mooncake/pull/1719)

[15] [Mooncake #2080: Add DSA-like workload allocation strategy](https://github.com/kvcache-ai/Mooncake/pull/2080)

[17] [vLLM #39487: Support custom callable proposer backend for speculative decoding](https://github.com/vllm-project/vllm/pull/39487)

[18] [vLLM #39949: Support hybrid attention models in extract_hidden_states](https://github.com/vllm-project/vllm/pull/39949)

[19] [vLLM #42536: Remove verifier model type check](https://github.com/vllm-project/vllm/pull/42536)

[21] [TokenSpeed #78: Fuse embeds and hidden norm in MLA eagle3](https://github.com/lightseekorg/tokenspeed/pull/78)

[22] [TokenSpeed #124: Enable AR-Norm fusion and fused FP8 decode for MLA Eagle3](https://github.com/lightseekorg/tokenspeed/pull/124)

[23] [TokenSpeed #126: Optimize lm_head](https://github.com/lightseekorg/tokenspeed/pull/126)

[26] [SGLang #24925: Integrate tokenspeed_mla prefill/decode kernels](https://github.com/sgl-project/sglang/pull/24925)

[27] [SGLang #25001: MLA attention LoRA q_b_proj / kv_b_proj support](https://github.com/sgl-project/sglang/pull/25001)

[28] [SGLang #24148: Add _skip_rope_for_aiter_fused_mla](https://github.com/sgl-project/sglang/pull/24148)

[29] [vLLM #41946: Add aiter mhc support](https://github.com/vllm-project/vllm/pull/41946)

[33] [TokenSpeed #97: ts serve — smg gateway + gRPC engine](https://github.com/lightseekorg/tokenspeed/pull/97)

[34] [TokenSpeed #114: Default SMG serve port to 8000](https://github.com/lightseekorg/tokenspeed/pull/114)

[36] [TokenSpeed #127: Print TokenSpeed banner on ts serve startup](https://github.com/lightseekorg/tokenspeed/pull/127)

[37] [TokenSpeed #128: Accept positional model arg in ts serve](https://github.com/lightseekorg/tokenspeed/pull/128)

[39] [TokenSpeed #77: Optimize mamba prefix cache performance](https://github.com/lightseekorg/tokenspeed/pull/77)
