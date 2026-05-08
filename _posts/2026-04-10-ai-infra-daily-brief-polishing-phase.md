---
title: AI Infra 早报｜推理框架持续迭代，多模态与工具调用能力深化
date: 2026-04-10 04:55:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: 过去 24 小时，AI Infra 各项目持续迭代深化。推理侧 vLLM 完善量化 MoE 与异构架构，SGLang 新增多模态 diffusion 与 Qwen3-VL EAGLE3 支持；生产部署侧 Mooncake 增加分层存储与 HA 能力；训练侧 TRL 强化 tool calling 训练支持；应用侧 OpenClaw 深化 Memory/Dreaming 记忆系统。本窗口以 A/B 级迭代为主，无 S 级重大发布。
tags: [Inference, Multimodal, Quantization]
---

过去 24 小时，AI Infra 领域呈现持续迭代深化的态势。推理框架侧，vLLM 在量化 MoE 与异构架构支持上取得新进展，SGLang 新增多模态 diffusion 与 Qwen3-VL EAGLE3 支持，llama.cpp 继续优化 SYCL/Vulkan 多后端；生产部署侧，Mooncake 完善分层存储与高可用能力，TensorRT-LLM 推进 disaggregation 部署；训练侧 TRL 强化 tool calling 训练支持；应用侧 OpenClaw 深化 Memory/Dreaming 记忆系统。本窗口属于"打磨迭代期"，无 S 级重大发布。

## 一、推理框架：量化与多模态并行演进

vLLM 过去 24 小时合并 30+ PRs，重点推进 Gemma4 量化 MoE 支持[[1]](https://github.com/vllm-project/vllm/pull/39045)，解决 FlashInfer MXINT4 MoE crash 问题[[2]](https://github.com/vllm-project/vllm/pull/39315)，并修复 PD 异构架构下 CPU_ATTN/Flash_ATTN 混合的精度问题[[4]](https://github.com/vllm-project/vllm/pull/38935)。同时优化 NUMA binding 适配 Grace-Blackwell 系统[[5]](https://github.com/vllm-project/vllm/pull/39361)，Pooling 模型通过减少冗余同步实现 3.7% 吞吐量提升[[6]](https://github.com/vllm-project/vllm/pull/39113)。

SGLang 新增 Qwen3-VL EAGLE3 推理支持[[7]](https://github.com/sgl-project/sglang/pull/22230)，FLUX.2-small-decoder 扩散模型支持[[8]](https://github.com/sgl-project/sglang/pull/22414)，以及基于 chunk 的流式 ASR 功能[[12]](https://github.com/sgl-project/sglang/pull/22089)。FlashAttention V4 改为懒加载避免启动时开销[[9]](https://github.com/sgl-project/sglang/pull/22306)，DSA 模型支持 AllReduce 融合优化[[10]](https://github.com/sgl-project/sglang/pull/22390)，DeepSeekV3 MoE 的 MLA LoRA 也获得支持[[11]](https://github.com/sgl-project/sglang/pull/22323)。

llama.cpp 在多后端优化上持续发力：SYCL FlashAttention 新增头长 512 支持[[13]](https://github.com/ggml-org/llama.cpp/pull/21654)，Vulkan 类型宏统一为 Vx 规范[[14]](https://github.com/ggml-org/llama.cpp/pull/21605)，Metal 后端完善 Q1_0 mm-id 特化[[15]](https://github.com/ggml-org/llama.cpp/pull/21662)，HIP 新增 CDNA4 (gfx950) 架构支持适配 MI350X/MI355X[[16]](https://github.com/ggml-org/llama.cpp/pull/21570)。

## 二、训练框架：Tool Calling 与参数管理

TRL 过去 24 小时在 tool calling 训练支持上动作频频。新增 `supports_tool_calling` 工具[[17]](https://github.com/huggingface/trl/pull/5462)用于验证模型初始化时的 tool 支持能力，新增 GPT-OSS Tool Calling 支持[[18]](https://github.com/huggingface/trl/pull/5464)，Chat template 新增 `generation` block 支持[[19]](https://github.com/huggingface/trl/pull/5470)用于训练时生成模板，并优化 Image deepcopy 减少多模态消息处理开销[[20]](https://github.com/huggingface/trl/pull/5475)。同时新增 DistillationTrainer[[21]](https://github.com/huggingface/trl/pull/5407)用于高效 on-policy 蒸馏训练，废弃 pad_token 配置参数[[22]](https://github.com/huggingface/trl/pull/5480)以清理废弃接口。

Megatron-LM 方面，将 async_allgather 重命名为 overlap_param_gather[[23]](https://github.com/NVIDIA/Megatron-LM/pull/4217)以更清晰地表达异步参数聚合语义，修复多项 MTP 推理问题[[24]](https://github.com/NVIDIA/Megatron-LM/pull/4191)，为 Mamba 模型启用细粒度 activation offloading[[25]](https://github.com/NVIDIA/Megatron-LM/pull/4173)，并为 quick_gelu 激活函数添加 fused grouped MLP 支持[[26]](https://github.com/NVIDIA/Megatron-LM/pull/4219)。

## 三、生产部署：分层存储与高可用

Mooncake 过去 24 小时新增 DRAM-CXL-SSD 多协议分层存储支持[[27]](https://github.com/kvcache-ai/Mooncake/pull/1832)，引入 HA OpLog 抽象与 LocalFS oplog store[[28]](https://github.com/kvcache-ai/Mooncake/pull/1804)提升高可用能力。修复 BufferDesc 子分配 GPU tensor 地址注册问题[[29]](https://github.com/kvcache-ai/Mooncake/pull/1837)，修复 stale segment cache 问题[[30]](https://github.com/kvcache-ai/Mooncake/pull/1826)，并通过 HTTP API 暴露 drain job 控制接口[[31]](https://github.com/kvcache-ai/Mooncake/pull/1815)。

TensorRT-LLM 推进 disaggregation 部署：修复 Nano chunked prefill 问题[[32]](https://github.com/NVIDIA/TensorRT-LLM/pull/12782)，新增 DeepSeek-R1 AutoDeploy 支持[[33]](https://github.com/NVIDIA/TensorRT-LLM/pull/12601)，并添加生命周期竞争条件测试[[34]](https://github.com/NVIDIA/TensorRT-LLM/pull/12803)。

LMCache 修复多路径场景下的 CUDA launch host func 死锁问题[[35]](https://github.com/LMCache/LMCache/pull/2952)，并锁定 CI cu128 nightly wheel 版本[[36]](https://github.com/LMCache/LMCache/pull/2987)。

Ray 新增 GCS io_service 健康检查[[37]](https://github.com/ray-project/ray/pull/62374)，Serve 新增 Central capacity queue[[38]](https://github.com/ray-project/ray/pull/62323)用于基于 token 的请求路由，并新增standalone KubeRay IPPR provider[[39]](https://github.com/ray-project/ray/pull/62215)。

## 四、应用层：记忆系统深化

OpenClaw 过去 24 小时深化 Memory/Dreaming 记忆系统：Dreaming 暴露 grounded scene lane[[40]](https://github.com/openclaw/openclaw/pull/63395)实现可视化场景呈现，修复 Dreaming cron 在 runtime lifecycle 的协调问题[[41]](https://github.com/openclaw/openclaw/pull/63873)，添加 narrative idempotency[[42]](https://github.com/openclaw/openclaw/pull/63876)确保叙事生成的幂等性，支持 slot-owned memory 配置[[43]](https://github.com/openclaw/openclaw/pull/63874)，并修复 UI trace 布局溢出问题[[44]](https://github.com/openclaw/openclaw/pull/63875)。

多渠道与安全方面：修复 msteams thread 会话隔离[[45]](https://github.com/openclaw/openclaw/pull/62713)，WhatsApp 重连后消息保留[[46]](https://github.com/openclaw/openclaw/pull/62892)，Browser 自动生成控制认证 token[[47]](https://github.com/openclaw/openclaw/pull/63280)，QQBot 图片尺寸探测 SSRF 防护[[48]](https://github.com/openclaw/openclaw/pull/63495)，以及 Cron nextRunAtMs=0 修复[[49]](https://github.com/openclaw/openclaw/pull/63507)。

---

## 参考来源

[1] [[Gemma4] Support quantized MoE](https://github.com/vllm-project/vllm/pull/39045)

[2] [[Bugfix] FlashInfer MXINT4 MoE crashes](https://github.com/vllm-project/vllm/pull/39315)

[3] [[XPU] Skip VLLM_BATCH_INVARIANT for XPU](https://github.com/vllm-project/vllm/pull/39164)

[4] [[PD][HeteroArch] Fix accuracy issue](https://github.com/vllm-project/vllm/pull/38935)

[5] [Fix NUMA binding on Grace-Blackwell](https://github.com/vllm-project/vllm/pull/39361)

[6] [[Perf] Optimize redundant sync for pooling model](https://github.com/vllm-project/vllm/pull/39113)

[7] [[Feature] Support eagle3 for qwen3-vl](https://github.com/sgl-project/sglang/pull/22230)

[8] [[diffusion] FLUX.2-small-decoder support](https://github.com/sgl-project/sglang/pull/22414)

[9] [Lazy import flash_attention_v4](https://github.com/sgl-project/sglang/pull/22306)

[10] [[DSA] Enable all reduce fusion](https://github.com/sgl-project/sglang/pull/22390)

[11] [[Lora] Support deepseekv3 mla lora](https://github.com/sgl-project/sglang/pull/22323)

[12] [[Feature] Chunk-based streaming ASR](https://github.com/sgl-project/sglang/pull/22089)

[13] [SYCL flash-attn head size 512](https://github.com/ggml-org/llama.cpp/pull/21654)

[14] [Vulkan unify type macros](https://github.com/ggml-org/llama.cpp/pull/21605)

[15] [Metal Q1_0 mm-id specializations](https://github.com/ggml-org/llama.cpp/pull/21662)

[16] [HIP CDNA4 gfx950 support](https://github.com/ggml-org/llama.cpp/pull/21570)

[17] [Add supports_tool_calling utility](https://github.com/huggingface/trl/pull/5462)

[18] [Add GPT-OSS tool calling support](https://github.com/huggingface/trl/pull/5464)

[19] [Add generation support to chat templates](https://github.com/huggingface/trl/pull/5470)

[20] [Avoid image deepcopy](https://github.com/huggingface/trl/pull/5475)

[21] [Add DistillationTrainer](https://github.com/huggingface/trl/pull/5407)

[22] [Deprecate pad_token config](https://github.com/huggingface/trl/pull/5480)

[23] [rename async_allgather to overlap_param_gather](https://github.com/NVIDIA/Megatron-LM/pull/4217)

[24] [Miscellaneous MTP inference fixes](https://github.com/NVIDIA/Megatron-LM/pull/4191)

[25] [Enable fine-grained activation offloading for Mamba](https://github.com/NVIDIA/Megatron-LM/pull/4173)

[26] [fused grouped MLP for quick_gelu](https://github.com/NVIDIA/Megatron-LM/pull/4219)

[27] [DRAM-CXL-SSD multi-protocol support](https://github.com/kvcache-ai/Mooncake/pull/1832)

[28] [HA OpLog abstraction](https://github.com/kvcache-ai/Mooncake/pull/1804)

[29] [BufferDesc address registration fix](https://github.com/kvcache-ai/Mooncake/pull/1837)

[30] [Fix stale segment cache](https://github.com/kvcache-ai/Mooncake/pull/1826)

[31] [expose drain job control via HTTP API](https://github.com/kvcache-ai/Mooncake/pull/1815)

[32] [Fix Nano chunked prefill](https://github.com/NVIDIA/TensorRT-LLM/pull/12782)

[33] [AutoDeploy onboard DeepSeek-R1](https://github.com/NVIDIA/TensorRT-LLM/pull/12601)

[34] [Lifecycle Race Condition test](https://github.com/NVIDIA/TensorRT-LLM/pull/12803)

[35] [Fix deadlock from cuda launch host func](https://github.com/LMCache/LMCache/pull/2952)

[36] [Pin cu128 nightly wheel](https://github.com/LMCache/LMCache/pull/2987)

[37] [GCS health check on io_service](https://github.com/ray-project/ray/pull/62374)

[38] [Central capacity queue](https://github.com/ray-project/ray/pull/62323)

[39] [standalone KubeRay IPPR provider](https://github.com/ray-project/ray/pull/62215)

[40] [Dreaming surface grounded scene lane](https://github.com/openclaw/openclaw/pull/63395)

[41] [Memory cron reconciliation fix](https://github.com/openclaw/openclaw/pull/63873)

[42] [Dreaming narrative idempotency](https://github.com/openclaw/openclaw/pull/63876)

[43] [Slot-owned memory support](https://github.com/openclaw/openclaw/pull/63874)

[44] [UI Dreaming trace layout fix](https://github.com/openclaw/openclaw/pull/63875)

[45] [msteams thread session isolation](https://github.com/openclaw/openclaw/pull/62713)

[46] [WhatsApp preserve replies across reconnects](https://github.com/openclaw/openclaw/pull/62892)

[47] [Browser auto-generate auth token](https://github.com/openclaw/openclaw/pull/63280)

[48] [QQBot SSRF guard](https://github.com/openclaw/openclaw/pull/63495)

[49] [Cron nextRunAtMs=0 fix](https://github.com/openclaw/openclaw/pull/63507)
