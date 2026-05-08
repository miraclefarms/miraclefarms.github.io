---
title: AI Infra 早报｜推理框架迭代深化，生产部署侧持续优化
date: 2026-04-09 04:55:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: 过去 24 小时，AI Infra 各项目继续深化迭代。vLLM 推进 NVFP4 批量不变支持，llama.cpp 完善 Q1_0 Metal 后端，SGLang 升级 EAGLE3；Mooncake 修复 NVLink IPC 通信并增加昆仑 RDMA 支持；TRL 完成 KTOConfig 重构；OpenClaw 强化 Memory/Dreaming 功能。
tags: [Inference, Quantization, Training]
---

过去 24 小时，AI Infra 各项目继续深化迭代，整体处于"持续打磨期"，无 S 级重大发布。推理框架侧重点从功能可用转向生产优化，生产部署侧则在跨设备通信和缓存层持续改进。

## 一、推理框架：量化与后端完善

vLLM 持续推进 NVFP4 量化支持，新增批量不变 Linear 支持以解决批量推理下的兼容性问题。同时修复了 DeepSeek EP 精度问题（影响 Qwen3.5、Qwen3-Next），以及 Gemma4 流式 tool call 的解析崩溃。值得关注的是 max-model-len=-1 导致的 hang 问题被修复，避免过限请求 starve 整个服务。

llama.cpp 方面最亮眼的进展是 Q1_0 1-bit 量化 Metal 后端就绪，这标志着 Apple Silicon GPU 上的极致压缩方案正式可用。同时继续完善 KV cache 量化检查和 CUDA 融合决策优化。

SGLang 将 eagle_infer_beta 切换至 EAGLE3，并升级 FlashInfer 至 0.6.7.post3 版本。AMD 路径上修复了 GLM-5 fp8 KV 量化在 MI300 上的 dispatch 问题。

## 二、训练框架：Config 重构与路由优化

TRL 完成两项重要重构：KTOConfig 与 DPO Config 结构对齐，以及 chat templates 从内联字符串迁移至 .jinja 文件。Gemma4 CI 支持也已就绪。

Megatron-LM 新增 MoE Router 的 score function 扩展，增强路由灵活性。Checkpointing 文档也更新以适配异步变更。

## 三、生产部署：通信层与跨设备传输

Mooncake 本窗口修复了关键通信问题：NVLink IPC 地址问题（影响子分配的 GPU tensor），以及 TCP 端口冲突。更重要的是，昆仑芯片的 RDMA 支持进入可用状态，TE 模块完成了 RDMA 初始化。

LMCache 新增多路径本地盘后端，支持多设备并行 I/O。CI 基础镜像升级至 CUDA 13.0。

TensorRT-LLM 修复了 LoRA 在 Qwen3 模型上的适配问题，并解决了大规模 LoRA adapter 的 OOM 问题。

Ray 继续优化测试稳定性，减少 bundle placement group 测试的节点需求，并移除不必要的 lazy imports。

## 四、应用层：Memory/Dreaming 与安全

OpenClaw v2026.4.7/4.8 版本带来 Memory/Dreaming 功能的显著强化，包括 grounded REM 提取、backfill lane 和 diary controls。同时改善了 OAuth reauth 错误的可诊断性，修复了 Telegram/Channel 打包问题，并增加了 Slack Socket Mode 的 proxy 支持。SSRF guard 也得到强化，跨域重定向时默认丢弃 request body。

整体来看，本窗口属于"打磨迭代期"，各项目在各自领域持续优化而非激进突破。NVFP4 量化支持、Q1_0 Metal 后端、昆仑 RDMA 支持是值得关注的长期趋势。

## 参考来源

[1] [Feature] Batch invariant nvfp4 linear support：https://github.com/vllm-project/vllm/pull/39322

[2] [Bugfix]Fix EP precision for Qwen3.5, Qwen3-Next：https://github.com/vllm-project/vllm/pull/39181

[3] [Bugfix] Fix Gemma4 streaming tool call corruption：https://github.com/vllm-project/vllm/pull/39114

[4] [BugFix] --max-model-len=-1 causes over-limit requests to hang：https://github.com/vllm-project/vllm/pull/39102

[5] metal: Q1_0 backend：https://github.com/ggml-org/llama.cpp/pull/21528

[6] CUDA: also store `node->src->data` ptrs for equality check：https://github.com/ggml-org/llama.cpp/pull/21635

[7] kv-cache : extend cache quantization checks：https://github.com/ggml-org/llama.cpp/pull/21586

[8] autoparser: fix MiniMax handling：https://github.com/ggml-org/llama.cpp/pull/21573

[9] Switch eagle_infer_beta to EAGLE3：https://github.com/sgl-project/sglang/pull/22303

[10] chore: bump flashinfer version to 0.6.7.post3：https://github.com/sgl-project/sglang/pull/22382

[11] [AMD] Fix GLM-5 fp8 KV quant path dispatch on MI300：https://github.com/sgl-project/sglang/pull/22314

[12] Align KTO with DPO: Reorganize KTOConfig：https://github.com/huggingface/trl/pull/5477

[13] Move chat templates from inline strings to `.jinja` files：https://github.com/huggingface/trl/pull/5459

[14] [Dev][MoE] Add a new score function to the router：https://github.com/NVIDIA/Megatron-LM/pull/4193

[15] [TENT] Fix NVLink IPC address for sub-allocated GPU tensors：https://github.com/kvcache-ai/Mooncake/pull/1831

[16] [Bug fix] Fix get tcp port collision：https://github.com/kvcache-ai/Mooncake/pull/1816

[17] [TE] feat: setup the RDMA for mlu device：https://github.com/kvcache-ai/Mooncake/pull/1799

[18] feat(disk): support multi-path local disk backend：https://github.com/LMCache/LMCache/pull/2801

[19] [None][fix] Fix LoRA support for Qwen3 models：https://github.com/NVIDIA/TensorRT-LLM/pull/12785

[20] Memory/dreaming: feed grounded backfill into short-term promotion：https://github.com/openclaw/openclaw/pull/63370

[21] Reply: surface OAuth reauth failures：https://github.com/openclaw/openclaw/pull/63217

[22] fix(slack): honor HTTPS_PROXY for Socket Mode WebSocket：https://github.com/openclaw/openclaw/pull/62878

[23] Network/fetch guard: drop request bodies on cross-origin redirects：https://github.com/openclaw/openclaw/pull/62357