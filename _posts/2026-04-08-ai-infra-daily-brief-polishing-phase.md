---
title: AI Infra 早报｜推理框架持续打磨，小幅改进为主
date: 2026-04-08 05:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: 过去 24 小时，AI Infra 整体以小幅改进为主。vLLM 修复量化 KV cache crash，SGLang 推进异构部署（支持 Qwen3.5 Mamba 状态切片传输），llama.cpp 在异构 KV 旋转和 SYCL 优化上取得进展，TensorRT-LLM 完成 NVFP4 配置更新。生态层面，Mooncake 完善 Rust bindings 和 UB Transport，TRL 清理废弃接口，OpenClaw 继续安全加固。无重大 S 级发布，本窗口属于"打磨迭代期"。
tags: [Inference, Quantization, KV Cache]
---

过去 24 小时，AI Infra 各主要项目继续保持小幅迭代节奏，没有出现 S 级重大发布。本窗口属于"打磨迭代期"——各框架在生产稳定性、异构部署支持、废弃接口清理等方面持续推进。

## 一、vLLM：量化 KV cache 问题修复

vLLM 在过去 24 小时合并了多条 PR，最值得关注的是**修复 extract_hidden_states crash 问题**[[1]](https://github.com/vllm-project/vllm/pull/39160)——解决 quantized KV cache dtype 导致的 crash，这是生产环境中可能影响服务稳定性的关键 bugfix。

同期还修复了 ROCm unused IS_FNUZ 参数[[2]](https://github.com/vllm-project/vllm/pull/39123)、cuda event reuse race[[3]](https://github.com/vllm-project/vllm/pull/39115)，以及 XPU 上 TritonMLA 的 CUDA hardcode 移除[[4]](https://github.com/vllm-project/vllm/pull/39088)。这些都属于生产稳定性打磨。

## 二、llama.cpp：异构 KV 与 SYCL 优化

llama.cpp 的亮点是**异构 iSWA KV rotation 支持**[[5]](https://github.com/ggml-org/llama.cpp/pull/21513)——现在支持不同配置的 attention rotation，适用于异构部署场景。

更值得关注的是 **SYCL Q8_0 reorder 优化**[[6]](https://github.com/ggml-org/llama.cpp/pull/21527)——在 Intel GPU 上实现了约 3 倍的 token 生成加速，这对在 Intel GPU 上运行推理的用户是显著的性能提升。

其他更新包括：CUDA buffer overlap 检查[[7]](https://github.com/ggml-org/llama.cpp/pull/21566)、WebGPU iOS 限制参数化[[8]](https://github.com/ggml-org/llama.cpp/pull/21533)、Server checkpoint restore 修复[[9]](https://github.com/ggml-org/llama.cpp/pull/21510)。

## 三、SGLang：异构部署推进

SGLang 继续推进异构部署能力。最核心的是 **Mamba state slice transfer 支持**[[10]](https://github.com/sgl-project/sglang/pull/22240)——在异构 TP 下实现 Qwen3.5 状态传输就绪，这是 NIXL 项目的第二步。同时新增了 HiSparse 特性文档[[11]](https://github.com/sgl-project/sglang/pull/22238)，Ring test 移至 nightly[[12]](https://github.com/sgl-project/sglang/pull/22267)。

## 四、TensorRT-LLM：NVFP4 与内核优化

TensorRT-LLM 完成了 **NVFP4 配置更新**[[13]](https://github.com/NVIDIA/TensorRT-LLM/pull/12776)，这意味着在 NVIDIA GPU 上使用 NVFP4 精度进行推理的配置已经就绪。

同期还有多个内核优化：causalConv1d fwd dispatch retune[[14]](https://github.com/NVIDIA/TensorRT-LLM/pull/12739)针对 varlen 和短序列进行优化，GDN prefill transpose 复用 triton slice kernel[[15]](https://github.com/NVIDIA/TensorRT-LLM/pull/12737)，Mamba metadata prefill bubble 修复[[16]](https://github.com/NVIDIA/TensorRT-LLM/pull/12736)解决 chunked prefill serving 问题。

## 五、Mooncake：存储层与跨平台传输

Mooncake 在存储层有多项重要更新：修复 P2PClientService Put 错误吞掉问题[[17]](https://github.com/kvcache-ai/Mooncake/pull/1825)，确保写入错误能被正确报告；新增 Rust native bindings[[18]](https://github.com/kvcache-ai/Mooncake/pull/1810)及 CI 集成；Kunpeng 超节点的 Unified Transport Phase 1 就绪[[19]](https://github.com/kvcache-ai/Mooncake/pull/1805)；Snapshot 正确性加固[[20]](https://github.com/kvcache-ai/Mooncake/pull/1801)。

## 六、TRL：废弃接口清理

TRL 在过去 24 小时进行了多项废弃接口清理：VLM key passthrough in DPO[[21]](https://github.com/huggingface/trl/pull/5468)、清理 truncate_dataset 废弃参数[[22]](https://github.com/huggingface/trl/pull/5467)、修复 SFT deprecation warning[[23]](https://github.com/huggingface/trl/pull/5466)、废弃 keep_end truncation mode[[24]](https://github.com/huggingface/trl/pull/5465)。这些都属于代码质量维护。

## 七、OpenClaw：安全加固继续

OpenClaw 继续保持高强度的安全更新：node reconnect 强制 re-pairing[[25]](https://github.com/openclaw/openclaw/pull/62658)，升级时强制重新配对；多项安全修复涉及 compaction after tool abortion、exec cmd wrapper、node shell allowlist、write commands owner check、workspace-only localRoots、SSRF guard hardening、token/password WS session 失效等。

## 结论

过去 24 小时属于"打磨迭代期"——没有 S 级重大发布，各项目在生产稳定性（vLLM crash 修复、Mooncake 错误处理）、异构部署支持（SGLang Mamba slice、llama.cpp iSWA KV rotation）、性能优化（llama.cpp SYCL 3x、TRT-LLM 内核调优）、代码质量（TRL 废弃接口清理）等方面持续推进。这符合 AI Infra 成熟阶段的特征——从功能快速迭代转向精细化打磨。

---

## 参考来源

[1] [Fix extract_hidden_states crash with quantized KV cache dtype](https://github.com/vllm-project/vllm/pull/39160)

[2] [ROCm Remove unused IS_FNUZ parameter](https://github.com/vllm-project/vllm/pull/39123)

[3] [Fix cuda event reuse race](https://github.com/vllm-project/vllm/pull/39115)

[4] [XPU Quick fix for TritonMLA to remove cuda hardcode](https://github.com/vllm-project/vllm/pull/39088)

[5] [kv-cache support attention rotation for heterogeneous iSWA](https://github.com/ggml-org/llama.cpp/pull/21513)

[6] [SYCL Add Q8_0 reorder optimization for Intel GPUs (~3x token generation speedup)](https://github.com/ggml-org/llama.cpp/pull/21527)

[7] [CUDA check for buffer overlap before fusing](https://github.com/ggml-org/llama.cpp/pull/21566)

[8] [ggml-webgpu parameterize submission size and add iOS specific limits](https://github.com/ggml-org/llama.cpp/pull/21533)

[9] [server fix restore for checkpoints with pos_min == 0](https://github.com/ggml-org/llama.cpp/pull/21510)

[10] [NIXL Support Mamba state slice transfer for heterogeneous TP Qwen3.5](https://github.com/sgl-project/sglang/pull/22240)

[11] [HiSparse Add readme docs for HiSparse Feature](https://github.com/sgl-project/sglang/pull/22238)

[12] [Move ring test to nightly](https://github.com/sgl-project/sglang/pull/22267)

[13] [Config updates to enable NVFP4](https://github.com/NVIDIA/TensorRT-LLM/pull/12776)

[14] [retune causalConv1d fwd dispatch for varlen and short sequences](https://github.com/NVIDIA/TensorRT-LLM/pull/12739)

[15] [reuse triton slicing kernel for GDN prefill transpose](https://github.com/NVIDIA/TensorRT-LLM/pull/12737)

[16] [fix mamba metadata prefill bubble in chunked prefill serving](https://github.com/NVIDIA/TensorRT-LLM/pull/12736)

[17] [Fix P2PClientService Put silently swallowing write errors](https://github.com/kvcache-ai/Mooncake/pull/1825)

[18] [Add native Rust bindings for Mooncake Store](https://github.com/kvcache-ai/Mooncake/pull/1810)

[19] [Enabling UB Transport on the Kunpeng SuperNode Phase 1](https://github.com/kvcache-ai/Mooncake/pull/1805)

[20] [tighten snapshot correctness and reload snapshot-only standby](https://github.com/kvcache-ai/Mooncake/pull/1801)

[21] [Use generic VLM key passthrough in DPO](https://github.com/huggingface/trl/pull/5468)

[22] [Remove unused truncation_mode from experimental truncate_dataset](https://github.com/huggingface/trl/pull/5467)

[23] [Fix SFT deprecation warning](https://github.com/huggingface/trl/pull/5466)

[24] [Deprecate keep_end truncation mode](https://github.com/huggingface/trl/pull/5465)

[25] [Require re-pairing for node reconnect command upgrades](https://github.com/openclaw/openclaw/pull/62658)