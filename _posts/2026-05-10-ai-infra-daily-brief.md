---
title: AI Infra 早报｜推理框架从"能跑"转向"跑得稳"：约束解码体系化、KV 缓存韧性、异构后端全覆盖
date: 2026-05-10 08:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: SGLang 约束解码从一次性功能变成有层级架构的系统能力，LMCache 补齐断线重连、多后端和可观测三项生产刚需，llama.cpp 异构后端从"能编译"走向"能跑好"。
tags: [Constrained Decoding, KV Cache, MoE, Heterogeneous Inference]
---

![题图](/assets/2026-05-10-ai-infra-daily-brief/cover.png)


本周推理框架的 PR 密度指向一个清晰方向：不再追求"能跑起来"，而是把已有能力的正确性、稳定性和平台覆盖推上一个台阶。SGLang 的约束解码从一次性 hack 变成分层设计的系统能力；LMCache 密集补齐断线重连、多后端和 token 级可观测三项生产刚需；llama.cpp 的 SYCL/Hexagon/Adreno 后端从"能编译"跨入"每个后端都能跑到合理性能"。vLLM、SGLang、TRT-LLM 各自出现的 MoE 正确性修复则从反面验证了这一步的紧迫——模型越复杂，静默精度丢失越容易发生。

## 一、SGLang 约束解码体系化

SGLang 本周在约束解码上推了三件互相补位的事。

Two-phase reasoning grammar[[1]](https://github.com/sgl-project/sglang/pull/23953) 解决了一个根因问题：reasoning 模型在 grammar 约束下容易丢失 `think_end` token。单阶段 grammar 无法区分 thinking 和 answer 两个语义区间的约束边界——thinking 阶段需放行 reasoning-content 类 token，answer 阶段才切换输出语法，同一套规则放不下两种矛盾的约束。Two-phase 方案配合 `--enable-strict-thinking` 标志，在 grammar 后端时序上显式划分两个阶段。**这不是格式补丁，而是 constrained generation 在 reasoning 模型上的架构级修正。**

PDL（Programmable Dependency Latency）[[2]](https://github.com/sgl-project/sglang/pull/23965) 把问题拉到了延迟维度。MoE 推理中 kernel launch 的依赖关系过去是硬编码的——哪些 kernel 必须串行、哪些可以并行，全写死在手写的 launch sequence 里。PDL 把调度权交给一个可编程的依赖描述层，可根据模型结构和硬件拓扑动态编排时序。代码注释点名了 DSV32 和 GLM5 的 kernel 路径，大 MoE 的端到端延迟首次有了可调的调度自由度。

Eagle3 扩展到 Gemma3/4[[3]](https://github.com/sgl-project/sglang/pull/23976) 将 speculative decoding 的模型覆盖从 Llama/Qwen 推向 Google 模型线；伴随的 FA3 page table 地址翻译修复保证 topk>1 场景下 spec metadata 正确性。三件事合在一起看：**grammar 层保正确性，PDL 层控延迟，Eagle3 层扩吞吐覆盖——约束解码在 SGLang 有了清晰的层级分工。**

## 二、LMCache 生产韧性三件套

LMCache 本周的 PR 密集覆盖了 KV 缓存从"能连上"到"连得稳"的三个维度。

断线重连[[4]](https://github.com/LMCache/LMCache/pull/3208) 解决了一个生产里很常见的场景：LMCache MP server 重启后 vLLM worker 的 KV-cache 注册信息丢失，所有 STORE/RETRIEVE 调用失败。这意味着缓存静默降级为无缓存，调用方完全无感知。此次 PR 通过注册恢复机制让服务重启后的缓存关系自动重建。

多后端扩展打破了 KV 缓存此前几乎绑死在 NVIDIA + 本地盘上的硬约束。Azure Blob Storage[[5]](https://github.com/LMCache/LMCache/pull/3160) 作为 NIXL 对象存储后端的 drop-in replacement 被纳入，ROCm CacheBlend[[6]](https://github.com/LMCache/LMCache/pull/3092) 用 Triton kernel 替代 FlashInfer 让 AMD GPU 也能用非 prefix KV cache 复用。

可观测性下沉到 token 级[[7]](https://github.com/LMCache/LMCache/pull/3196)：新增的 `lmcache_blend.lookup_requested_tokens` 和 `lmcache_blend.lookup_hit_tokens` 计数器把缓存命中率从 request 级变为 token 级。**request 级命中率说"这轮请求有没有命中"，是二值判断；token 级命中率说"命中了多少比例"，对成本核算才有实操意义。**

## 三、新模型跨框架同步登陆

本周四个新模型/架构在 vLLM、TRT-LLM、llama.cpp 中同步获得支持，且不是简单的 config 适配。

TRT-LLM[[8]](https://github.com/NVIDIA/TensorRT-LLM/pull/12932) 完整支持了 Gemma4 四个变体（26B-A4B-it MoE / E2B-it KV sharing PLE / 300M / 1.7B），覆盖 text + vision + audio 多模态。llama.cpp[[9]](https://github.com/ggml-org/llama.cpp/pull/22493) 完成 MiMo-V2.5 text-to-text 推理，但非对称 head size（K=192, V=128）导致 flash attention 回退到 CPU，被迫追加 MMA/Tiles 模板[[10]](https://github.com/ggml-org/llama.cpp/pull/22812)。vLLM[[11]](https://github.com/vllm-project/vllm/pull/42078) 新增 Cohere Eagle 模型支持，附带 speculative decoding 配置能力。

**新模型落地的工程代价在上升，但跨框架响应速度也在加快。** Gemma4 需要 NvFp4 权重转换修复才能跑起来，MiMo 的非对称 KV 尺寸直接逼出了一个 kernel 级改动——这不是"换个 config 就行"的时代了。

## 四、llama.cpp 异构后端全面开花

llama.cpp 本周的 SYCL 后端集中补齐了六个此前缺失的算子[[12]](https://github.com/ggml-org/llama.cpp/pull/22149)——从 FILL 到 GATED_DELTA_NET，让依赖这些算子的模型不再回退到 CPU。Q5_K/Q8_0 的 reorder-quantized 快速路径、BF16 GET_ROWS、flash attention buffer 复用策略同步落地，SYCL 后端的推理性能正从"能编译就行"的基线往上升。

移动端的加速同样密集。Hexagon HTP 拿到了 GATED_DELTA_NET[[13]](https://github.com/ggml-org/llama.cpp/pull/22837) 和 L2_NORM 专用 HVX kernel——前者让 Qwen3.5 等 GDN 模型的 recurrence 完全在端侧运行。Adreno OpenCL 新增 Q4_0 MoE GEMM[[14]](https://github.com/ggml-org/llama.cpp/pull/22731)。**"每个后端都能编译"和"每个后端都有合理推理性能"之间的差距，是 6 个算子、一批量化调优和几次内存分配策略改动填上的。**

## 五、MoE 正确性：同一天三个框架各自踩坑

vLLM[[15]](https://github.com/vllm-project/vllm/pull/42076) 修复了 Hopper GPU（sm_90, H20）上 GDN `chunk_scaled_dot_kkt` 的精度丢失——根本原因是 `tl.dot` 操作数布局与 WGMMA 不兼容，直接导致 GDN 模型在 lm_eval gsm8k 上得分为 0。这是静默错误的典型：模型能跑、不报错、输出像合法 token，但本质是随机结果。

SGLang[[16]](https://github.com/sgl-project/sglang/pull/24562) 则在 PyTorch 升级 2.11 后发现 DeepSeek V3 Triton MoE 性能回退——Triton 3.6.0 缺少 tuned config 导致回退到旧版配置。这不是正确性 bug，但同样暴露了 MoE 推理路径对底层软件栈版本的敏感。

TRT-LLM[[17]](https://github.com/NVIDIA/TensorRT-LLM/pull/13932) 在 DSv4 门控单元测试的 multi-GPU CI 上暴露了 FP32 reference 运算错误，伴随 FP8 workspace 尺寸计算修复和 Hadamard rotation 条件门控。**三个框架在同一天各自修复 MoE 路径的关键缺陷——不是巧合，而是 MoE 推理工程复杂度越过临界点后集中暴露 bug。**

## 六、今天真正值得记住的判断

推理框架的竞争正在换挡。"能跑起来"这条基线在过去一年被反复拉高后，接下来的竞争在三个维度上摊开：**正确性验证是生产部署的硬前置条件（三项 MoE 修复在同一周发生不是巧合），异构覆盖决定硬件选择自由度（llama.cpp 的后端补齐是这一趋势最清晰的信号），系统架构的体系化程度决定能力可组合性（SGLang 约束解码和 LMCache 生产韧性分别从设计端和运维端证明了这一点）。**

---

## 参考来源

[1] [Two-phase reasoning grammar + --enable-strict-thinking](https://github.com/sgl-project/sglang/pull/23953)

[2] [Enable PDL for DSV32/GLM5 kernels](https://github.com/sgl-project/sglang/pull/23965)

[3] [Gemma3/4 + Eagle3 speculative decoding](https://github.com/sgl-project/sglang/pull/23976)

[4] [vLLM reconnect after LMCache restart](https://github.com/LMCache/LMCache/pull/3208)

[5] [Azure Blob NIXL backend](https://github.com/LMCache/LMCache/pull/3160)

[6] [ROCm Triton block-sparse attention for CacheBlend](https://github.com/LMCache/LMCache/pull/3092)

[7] [Blend token-level hit-rate counters](https://github.com/LMCache/LMCache/pull/3196)

[8] [Gemma4 multimodal in TRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/pull/12932)

[9] [MiMo-V2.5 text-to-text in llama.cpp](https://github.com/ggml-org/llama.cpp/pull/22493)

[10] [Flash attention MMA/Tiles for MiMo-V2.5](https://github.com/ggml-org/llama.cpp/pull/22812)

[11] [Cohere Eagle + MoE fix in vLLM](https://github.com/vllm-project/vllm/pull/42078)

[12] [SYCL: FILL, CUMSUM, DIAG, SOLVE_TRI, SSM_SCAN, GATED_DELTA_NET](https://github.com/ggml-org/llama.cpp/pull/22149)

[13] [Hexagon HTP: GATED_DELTA_NET HVX kernel](https://github.com/ggml-org/llama.cpp/pull/22837)

[14] [OpenCL: Adreno Q4_0 MoE GEMM](https://github.com/ggml-org/llama.cpp/pull/22731)

[15] [GDN KKT precision loss on Hopper GPUs](https://github.com/vllm-project/vllm/pull/42076)

[16] [DSV3 Triton MoE perf regression on SM90](https://github.com/sgl-project/sglang/pull/24562)

[17] [DSv4 gate test fix](https://github.com/NVIDIA/TensorRT-LLM/pull/13932)