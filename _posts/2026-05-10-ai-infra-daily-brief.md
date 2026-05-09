---
title: AI Infra 早报｜FP4 量化跨越可用性门槛，三个框架同日推进
date: 2026-05-10 08:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: FP4 量化在 vLLM、SGLang、llama.cpp 中同步跨越可用性门槛，TRL v1.4.0 修复两个静默训练 bug，TokenSpeed 密集迭代瞄准国产模型生态。
---

![题图](/assets/2026-05-10-ai-infra-daily-brief/cover.jpg)


一个推理框架在 H20 GPU 上跑 GSM8K 评测，得分是零——不是模型不行，是 FP4 精度路径根本没被认真测过。vLLM 本周修了这个 bug，而就在同一个时间窗口内，SGLang 重写了 FP4 GEMM kernel，llama.cpp 修了 NVFP4 checkpoint 转换的崩溃。**FP4 量化正在从"格式能加载"进入"数字对得上、kernel 跑得快"的阶段——这不是巧合，是行业节奏到了。**

## 一、FP4 量化：三个框架同日推进，H20 精度 bug 最说明问题

vLLM 本周做了两件事。一是添加原生 NVFP4 W4A16 支持[[1]](https://github.com/vllm-project/vllm/pull/41769)，这是功能补齐。二是修了一个更关键的 bug：Hopper 架构 GPU 上 GDN-based 模型输出垃圾结果，GSM8K 得分归零[[2]](https://github.com/vllm-project/vllm/pull/42076)。**一个生产级 GPU 上完整跑完评测流程而精度完全崩坏，说明 FP4 的端到端路径此前根本没有被认真验证过。**vLLM 另一个 FP4 相关修复是移除 GDN rearrange_mixed_qkv 中的嵌套 torch.compile，解决了 CUDA graph capture 失败的问题[[3]](https://github.com/vllm-project/vllm/pull/42070)。

SGLang 同一天重新合入了用 CuteDSL 实现的 FP4 稠密 GEMM kernel[[4]](https://github.com/sgl-project/sglang/pull/23590)，这是 kernel 层面的深度优化，不是为了跑通，是为了跑快。llama.cpp 则聚焦于格式转换——Gemma4 26B NVFP4 checkpoint 的 weight_scale/input_scale 重命名和 FP8 KV-cache scales 剥离[[5]](https://github.com/ggml-org/llama.cpp/pull/22804)，以及因字典迭代变更导致的转换崩溃修复[[6]](https://github.com/ggml-org/llama.cpp/pull/22818)。

在量化训练侧，Megatron-LM 修了 FP8/MXFP8/FP4 参数 gather 在 eval 模式插入训练步之间时的收敛问题[[7]](https://github.com/NVIDIA/Megatron-LM/pull/4563)。TokenSpeed 修了 GPT-OSS MXFP4 的 scale dtype，保持共享 scale 参数为 checkpoint bytes 以避免加载时精度损坏[[8]](https://github.com/lightseekorg/tokenspeed/pull/42)。

三件事儿放在一起看：修精度、写 kernel、通格式转换——各自解决的是 FP4 落地链路的不同环节，但进度几乎同步。如果把 2025 年看成 FP8 从实验走向生产的过渡年，2026 年下半年很可能就是 FP4 的同类周期。

## 二、TRL v1.4.0：SFT 显存减半，两个静默训练 bug 比新特性更重要

TRL v1.4.0 正式发布[[9]](https://github.com/huggingface/trl/releases/tag/v1.4.0)，核心特性 `loss_type="chunked_nll"` 将 SFT 阶段峰值显存降低最高 50%，并已扩展到 VLM 和 MoE 模型[[10]](https://github.com/huggingface/trl/pull/5684)。

但最值得关注的不是这个显存优化，而是两个 GKD 训练路径的 bug 修复。一个是 `use_liger_kernel=True` 时 JSD 路径的 `weight_hard_loss` 默认值与标准路径不一致，导致训练目标静默错误[[11]](https://github.com/huggingface/trl/pull/5731)。另一个是 `seq_kd=True` 时 teacher 前向结果被 student 覆盖而完全浪费[[12]](https://github.com/huggingface/trl/pull/5726)。**这两个 bug 的共同特征是：训练能跑、loss 在降、没有报错，但最终模型不对。在训练框架里，这类 bug 比显式崩溃危险得多——你不知道自己浪费了多少 GPU 小时。**

此外，Qwen2.5 工具调用响应格式支持补齐[[13]](https://github.com/huggingface/trl/pull/5728)，新增了 MFU 计算工具函数覆盖 dense 和 MoE 模型的训练 FLOPs 估算[[14]](https://github.com/huggingface/trl/pull/5698)，以及修复了 activation offloading 中 5GB+ CUDA 内存泄漏和 BNB 反量化缓冲未释放问题[[15]](https://github.com/huggingface/trl/pull/5700)[[16]](https://github.com/huggingface/trl/pull/5730)。

## 三、TokenSpeed：用"窄而深"的策略切入国产模型生态

TokenSpeed 本周的 PR 密度惊人。其瞄准的模型矩阵——DeepSeek V4、MiniMax-M2、Qwen3.5、Kimi K2.5、GPT-OSS——与其他推理框架形成清晰差异化：不是通吃所有模型，而是针对国产模型做深度优化。

DeepSeek V4 路径合并了 DeepGEMM mega_moe experts、fast mHC fused kernels 和 ratio-aware compressed KV cache layout[[17]](https://github.com/lightseekorg/tokenspeed/pull/30)。MiniMax-M2.5 FP8 推理优化在 TP8/EP8 路径落地[[18]](https://github.com/lightseekorg/tokenspeed/pull/10)。Qwen3.5 运行时优化减少了前向准备阶段不必要的 GPU kernel launch、DtoD memcpy 和冗余通信[[19]](https://github.com/lightseekorg/tokenspeed/pull/32)。Mamba prefix cache 和 disaggregated prefill 的 Mamba cache 一并合入[[20]](https://github.com/lightseekorg/tokenspeed/pull/15)[[21]](https://github.com/lightseekorg/tokenspeed/pull/14)，为 hybrid linear attention 模型提供缓存支持。

Kimi K2.5 建立了 NVFP4 agentic 性能 CI 管线，包含 router_gemm token 阈值调优和 tokenize 性能修复[[22]](https://github.com/lightseekorg/tokenspeed/pull/29)。推测解码默认参数调至 3 steps / EAGLE top-k 1[[23]](https://github.com/lightseekorg/tokenspeed/pull/40)。AMD 平台方面，MI355/MI300 的 CI、triton kernels 升级、通信后端切换、硬件拓扑探测也已合入[[24]](https://github.com/lightseekorg/tokenspeed/pull/36)。

**TokenSpeed 的策略是"窄而深"——不做全量模型兼容，而是针对国产模型做端到端的性能优化。在推理框架竞争已经很拥挤的 2026 年，这是一个值得跟踪的差异化信号。**

## 四、LMCache：多云部署的最后几块拼图

LMCache 本周合入的 PR 有一个共同特点：不改核心技术架构，但决定了能否在多云异构环境中交付。

ROCm 平台上通过 Triton block-sparse attention 启用了 CacheBlend 非 prefix KV cache 复用[[25]](https://github.com/LMCache/LMCache/pull/3092)。新增 Azure Blob Storage 作为 NIXL 对象存储后端[[26]](https://github.com/LMCache/LMCache/pull/3160)。修复了 operator 硬编码 `runtimeClassName: nvidia` 导致 AMD 集群 Pod 无法调度的问题[[27]](https://github.com/LMCache/LMCache/pull/3211)。解决了 MP server 重启后 vLLM worker 的 KV cache 注册信息丢失导致的 STORE/RETRIEVE 失败[[28]](https://github.com/LMCache/LMCache/pull/3208)。MP 模式下新增 Device-DAX L2 缓存适配[[29]](https://github.com/LMCache/LMCache/pull/3161) 和 Mooncake 批量操作接口[[30]](https://github.com/LMCache/LMCache/pull/3172)。暴露了 token 级别的 lookup/hit 计数器用于可观测性[[31]](https://github.com/LMCache/LMCache/pull/3196)。

**这一批改动凑在一起，把 LMCache 从"实验室能跑"拉到了"多云异构能交付"——对生产落地来说，这种"无聊但必要"的补缺比炫技性特性重要得多。**

## 五、今天真正值得记住的判断

FP4 量化在三个主流推理框架中同日推进精度修复和 kernel 优化，标志着它正跨越从"实验支持"到"生产可用"的门槛。如果把 2025 年看作 FP8 的过渡年，2026 年下半年很可能进入 FP4 的同类周期。TRL 两个静默训练 bug 比新特性更值得关注——训练框架的工程成熟度决定了多少人正在浪费 GPU 小时而不自知。TokenSpeed 用"窄而深"的策略切国产模型生态，在推理框架竞争趋于同质化时，这个差异化方向值得持续跟踪。

---

## 参考来源

[1] [vLLM NVFP4 W4A16 support PR #41769](https://github.com/vllm-project/vllm/pull/41769)

[2] [vLLM Fix GDN KKT precision loss PR #42076](https://github.com/vllm-project/vllm/pull/42076)

[3] [vLLM Remove nested torch.compile PR #42070](https://github.com/vllm-project/vllm/pull/42070)

[4] [SGLang Cute-DSL FP4 dense GEMM PR #23590](https://github.com/sgl-project/sglang/pull/23590)

[5] [llama.cpp Gemma4 NVFP4 checkpoint convert PR #22804](https://github.com/ggml-org/llama.cpp/pull/22804)

[6] [llama.cpp FP8 KV-cache scales fix PR #22818](https://github.com/ggml-org/llama.cpp/pull/22818)

[7] [Megatron-LM MXFP8/FP4 param gather PR #4563](https://github.com/NVIDIA/Megatron-LM/pull/4563)

[8] [TokenSpeed MXFP4 scale dtype fix PR #42](https://github.com/lightseekorg/tokenspeed/pull/42)

[9] [TRL v1.4.0 Release](https://github.com/huggingface/trl/releases/tag/v1.4.0)

[10] [TRL chunked NLL VLM/MoE PR #5684](https://github.com/huggingface/trl/pull/5684)

[11] [TRL GKD Liger JSD fix PR #5731](https://github.com/huggingface/trl/pull/5731)

[12] [TRL GKD seq_kd teacher forward fix PR #5726](https://github.com/huggingface/trl/pull/5726)

[13] [TRL Qwen2.5 response schema PR #5728](https://github.com/huggingface/trl/pull/5728)

[14] [TRL MFU helpers PR #5698](https://github.com/huggingface/trl/pull/5698)

[15] [TRL activation offloading fix PR #5700](https://github.com/huggingface/trl/pull/5700)

[16] [TRL BNB dequant buffer fix PR #5730](https://github.com/huggingface/trl/pull/5730)

[17] [TokenSpeed DeepSeek V4 perf PR #30](https://github.com/lightseekorg/tokenspeed/pull/30)

[18] [TokenSpeed MiniMax-M2 FP8 PR #10](https://github.com/lightseekorg/tokenspeed/pull/10)

[19] [TokenSpeed Qwen3.5 runtime optimization PR #32](https://github.com/lightseekorg/tokenspeed/pull/32)

[20] [TokenSpeed Mamba prefix cache PR #15](https://github.com/lightseekorg/tokenspeed/pull/15)

[21] [TokenSpeed PD Mamba cache PR #14](https://github.com/lightseekorg/tokenspeed/pull/14)

[22] [TokenSpeed Kimi K2.5 NVFP4 agentic perf CI PR #29](https://github.com/lightseekorg/tokenspeed/pull/29)

[23] [TokenSpeed speculative decoding defaults PR #40](https://github.com/lightseekorg/tokenspeed/pull/40)

[24] [TokenSpeed AMD platform support PR #36](https://github.com/lightseekorg/tokenspeed/pull/36)

[25] [LMCache ROCm CacheBlend PR #3092](https://github.com/LMCache/LMCache/pull/3092)

[26] [LMCache Azure Blob NIXL PR #3160](https://github.com/LMCache/LMCache/pull/3160)

[27] [LMCache GPU vendor operator PR #3211](https://github.com/LMCache/LMCache/pull/3211)

[28] [LMCache MP reconnect PR #3208](https://github.com/LMCache/LMCache/pull/3208)

[29] [LMCache Device-DAX L2 PR #3161](https://github.com/LMCache/LMCache/pull/3161)

[30] [LMCache Mooncake batch ops PR #3172](https://github.com/LMCache/LMCache/pull/3172)

[31] [LMCache blend hit-rate counters PR #3196](https://github.com/LMCache/LMCache/pull/3196)