---
title: AI Infra 早报｜内核层提速与应用侧新版本，Blackwell 优化进入多项目协同阶段
date: 2026-04-02 05:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: 今天的主线是内核层：llama.cpp 合入 NVFP4 MMQ 通用矩阵乘内核并连发三个构建；SGLang 将 TRT-LLM 稀疏 MLA 内核接入 DSA 预填充路径，并通过 JIT 编译 RMSNorm 减少归一化开销；TRT-LLM 版本号推至 rc11，Mamba2 MTP SSM 缓存 CUDA 内核同期合入。应用侧，OpenClaw 发布 v2026.4.1，是今年功能覆盖最广的一次：任务看板、SearXNG、Bedrock Guardrails、Provider 插件化回放钩子，以及 exec 审批耐久化修复一并上线。
---

上一周期密集的版本发布已经收尾——TRL v1.0.0、vLLM v0.18.1、DeepSpeed v0.18.9 都在过去两天完成了各自的里程碑落地。今天进入的，是内核和协议层的精度打磨阶段：多个项目同日推进 Blackwell（NVFP4）量化支持，投机解码路径补入 SSM 专属内核，Ray Serve 修复了一个在高并发流式推理中才会暴露的协议细节，OpenClaw 用一次大版本把过去几个月积压的架构欠账一并清偿。

## 一、Blackwell 量化内核：多后端并行推进

llama.cpp 今天完成了一项对 Blackwell 用户来说实质性的补丁：PR #21074 新增了 NVFP4 通用矩阵乘量化（MMQ）内核。[[1]](https://github.com/ggml-org/llama.cpp/pull/21074) 此前 llama.cpp 在 Blackwell GPU 上运行 NVFP4 权重时会退回到性能更低的 MMVQ/dp4a 路径，本次新增的内核通过复用现有 `vec_dot_q8_0_16_q8_1_mma` 和 `vec_dot_q8_0_16_q8_1_dp4a` 两条向量点积路径，以与 Q3_K 相同的 tile 尺寸对 NVFP4 做矩阵乘，在预填充速度上相比回退路径有显著提升。提交说明中提到当前寄存器压力较高、占用率约 16.7%，Blackwell 专用 MMA 内核将在后续 PR 中跟进。

同日，SYCL 后端也通过 PR #21227 补入了 NVFP4 `mul_mat` 支持。[[2]](https://github.com/ggml-org/llama.cpp/pull/21227) CUDA 和 SYCL 两条路径并行推进，反映了 llama.cpp 当前对 Intel 和 NVIDIA 下一代精度格式覆盖的重视。

4 月 1 日 llama.cpp 连发了三个构建：b8609、b8610、b8611，频率明显高于日常单次节奏，背后是多个并行 bugfix 的叠加——CUDA FA 内核选择逻辑修复（#21271）、Hexagon RMSNorm/DIV 精度改善（#21251）、LFM2 工具调用解析修复（#21242）、RWKV 线程分配修复（#21226）几个修复分批合入，触发了三次独立构建。

## 二、SGLang：跨项目内核复用与 JIT 归一化

SGLang 今天合入了两个值得关注的 kernel 层变更。

PR #21783 将 TRT-LLM 的稀疏 MLA（Multi-head Latent Attention）内核接入 SGLang 的 DSA（Dynamic Sparse Attention）预填充批次路径。[[3]](https://github.com/sgl-project/sglang/pull/21783) DSA 是 SGLang 处理 DeepSeek 系 MLA 模型的专用路径，此前使用自己的 dense 计算实现；接入 TRT-LLM 稀疏 MLA 内核后，可以跳过部分不必要的 dense 注意力计算，在保持精度的前提下降低预填充的计算量。这一变更依赖 flashinfer v0.6.7，是推理框架间内核级复用正在常态化的一个具体例子。

PR #21834 为 RMSNorm 引入了 JIT 编译路径，[[4]](https://github.com/sgl-project/sglang/pull/21834) 通过 Triton-based JIT 减少归一化内核的 launch 开销——RMSNorm 在 Transformer 推理中高频调用，每次调用的 kernel launch overhead 在高并发场景下可以显著累积。这个 PR 的注释中明确声明由 Claude 辅助生成，是 SGLang 仓库中首次出现此类说明，某种程度上也是推理框架社区 AI 辅助开发实践的一个信号。

此外，MooncakeSpec CI 测试（#21794）将模型升级为 EAGLE3 + Llama-3.1 组合，测试阈值从 0.20 提升至 0.74，得分约 0.775，完成了 speculative decoding CI 测试的代际切换。

## 三、TRT-LLM 向 stable 收拢，Mamba2 SSM 内核补位

TRT-LLM 主干版本号通过 PR #12627 推进至 1.3.0rc11，[[5]](https://github.com/NVIDIA/TensorRT-LLM/pull/12627) 意味着 v1.3.0 stable 的最终 RC 阶段已经开启。昨日早报已经覆盖了 rc10 的核心功能（FlexKV、KV cache-aware ADP router、Qwen3.5 NVFP4、request priority API），今天在 rc11 窗口中合入的最值得记录的是 PR #12537：为 Mamba2 MTP（Multi-Token Prediction）树状投机解码路径新增专用的 SSM 缓存 CUDA 内核。[[6]](https://github.com/NVIDIA/TensorRT-LLM/pull/12537)

Mamba2 是近期被引入混合架构（如 Nemotron-H）的 SSM 组件，与标准 Transformer 不同，它的状态计算需要专门的 cache 更新逻辑。此前 TRT-LLM 的树状投机解码缺乏针对 SSM 的原生内核，这个 PR 填补了这一空白，支持 float32/float16/bfloat16 精度，并暴露了可配置的 gating 和 caching 参数。

同期合入的 MoE autotuner OOM 修复（#12523）解决了大 context 场景下 trtllmGen MoE runner 的内存溢出问题。

## 四、协议细节与生产稳定性

Ray Serve 修复了一个在流式推理高并发场景下才会稳定复现的协议 bug：PR #62246 修复了 LLM 路径在 SSE 流式响应结束时 `data: [DONE]` 被发送两次的问题。[[9]](https://github.com/ray-project/ray/pull/62246) 这类双 `[DONE]` bug 的后果因客户端实现不同而异，严格的 OpenAI 兼容客户端会在第二个 `[DONE]` 处抛出解析错误，宽松实现则可能产生一个空回复帧——任何一种都是生产中难以追踪的噪音。

LMCache 的 PR #2893 为分块（Chunk-Based，CB）KV 缓存模式补入了 LRU 驱逐逻辑。[[10]](https://github.com/LMCache/LMCache/pull/2893) 此前 CB 模式没有内存上限，长时间运行后内存会无限增长，这个限制使其只能用于实验环境；引入驱逐后，CB 模式可以设定内存预算并在超限时自动淘汰最久未用的 chunk，生产可用性从"实验性"提升到"可托管"。

Megatron-LM 修复了 FSDP 路径中精度感知优化器配置的传递缺口（#4024），[[7]](https://github.com/NVIDIA/Megatron-LM/pull/4024) 并为 NVshmem 已知问题加入了防护逻辑（#4093），[[8]](https://github.com/NVIDIA/Megatron-LM/pull/4093) 两个修复对 FP8/BF16 混合精度训练的正确性和稳定性都有影响。

## 五、OpenClaw v2026.4.1：架构欠账与功能补完

OpenClaw 发布了 v2026.4.1，从 release notes 的长度来看，这是今年以来单版本变更密度最高的一次。[[11]](https://github.com/openclaw/openclaw/releases/tag/v2026.4.1)

功能侧，`/tasks` 命令现在可以在聊天界面直接查看当前 session 的后台任务状态，不再需要切换到外部工具面板；SearXNG 捆绑提供商插件上线，提供可自托管的开源搜索后端；Amazon Bedrock Guardrails 支持进入捆绑提供商，Bedrock 用户可以配置内容过滤策略而无需自建拦截层；macOS Voice Wake 支持让用户可以通过语音触发 Talk Mode。

架构侧，PR #59143 将回放、历史记录、工具 schema 和推理模式的运行时钩子从 Core 迁移到 Provider 插件层，[[12]](https://github.com/openclaw/openclaw/pull/59143) 实现核心与提供商特定逻辑的解耦。这是一个用户层面不直接感知的变化，但它是后续多 provider 运行时切换和 session 回放能否做得干净的前提条件。

exec 审批路径也完成了一次重要修复：`allow-always` 现在被写入耐久信任存储，不再每次重启后失效；cron 隔离 session 下因无可用路由触发的审批死循环也已修复；PR #58872 修复了 recurring main job 的 busy-wait 漂移问题。[[13]](https://github.com/openclaw/openclaw/pull/58872)

## 参考来源

[1] [llama.cpp #21074 ggml-cuda: Add generic NVFP4 MMQ kernel](https://github.com/ggml-org/llama.cpp/pull/21074)

[2] [llama.cpp #21227 SYCL: Support nvfp4 type in mul_mat](https://github.com/ggml-org/llama.cpp/pull/21227)

[3] [SGLang #21783 DSA: Support trtllm sparse mla kernel for prefill batches](https://github.com/sgl-project/sglang/pull/21783)

[4] [SGLang #21834 Feature: JIT rmsnorm update](https://github.com/sgl-project/sglang/pull/21834)

[5] [TRT-LLM #12627 Bump version to 1.3.0rc11](https://github.com/NVIDIA/TensorRT-LLM/pull/12627)

[6] [TRT-LLM #12537 Add Mamba2 MTP SSM cache CUDA kernel for tree-based speculative decoding](https://github.com/NVIDIA/TensorRT-LLM/pull/12537)

[7] [Megatron-LM #4024 m-fsdp: wire use_precision_aware_optimizer from ddp_config](https://github.com/NVIDIA/Megatron-LM/pull/4024)

[8] [Megatron-LM #4093 Guard NVshmem issues](https://github.com/NVIDIA/Megatron-LM/pull/4093)

[9] [Ray Serve #62246 Fix duplicate 'data: [DONE]' in streaming SSE responses](https://github.com/ray-project/ray/pull/62246)

[10] [LMCache #2893 Add eviction for CB KV cache](https://github.com/LMCache/LMCache/pull/2893)

[11] [OpenClaw v2026.4.1 release](https://github.com/openclaw/openclaw/releases/tag/v2026.4.1)

[12] [OpenClaw #59143 refactor: add provider replay runtime hook surfaces](https://github.com/openclaw/openclaw/pull/59143)

[13] [OpenClaw #58872 fix(cron): avoid busy-wait drift for recurring main jobs](https://github.com/openclaw/openclaw/pull/58872)
