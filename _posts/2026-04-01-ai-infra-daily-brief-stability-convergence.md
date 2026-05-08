---
title: AI Infra 早报｜版本节点落地，基础设施向健壮性收拢
date: 2026-04-01 05:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: vLLM v0.18.1 修复 Blackwell 上 Qwen3.5 FP8 精度退化并稳定 MLA 后端；TRT-LLM v1.3.0rc10 打包 FlexKV、KV cache-aware 路由和 Qwen3.5 NVFP4；TRL v1.0.0 tag 确认里程碑；DeepSpeed 补上 ZeRO Stage 3 内存碎片整理工具和进程组竞态修复；OpenClaw v2026.3.31 完成任务控制平面统一，原子写入与 owner-key 访问边界同日上线。
tags: [Inference, Quantization, Attention, vLLM]
---

今天没有大规模功能上线，但一批重要的稳定性和安全修复密集落地，几个主要框架都在把前一阶段的功能债和安全欠账结清。

## 一、推理侧：精度、稳定与架构铺垫并进

vLLM 发布了 v0.18.1 patch [[1]](https://github.com/vllm-project/vllm/releases/tag/v0.18.1)，主要解决两个问题。第一个是 Qwen3.5 FP8 在 Blackwell（B200/B100，SM100）上的精度退化：DeepGemm 对 Qwen3.5 E8M0 scaling factor 的处理路径存在数值误差，v0.18.0 上线以来一直有用户反映生成质量在新卡上低于预期，此次修复补全了 E8M0 格式的处理路径。第二个是 MLA prefill 后端的稳定性问题：v0.18.0 把 SM100 MLA prefill 默认后端切换到了一个实验性路径，但实测稳定性不及 TRT-LLM 后端，于是回退到 TRT-LLM 作为默认 [[1]](https://github.com/vllm-project/vllm/releases/tag/v0.18.1)。

同日合入的 Helion kernel torch.compile 集成（PR #38592 [[2]](https://github.com/vllm-project/vllm/pull/38592)）是一个更长期的架构铺垫：Helion 是基于 Triton 的 kernel DSL，此 PR 将其接入 dynamo 拦截点，为未来 vLLM kernel 动态编译提供统一入口。Responses API 路径也补全了 `presence_penalty` 和 `frequency_penalty` 字段（PR #38613 [[3]](https://github.com/vllm-project/vllm/pull/38613)），消除与 completions 路径的参数对齐缺口；PD 分离的 NIXL 兼容矩阵文档（PR #38628 [[4]](https://github.com/vllm-project/vllm/pull/38628)）明确了哪些 Prefill-Decode 拓扑在 NIXL transport 下经过了验证。

TRT-LLM v1.3.0rc10 [[5]](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.3.0rc10) 是近期迭代中功能密度较高的一个 RC。FlexKV 支持异构 KV cache 布局，允许不同 transformer 层使用不同的 page size 或存储精度，这对 MoE 和 hybrid attention 模型（如 Nemotron-H）的内存利用率优化很有价值；KV cache-aware ADP router 在做前缀路由时会感知节点的 KV cache 实际占用状态，改善 prefix reuse 命中率；Qwen 3.5 NVFP4 支持（PR #12302）进入 RC，Nemotron-H all-reduce 与 norm 融合优化也一并纳入。这个版本还有一个 BREAKING 变更：log prob 默认不再按 token 数量归一化，与 OpenAI 标准对齐（PR #12366）。

SGLang 在边缘修复上也推进了一件事：Sliding Window Attention 模型（如 Mistral 系列）在使用 SGLang 的 KV cache 驱逐时，piecewise CUDA graph 之前不能同时启用。PR #21754 [[6]](https://github.com/sgl-project/sglang/pull/21754) 解除了这一限制，两者可以共存，大 batch 推理下的吞吐稳定性得到改善。

llama.cpp 今天解决了一个推理行为方面的隐患：启用 `--reasoning-budget N` 后，采样设置仍然生效，会在推理预算尚未耗尽时提前截断模型输出。PR #21209 [[7]](https://github.com/ggml-org/llama.cpp/pull/21209) 让推理预算模式下自动旁路 backend sampling，推理过程和采样过程不再互相干扰。同日，PR #21207 [[8]](https://github.com/ggml-org/llama.cpp/pull/21207) 打通了 ARM64 的 CPU 和 Vulkan CI 发布流程，苹果 Silicon 和 ARM 服务器部署场景终于有了持续的自动化质量保证。

## 二、训练侧：TRL 确认里程碑，DeepSpeed 继续清账

TRL v1.0.0 tag [[9]](https://github.com/huggingface/trl/releases/tag/v1.0.0) 今日在 Releases 正式落地。昨日早报已详细报道了 AsyncGRPO/VESPO/DPPO/SDPO 的技术细节，今天补充一个后续合并的细节：PR #5405 [[9]](https://github.com/huggingface/trl/releases/tag/v1.0.0) 为 Qwen 3.5 添加了第二版 chat template 支持。Qwen 3.5 存在两种对话格式变体，TRL 的 reward 和 policy trainer 在多轮对话中处理这些变体时需要正确区分，避免 tokenization 序列错位导致 reward 计算偏移。

DeepSpeed 在 v0.18.9 发布后继续补稳定性欠账。PR #7940 [[10]](https://github.com/deepspeedai/DeepSpeed/pull/7940) 新增了 ZeRO Stage 3 defragment utility：长时间训练中频繁分配和释放 parameter shards 会在 GPU 内存留下碎片，导致明明有足够总量的内存却分配失败。这个工具允许在训练检查点处主动整理内存碎片，对超长训练任务（如数百亿参数模型的多日训练）有实际价值。PR #7941 [[11]](https://github.com/deepspeedai/DeepSpeed/pull/7941) 修复了进程组关闭时的竞态条件——多卡训练正常结束时，`destroy_process_group()` 路径在部分 rank 提前退出的情况下会 hang，通过加锁和超时保护完成修复。

## 三、生产部署侧：Ray Serve 零副本场景修复

Ray Serve 修复了一个影响高可用部署的边界问题（PR #62213 [[12]](https://github.com/ray-project/ray/pull/62213)）：当 ingress replica 数量被显式缩容为 0（例如临时下线某个服务）时，HAProxy 的 healthz 探测会收到不健康的响应，导致整个集群从负载均衡中被移除。修复后即使零副本状态，head node 的健康检查端点也返回健康，缩容不等于不可用。同期，Serve controller 的压测矩阵扩展到 3072/4096 副本规模（PR #62211 [[13]](https://github.com/ray-project/ray/pull/62211)），为千级模型副本的 serving 部署积累基准数据。Ray LLM 的 PD/DP 相关 API 继续保持 alpha 状态，表明 PD 分离方案在 Ray 侧还处于迭代阶段。

## 四、OpenClaw：任务系统从"书记员"升级为"控制平面"

OpenClaw 发布 v2026.3.31 [[14]](https://github.com/openclaw/openclaw/releases/tag/v2026.3.31)，其中最重要的架构变化是任务控制平面的统一。此前，background tasks 主要是 ACP run 的状态记录，其他类型的后台执行（subagent、cron job、background CLI）各自维护状态。此次重构将四种执行类型全部接入同一 SQLite-backed 任务账本，并引入 `openclaw flows list|show|cancel` 作为统一的任务流可见性接口。

配套当日合入的两个安全增强值得关注。PR #58521 [[15]](https://github.com/openclaw/openclaw/pull/58521) 将 task store 的写入从直接覆盖改为 write-to-temp-then-rename 的原子操作，防止节点崩溃时任务记录出现部分写入的损坏状态。PR #58516 [[16]](https://github.com/openclaw/openclaw/pull/58516) 为每个 task 引入 owner-key 访问边界，不同 session 和 agent 的任务记录互相隔离，防止跨 agent 的任务越权读写——在多 agent 协作场景下这是一个必要的隔离层。

Gateway 层同日修复了两个稳定性问题：高负载模型 fallback 时 session 进入死亡循环（PR #58379），以及 exec 审批中 shell init-file 脚本注入绕过漏洞（PR #58369）——后者通过完全拒绝 shell init-file 类脚本的审批匹配来关闭这条绕过路径。

---

今天的整体节奏是"收拢固化"而非扩张。v0.18.1 清了 Blackwell 的精度债，TRL 的里程碑 tag 落定，DeepSpeed 继续补 ZeRO 的长尾稳定性，llama.cpp 的推理预算解耦让边缘部署行为更可预期。OpenClaw 的任务控制平面统一是一个不显眼但重要的地基工程，后续功能层的可靠性依赖于此。没有突破性发布，但积累在继续。

## 参考来源

[1] [vLLM v0.18.1 patch release](https://github.com/vllm-project/vllm/releases/tag/v0.18.1)

[2] [vLLM #38592 Helion kernel torch.compile 集成](https://github.com/vllm-project/vllm/pull/38592)

[3] [vLLM #38613 Responses API presence/frequency penalty](https://github.com/vllm-project/vllm/pull/38613)

[4] [vLLM #38628 PD+NIXL 兼容矩阵文档](https://github.com/vllm-project/vllm/pull/38628)

[5] [TRT-LLM v1.3.0rc10](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.3.0rc10)

[6] [SGLang #21754 SWA KV cache 驱逐与 piecewise CUDA graph 联合启用](https://github.com/sgl-project/sglang/pull/21754)

[7] [llama.cpp #21209 reasoning budget 与采样器解耦](https://github.com/ggml-org/llama.cpp/pull/21209)

[8] [llama.cpp #21207 ARM64 CPU/Vulkan CI 打通](https://github.com/ggml-org/llama.cpp/pull/21207)

[9] [TRL v1.0.0 正式发布](https://github.com/huggingface/trl/releases/tag/v1.0.0)

[10] [DeepSpeed #7940 ZeRO Stage 3 defragment utility](https://github.com/deepspeedai/DeepSpeed/pull/7940)

[11] [DeepSpeed #7941 进程组关闭竞态修复](https://github.com/deepspeedai/DeepSpeed/pull/7941)

[12] [Ray Serve #62213 零副本缩容健康检查修复](https://github.com/ray-project/ray/pull/62213)

[13] [Ray Serve #62211 3072/4096 副本规模 benchmark](https://github.com/ray-project/ray/pull/62211)

[14] [OpenClaw v2026.3.31 release](https://github.com/openclaw/openclaw/releases/tag/v2026.3.31)

[15] [OpenClaw #58521 task-store 原子写入](https://github.com/openclaw/openclaw/pull/58521)

[16] [OpenClaw #58516 owner-key task 访问边界](https://github.com/openclaw/openclaw/pull/58516)
