---
title: AI Infra 早报｜首包延迟、指标语义与兼容层正确性进入主路径
date: 2026-04-24 05:30:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: Mooncake 重写 EFA SRD 共享端点后，把跨节点首包延迟和 QP 扩展性一起推进到可部署区间；LMCache、Ray、SGLang 同时修正”指标有值但语义失真”的观测路径；llama.cpp、TRT-LLM、DeepSpeed 则继续为 Anthropic 兼容、MoE 多流和 ZeRO offload 补正确性。
tags: [Inference, KV Cache, MoE]
---

今天这批更新的共同点，不再是“谁又多快了几个点”，而是几个项目同时把过去容易被当成边角料的问题收回了主路径：第一包请求为什么慢、指标看起来正常却不代表真实状态、兼容层和训练路径在复杂组合下会不会悄悄出错。

这背后其实是一个更成熟的阶段信号。AI Infra 现在已经不缺 headline 级优化点，真正开始拉开差距的，是谁能把首包时延、观测语义和边缘正确性写成默认能力。因为这些问题一旦没处理好，后面所有 benchmark 和平台能力都会变得不可信。

## 一、跨节点 KV 传输开始把首包延迟当成正式指标

**Mooncake 重构了 EFA SRD 传输层，把每个本地 NIC 共享一个 `fid_ep` 端点，用 `fi_addr_t` 来寻址所有 peer[[1]](https://github.com/kvcache-ai/Mooncake/pull/1944)**。这条 PR 最重要的地方不是“又做了一次 transport 重构”，而是它正面拆掉了过去 per-peer endpoint 模型的两个现实瓶颈。旧路径下，每增加一个 peer 就会额外消耗 16 个 QP，在 16 NIC 主机上 48 个 peer 就会撞上 768 QP 上限；同时，首包前的握手和 `fi_enable` 也把 warmup 拖得很长。新实现把 SRD connectionless 模型真正用起来之后，这两个问题一起消失了。

Mooncake 在 PR 里给出的数据也很有说服力。p5en 上 cold submit 从 99ms 降到 26ms，`warmupSegment()` 从 17 秒降到 1.1 秒，写带宽仍能维持在约 365 GB/s。也就是说，这不是单纯为了好看而改代码结构，而是在告诉上层系统一件更重要的事：跨节点 KV transport 已经不能只讨论稳态带宽，首包和扩展性正在变成同级指标。

**vLLM 紧接着把 KV offload connector 的 load 路径扩展到支持 multiple KV groups[[2]](https://github.com/vllm-project/vllm/pull/39402)**。这条改动表面不显眼，但它说明上层推理框架也在把 offload 看成正式资源层，而不是单一 cache group 的临时外挂。只要 KV pool 开始分组、异构或分层，装载语义就必须先被补齐，否则 transport 层再快，主框架也接不住。

## 二、可观测性开始从“有指标”转向“指标语义必须真实”

**LMCache 把 CUDA host callback 的时间戳源从 `system_clock` 改成了基于 `steady_clock` 的单调时间映射[[3]](https://github.com/LMCache/LMCache/pull/3103)**。这个修复很典型。过去 `mp.store` 和 `mp.retrieve` 这类事件在 NTP 微调时会出现 end 比 start 更早，Jaeger 最后看到的是一个巨大的无符号溢出时长。表面上 trace 在，span 也在，但这个观测结果本身是假的。LMCache 这次不是给 dashboard 再多加一张图，而是先把“时间”这个最底层的观测语义修正回来。

**Ray Serve 同时修了两个类似的问题：一是过滤 `serve_long_poll_latency_ms` 里由 bootstrap 造成的陈旧观测值[[4]](https://github.com/ray-project/ray/pull/62868)，二是让常规 HTTP proxy 路径也真正支持 per-request timeout 和 disconnect header[[5]](https://github.com/ray-project/ray/pull/62867)**。前者解决的是新 client 启动时把“旧配置最后一次变化时间”误记成当前传播延迟，结果看上去像长轮询慢了几十分钟；后者解决的是同样一条请求走 direct ingress 和 proxy 时，超时语义竟然不一致。两条 PR 都说明一件事：Serve 平台的难点已经不在“有没有指标”或“有没有这个 header”，而在默认路径里的语义是否统一。

**SGLang 也把 OpenTelemetry tracing 补进了 DiffGenerator 和 diffusion worker 路径[[6]](https://github.com/sgl-project/sglang/pull/21254)**。这件事单看像 observability 补课，但放到今天这组更新里看，它更像多模态 runtime 正式从“能跑出图”转向“可接入现有生产观测体系”。当 LLM 路径已经有完整 tracing，而 diffusion 还是黑盒时，多模态系统其实还没进入同一个生产面。

## 三、兼容层和复杂执行路径继续补生产正确性

**llama.cpp 修复了 Anthropic API 路径下的 prefix caching[[7]](https://github.com/ggml-org/llama.cpp/pull/21793)**。问题根源很工程化，也很现实：`x-anthropic-billing-header` 里的 `cch` 值会变，导致长提示词虽然只有后面几千行变了，前缀匹配却总在很早的位置断掉。修完之后，Anthropic 兼容层终于不再持续破坏 cache reuse。这个信号很强，因为 agentic coding 和 tool-use workload 正在越来越多地走 OpenAI/Anthropic 兼容接口，兼容层本身已经是主路径，不再只是 demo adapter。

**llama.cpp 同一天还修掉了 `n_discard` 可被负值触发的 heap-buffer-overflow 漏洞[[8]](https://github.com/ggml-org/llama.cpp/pull/22267)**。这条安全修复的意义不只在 CVE 本身，还在于 server 参数边界开始被当成公开攻击面看待。随着越来越多本地或私有化部署直接把 `llama-server` 暴露给工具链和外部客户端，解析层的一条小检查缺失，影响的就是整个 serving 面。

**TensorRT-LLM 把 AutoDeploy 多流 MoE 路径里的 TP deadlock 从 `caller_stream.synchronize()` 改成 `record_stream` 语义[[9]](https://github.com/NVIDIA/TensorRT-LLM/pull/13220)**，**DeepSpeed 则修复了 ZeRO-1/2 + CPU offload 在一个 step 里多次 `backward()` 时只保留最后一段梯度的问题[[10]](https://github.com/deepspeedai/DeepSpeed/pull/7981)**。前者说明 runtime 在 NCCL collective、stream allocator 和多流执行叠在一起时，CPU 侧同步是会把系统锁死的；后者说明训练框架一旦支持更复杂的 accumulation 边界，老路径里那些“默认只会 backward 一次”的假设就会开始漏数。两条更新都在补同一种债：复杂执行路径正在成为真实用户路径，正确性不能再靠默认前提兜住。

## 四、今天真正值得记住的判断

今天真正值得记住的，不是又出现了某个新的性能开关，而是几个主流项目几乎同时承认了一件事：首包延迟、指标语义和兼容层正确性，本身就是性能系统的一部分。只要这些地方不真实，后面的吞吐、延迟和调度优化都没有稳定地基。

接下来 AI Infra 的分水岭，很可能来自谁先把这三类问题写成默认能力。能把 steady-state benchmark 做漂亮已经不够了，系统还得在第一包、异常时间源、兼容 header、多流并发和复杂训练边界里继续成立。

---

## 参考来源

[1] [Mooncake 重构 EFA SRD 共享端点，消除 per-peer QP cliff 并降低首包时延](https://github.com/kvcache-ai/Mooncake/pull/1944)

[2] [vLLM 为 KV offload connector 增加 multiple KV groups 的 load 支持](https://github.com/vllm-project/vllm/pull/39402)

[3] [LMCache 使用单调时钟修复 CUDA host callback trace 时间戳](https://github.com/LMCache/LMCache/pull/3103)

[4] [Ray Serve 过滤 `serve_long_poll_latency_ms` 中的 bootstrap 陈旧观测](https://github.com/ray-project/ray/pull/62868)

[5] [Ray Serve 在 HTTP proxy 路径支持 per-request timeout 和 disconnect 语义](https://github.com/ray-project/ray/pull/62867)

[6] [SGLang 为 DiffGenerator 和 diffusion worker 增加 OpenTelemetry tracing](https://github.com/sgl-project/sglang/pull/21254)

[7] [llama.cpp 修复 Anthropic API 路径下的 prefix caching](https://github.com/ggml-org/llama.cpp/pull/21793)

[8] [llama.cpp 修复 `n_discard` 负值触发的 heap-buffer-overflow 漏洞](https://github.com/ggml-org/llama.cpp/pull/22267)

[9] [TensorRT-LLM 修复 AutoDeploy 多流 MoE 中的 TP deadlock](https://github.com/NVIDIA/TensorRT-LLM/pull/13220)

[10] [DeepSpeed 修复 ZeRO-1/2 CPU offload 多次 backward 时的梯度丢失](https://github.com/deepspeedai/DeepSpeed/pull/7981)
