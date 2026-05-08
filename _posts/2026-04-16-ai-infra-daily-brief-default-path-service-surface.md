---
title: AI Infra 早报｜默认路径优化开始吞并服务面工程
date: 2026-04-16 05:30:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: 过去三天，SGLang 与 vLLM 把并行策略、CUDA Graph 与前后处理线程模型继续收进默认路径；TensorRT-LLM 开始系统补齐 Prometheus 指标和 disaggregated serving 稳定性，Mooncake 则把 SSD offload 与异构传输进一步下沉到底座。
tags: [Inference, Disaggregation]
---

过去三天，AI Infra 的变化已经不太像前一阶段那种“再多一个新特性”的竞赛，而更像一场默认路径收口。推理框架不再满足于把某个 kernel、某种并行策略或某段服务逻辑做成可选开关，而是开始把这些能力直接变成默认配置、稳定路径和可观测指标；真正的竞争点，正在从“你能不能跑起来”转向“默认情况下能不能跑得又快又稳”。

这也是今天最值得记住的判断：**性能优化、服务可观测性和数据面扩展，正在被一起推进到默认生产路径。** SGLang 和 vLLM 在做的是把手工调优收进框架内部，TensorRT-LLM 在补齐服务侧指标与 disaggregated serving 的稳定性，而 Mooncake 则继续把 SSD offload、ROCm HIP 和 Kunpeng 传输能力往统一底座里收。把这些变化放在一起看，默认路径质量已经成为新一轮 AI Infra 竞争的主战场。

## 一、默认路径正在接管手工调优

**SGLang 将多 GPU diffusion 的默认并行策略从纯 sequence parallel 切到 CFG parallel[[1]](https://github.com/sgl-project/sglang/pull/22763)**，而且触发条件非常直接：只要用户以 `--num-gpus >= 2` 启动、又没有显式指定并行参数，框架就会自动启用 CFG parallel。官方给出的 H200 双卡数据里，Qwen-Image 1024x1024 场景从 11.21 秒降到 7.14 秒，Wan-1.3B T2V 从 5.95 秒降到 4.77 秒，说明这类收益已经大到值得直接吞进默认行为里，而不是继续让用户手工调 `--sp-degree` 或 `--ulysses-degree`。

同一条主线也出现在 vLLM。**vLLM 为 Qwen3-VL 视频推理补上了 ViT full CUDA graph 正式支持[[2]](https://github.com/vllm-project/vllm/pull/38061)**，把原本只覆盖图像编码器的路径扩展到视频输入，并给出多组 A100 benchmark。在多卡 Qwen3-VL-32B 配置下，平均耗时改善约 13% 到 21%，P99 改善超过 80%。这类改动的意义不只是“又快一点”，而是多模态编码器也开始进入可以被系统级捕获和复用的默认优化面。

**vLLM 还把 pooling entrypoints 的前后处理阻塞操作下放到线程池[[3]](https://github.com/vllm-project/vllm/pull/39763)**，直接对准 async tokenizer 引入后的约 2ms 延迟回归和高并发吞吐波动。这里可以看出另一个很清晰的趋势：框架团队不再只盯着模型算子本身，而是开始把前后处理、调度和服务线程模型也纳入“默认路径优化”的范围。换言之，默认路径的定义已经从 kernel 扩展到了整个 serving surface。

## 二、服务系统开始补齐“看得见”和“不卡死”

**TensorRT-LLM 增加了生产级 Prometheus 指标[[4]](https://github.com/NVIDIA/TensorRT-LLM/pull/12545)**，把原本停留在 JSON 输出里的 iteration stats、KV cache、inflight batching、spec decode 统计和 per-request 时延指标，统一变成可抓取的 Prometheus metrics。更关键的是，它不只加了 token counters 和 phase histograms，还把模型、并行配置、投机解码配置和 KV cache 配置一起暴露出来。这说明框架团队开始把“系统当前是怎么配置出来的”也视为线上可观测性的一部分，而不只是留给开发者去翻日志。

与指标补齐同样重要的，是服务循环不能在关键阶段自己把自己卡住。**TensorRT-LLM 修复了 benchmark disaggregated serving 模式里的死锁问题[[5]](https://github.com/NVIDIA/TensorRT-LLM/pull/12208)**，核心改动是去掉阻塞式 fill loop，让 executor 在等待生成批次填满时，仍能继续处理 KV transfer、timeout、error detection 和 control requests。这个修复背后的信号很直接：disaggregated serving 已经从“能否演示”进入“能否长期运行”的阶段，而一旦进入这个阶段，控制流是否阻塞就会和吞吐优化一样重要。

把这两条更新放在一起看，一个更大的变化就浮出来了。过去框架常把“性能路径”和“运维路径”分开处理，但现在 production metrics 和 deadlock fix 正在一起进入主线，这意味着服务框架正在默认承担更多 SRE 语义。对部署者来说，接下来真正稀缺的将不是单点 benchmark，而是框架是否能在复杂执行路径里同时保持可观测和可恢复。

## 三、数据面继续向分层存储和异构硬件外扩

**Mooncake 在 Python `setup()` 接口里直接支持 SSD offload[[6]](https://github.com/kvcache-ai/Mooncake/pull/1857)**，允许在 embedded RealClient 模式下通过 `enable_offload=True` 自动拉起内部 RPC server，并动态分配端口处理 offload RPC。这类改动看起来不像 headline feature，但它实质上是在把原本需要额外部署和拼装的 offload 能力，收进更接近应用入口的默认接口里。只要这条路径稳定下来，分层存储就会更容易从“实验开关”变成常规配置。

同时，**Mooncake 继续把传输底座往异构硬件上铺开：一方面在 Python 包里补齐了 ROCm HIP transport[[7]](https://github.com/kvcache-ai/Mooncake/pull/1742)**，让 AMD GPU 传输路径能够真正从 wheel、runtime 到容器环境打通；另一方面又在 Kunpeng 950 平台上推进 UB Transport 第二阶段[[8]](https://github.com/kvcache-ai/Mooncake/pull/1855)，把 URMA mock、测试和构建集成补到位。前者解决的是 AMD 路径可不可用，后者解决的是非 NVIDIA 平台值不值得继续纳入统一传输栈。

这组变化的共同点，是 Mooncake 已经不再只是在做“某一种远端缓存实现”，而是在持续把不同介质和不同硬件的传输能力往统一接口里压。对整个 AI Infra 生态来说，这意味着缓存、offload 和传输层正在更像基础设施抽象，而不是每个项目都各写一套适配层。

## 四、今天真正值得记住的判断

过去三天真正的变化，不是哪一个项目单独快了 10% 或 20%，而是越来越多框架开始把这些优化、指标和传输能力收进默认生产路径。默认路径一旦变强，用户就不必再靠经验去拼装一堆开关；框架之间的差距，也会从“谁功能更多”逐步转向“谁在默认情况下更快、更稳、更容易观测”。

如果这个判断成立，那么下一阶段最值得追踪的就不是零散 feature，而是三条主线会不会继续汇合：并行和图捕获能否进一步自动化，服务侧指标与恢复逻辑能否继续内建，以及缓存与传输底座能否跨硬件真正收敛。谁先把这三件事同时做成默认能力，谁就更可能吃到下一轮生产部署红利。

---

## 参考来源

[1] [SGLang 为多 GPU diffusion 默认启用 CFG parallel](https://github.com/sgl-project/sglang/pull/22763)

[2] [vLLM 为 Qwen3-VL 视频推理支持 ViT full CUDA graph](https://github.com/vllm-project/vllm/pull/38061)

[3] [vLLM 将 pooling entrypoints 的阻塞前后处理下放到线程池](https://github.com/vllm-project/vllm/pull/39763)

[4] [TensorRT-LLM 增加生产级 Prometheus 指标](https://github.com/NVIDIA/TensorRT-LLM/pull/12545)

[5] [TensorRT-LLM 修复 benchmark disaggregated serving 死锁](https://github.com/NVIDIA/TensorRT-LLM/pull/12208)

[6] [Mooncake 在 Python setup 接口中支持 SSD offload](https://github.com/kvcache-ai/Mooncake/pull/1857)

[7] [Mooncake Python 包增加 ROCm HIP transport 支持](https://github.com/kvcache-ai/Mooncake/pull/1742)

[8] [Mooncake 在 Kunpeng 平台推进 UB Transport 第二阶段](https://github.com/kvcache-ai/Mooncake/pull/1855)
