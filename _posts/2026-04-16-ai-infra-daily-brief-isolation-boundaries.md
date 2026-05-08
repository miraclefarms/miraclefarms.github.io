---
title: AI Infra 早报｜隔离边界开始进入推理基础设施默认设计
date: 2026-04-16 05:30:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: 过去三天，vLLM 与 LMCache 开始把按用户隔离的 cache_salt 透传进 KV 复用与驱逐接口，Mooncake 把 failover 和 SSD 指标做成可观测状态，llama.cpp 与 SGLang 则收紧 GPU 互联和上下文 ownership，推理系统正在系统改写那些过去默认成立的边界假设。
tags: [KV Cache]
---

过去三天，AI Infra 最值得写的变化，不是哪一个项目又多了一项 headline feature，而是多个核心项目都开始重写“边界条件”。缓存系统不再默认所有请求共享同一套复用命名空间，传输层不再默认一条链路出问题后只能让上层自己兜底，GPU 互联也不再默认只要硬件宣称可用就应该自动打开。过去那些被当作“运行环境细节”的东西，正在被正式写进推理基础设施的主设计里。

这也是今天最值得记住的判断：**推理系统开始把租户边界、故障边界和设备边界当成默认设计对象。** 这类改动通常不如新模型支持、新 kernel 或新 benchmark 数字显眼，但它决定了系统从“单一理想环境里跑得通”走向“多租户、会出错、硬件形态复杂的真实生产环境”时能不能继续稳定成立。

## 一、多租户隔离开始进入 KV 复用主链路

最直接的信号来自 vLLM 与 LMCache 的联动。**vLLM 把 `request.cache_salt` 正式透传进 LMCache 的多进程 connector[[1]](https://github.com/vllm-project/vllm/pull/39837)**，让原本只存在于 OpenAI API 层、用于 prefix cache 隔离的字段，开始真正参与 lookup、retrieve 和 store 这类底层 KV 操作。这个变化的重要性不在于多了一个参数，而在于缓存系统终于承认“命中”不只是 token 序列相同，还可能取决于这是谁的缓存、应该归到哪一个租户空间。

LMCache 自己也在同步把这条路径补完整。**它先给 MP adapter 全面加上 `cache_salt` 接口形状[[2]](https://github.com/LMCache/LMCache/pull/3029)**，明确后续每一次 lookup、store 与 retrieve 都可以带着租户级上下文进入内部 key 构造；随后又在 eviction policy 抽象里加入 `is_user_level` 和 `cache_salt` 参数[[3]](https://github.com/LMCache/LMCache/pull/3032)，为按用户配额、按用户驱逐这类策略预留正式扩展点。换言之，LMCache 不再把“全局 LRU”当成唯一真相，而是在为“每个用户有各自边界”的缓存治理方式铺路。

同一主线还出现在观测面。**LMCache 新增了 L1 lifecycle subscriber[[4]](https://github.com/LMCache/LMCache/pull/2986)**，开始按采样记录 chunk lifetime、idle-before-evict、reuse gap 等生命周期指标。缓存一旦进入多租户和分层治理阶段，光知道命中率已经不够了；系统还得知道某一类 block 为什么长期滞留、为什么总在被驱逐后又迅速回读。这意味着 KV 复用正在从“尽量命中”走向“带着隔离语义去管理生命周期”。

## 二、故障切换和分层存储开始被当成正常状态

**Mooncake 的 TENT 这次把原本存在但没有真正走通的 failover 逻辑接上线[[5]](https://github.com/kvcache-ai/Mooncake/pull/1878)**，在传输任务失败时自动沿优先级链路切到下一条 transport，并为 failover 次数设上限，同时把 `RDMA -> TCP` 这类切换过程和计数导出成结构化日志与 Prometheus 指标。它真正说明的问题是，系统已经不再把“主 transport 应该一直可用”当作默认前提，而是开始把链路失败视为必须被控制、限制和观测的日常事件。

**同一时间，Mooncake 又把 SSD offload 的吞吐、IOPS 和 P50/P90/P99 延迟做成标准指标[[6]](https://github.com/kvcache-ai/Mooncake/pull/1879)**。这一步很关键，因为一旦 KV 系统开始落到 SSD 甚至更慢的层级上，团队讨论的就不只是“能不能 offload”，而是读写到底慢在哪、抖动集中在哪个百分位、吞吐和延迟在不同批次下如何变化。只要这些指标进入默认导出，分层存储就不再只是容量应急方案，而会逐渐变成可被工程化优化的常规路径。

把这两条更新放在一起看，Mooncake 正在把“失败”与“慢存储”一起拉进默认可观测面。过去许多 KV 系统更像是在 steady state 下展示性能曲线；现在更关键的是，当链路失效、介质变慢、回退路径被触发时，系统能不能继续有秩序地运行，并且让运维团队看得到发生了什么。

## 三、GPU 互联和上下文 ownership 不再被默认放大

设备边界的收紧在 llama.cpp 和 SGLang 上表现得尤其明显。**llama.cpp 把 CUDA P2P access 改成显式 opt-in[[7]](https://github.com/ggml-org/llama.cpp/pull/21910)**，原因不是性能不重要，而是某些主板和 BIOS 组合下，即便 `cudaDeviceCanAccessPeer` 返回可访问，也仍然会出现崩溃和输出损坏。这个改动等于明确承认：硬件层“理论上支持”不代表框架就应该默认替你打开。对部署者来说，这是一种更成熟的边界策略，先把潜在破坏性路径关到显式配置里，再让用户按自己的硬件拓扑去承担收益与风险。

**llama.cpp 还把 NCCL communicator 的管理收回到外部 context[[8]](https://github.com/ggml-org/llama.cpp/pull/21891)**，不再让通信资源以更隐式的方式散落在 backend 生命周期里。这和 P2P opt-in 放在一起看，方向非常一致：多 GPU 通信不再只是“能建链就建链”，而是逐步被改造成带 ownership、带生命周期的明确对象。

**SGLang 对 Kimi-K2.5 GPU 图像预处理路径的修复[[9]](https://github.com/sgl-project/sglang/pull/22858)** 则把这个问题暴露得更具体。它处理的是 `grid_thws` 作为 CUDA tensor 在 TP rank 之间被 pickle / 广播后，触发所有 rank 在 `cuda:0` 上被动打开完整 CUDA context 的问题，每个 rank 会额外吃掉约 500 MiB 显存。SGLang 最终把这条路径改回更受控的 ownership 模式，本质上也是在拒绝“上下文副作用无害”这种默认假设。框架一旦进入大模型、多卡和多模态共存阶段，这类上下文泄漏就不再是边角 bug，而是直接决定部署成本和稳定性的硬约束。

## 四、今天真正值得记住的判断

如果把今天这些更新放在一起看，一个很清晰的阶段变化已经出现了。AI Infra 团队不再只是在比谁能把新能力接进主线，而是在补那些过去被当作环境前提的边界条件：缓存是不是天然共享，失败是不是异常而不是常态，设备互联是不是只要可用就该默认开启，上下文副作用是不是可以忽略。

这类边界工程往往不会带来最亮眼的 benchmark，却更接近真正的基础设施竞争。下一轮谁更占优势，未必取决于谁再多发明一个新优化，而更可能取决于谁更早把隔离、故障和 ownership 做成默认可用、默认可观测、默认可控制的系统能力。

---

## 参考来源

[1] [vLLM 透传 cache_salt 到 LMCache MP connector 以支持按用户隔离](https://github.com/vllm-project/vllm/pull/39837)

[2] [LMCache 为多进程 adapter 接口加入 cache_salt 参数](https://github.com/LMCache/LMCache/pull/3029)

[3] [LMCache 为 EvictionPolicy 加入 is_user_level 与 cache_salt 扩展点](https://github.com/LMCache/LMCache/pull/3032)

[4] [LMCache 新增 L1 lifecycle subscriber 观测缓存块生命周期](https://github.com/LMCache/LMCache/pull/2986)

[5] [Mooncake TENT 接通跨 transport failover 并加入安全限制与指标](https://github.com/kvcache-ai/Mooncake/pull/1878)

[6] [Mooncake 为 SSD offload 增加 throughput、IOPS 与延迟指标](https://github.com/kvcache-ai/Mooncake/pull/1879)

[7] [llama.cpp 将 CUDA P2P access 改为显式 opt-in](https://github.com/ggml-org/llama.cpp/pull/21910)

[8] [llama.cpp 将 NCCL communicator 管理收回到 context](https://github.com/ggml-org/llama.cpp/pull/21891)

[9] [SGLang 修复 Kimi-K2.5 路径下 TP ranks 被动创建 cuda:0 context 的问题](https://github.com/sgl-project/sglang/pull/22858)
