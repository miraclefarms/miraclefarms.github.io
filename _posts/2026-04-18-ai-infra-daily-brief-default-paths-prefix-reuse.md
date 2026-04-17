---
title: AI Infra 早报｜低比特与前缀复用开始进入默认主路径
date: 2026-04-18 05:30:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: 过去三天，vLLM、llama.cpp 与 Megatron Core 正把 FP8/NVFP4 从“能跑”推进到默认选路；TensorRT-LLM 则把前缀复用写进调度器核心开销模型，SGLang 和 Ray 进一步把拆分式 serving、流式入口与 HA 能力从专项路径推进为平台主线。
---

过去三天，更值得写的变化不是谁又补了一页模型支持表，而是两条过去经常被当作“优化加分项”的路线，正在被正式写进默认主路径。一条是低比特执行栈：框架不再满足于“模型能加载”，而开始把 FP8、NVFP4 对应的 kernel 选路、图执行和量化上下文变成稳定默认。另一条是前缀复用与拆分式 serving：它不再只是 cache 命中率好不好看的问题，而是直接进入调度器开销、容量计算和服务拓扑设计。

这也是今天真正值得记住的判断。AI Infra 的竞争正在从“功能有没有”转向“默认路径是不是已经按真实生产条件重写过”。低比特如果仍停留在实验分支，前缀复用如果仍只是事后统计，拆分式 serving 如果仍需要手工拼接魔法参数，它们都还算不上真正的主干能力。最近三天的更新说明，几个核心项目已经开始越过这条线。

## 一、低比特不再只是模型兼容表，而是默认执行栈的一部分

**vLLM 把 Marlin kernel 正式接入 block-scaled mm 的选路逻辑[[1]](https://github.com/vllm-project/vllm/pull/40105)**，表面看只是 16 行代码的小修，实际含义却很明确：FP8 路径已经不再只是“模型声明自己是 FP8”，而是连具体 linear kernel 的默认选择都开始围绕这类模型重排。PR 给出的测试直接用 `Qwen/Qwen3.5-27B-FP8` 在 A100 上跑通，并在日志里确认 `MarlinFP8ScaledMMLinearKernel` 被选中。框架一旦开始把低比特 kernel 选路写进默认逻辑，而不是让用户自己猜 backend 是否命中，低比特才真正从兼容性选项进入主执行栈。

**llama.cpp 也在同一时间为 Gemma4 接入 NVFP4 tensor 支持[[2]](https://github.com/ggml-org/llama.cpp/pull/21971)**。这条 PR 的体量很大，直接改到了图构建和大批模型定义文件，不是补一个 reader 就结束，而是让 NVFP4 这种格式能在现有图和模型体系里被真正接住。更关键的是，它指向了一个越来越清楚的现实：低比特格式的竞争已经从“谁先发布一种量化”转向“谁能把这种量化接进通用图执行与模型装配流程”。一旦还需要围绕某个模型单独打补丁，那就还谈不上默认能力。

**Megatron Core 0.17.0 的 release 则把这件事推进到训练与推理一体化的层面[[3]](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.17.0)**。这版更新里同时出现了 FP4 context for Mamba、inference linear 层的 MXFP8 quantization、对更细粒度 CUDA graph 覆盖的补强，以及针对 full-iteration 图和 offload 交互的修正。把这些条目放在一起看，Megatron 的方向已经不是“给某个 recipe 多一个低比特开关”，而是把低比特、图执行和 offload 这些原本容易彼此打架的路径收敛到同一套可持续维护的训练/推理主干里。

## 二、前缀复用开始进入调度器数学，而不是留在命中率面板里

**TensorRT-LLM 把原先分散的 prefix reuse 查询收敛成一次 `analyzePrefixReuse()`[[4]](https://github.com/NVIDIA/TensorRT-LLM/pull/13139)**，这是今天最值得写的单条调度器更新之一。PR 直接把默认调度器里每个 pending request 的 radix tree 遍历次数从 5 次压到 1 次，并且给出 B200 与 H100 上的并发测量。更关键的是，结论不是“prefix-aware scheduling 便宜了一点”，而是“它终于不再需要为了存在而支付额外的结构性开销”。这意味着前缀复用开始从一项可选优化，变成调度器本来就该内建的预算前提。

这条主线随后又被两条修复补齐。**TensorRT-LLM 修正了 KVCacheManagerV2 在 SWA 场景下的容量计算[[5]](https://github.com/NVIDIA/TensorRT-LLM/pull/12968)**，避免 generation 阶段因 reserved blocks 与 applied blocks 关系错乱而在 warmup 里触发非法内存访问；**Nemotron Nano VL 的 chunked prefill API contract 也被补齐[[6]](https://github.com/NVIDIA/TensorRT-LLM/pull/13025)**，让多模态 encoder 返回值终于满足 chunked prefix caching 的约束。把这三条更新连起来看，框架团队已经在承认一个事实：前缀复用不是在 cache 层额外算一遍“可复用多少”就结束了，它会反过来决定调度器怎么估预算、KV 容量怎么预留、模型侧接口必须满足什么 contract。

如果说前两天行业还在补“谁有 cache_salt、谁有 tracing”，那今天更进一步的变化是，前缀复用本身正在进入调度器的核心成本模型。这比单纯提高命中率更重要，因为它决定的是系统能否把复用能力当成默认前提，而不是每多开一个优化就多背一层调度器开销。

## 三、拆分式 serving 开始把假路径、局部拓扑和压测工况都视作正式环境

**SGLang 给 benchmark 工具补上了 `--fake-prefill` 开关[[7]](https://github.com/sgl-project/sglang/pull/22973)**，看起来只是 CLI 便利性改进，实际却很能说明问题。过去要压测 decode-only 的 PD 拆分路径，用户得手工往请求体里注入 `bootstrap_host=2.2.2.2` 这类内部哨兵值；现在项目直接承认“没有真实 prefill 节点的 decode 压测”本身就是常规需求，并把它提升成正式参数。这意味着拆分式 serving 不再只围绕理想部署图去证明自己，而是开始认真面对那些不完整、半仿真、但非常接近真实压测流程的工况。

**同一批更新里，SGLang 还修掉了 pipeline parallelism 下 prefill 请求可能直接挂死的调度器逻辑[[8]](https://github.com/sgl-project/sglang/pull/23006)**。问题本身很朴素：一个早先被反转的条件判断，让 chunked prefill 请求在 `pp_size=2` 时会持续返回 `None`，主循环每轮都不再推进，最后整个 server 卡住。之所以值得写，不是因为修复难度大，而是它说明 PP 和 chunked prefill 的组合现在已经进入“不能挂、挂了就得立刻修”的主路径质量要求，而不是可以容忍边角失效的高级玩法。

**SGLang 进一步把 Ray scheduler actor 绑定到 GPU-local NUMA 节点[[9]](https://github.com/sgl-project/sglang/pull/22989)**，也把“局部拓扑”从附加优化升级成默认前提。这个 PR 指出一个此前很隐蔽的问题：Ray actor 不是通过 `multiprocessing.spawn` 启动，所以原来的 NUMA 绑定路径根本不会生效，默认配置下 scheduler actor 实际一直处于 unbound 状态。修复之后，PR 给出的测试里 Qwen 在 rate 8-16 区间的 mean E2E 改善约 9% 到 16%，TTFT 也有 7% 到 18% 的下降。换句话说，拆分式 serving 现在连“控制面进程应该落在哪个 NUMA 节点”这种过去常被忽略的细节，都已经进入默认性能模型。

## 四、Serve 平台开始把流式、HA 与 tracing 打包成一整套服务面

**Ray 2.55.0 这次 release 最值得写的，不是一条单独功能，而是它把服务平台该有的外层能力几乎成套推出[[10]](https://github.com/ray-project/ray/releases/tag/ray-2.55.0)**。Serve 侧同时加入了端到端 gRPC 双向流、基于 HAProxy 的高吞吐入口、面向 async inference 和 Taskiq 负载的队列式 autoscaling，以及更完整的 tracing 和运维指标；Ray LLM 侧则继续推进 decode-as-orchestrator 的 PD 架构，并把 SGLang engine、WideEP fault tolerance、NIXL 传输修复等能力整合进 release 里。

这类 release 之所以重要，在于它说明 Serve 平台的竞争点已经不再只是“能不能包一层 OpenAI API”。现在更关键的是，平台能不能同时提供稳定入口、高可用代理、流式链路、可观测性和调度/拆分架构，并把它们作为一套可部署的服务面交付出去。Ray 这次版本把这些条目集中打包，也意味着 serving 基座正在进一步远离“推理框架的薄包装”，朝真正的平台层能力移动。

## 五、今天真正值得记住的判断

今天真正值得记住的，不是低比特、前缀复用或拆分式 serving 各自又多了一条 PR，而是它们都在被重写成“默认路径必须成立”的问题。vLLM、llama.cpp 和 Megatron 正在把低比特从模型兼容性推进到 kernel、图执行和 recipe 主干；TensorRT-LLM 则把前缀复用塞进调度器最核心的开销模型；SGLang 与 Ray 则进一步说明，拆分式 serving 已经不能只在理想拓扑里证明自己，还要在假 prefill、NUMA 绑定、流式入口和 HA 代理这些真实环境条件下成立。

如果这个方向继续下去，下一阶段 AI Infra 的差异就不会主要体现在“谁多了一个 headline feature”，而会体现在“谁更早把这些 feature 改造成无需解释、无需绕路、默认就该可靠存在的主路径能力”。

---

## 参考来源

[1] [vLLM 将 Marlin kernel 接入 block-scaled mm 默认选路](https://github.com/vllm-project/vllm/pull/40105)

[2] [llama.cpp 为 Gemma4 增加 NVFP4 tensor 支持](https://github.com/ggml-org/llama.cpp/pull/21971)

[3] [Megatron Core 0.17.0 release](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.17.0)

[4] [TensorRT-LLM 将 prefix reuse 查询收敛为单次 analyzePrefixReuse](https://github.com/NVIDIA/TensorRT-LLM/pull/13139)

[5] [TensorRT-LLM 修复 SWA 场景下 KVCacheManagerV2 的容量问题](https://github.com/NVIDIA/TensorRT-LLM/pull/12968)

[6] [TensorRT-LLM 修复 Nemotron Nano VL 的 chunked prefill API contract](https://github.com/NVIDIA/TensorRT-LLM/pull/13025)

[7] [SGLang 为 decode-only 压测新增 fake-prefill 开关](https://github.com/sgl-project/sglang/pull/22973)

[8] [SGLang 修复 pipeline parallelism 下的 scheduler hang](https://github.com/sgl-project/sglang/pull/23006)

[9] [SGLang 将 Ray scheduler actor 绑定到 GPU-local NUMA 节点](https://github.com/sgl-project/sglang/pull/22989)

[10] [Ray 2.55.0 release](https://github.com/ray-project/ray/releases/tag/ray-2.55.0)
