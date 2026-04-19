---
wechat_published: true
---

# 今日焦点：低比特与前缀复用开始进入默认主路径

**📅 2026-04-18**

> 中文：清晨的数据中心控制室里，GPU 集群、推理网关与调度面板同时亮起，监控大屏展示 prefix reuse、流式请求、NUMA 拓扑和低比特 kernel 选路，工程师在多节点服务平台上观察吞吐与延迟变化，无文字，16:9
>
> English: A modern AI datacenter control room at dawn with GPU clusters, inference gateways, and scheduling dashboards showing prefix reuse, streaming requests, NUMA topology, and low-bit kernel routing, engineers monitoring throughput and latency across a multi-node serving platform, no text, 16:9

---

过去三天，AI Infra 领域有几个变化值得记录。但我更想说的不是这些变化本身，而是它们共同反映的一个规律——一项技术从"能跑"到"成为默认选项"，中间往往隔着好几年的工程打磨。低比特和前缀复用，正在经历这个过程。

## 推理侧：低比特的三路并进

先说推理侧。三个项目各自推进，但放在一条时间线上看，能看出一个共同方向。

vLLM 把 Marlin kernel 接入了 block-scaled mm 默认选路。这意味着 FP8 路径开始从"模型能加载"进入"默认 kernel 怎么选"的层面。框架团队不再把低比特当成兼容性展示，而是开始把它写进真正决定性能和稳定性的执行栈里。本质上，这是工程上的一个拐点——低比特从此不再是"可选插件"，而是默认路径的一部分。

llama.cpp 为 Gemma4 增加了 NVFP4 tensor 支持。Megatron Core 0.17.0 则把 FP4/MXFP8、CUDA graph 与 offload 相关更新一起推进。

我的观察是这样的：这三家的做法各不相同，但指向同一个方向。低比特现在已经不像早期那样只是"能跑就算支持"，而是在逼近默认化、体系化的执行路径。这个过程大概始于两到三年前，今天算是到了一个节点。

## 生产部署侧：前缀复用终于进入调度器成本模型

再来看生产部署侧，这块的进展更值得关注。

TensorRT-LLM 把 prefix reuse 查询压缩成单次 `analyzePrefixReuse`。过去默认调度器里，一个 pending request 可能要重复走多次 radix tree；现在核心信息一次算完再复用，前缀复用终于开始真正进入调度器的成本模型，而不是挂在 cache 命中率面板上的附加优化。

这事说起来简单，但做过调度系统的人会明白这意味着什么——它意味着前缀复用从"锦上添花"变成了"有它没它不一样"。

随后 TensorRT-LLM 又补上两件事：SWA 容量修复和 Nemotron Nano VL 的 chunked prefill contract 修正。前者解决 warmup 阶段 reserved blocks 和 applied blocks 关系错乱的问题，后者让多模态 encoder 真正满足 chunked prefix caching 的接口要求。这组变化加在一起看，前缀复用现在不只是"复用了多少块"，而是在反过来约束容量预算、模型接口和 warmup 安全。这是一个值得关注的变化。

SGLang 这边有三个更新值得关注：为 decode-only 压测提供 `--fake-prefill` 正式开关；修掉了 pipeline parallelism 下的 scheduler hang；将 Ray scheduler actor 绑定到 GPU-local NUMA 节点。这三件事放在一起，说明拆分式 serving 连"假路径怎么测""控制面进程绑在哪"都不再是边角问题，而是主路径质量的一部分。

## 平台侧：Ray 2.55.0 的整包交付

最后看平台侧。

Ray 2.55.0 release 一次性把流式、HA 和 tracing 打成了整套服务面。Serve 同时加入端到端 gRPC 双向流、HAProxy 入口、队列式 autoscaling 和更完整的 tracing；Ray LLM 则继续推进 decode-as-orchestrator 的 PD 架构以及 SGLang engine、WideEP fault tolerance 等能力。

平台竞争点已经不再只是"包一层 API"，而是能否把入口、高可用、观测和拆分架构一起交付。这个趋势大概始于三年前，今天算是比较清晰地显现出来了。

---

**一句话结论：AI Infra 正在从"功能是否存在"转向"这些能力能否默认、稳定、低绕路地成立"。** 之所以这么说，是因为我观察到一个规律：任何技术从出现到真正普及，都要经历这么一个阶段——不再是"有没有"，而是"默认走这条路的代价有多低"。低比特和前缀复用，正在经历这个阶段。

---

## 参考

[1] vLLM 将 Marlin kernel 接入 block-scaled mm 默认选路：https://github.com/vllm-project/vllm/pull/40105

[2] llama.cpp 为 Gemma4 增加 NVFP4 tensor 支持：https://github.com/ggml-org/llama.cpp/pull/21971

[3] Megatron Core 0.17.0 release：https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.17.0

[4] TensorRT-LLM 将 prefix reuse 查询收敛为单次 analyzePrefixReuse：https://github.com/NVIDIA/TensorRT-LLM/pull/13139

[5] TensorRT-LLM 修复 SWA 场景下 KVCacheManagerV2 的容量问题：https://github.com/NVIDIA/TensorRT-LLM/pull/12968

[6] TensorRT-LLM 修复 Nemotron Nano VL 的 chunked prefill API contract：https://github.com/NVIDIA/TensorRT-LLM/pull/13025

[7] SGLang 为 decode-only 压测新增 fake-prefill 开关：https://github.com/sgl-project/sglang/pull/22973

[8] SGLang 修复 pipeline parallelism 下的 scheduler hang：https://github.com/sgl-project/sglang/pull/23006

[9] SGLang 将 Ray scheduler actor 绑定到 GPU-local NUMA 节点：https://github.com/sgl-project/sglang/pull/22989

[10] Ray 2.55.0 release：https://github.com/ray-project/ray/releases/tag/ray-2.55.0