---
wechat_published: true
---
# 低比特与前缀复用：当基础设施从"能用"走向"默认"

**📅 2026-04-18**

> 中文：清晨的数据中心控制室里，GPU 集群、推理网关与调度面板同时亮起，监控大屏展示 prefix reuse、流式请求、NUMA 拓扑和低比特 kernel 选路，工程师在多节点服务平台上观察吞吐与延迟变化，无文字，16:9
>
> English: A modern AI datacenter control room at dawn with GPU clusters, inference gateways, and scheduling dashboards showing prefix reuse, streaming requests, NUMA topology, and low-bit kernel routing, engineers monitoring throughput and latency across a multi-node serving platform, no text, 16:9

过去三天，AI Infra 领域有一组变化值得注意。不是哪个框架又加了什么新功能——这类消息几乎每周都有——而是低比特计算、前缀复用、拆分式 serving 这三件事，几乎同时开始被当作默认路径来要求。

这个信号比单个功能点重要得多。

---

推理框架里，低比特选路正在从"兼容层"升到"执行栈"。

vLLM 把 Marlin kernel 接进了 block-scaled mm 的默认选路 [1]。这件事的意义不在于 Marlin 本身——Marlin 去年就有了——而在于框架团队不再把 FP8 当成一种"加载后告诉你能跑"的展示，而是把它写进了真正决定性能和稳定性的 kernel 调度逻辑。之所以要这么干，原因很简单：当半数以上的新模型出厂就是 FP8 量化格式时，"低比特路径"就不再是可选的旁路，而是主路。

同一周，llama.cpp 为 Gemma4 加入了 NVFP4 tensor 支持 [2]，Megatron Core 0.17.0 把 FP4/MXFP8、CUDA graph 和 offload 相关更新一起推进 [3]。前者把 NVFP4 接进了现有的计算图体系，后者则把低比特、图执行和 recipe 主干一并收敛。如果回忆一下 2018 年前后混合精度训练走过的路——从"能跑 FP16 就不错了"到"FP16 是默认、FP32 是回退"——低比特推理现在正处于类似的拐点。

不是"能跑就算支持"。是"不默认就落后"。

---

生产部署这一侧，前缀复用开始进入调度器的成本模型，而不再是挂在缓存命中率面板上的附加指标。

TensorRT-LLM 把 prefix reuse 的查询压缩成了单次 `analyzePrefixReuse` [4]。过去默认调度器里，一个 pending request 可能要在 radix tree 上反复走好几遍。现在核心信息一次算完再复用。我的观察是，这种改动看起来小，其实标志着一个转变：前缀复用不再只是"复用了多少块"的统计问题，而是开始影响调度器本身的决策路径。

TensorRT-LLM 紧接着又修了两件事——SWA 容量修复 [5] 和 Nemotron Nano VL 的 chunked prefill contract 修正 [6]。前者解决 warmup 阶段 reserved blocks 和 applied blocks 的关系错乱，后者让多模态 encoder 真正满足 chunked prefix caching 的接口要求。放在一起看，前缀复用正在反过来约束容量预算、模型接口和 warmup 安全。这不再是优化，是约束条件。

SGLang 这边也有类似趋势。decode-only 压测的 `--fake-prefill` 开关从内部哨兵值变成了正式参数 [7]。pipeline parallelism 下的 scheduler hang 被修掉 [8]，Ray scheduler actor 绑定到了 GPU-local NUMA 节点 [9]。"假路径怎么测""控制面进程绑在哪"——这些过去是边角问题，现在是主路径质量的一部分。

话说回来，也许我判断得太乐观了。前缀复用真正进入默认路径，前提是大部分请求确实共享前缀。如果 workload 特征不匹配，这些优化的价值会大打折扣。但目前主流的对话式推理场景——系统提示词长、用户提示词短——恰好满足这个条件，所以大体上方向没错。

---

平台层面，Ray 2.55.0 一次性把流式、高可用和观测打成了整套服务面 [10]。Serve 加入了端到端 gRPC 双向流、HAProxy 入口、队列式 autoscaling 和更完整的 tracing。Ray LLM 继续推进 decode-as-orchestrator 的 PD 架构以及 SGLang engine、WideEP fault tolerance 等能力。

平台竞争的本质已经变了。不是"包一层 API 就能交付"，而是能不能把入口、高可用、观测和拆分架构一起兜住。

---

2018 年混合精度训练用了大约两年走完从"实验性支持"到"默认开启"的路。低比特推理和前缀复用现在走到哪一步了？我判断大致在 2018 年混合精度训练的中期——不再是实验，但还没到人人默认。快则一年，慢则两年，这些能力会从"框架支持"变成"框架默认"。

到那个时候，谁没跟上，谁就是 2018 年还在坚持 FP32 训练的那个。

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