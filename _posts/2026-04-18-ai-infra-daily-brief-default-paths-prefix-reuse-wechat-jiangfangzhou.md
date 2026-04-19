---
title: AI Infra 早报｜低比特与前缀复用开始进入默认主路径
date: 2026-04-18 08:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: 低比特、前缀复用和拆分式 serving 正在从"功能展示"转向"默认主路径"。三个变化信号，零废话。
---

过去三天我一直在想一件事：AI Infra 这行最怕的不是方向错，而是方向对了但没人当回事——就是那种"技术上早就有，但没人往默认路径里写"的尴尬。

现在看，这次好像不太一样。

## 推理侧

vLLM 把 Marlin kernel 接进了 block-scaled mm 默认选路。FP8 这条路，算是从"模型能加载"正式进到了"默认 kernel 怎么选"这个层面。我记得早期低比特支持刚出来的时候，大家都当它是个彩蛋——能用，但默认不打开。现在不一样了，开始往执行栈里写。

llama.cpp 那边给 Gemma4 加了 NVFP4 tensor 支持。Megatron Core 0.17.0 也把 FP4/MXFP8、CUDA graph 和 offload 一起推进去了。

说实话，低比特早期给我的感觉就是"能跑就算支持"，现在再看——它正在往默认化、体系化的路上走。这话说出来我也不知道会不会被打脸。

## 生产部署侧

TensorRT-LLM 把 prefix reuse 查询压缩成单次 `analyzePrefixReuse`。之前 pending request 可能要反复走 radix tree，现在核心信息一次算完再复用。

我承认这个变化一开始我没太当回事——前缀复用嘛，不就是 cache 命中率的事。但仔细看，它现在是在反过来约束容量预算、模型接口和 warmup 安全了。这就不是单纯的 cache 优化。

SGLang 那边也有意思：给 decode-only 压测加了 `--fake-prefill` 正式开关。以前要手工塞内部哨兵值才能测假 prefill，现在直接提升成正式参数。同时修掉了 pipeline parallelism 下的 scheduler hang，把 Ray scheduler actor 绑到了 GPU-local NUMA 节点。

这说明什么？拆分式 serving 连"假路径怎么测""控制面进程绑在哪"都不再是边角问题，而是主路径质量的一部分。嗯。

## 平台侧

Ray 2.55.0 release 把流式、HA 和 tracing 打成了整套服务面。Serve 加了端到端 gRPC 双向流、HAProxy 入口、队列式 autoscaling 和更完整的 tracing。Ray LLM 继续推进 decode-as-orchestrator 的 PD 架构以及 SGLang engine、WideEP fault tolerance。

平台竞争点已经不再只是"包一层 API"。

够不够交付一套完整的，说实话我现在也判断不了。先记下来吧。

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