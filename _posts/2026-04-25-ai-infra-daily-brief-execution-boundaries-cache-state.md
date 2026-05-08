---
title: AI Infra 早报｜执行边界与缓存状态开始被写回主路径
date: 2026-04-25 05:30:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: 过去一天，推理框架开始拆掉”整段静态执行”的默认前提，缓存系统也不再只保存 token，而把路由、分层对象与恢复状态一起纳入主链路。AI Infra 的竞争点，正在继续从单点性能转向复杂执行边界下的可成立性。
tags: [Inference, KV Cache, MoE]
---

今天最值得写的，不是谁又把某个 kernel 提快了几个点，而是几个主干项目同时承认了一件更现实的事：真正上线后的系统，很少活在一条干净、连续、永远不被打断的执行路径里。CUDA graph 会遇到 offload，prefix cache 会遇到 MoE 路由，控制面会在 shutdown 中途崩掉，浏览器后端也未必具备完整的 subgroup matrix 能力。

这意味着 AI Infra 正在进入一个更硬的阶段。过去大家常把复杂边界当成“之后再补”的尾项，现在这些边界开始直接进入默认主路径。谁先把这些条件写成稳定行为，谁才更接近真正可用。

## 一、运行时开始拆掉“整段静态执行”的默认前提

**SGLang 引入实验性的 breakable piecewise CUDA graph[[1]](https://github.com/sgl-project/sglang/pull/22218)**。这条更新的重点，不只是又多了一个 graph 选项，而是它把“图必须整段连续捕获”的假设主动松开了。新开关 `--enable-breakable-cuda-graph` 允许 runtime 在更细粒度的位置切开图执行，PR 里的 mGSM8K 数据也说明它不是纯概念特性，在多种 Qwen3 配置上吞吐和显存都已经进入可比较区间。图执行开始从“一整块要么成立要么失效”，变成更接近真实线上环境的局部组织能力。

**TensorRT-LLM 则修掉了 DSA + host KV cache offload + CUDA graph 叠加时的非法地址访问[[2]](https://github.com/NVIDIA/TensorRT-LLM/pull/13124)**。问题根源在于 host offload 发生交换后，C++ 侧的 block ID 和真实 memory pool index 会分离，draft replay 仍把 raw block ID 当偏移去算，最后在 CUDA graph replay 时把地址打到 buffer 之外。这个修复很关键，因为它说明一旦 graph、spec decode 和 host offload 真的叠在一起，很多“平时单独都能跑”的路径会立刻暴露出隐藏的索引契约。

训练侧也在补同类边界。**DeepSpeed 让 dynamic offload 能和 static optimizer offload 共存[[3]](https://github.com/deepspeedai/DeepSpeed/pull/7979)**，它解决的同样是组合条件下的默认假设冲突。单看这条 PR 很小，但它释放出的信号很明确：框架已经不能继续假设用户只会选一种 offload 形态，复杂资源编排本身正在进入默认配置空间。

## 二、缓存系统不再只保存 token，而开始保存更细的中间状态

**Megatron-LM 为 prefix caching 增加 per-block MoE routing storage[[4]](https://github.com/NVIDIA/Megatron-LM/pull/4301)**。这条更新很值得记，因为它把 prefix cache 的语义从“保存一段前缀文本”推进到了“保存生成这段前缀时实际走过的路由状态”。路由索引在 forward 后被转成 CPU numpy，再按 block 散落进 KV cache，等请求完成时再拼回来，连 LRU 命中的 prefix-cached blocks 也能保留原请求的 routing。只要服务对象是 MoE 模型，这类状态就不再是可有可无的附属信息，而是能不能继续复用前缀的必要条件。

**LMCache 在 MP 模式下加入 S3 L2 adapter[[5]](https://github.com/LMCache/LMCache/pull/3064)**，把对象命名、容量统计、DeleteObject 驱逐和 circuit breaker 一起纳入 L2AdapterInterface。它的意义不只是“又多一个存储后端”，而是多进程缓存分层终于开始把远端对象存储当成正式层级来治理。只要 L2 进入 S3，缓存就不再只是机内页表问题，而变成跨进程、跨容量边界、跨失败语义的对象生命周期问题。

**Mooncake 也在 Get 路径里加了 batch route query[[6]](https://github.com/kvcache-ai/Mooncake/pull/1970)**。这看上去像一次 P2P client service 的重构，但真正重要的是读路径终于开始承认“路由查询本身就是成本中心”。当缓存命中判断需要频繁向多处查路由时，批量化查询就不是小优化，而是把读路径重新写回正式调度模型。

## 三、恢复语义和轻量后端都在向“真实可用”推进

**Ray Serve 修复了 controller 在 shutdown 中途崩溃时产生 orphaned actors 的问题[[7]](https://github.com/ray-project/ray/pull/62823)**。旧路径会在 shutdown 一开始就删掉 KV checkpoint，一旦 controller 在 actor teardown 完成前重启，新的 controller 根本不知道还有哪些 deployment 需要清理。现在它先持久化 `SHUTDOWN_IN_PROGRESS_KEY`，再把 checkpoint 删除推迟到最后，控制面终于开始把“恢复到一半的 shutdown”当成真实状态，而不是异常角落。

**LMCache 还给 BlendEngineV2 补上了 per-request root OTel span 和完整的 CB 事件订阅体系[[8]](https://github.com/LMCache/LMCache/pull/3062)**。这条更新之所以重要，不是因为 trace 更多了，而是 cache blending 这类异步路径终于能和主请求生命周期挂到同一棵 span 树上。只要 retrieve、inference、store_final 之间的间隙还会让 span 提前闭合，观测就永远只能看到碎片。

另一条很有代表性的变化来自边缘后端。**llama.cpp 让 WebGPU 在没有 subgroup matrix 的浏览器里也能启用 `FLASH_ATTN_EXT`[[9]](https://github.com/ggml-org/llama.cpp/pull/22199)**，通过补 tile flash attention kernel 和清理 vec path，把浏览器能力门槛又往下压了一截。它当然还没有 subgroup matrix 版本那么快，但方向已经很清楚了：浏览器和轻量设备后端不再只是“能跑就行”，而是开始补真正可用的 attention 快路径。

## 四、今天真正值得记住的判断

今天真正值得记住的，是 AI Infra 正在把“执行边界”当成主战场。图执行要面对 offload，前缀复用要带着路由状态继续成立，控制面要能穿过半途崩溃把 shutdown 收完，浏览器后端也要在残缺硬件能力上继续给出像样的快路径。

接下来真正拉开差距的，未必是谁再写出一个更亮眼的 benchmark，而是谁先把这些复杂条件写进默认主路径，让系统在被打断、被分层、被恢复的时候仍然成立。

---

## 参考来源

[1] [SGLang 引入实验性的 breakable piecewise CUDA graph](https://github.com/sgl-project/sglang/pull/22218)

[2] [TensorRT-LLM 修复 DSA 与 host KV cache offload 在 CUDA graph 下的非法地址访问](https://github.com/NVIDIA/TensorRT-LLM/pull/13124)

[3] [DeepSpeed 让 dynamic offload 兼容 static optimizer offload](https://github.com/deepspeedai/DeepSpeed/pull/7979)

[4] [Megatron-LM 为 prefix caching 增加 per-block MoE routing storage](https://github.com/NVIDIA/Megatron-LM/pull/4301)

[5] [LMCache 为 MP 模式增加 S3 L2 adapter](https://github.com/LMCache/LMCache/pull/3064)

[6] [Mooncake 在 Get 路径实现 batch route query](https://github.com/kvcache-ai/Mooncake/pull/1970)

[7] [Ray Serve 修复 shutdown 中途崩溃导致的 orphaned actors](https://github.com/ray-project/ray/pull/62823)

[8] [LMCache 为 BlendEngineV2 增加 per-request root OTel span 与 SpanRegistry](https://github.com/LMCache/LMCache/pull/3062)

[9] [llama.cpp 让 WebGPU 在无 subgroup matrix 的浏览器中启用 FLASH_ATTN_EXT](https://github.com/ggml-org/llama.cpp/pull/22199)
