---
title: AI Infra 早报｜MoE 与 KV 数据面开始为真实生产负载补课
date: 2026-04-15 05:30:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: 过去三天，vLLM 与 Megatron-LM 正在拆掉 MoE 路径对理想 hidden dim、理想路由和理想 overlap 条件的依赖；与此同时，vLLM、Mooncake 与 LMCache 把 offload、恢复与生命周期观测补进 KV 数据面，推理基础设施开始为真实生产负载补工程课。
tags: [MoE, KV Cache]
---

过去三天，AI Infra 更值得写的变化，不是又多了一个 headline feature，而是几个核心项目开始系统拆掉“默认假设”。MoE 内核不再只服务于对齐得刚刚好的 hidden dim，router 不再满足于用旧精度路径勉强工作，共享专家 overlap 也不再停留在理想 dispatcher 条件下。另一边，KV cache 相关项目同步在补另一类短板：offload 布局、主从恢复、生命周期观测和 eviction 语义，正在从“能跑通”走向“在真实生产场景下也能接住问题”。

这也是今天最值得记住的判断：**推理框架和 KV 基础设施正在一起为“真实工作负载”补课。** 前一阶段大家更关心新 kernel、新并行策略和新缓存层能不能跑出 benchmark；现在更关键的问题变成了，当模型形状不完美、路由精度更敏感、节点会重启、缓存 key 会被锁住时，系统还能不能继续稳定工作。谁先把这些边角条件吃进主路径，谁就更接近真正的生产默认方案。

## 一、MoE 路径开始摆脱“理想形状”前提

**vLLM 为 TRTLLM GEN NVFP4 MoE 内核补上了非 512 对齐 hidden dim 支持[[1]](https://github.com/vllm-project/vllm/pull/39510)**，做法不是放弃现有高性能内核，而是在加载时对权重补零，把原本受限于严格对齐条件的路径扩展到更多真实模型形状。这个改动的价值不只是“支持更多 case”，而是把 benchmark 友好的内核能力往真实模型分布上推了一步。官方在 Nemotron-3-Nano-30B-A3B-NVFP4 上给出的结果也说明，这种补齐不是纯兼容性修补：请求吞吐提升约 22%，输出 token 吞吐提升约 22%，P99 TTFT 下降超过 33%。

**Megatron-LM 则把 shared expert overlap 正式扩到 FlexDispatcher[[2]](https://github.com/NVIDIA/Megatron-LM/pull/2207)**。这背后针对的是 MoE 推理里一个更难察觉的问题：即便你已经有 shared experts，如果 overlap 只能在特定 dispatcher 或特定连接数假设下工作，它就很难成为可复用的系统能力。这个 PR 一边补上 FlexDispatcher 支持，一边为 `CUDA_DEVICE_MAX_CONNECTIONS > 1` 的情形增加 stream wait，避免 shared expert GEMM 过早启动。换言之，Megatron 开始把 overlap 从“论文里成立的优化”改造成“不同执行条件下都能安全启用的路径”。

同一主线还体现在路由器本身。**Megatron-LM 为 MoE router 增加了新的 `sqrtsoftplus` score function，并把 routing 中间计算统一收回 FP32[[3]](https://github.com/NVIDIA/Megatron-LM/pull/3673)**，只在返回时再转回原 dtype。这个改动表面上像一次 score function 扩展，实质上是在承认 router 精度本身已经成为产线问题。MoE 进入大规模部署后，route score 的数值稳定性、aux loss 计算和 top-k 选择都不再只是研究细节，而是决定吞吐与质量能否同时守住的底层约束。

## 二、KV 数据面正在从“搬运数据”转向“管理生命周期”

**vLLM 为 offloading workers 引入统一的 mmap 共享内存布局[[4]](https://github.com/vllm-project/vllm/pull/37206)**，把过去每个 TP worker 各自持有一块 pinned CPU buffer 的方式，改成所有 worker 共享同一份物理内存页，并按 interleaved block 布局组织。这个变化的意义在于，offloaded KV block 不再强依赖当时的 TP 配置才能被理解和重建，跨实例、跨并行度迁移开始具备了真实基础。KV offload 到这一步，讨论的已经不是“CPU 能不能接住显存溢出的块”，而是这些块能否成为可迁移、可共享的数据资产。

**Mooncake 则把主节点崩溃后的恢复逻辑正式做成 client-based HA recovery[[5]](https://github.com/kvcache-ai/Mooncake/pull/1876)**。它没有停留在“Master 重启后让客户端手工重报 metadata”的粗糙策略，而是补了一个明确的状态机和三阶段恢复流水线，先回放 hot keys，再补 DRAM entries，最后同步 storage tier，并在恢复期间用双优先级队列区分正常流量与 recovery 流量。这是一个很典型的信号：KV 系统已经进入必须面对故障恢复排序、锁竞争和回放优先级的阶段，而不是只展示 steady state benchmark。

**LMCache 最近三天的两条更新则说明，多级缓存系统开始认真补“观测”和“回收”这两块基础语义。** 一方面，它新增了 L0 subscriber，用于跟踪 GPU KV cache block 的生命周期、空闲时间和复用间隔[[6]](https://github.com/LMCache/LMCache/pull/2974)；另一方面，它修掉了 LRU eviction 里“命中锁住 key 仍硬删”的低效路径，开始跳过 read/write-locked keys 再选择驱逐对象[[7]](https://github.com/LMCache/LMCache/pull/2978)。这两条改动合在一起很有代表性：缓存系统的成熟度，不只体现在命中率和带宽上，也体现在你是否知道 block 为什么留下、为什么被逐出，以及逐出时会不会因为锁语义失效而白白浪费一次 eviction 机会。

## 三、今天真正值得记住的判断

如果把这几条更新放在一起看，会发现 AI Infra 正在进入一个更不讨巧、但也更关键的阶段。系统团队不再只追求“再快一点”的 headline，而是开始清理那些只在理想条件下才成立的默认假设：hidden dim 必须整齐，dispatcher 行为必须单一，Master 不会掉线，缓存 key 大多可直接回收，offload block 只会在当前 TP 配置里被读写。

这类补课往往不如新 feature 显眼，却更决定一个项目能否成为长期基础设施。下一轮竞争未必是谁先发明新算法，而更可能是谁更早把这些真实生产约束吃进主路径，并把它们做成默认可用、默认可观测、默认可恢复的能力。

---

## 参考来源

[1] [vLLM 为 TRTLLM GEN NVFP4 MoE 内核补上非 512 对齐 hidden dim 支持](https://github.com/vllm-project/vllm/pull/39510)

[2] [Megatron-LM 为 FlexDispatcher 支持 shared expert overlap](https://github.com/NVIDIA/Megatron-LM/pull/2207)

[3] [Megatron-LM 为 MoE router 增加 sqrtsoftplus 并统一 FP32 中间计算](https://github.com/NVIDIA/Megatron-LM/pull/3673)

[4] [vLLM 为 offloading workers 引入统一 mmap 共享内存布局](https://github.com/vllm-project/vllm/pull/37206)

[5] [Mooncake 实现基于客户端的 Master HA recovery](https://github.com/kvcache-ai/Mooncake/pull/1876)

[6] [LMCache 新增 L0 subscriber 追踪 GPU KV cache 生命周期](https://github.com/LMCache/LMCache/pull/2974)

[7] [LMCache 在 LRU eviction 中跳过 locked keys 以提升驱逐效率](https://github.com/LMCache/LMCache/pull/2978)
