---
title: AI Infra 早报｜推理快路径进入生产护栏期
date: 2026-04-22 05:30:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: 过去一天，speculative decoding、PD disaggregation、KV Store 和 Serve 路由的更新都指向同一个方向：推理快路径正在从单纯追求速度，转向补齐正确性、回退、批处理、观测和运行时控制。
tags: [Speculative Decoding, Inference, KV Cache]
---

今天这批更新最有意思的地方，是“快路径”开始变得保守。过去 speculative decoding、PD disaggregation、KV offload 和 replica 直连路由常被当成性能增强项来讲；最近一天，主干项目合进来的变化却更多在补状态初始化、默认模式、unsupported config guard、批处理失败收敛、trace replay 和 endpoint metadata。

这说明推理系统已经走到一个新阶段：只要某条路径准备进入默认生产流量，它就不能只回答“能不能更快”，还要回答“错了怎么办、遇到不支持的组合怎么退、真实负载能不能回放、局部失败会不会拖垮整批请求”。快路径开始长出护栏，这比单点 benchmark 再漂亮一点更值得写。

## 一、speculative decoding 从性能技巧变成需要状态护栏的默认能力

**vLLM 修复了 MRv2 在 PD disaggregation 下因复用 stale `last_sampled_tokens` 和 `draft_tokens` 导致的准确率回退[[1]](https://github.com/vllm-project/vllm/pull/39833)**。这条 PR 的信号很强：decode 侧接收 fully prefilled 请求时，第一步就是 decode，没有经过常规 prefill 的 `postprocess()` 初始化；如果模型 runner 复用旧 request slot，speculative combine kernel 就可能读到上一个请求留下来的 token。PR 给出的验证结果也很直接，GSM8K 分数从修复前的 0.82 回到 0.94，接近 V1 baseline 的 0.93。

这条修复背后，是 speculative decoding 进入拆分式服务之后必须面对的状态语义问题。prefill 和 decode 一旦拆开，系统里就会出现“请求已经算完前缀，但 decode 状态尚未按旧路径初始化”的新入口。快路径要成为默认路径，第一步就是把这些状态边界写清楚。

**vLLM 同时把 Mamba-based 模型在 speculative decoding 下的 cache mode 默认切到 `align`[[2]](https://github.com/vllm-project/vllm/pull/40454)**。项目方承认当前 `all` mode 在 SpecDec 组合下存在不稳定性，因此先选择一个前缀缓存效率可能较低、但行为更一致的默认值。这个取舍很现实：当某个优化组合还没有完全站稳，默认配置应该先保护正确性，再继续追性能上限。

**SGLang 则给 adaptive speculative decoding 增加 unsupported config guard[[3]](https://github.com/sgl-project/sglang/pull/23289)**。当 `--speculative-adaptive` 遇到 DP attention、overlap scheduler / spec v2、多层 EAGLE、two batch overlap、PDMux 这些尚未实现的组合时，它会回退到静态 speculative 参数，避免继续沿着半支持路径跑下去。再加上 **llama.cpp 新增 `--spec-default`，把 speculative decoding 的默认配置显式暴露出来[[4]](https://github.com/ggml-org/llama.cpp/pull/22223)**，可以看到几个项目都在做同一件事：speculative decoding 正在从“高级用户手动拼参数”走向“默认可开启，但必须有清晰边界”。

## 二、速度优化开始围绕真实请求形态做减法

快路径补护栏，并不代表性能优化停下来。更准确地说，优化开始贴近真实请求形态。**vLLM 把 batch-invariant 场景下的 fused RMSNorm 调用收掉，给出约 2.1% 的端到端 latency 改善[[5]](https://github.com/vllm-project/vllm/pull/40413)**。这里的关键不只在 2.1% 本身，还在于它利用了“这个算子在 batch 维度上不变”这类运行时不变量，减少不必要 kernel 调用。主路径越来越成熟之后，性能空间往往来自这种对真实执行语义的重新审视。

**SGLang 在 KDA 路径里融合 gate 与 cumsum，并复用 chunk index[[6]](https://github.com/sgl-project/sglang/pull/23038)**，也属于同一类变化。PR 里的 microbench 显示，fused gate+cumsum 在多个 batch / sequence 配置下有 2 倍以上速度提升；更深一层看，它是在把 linear attention / KDA 这种新 attention 路径里的中间张量往更少 kernel、更少 global memory roundtrip 上压。随着 Kimi、Mamba、linear attention 这类非标准 attention 路径进入服务框架，优化目标已经不再局限于传统 dense attention。

**llama.cpp 的 server speculative checkpointing[[7]](https://github.com/ggml-org/llama.cpp/pull/19493)** 也值得放进这个主题里看。它为 recurrent modules 下的 speculative decoding 引入 checkpoint，在部分 draft 被接受后可以回到 checkpoint 重新执行更短 batch。PR 也坦率写出这条路径并不总是最快，但在重复性很强的 quicksort 这类场景里能带来明显收益。这里体现的是另一种成熟度：项目没有把 speculative decoding 讲成万能加速，而是在承认可接受率、回滚成本、状态保存都会决定它是否真正划算。

## 三、KV Store 和 Serve 路由开始补“失败之后还能看清楚”的能力

另一条主线落在分离式数据面。**Mooncake Store 把 batch put 的 write route 查询聚合成 `BatchGetWriteRoute` RPC[[8]](https://github.com/kvcache-ai/Mooncake/pull/1947)**，同时修正本地 write route 直接被选中、却不一定符合 write config 的问题。**紧接着，Mooncake 又修复了 transfer failure 时提前完成 batch 导致 use-after-free 的风险[[9]](https://github.com/kvcache-ai/Mooncake/pull/1906)**。这两条放在一起看，说明 KV Store 的批处理语义正在被重新写严：批量请求不能只是把多个单请求打包，还要保证路由选择、失败传播和生命周期结束都按整批语义成立。

观测层也在补齐。**Mooncake 为 Client 增加 Prometheus-compatible HTTP metrics、`/metrics/summary` 和 `/health` 端点[[10]](https://github.com/kvcache-ai/Mooncake/pull/1934)**；**LMCache 则新增 `lmcache trace` CLI，可对 storage-level trace 做离线 info、replay 和性能统计[[11]](https://github.com/LMCache/LMCache/pull/3075)**。前者让客户端侧不再只是 Master 的黑盒从属，后者把真实缓存工作负载拿到离线环境里回放，用于 regression hunting、L1/L2 延迟分析和配置调参。这类能力看起来没有 headline 性能数字，但它们决定了系统出问题后能不能被复现。

Serve 路由也在做类似铺垫。**Ray Serve 暴露 backend HTTP endpoint metadata，让控制器和 HAProxy 层可以把选中的 replica_id 解析成具体 host:port[[12]](https://github.com/ray-project/ray/pull/62667)**；随后 **Ray 又把 experimental `ray-haproxy` binary resolution 接进 Serve，并提供环境变量、pip bundle、系统二进制的多级回退[[13]](https://github.com/ray-project/ray/pull/62589)**。这些改动本身还没有改变请求流，但它们把下一步 replica 直连、HAProxy 接入和后端 endpoint 管理需要的底座先写进了控制面。

## 四、今天真正值得记住的判断

今天真正值得记住的，是快路径正在被迫变得可治理。speculative decoding 要有状态初始化和 unsupported config 回退，KDA / RMSNorm 优化要贴合真实执行不变量，KV Store 批处理要能正确收尾，trace 和 metrics 要能把生产负载带回离线分析，Serve 路由也要先补 endpoint metadata。

下一阶段 AI Infra 的差异，很可能不在于谁先把某个优化开关挂出来，而在于谁能把这个开关背后的状态、失败、观测和默认值一起处理好。快路径只有长出这些护栏，才有资格成为默认路径。

---

## 参考来源

[1] [vLLM 修复 MRv2 stale token 导致的 speculative decoding 准确率回退](https://github.com/vllm-project/vllm/pull/39833)

[2] [vLLM 在 Mamba speculative decoding 下默认使用 align cache mode](https://github.com/vllm-project/vllm/pull/40454)

[3] [SGLang 为 adaptive speculative decoding 增加 unsupported config guard](https://github.com/sgl-project/sglang/pull/23289)

[4] [llama.cpp 新增 speculative decoding 默认配置开关](https://github.com/ggml-org/llama.cpp/pull/22223)

[5] [vLLM 利用 batch invariant 优化 fused RMSNorm 调用](https://github.com/vllm-project/vllm/pull/40413)

[6] [SGLang 在 KDA 中融合 gate+cumsum 并复用 chunk index](https://github.com/sgl-project/sglang/pull/23038)

[7] [llama.cpp server 引入 speculative checkpointing](https://github.com/ggml-org/llama.cpp/pull/19493)

[8] [Mooncake Store 使用 BatchGetWriteRoute 优化 batch_put 路由查询](https://github.com/kvcache-ai/Mooncake/pull/1947)

[9] [Mooncake Store 等待整批 transfer task 完成后再结束 batch](https://github.com/kvcache-ai/Mooncake/pull/1906)

[10] [Mooncake Store 为 Client 增加 HTTP metrics endpoint](https://github.com/kvcache-ai/Mooncake/pull/1934)

[11] [LMCache 新增 lmcache trace CLI 支持离线 trace 分析与 replay](https://github.com/LMCache/LMCache/pull/3075)

[12] [Ray Serve 暴露 backend HTTP endpoint metadata](https://github.com/ray-project/ray/pull/62667)

[13] [Ray Serve 增加 experimental ray-haproxy binary resolution](https://github.com/ray-project/ray/pull/62589)
