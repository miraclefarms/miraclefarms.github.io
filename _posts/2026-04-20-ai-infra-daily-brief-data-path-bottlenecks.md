---
title: AI Infra 早报｜系统级数据路径开始压过单点算子优化
date: 2026-04-20 05:30:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: 这两天最值得记住的不是又多了几个快 kernel，而是越来越多主干项目开始正面处理调度气泡、KV 传输批处理、缓存记账与 checkpoint handoff 的系统瓶颈。
tags: [Inference, KV Cache]
---

这两天翻完几家主干仓库，我最强烈的感受不是“又快了多少”，而是“大家终于开始承认真正卡住生产系统的，往往不是那一个 kernel”。

算子优化当然还在推进，但今天更有分量的变化，已经落在调度器、KV 传输、缓存记账、异步 checkpoint 这些过去常被当成工程细节的位置。换句话说，AI Infra 的性能竞争，正在从单点 kernel 向整条数据路径外溢。

## 一、快路径还在前进，但目标已经变成消掉系统气泡

**Megatron-LM 把 FlashAttention 4 接进 inference 主路径[[1]](https://github.com/NVIDIA/Megatron-LM/pull/4186)**，**TensorRT-LLM 扩展了 FP4 quant + rms_norm kernel 的维度覆盖范围[[2]](https://github.com/NVIDIA/TensorRT-LLM/pull/13033)**，这类更新看上去还是熟悉的“算子继续卷”。但如果只看到这里，反而会错过今天更重要的信号。

更能说明问题的是 **vLLM 开始直接处理多模态调度器本身的 CPU 开销[[3]](https://github.com/vllm-project/vllm/pull/40143)**。这个改动没有换新模型，也不是再塞一个花哨 backend，而是把 `get_num_embeds` 的热路径访问成本收掉，避免调度线程拖慢 GPU 重叠执行。PR 给出的结果很直接：在 Gemma 4 多图场景里，请求吞吐提升约 26.9%，TPOT 下降约 27.3%。这已经不是“某个 kernel 更快”，而是服务栈开始补掉原来被忽视的调度气泡。

同样的味道也出现在 **SGLang 针对 AMD NSA indexer 把多步 kernel 收敛成单步路径[[4]](https://github.com/sgl-project/sglang/pull/22850)**。它不是为了做一篇漂亮 benchmark，而是在真实部署里把多余的 dtype cast、GEMM fallback 和 cache store 分拆收掉。这里的重点不只是省了几个微秒，而是主干项目越来越少接受“功能上能跑，但路径上有一堆小气泡”这种状态。

## 二、KV 数据面开始被当成第一等公民来补课

今天最值得继续盯的，还是分离式 serving 里的 KV 数据面。**Mooncake 把 transfer request 批量化到 `cudaMemcpyBatchAsync`[[5]](https://github.com/kvcache-ai/Mooncake/pull/1890)**，背后的动机非常直白：prefill 和 decode 使用不同 TP 布局时，系统会抛出大量 token-size 的传输请求，运行时开销本身就能把收益吃掉。现在它直接利用 CUDA 12.8 的批量异步拷贝把这些请求合并，并用 event 查询单批状态，说明 KV 传输已经不再是“搬运工作”，而是必须单独优化的数据路径。

更现实的是，**Mooncake 紧接着又补了一组 production 暴露出来的 transfer/store 故障[[6]](https://github.com/kvcache-ai/Mooncake/pull/1895)**：超时判断错误会把单批超时拉长到不可接受的量级，peer 挂住时可能导致读取悬挂，IB 异步事件还会带来死锁风险。这类修复看起来不性感，但它们恰好说明分离式 KV 的问题已经从“有没有性能”进到了“能不能稳定跑在生产里”。

LMCache 的动作也很典型。**它给 storage manager 加了二进制 trace 录制能力[[7]](https://github.com/LMCache/LMCache/pull/3063)**，支持离线检查和回放缓存工作负载；另一边，**它还修掉了 MP connector 在 store 记账上的两个错误，避免 KV block 在混合命中场景里被静默丢弃[[8]](https://github.com/LMCache/LMCache/pull/3012)**。前者是在补可观测，后者是在补正确性。放在一起看，意思很明确：缓存系统已经不是“命中率高就行”，而是必须同时回答“到底发生了什么”和“有没有悄悄写错”。

## 三、训练侧也开始绕开系统资源上限

这种变化不只发生在推理。**Megatron-LM 新增 `--async-ckpt-use-cpu-shm` 参数，把异步 checkpoint 里的 GPU tensor 先复制到 CPU shared memory 再交给后台 worker[[9]](https://github.com/NVIDIA/Megatron-LM/pull/4355)**。它要解决的不是算法问题，而是 MNNVL 系统里 CUDA IPC 和 NVLink fabric handle 被耗尽之后，异步保存流程会被系统资源上限反咬一口。

这个细节很重要。过去大家谈训练基础设施，注意力常常只落在吞吐、并行策略和 optimizer 状态怎么切；但现在连 checkpoint handoff 这种边缘环节，也开始被明确视作容量与稳定性的瓶颈。基础设施走到这个阶段，说明“模型跑得快”已经不够，真正决定上限的是整条训练/推理数据路径能不能稳定穿过去。

## 四、今天真正值得记住的判断

今天这波更新真正说明的，不是 AI Infra 暂时放弃了算子优化，而是行业正在承认另一件更麻烦的事实：单点 kernel 再快，也救不了一条充满调度气泡、传输碎片、记账错误和系统句柄瓶颈的主路径。

接下来一段时间，我会更关注那些看起来“没那么炫”的提交。因为真正开始决定生产胜负的，可能恰恰不是又多快了 3%，而是谁先把整条数据路径里的隐性摩擦系数降下来。

---

## 参考来源

[1] [Megatron-LM 为 inference 接入 FlashAttention 4](https://github.com/NVIDIA/Megatron-LM/pull/4186)

[2] [TensorRT-LLM 扩展 FP4 quant 与 rms_norm kernel 的维度支持](https://github.com/NVIDIA/TensorRT-LLM/pull/13033)

[3] [vLLM 降低多模态调度器与 get_num_embeds 的 CPU 开销](https://github.com/vllm-project/vllm/pull/40143)

[4] [SGLang 在 AMD NSA indexer 上减少冗余 kernel](https://github.com/sgl-project/sglang/pull/22850)

[5] [Mooncake 使用 cudaMemcpyBatchAsync 批量处理 transfer request](https://github.com/kvcache-ai/Mooncake/pull/1890)

[6] [Mooncake 修复生产部署中暴露的 transfer 与 store 稳定性问题](https://github.com/kvcache-ai/Mooncake/pull/1895)

[7] [LMCache 为 storage manager 增加 trace 录制能力](https://github.com/LMCache/LMCache/pull/3063)

[8] [LMCache 修复 MP connector 导致 KV block 静默丢失的记账错误](https://github.com/LMCache/LMCache/pull/3012)

[9] [Megatron-LM 为异步 checkpoint 增加 CPU shared memory handoff 选项](https://github.com/NVIDIA/Megatron-LM/pull/4355)
