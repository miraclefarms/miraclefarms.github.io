---
title: LMSYS P2P 权重更新：把 RL Rollout 的停机窗口压到秒级
date: 2026-05-09 12:00:00 +0800
author: Ethan
kind: reading
category: Reading
intro: 本文阅读 LMSYS 的 P2P weight transfer 方案，拆解它如何用 CPU 侧 SGLang replica 和 RDMA 直写，把大规模 RL 的权重更新停机窗口压到秒级。
tags: [Networking, Inference, Training, SGLang]
---

> **版本声明**：本文分析基于 LMSYS 博客《Updating 1T parameters in seconds — P2P weight transfer in Large Scale Distributed RL》（2026-04-29）<a href="https://www.lmsys.org/blog/2026-04-29-p2p-update/">[1]</a>，并参考 Miles 文档、TransferEngine 论文和相关 SGLang PR；开放 PR 状态以 2026-05-09 查询结果为准。

大规模 RL 系统里，训练刚更新完权重，rollout 引擎必须停下来等新权重灌进去，这段等待时间往往比单次计算更刺眼。LMSYS 这篇博客给了一个很硬的数字：在 1T 参数 Kimi-K2 FP8 配置上，传统 NCCL broadcast 的权重更新时间是 53.3 秒，RDMA P2P 路径降到 7.2 秒，约 7.37 倍加速<a href="https://www.lmsys.org/blog/2026-04-29-p2p-update/">[1]</a>。代价也同样明确：每个 training rank 需要额外放一个 CPU 侧 inference engine replica，约 32GB CPU memory。

这篇文章最值得读的地方，是它把“RDMA 比 NCCL 快”这个表层结论压低，转而讨论 RL 训练到 rollout 的权重同步怎样从 collective communication 问题变成数据路径编排问题。NCCL broadcast 适合对称、静态、所有 rank 同步参与的训练通信；rollout weight update 面对的是训练 rank 和推理 rank 的非对称映射、MoE expert shard、动态 rollout engine，以及每次更新时整条 RL pipeline 都在空转的停机窗口。

## 一、NCCL Broadcast 卡住的是数据路径，不只是带宽

传统开源 RL 框架里，Megatron / FSDP 侧会先把 TP、EP 维度的权重 shard all-gather 成 bucket，然后通过 SGLang 的 `update_weights_from_distributed` 之类接口 broadcast 到 rollout engine。这个路径的问题在于，真正参与外发的通常是每个 PP group 的 head rank，其他 training rank 在很长一段时间里只是等待；同一份权重还会被重复发给多个目标 rank。网络带宽没有被整个训练集群共同使用，反而集中压在少数源 rank 上。

LMSYS 在博客里把几条路径放到同一张表里比较：磁盘加载是数分钟级，NCCL broadcast 约 50 秒，Perplexity fabric-lib 的 P2P 约 1.2 秒，本文实现的 RDMA P2P 约 7 秒<a href="https://www.lmsys.org/blog/2026-04-29-p2p-update/">[1]</a>。这里的 7 秒更像一个工程折中点：保留对 Megatron FSDP、SGLang 常规模型加载逻辑、常见开源模型和量化后处理的兼容性，同时把最重的 broadcast 瓶颈拆散。

换成系统视角，NCCL broadcast 的根本限制是 collective 语义。它要求通信组成员、调用顺序、张量形状和同步节奏高度一致。RL rollout 更新恰好越来越不像这种工作负载：训练侧和推理侧 GPU 数可能不同，pipeline parallel 和 expert parallel 的布局可能不同，目标 rollout engine 也可能新增或扩缩容。此时用一个全局同步的集体通信原语去描述“某个训练 rank 给若干推理 rank 写入它们需要的 shard”，表达能力本身就不够顺手。

## 二、P2P 方案的关键，是在源侧造一个 CPU SGLang Replica

LMSYS 的 P2P 设计保留了前半段 all-gather。Megatron 侧仍然按 bucket 收集 TP / EP shard，并转换到 HuggingFace tensor 语义。变化发生在下一步：系统绕开 head rank 对所有 SGLang rank 的完整权重 broadcast，在 training rank 的 CPU 内存里创建一个 SGLang engine replica。这个 replica 按目标 rollout engine 的 parallelism layout 加载权重，把 HF tensor 重新切成 SGLang 目标 rank 正确需要的 shard，然后通过 Mooncake TransferEngine 做 RDMA P2P 写入<a href="https://www.lmsys.org/blog/2026-04-29-p2p-update/">[1]</a>。

![NCCL broadcast 与 P2P source replica 的数据路径对比](/assets/lmsys-p2p-weight-transfer-rl-serving/fig-1-nccl-vs-p2p.png)

*图 1：左侧 NCCL 路径把 gathered tensor broadcast 到多个 SGLang rank，少数源 rank 成为外发瓶颈；右侧 P2P 路径在源侧引入 CPU replica，由多个 training rank 直接发送目标 rank 需要的 shard。来源：LMSYS 博客。*

这个 CPU replica 是方案里最有工程味的部分。它看起来像额外绕了一步，实际是在换取两个能力：第一，训练 GPU 不需要长期放一份 rollout 形态的 replica，避免挤占训练 HBM；第二，SGLang 原有 `load_weight(huggingface_tensor)`、模型 sharding、量化和后处理路径可以继续复用。对于支持模型范围来说，这比手写一套只会处理特定 DTensor layout 的裸 RDMA 写入更实际。

Miles 文档把这个过程拆得更清楚：P2P mode 会建立 transfer plan，查询远端 rollout engine 的 weight memory registration 信息和 parallelism config，在本地 CPU 上构造匹配目标 sharding 的模型 replica，然后每个 source rank 只写目标 rollout rank 需要的 bucketed tensor<a href="https://github.com/radixark/miles/blob/main/docs/en/advanced/p2p-weight-transfer.md">[2]</a>。CPU replica 的价值不在计算，而在复用 SGLang 的权重解释器。它把模型格式、shard 规则、expert 映射这些复杂性挡在 RDMA 传输之前。

## 三、P2P 把网络冗余换成了源侧内存和注册成本

这套方案没有免费午餐。LMSYS 明确写出，P2P 路径用源侧额外的 CPU replica 换取更少的网络传输和更高的源侧带宽利用率。假设每个 inference rank 需要 P 参数，NCCL broadcast 下它可能要接收 `ep * P` 规模的数据；RDMA P2P 目标 rank 只接收自己需要的 P。源侧则从每个 PP group 的少数 head rank 参与，变成 M 个 source rank 都参与发送<a href="https://www.lmsys.org/blog/2026-04-29-p2p-update/">[1]</a>。

博客附录里有一个关键细节：初始设计曾尝试把源侧 replica 放在 GPU 上，但 GPU 显存占用和注册 / 反注册成本很快变成主要问题。现代集群里裸传输本身很快，真正慢的常常是 RDMA memory registration，整份 replica 的注册可以拖到数十秒。最后方案把 replica 移到 CPU，并复用同一块物理内存，按不同目标 engine shard 依次更新和发送<a href="https://www.lmsys.org/blog/2026-04-29-p2p-update/">[1]</a>。

这解释了为什么 7.2 秒不能按单纯网络带宽数字来读。它包含了 bucket 更新、CPU replica load、部分目标上的 GPU 侧 post-process，以及调度多个目标 rank 的等待关系。Miles 文档还补了一句对 Kimi-K2 很重要的注释：Kimi-K2 的 RDMA 时间包含约 884ms rollout engine 侧 `post_load_weights` requantization，因为该模型在 RDMA 后还需要 GPU 侧权重再量化<a href="https://github.com/radixark/miles/blob/main/docs/en/advanced/p2p-weight-transfer.md">[2]</a>。对生产团队来说，这类“传完以后还要做什么”通常比传输 API 本身更容易变成尾延迟来源。

## 四、真正难的是让 SGLang 知道“哪个 Tensor 已经完整可发”

读到实现部分会发现，这项工作很大一块不在 RDMA，而在模型权重映射。P2P update 是 bucketed 的，一个 bucket 里可能只有某个 SGLang tensor 的一部分 shard。比如 HF 侧的 `q_proj`、`k_proj`、`v_proj` 最后可能合到 SGLang 的 `qkv_proj`；某个 MoE expert 的 down projection 还要带着 expert id、本地 expert 数和 shard id。系统必须知道什么时候一个 SGLang tensor 所需的所有 shard 都已经到齐，才能把它写入 replica 并发给目标 rank。

![共享 CPU replica 下的 bucketed transfer flow](/assets/lmsys-p2p-weight-transfer-rl-serving/fig-2-shared-replica-flow.png)

*图 2：同一个 source rank 复用 CPU replica 向多个 SGLang engine rank 发送权重。部分 HF tensor 会先进入 buffer，等所有 shard 到齐后再更新 replica 并触发 RDMA send。来源：LMSYS 博客。*

这就是 SGLang PR #17326 的位置：它试图暴露统一的 parameter mapper，让 `model.load_weight()` 之后能回答一个问题：某个 HF tensor 对应哪个 SGLang parameter、总共有几个 shard、当前是第几个 shard，以及是否涉及 expert 映射<a href="https://github.com/sgl-project/sglang/pull/17326">[6]</a>。PR 描述里列出的目标模型包括 Llama3、Qwen2、Qwen3、Qwen3-MoE、GLM4、GLM4-MoE、DeepseekV2 等，这也解释了为什么博客强调“兼容主流开源模型”。兼容性来自这些模型加载路径被统一映射，RDMA 只负责后面的传输。

另一个依赖是 PR #20907。P2P 方案需要在 SGLang 外部构造一个和目标 engine shard 完全一致的 CPU replica，因此外部进程必须拿到 tp、pp、moe_ep、moe_tp、attn_tp、attn_cp、moe_dp 等 parallelism 信息<a href="https://github.com/sgl-project/sglang/pull/20907">[5]</a>。再加上 PR #15245 暴露 `/post_process_weights`，让 GPU 本地执行 Marlin conversion、量化后处理等逻辑<a href="https://github.com/sgl-project/sglang/pull/15245">[7]</a>，整套链路才接近“传完权重后，rollout engine 真的能继续生成”。

这里要看清边界：这些 PR 在 2026-05-09 查询时仍是 open 状态，博客也说相关接口合在 miles targeted `sglang-miles` branch，尚未进入 SGLang main 的稳定 release API<a href="https://www.lmsys.org/blog/2026-04-29-p2p-update/">[1]</a>。所以这篇博客展示的是一条正在工程化的 production path，通用 SGLang 功能还需要等待主线化。

## 五、性能曲线说明：它主要服务大 MoE 和高 EP 场景

LMSYS 的 profiling 覆盖 9B 到 1T 参数模型，硬件是 H100 8-GPU hosts 和 InfiniBand，计时范围从 engine pause 返回到 `continue_generation` 调用。结果很有层次：GLM-Z1-9B 上 RDMA P2P 是 707.1ms，NCCL 是 694.6ms，基本没有收益；GLM-4.7-9B-Flash 30B(3B) 在低节点、低 EP 配置下反而从 2.51 秒变成 4.23 秒，P2P 更慢<a href="https://www.lmsys.org/blog/2026-04-29-p2p-update/">[1]</a>。

优势从更大 MoE 开始显现。GLM-4.5-Air 106B 从 5.00 秒降到 2.64 秒，Qwen3-235B-A22B 从 10.75 秒降到 3.16 秒，GLM-5 744B 从 58.30 秒降到 8.48 秒，Kimi-K2 1T 从 53.28 秒降到 7.23 秒<a href="https://www.lmsys.org/blog/2026-04-29-p2p-update/">[1]</a>。这个趋势和设计预期一致：expert parallelism 越高，目标 rank 真正需要的 shard 越少；source rank 参与越多，集群总外发带宽越能被用起来。

这也给使用者一个很现实的判断标准。小模型、单节点、低 EP、CPU replica load 成本占比高的场景，P2P 很可能不划算。大 MoE、多节点、高 expert parallelism、rollout engine 数量多、权重更新处于训练 critical path 的场景，它才开始变成主线方案。它更适合那些权重同步已经挡住 RL 训练节奏的集群。

## 六、它和 TransferEngine / R-Fork 指向同一个趋势

LMSYS 这篇博客其实接在两条更长的线索上。第一条是 Tensor R-Fork：SGLang 之前已经把“从另一个正在运行的实例加载权重”做成一种远端权重加载路径，TransferEngine 后端可以把 GPU resident weight 通过 RDMA 暴露给新实例，避免磁盘和 DRAM 反复搬运<a href="https://www.lmsys.org/blog/2025-12-10-rfork/">[4]</a>。第二条是 TransferEngine 论文：它把 ConnectX-7 和 AWS EFA 这类异构 RDMA 硬件抽象成统一 P2P 通信接口，报告在两类硬件上达到 400Gbps 级峰值，并展示了 KV cache transfer、RL weight update、MoE dispatch/combine 三个系统用例<a href="https://arxiv.org/abs/2510.27656">[3]</a>。

这两条线索合起来看，AI 集群正在从“GPU 只负责算”变成“GPU memory 也是分布式数据平面的一部分”。KV cache、模型权重、MoE token routing 都在要求更灵活的 point-to-point 通信。Collective 仍然适合训练里的规则化通信，但推理和 RL 系统越来越多地需要：某个 rank 对某个远端 rank 写一段已经注册好的内存，目标侧按版本号或 immediate counter 确认完成，然后继续服务。

TransferEngine 论文里的 RL weight update 路径报告 1.3 秒级 trillion-parameter update，从 256 training GPUs 到 128 inference GPUs，直接对远端 GPU memory 做 one-sided RDMA Write<a href="https://arxiv.org/abs/2510.27656">[3]</a>。LMSYS 博客里的 7.2 秒更保守，也更接近开源生态落地路径：它要兼容 Megatron bucket、SGLang 模型加载、CPU replica、量化后处理和多模型支持。两个数字放在一起读，反而能看到工程化的代价：纯数据平面可以很快，接入真实框架和模型语义以后，系统还要为通用性买单。

## 七、边界与开放问题

第一，CPU memory 是实打实的代价。博客给出的 32GB per training rank 在 1T 模型规模下可以接受，因为它换回的是几十秒停机窗口；但如果训练集群 rank 数很大，或者同机还有 dataloader、checkpoint、optimizer offload、日志和控制面组件，这部分常驻内存仍然要进入容量规划。

第二，开放 PR 和 `sglang-miles` branch 意味着 API 还在移动。Parameter mapper、parallelism info、post-process endpoint 都是正确方向，但它们一旦进入主线，还要面对更多模型族、更多量化格式、pipeline parallel rollout、GB200、故障恢复和 backward compatibility。尤其是权重映射接口，它会把模型实现细节暴露给外部传输系统，长期维护成本不低。

第三，正确性验证比性能表更难。Miles 文档写到，Kimi-K2 的 `--check-weight-update-equal` 需要先 dequant 到 BF16 再 requant，块量化大小还从 `[128, 128]` 调到 `[64, 64]`，某些 rollout 侧初始化 tensor 需要跳过；这些 hard-coded workaround 不会并入主线<a href="https://github.com/radixark/miles/blob/main/docs/en/advanced/p2p-weight-transfer.md">[2]</a>。这提醒我们，RL weight transfer 的正确性要同时覆盖字节写入、训练权重、推理权重、量化 scale、post-process tensor 和版本号。

## 八、结论

LMSYS 这篇 P2P weight transfer 博客最重要的判断，是把 RL rollout 权重更新从 collective broadcast 里解放出来。训练和推理在大规模 RL 里已经是两个不同形态的系统，权重更新也应该按目标 rank、shard、parallelism layout 和可用网络路径来编排。CPU 侧 SGLang replica 看似笨重，却把 SGLang 的模型加载语义保留下来，让 RDMA P2P 不至于变成只能服务少数模型的专用黑盒。

我的阅读结论很直接：这条路径适合大 MoE、高 EP、多节点、权重更新处于 critical path 的 RL 训练；小模型或低并行场景未必划算。它展示的是一种推理系统基础设施转向：当模型权重和 KV cache 都开始在集群内高速流动，通信层必须从“大家一起做同一件事”扩展到“任意两个端点按语义精确交换状态”。

如果未来 agentic RL、online RLHF、self-play 和高频 rollout 继续扩大，权重更新的停机窗口会越来越像一个一等公民的调度问题。7.2 秒只是这篇博客里的数字，更长期的问题是：训练系统和推理系统之间，究竟需要怎样的数据平面才能让模型在持续更新时仍然保持高利用率。

---

## 参考资料

[1] [Updating 1T parameters in seconds — P2P weight transfer in Large Scale Distributed RL](https://www.lmsys.org/blog/2026-04-29-p2p-update/)

[2] [Miles P2P Weight Transfer documentation](https://github.com/radixark/miles/blob/main/docs/en/advanced/p2p-weight-transfer.md)

[3] [RDMA Point-to-Point Communication for LLM Systems](https://arxiv.org/abs/2510.27656)

[4] [Let Tensors Fly — Accelerating Large Model Weight Loading with R-Fork](https://www.lmsys.org/blog/2025-12-10-rfork/)

[5] [SGLang PR #20907: Expose Model Parallelism Information](https://github.com/sgl-project/sglang/pull/20907)

[6] [SGLang PR #17326: Parameter Mapper to convert hugging face parameter to sglang param and shard info](https://github.com/sgl-project/sglang/pull/17326)

[7] [SGLang PR #15245: add API for selective Int4 weight post-processing](https://github.com/sgl-project/sglang/pull/15245)

[8] [Mooncake Transfer Engine Python API](https://kvcache-ai.github.io/Mooncake/python-api-reference/transfer-engine.html)

### 版本对齐信息

- LMSYS P2P weight transfer 博客：发布于 2026-04-29；本文查询日期为 2026-05-09。
- Miles 文档 `docs/en/advanced/p2p-weight-transfer.md`：参考 `radixark/miles` main 分支文件，最近相关提交 `523552a1e1a3`（2026-04-21，`Finalize all weight transfer P2P examples`）。
- SGLang PR #20907：开放 PR，不视为稳定 release API；head commit `cce8c9702aa2`，查询日期 2026-05-09。
- SGLang PR #17326：开放 PR，不视为稳定 release API；head commit `31373a3297c9`，查询日期 2026-05-09。
- SGLang PR #15245：开放 PR，不视为稳定 release API；head commit `3567e15965f5`，查询日期 2026-05-09。
