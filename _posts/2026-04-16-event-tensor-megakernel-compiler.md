---
title: Event Tensor：为动态 Megakernel 设计一套编译器抽象
date: 2026-04-16 12:00:00 +0800
author: Ethan
kind: reading
category: Reading
intro: MLSys 2026 论文 Event Tensor 提出了一种统一的编译器抽象，让 megakernel 第一次能够处理动态形状和 MoE 数据依赖，在保住全部融合收益的同时将 vLLM warmup 时间从 123 秒降到 35 秒。
tags: [Inference, MoE]
---

LLM 推理系统在 GPU 上的执行效率，有一个长期被低估的成本中心：kernel launch overhead。每次调用 NCCL AllReduce 或 GEMM，CPU 都要向驱动发起一次 kernel 提交，GPU 等待这条命令的间隙就是白白烧掉的时间。CUDA Graph 把这些提交批量预录，消除了 CPU-GPU 往返延迟，但代价是计算图必须在录制时形状固定——一旦遇到 MoE 的 token routing，或者不同请求的动态序列长度，图就得重新 capture，warmup 时间以百秒计。

Megakernel 是另一条路：把多个算子融合进一个持久 kernel，彻底跳过 kernel launch 的边界，让通信和计算在同一个 SM 时间线上重叠。这个思路已经被几个系统证明有效，但受限于一个根本问题——现有的 megakernel 方案几乎都依赖静态形状假设，无法在编译期表达"MoE gate 决定哪些 expert 被激活"这类运行时才确定的数据依赖。MLSys 2026 的论文 Event Tensor<a href="https://arxiv.org/abs/2604.13327">[1]</a> 正是为了解开这个死结而提出。

## 一、从 kernel-by-kernel 到 megakernel：性能收益从哪里来

理解 Event Tensor 的价值，要先搞清楚三种 GPU 调度模式的区别。

第一种是 kernel-by-kernel：每个算子独立提交，CPU 在每个 kernel 之间同步一次。这是最朴素的方式，GPU utilization 差，因为 GPU 要反复等待 CPU 告知下一步。第二种是 CUDA Graph：把一组 kernel 预录为图，提交时一次性触发，省掉了 CPU 往返。但整张图在 capture 时就固定了依赖关系和张量形状，动态形状意味着多版本 capture 和百秒级的 warmup。

![三种 GPU 调度模式对比](/assets/event-tensor-megakernel-compiler/fig-1-gpu-scheduling-models.png)

*图 1：kernel-by-kernel 和 CUDA Graph 都是粗粒度顺序执行，megakernel 把算子拆成更小的任务块，使通信和计算能在 SM 粒度上并行重叠。来源：论文 Figure 1。*

第三种就是 megakernel：所有算子共存于同一个 persistent kernel 中，算子之间的同步降低到 SM 粒度的信号传递。一旦某个 tile 的 GEMM 计算完成，紧接着 Reduce-Scatter 需要的数据就已就位，AllReduce 的等待时间消失在 GEMM 执行的阴影下。论文测量了 GEMM+Reduce-Scatter 这一典型模式，在 8 张 B200 上相比 cuBLAS+NCCL 基线可以达到 1.40x 加速——这几乎全部来自消除了 kernel launch 边界后重叠暴露出来的计算并行。

但要让 megakernel 真正工作，需要一种方式把"哪些任务之间有依赖"精确地编码进内核。对于静态形状，这件事可以在编译期完全确定；而对于动态输入，这件事要求编译器能够推迟部分依赖解析到运行时，同时不付出 CUDA Graph 式 recapture 的代价。

## 二、Event Tensor 抽象：用张量表达任务依赖

Event Tensor 的核心思想出人意料地简洁：把"依赖同步信号"本身表示为一个多维数组，数组的每个元素对应某个 SM 上某组任务的完成事件。算子之间的依赖，就是一个 Event Tensor 的生产者和消费者关系。

![Event Tensor 抽象总览](/assets/event-tensor-megakernel-compiler/fig-2-event-tensor-overview.png)

*图 2：计算图被拆分为 tiled operator，Event Tensor 捕获这些 tile 之间的细粒度依赖，并统一处理形状动态性和数据依赖动态性。来源：论文 Figure 2。*

这个抽象的关键在于它对两类动态性的处理方式。

**形状动态性（shape dynamism）**：batch size、序列长度等运行时才确定的维度，用符号形状张量（symbolic-shape tensor）表达。Event Tensor 的依赖图在编译时只是一个模板，形状参数在 kernel launch 时注入，不需要重新编译或重新 capture。这和 CUDA Graph 的根本区别在于：ETC 生成的是可以接受符号参数的持久 kernel 代码，而不是一份固化快照。

**数据依赖动态性（data-dependent dynamism）**：MoE 的 token routing 是最典型的例子——gate 网络的输出决定每个 token 被哪个 expert 处理，这个决定在运行时才有结果，因此哪些 GEMM tile 需要被执行、它们之间的依赖关系，都是运行时才确定的稀疏图。

![数据依赖动态性：MoE 的 token routing 问题](/assets/event-tensor-megakernel-compiler/fig-5-data-dependent-dynamism.png)

*图 5：左侧是依赖关系静态可知的常规 workload，右侧是 MoE 的 token routing 场景——哪些 expert tile 被激活由运行时数据决定，形成不规则的任务图。来源：论文 Figure 5。*

Event Tensor 通过引入"动态 Event Tensor"解决这类问题：gate 的路由结果被写入一个特殊的事件张量，下游 expert 的 kernel tile 在读到自己对应位置的事件信号后才开始执行。整个流程不需要 CPU 介入，也不需要提前知道路由结果的分布。

## 三、编译器设计：静态与动态调度变换

有了 Event Tensor 这层抽象，ETC（Event Tensor Compiler）就有了足够的信息来做调度决策。编译器根据依赖图的特征，选择两种调度路径之一。

**静态调度**：如果依赖图在编译期完全确定，ETC 会在编译时把每个任务 tile 预先分配到特定 SM 的队列里，生成 notify-and-wait 风格的同步代码。SM 之间的协调通过共享内存里的信号位完成，不需要运行时调度开销。这是最低延迟的路径，适合 All-Gather + GEMM 这类通信-计算重叠模式。

**动态调度**：对于 MoE 这类数据依赖场景，编译时无法确定任务分配，ETC 生成的 kernel 内部包含一个 push-and-pop 调度器。GPU 上的 warp 在运行时从共享任务队列里取任务，路由结果写入队列触发下游 expert 的 GEMM。这个调度器完全运行在 GPU 上，没有 CPU 往返，且因为是编译进 kernel 的静态代码而不是通用运行时，overhead 极小。

两条路径都服务于同一个目标：消除 CPU-side runtime 的 task graph materialization。传统系统在每次 forward pass 时都要在 CPU 上实例化一张任务图，再把任务一条条提交给 GPU；ETC 把这个逻辑在编译时烧进 kernel，运行时只剩下一次 launch。

## 四、实验：延迟、warmup 与调度权衡

性能数据来自 NVIDIA B200 平台，覆盖微基准和端到端 serving 两个层面。

在通信-计算融合这个最干净的测试场景，GEMM+Reduce-Scatter 在 8 张 B200 上比 cuBLAS+NCCL 快 1.40x，All-Gather+GEMM 达到同等收益。MoE 层在单张 B200 上比 Triton 和 FlashInfer 基线快 1.23x（1024 token 规模）。

端到端服务延迟才是最有说服力的数字。

![端到端 serving 延迟对比](/assets/event-tensor-megakernel-compiler/fig-14-e2e-serving-perf.png)

*图 14：Qwen3-30B-A3B（MoE 模型）和 Qwen3-32B（dense 模型）的端到端 serving 延迟，ETC 在低 batch 场景下全面领先 vLLM 和 SGLang。来源：论文 Figure 14。*

Qwen3-30B-A3B（MoE 架构）在 batch size 1 时，ETC 比 vLLM 快 1.48x、比 SGLang 快 1.20x。Dense 模型 Qwen3-32B 提升相对温和，最高约 1.15x。这个差距直接反映了 MoE 场景下动态调度避免 CUDA Graph recapture 的价值——dense 模型用 CUDA Graph 本来就跑得不错，MoE 才是 CUDA Graph 的软肋。

warmup 开销的数据更能说明问题：vLLM 需要 123 秒完成 67 次 graph capture，SGLang 需要 583 秒完成 51 次；ETC 的 warmup 是 35 秒，且不做任何 JIT graph capture。3.5x 的 warmup 压缩，对于需要频繁冷启动的部署场景（autoscaling、spot 实例）意义相当直接。

## 五、结论

Event Tensor 解决的不只是性能问题，而是一个编译器设计问题：如何把"运行时才确定的依赖关系"纳入静态编译框架，同时不付出 JIT 编译或 graph recapture 的运行时代价。这个问题在 LLM 推理里之所以格外突出，是因为 MoE 架构让数据依赖动态性从边缘情况变成了主路径。

从这个视角看，Event Tensor 的核心贡献是一套表达力：用张量化的事件结构把"哪些 tile 之间有依赖、依赖何时确定"统一表达，让编译器可以在编译期把调度逻辑生成进 kernel，而不是推给运行时。这让 megakernel 第一次具备了处理真实 LLM workload 的必要条件。

开放问题仍然存在。ETC 目前测试在 B200 平台，能否在 H100/H200 或其他加速器上保持同等性能还需要验证；更复杂的多专家并行（EP）场景、跨节点的通信融合，也都还没有在论文里完整覆盖。但作为一篇 MLSys 2026 的工作，它提供了一个清晰的思路：megakernel 的动态化问题是编译器问题，而不是运行时问题。

---

## 参考资料

[1] Event Tensor: A Unified Abstraction for Compiling Dynamic Megakernel. https://arxiv.org/abs/2604.13327
