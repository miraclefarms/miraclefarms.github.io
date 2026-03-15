---
title: vLLM：从 PagedAttention 到推理基础设施平台
date: 2026-03-14 12:40:00 -0400
author: Ethan
kind: essay
category: Essay
intro: 基于官方博客、论文与源码，系统分析 vLLM 的整体架构、核心组件设计实现细节，以及从一项 KV Cache 分页技术走向推理基础设施平台的演进路径。
---

> **版本声明**：本文分析基于 vLLM 仓库（2026-03 前后）；除非特别说明，文中关于 PagedAttention、Block Manager、Scheduler、Prefix Cache 的实现描述，均以官方公开的代码和文档为依据。

这两年，vLLM 在 AI Infra 领域几乎是"推理框架"的代名词。每次谈到大模型推理性能，人们总会先问："vLLM 跑得怎么样？"但如果只把 vLLM 理解成一种"让推理更快"的技术优化，实际上会低估它的系统性。它的起点并不是"如何让 Transformer 跑得更快"，而是试图回答一个更根本的问题：**当大模型推理进入在线 serving 阶段时，KV Cache 到底应该怎么管理？**

也正因为这个起点不同，vLLM 后续的演进并没有停留在单点算子优化上，而是逐步构建起一套包含 PagedAttention、Block Manager、Scheduler、Prefix Cache、Offline/Online 部署模式的完整推理运行时。本文想回答的核心问题是：**vLLM 是如何出现的，它如何从一项 KV Cache 分页技术一步步走向更完整的推理基础设施平台，以及为什么它能在社区里持续保持领先地位。**

---

## 一、vLLM 整体架构

### 1.1 架构全景图

在深入各个组件之前，有必要先建立对 vLLM 整体架构的系统认知。从架构师的视角看，vLLM 可以划分为五个层次，各层之间职责清晰，形成了一个有序的抽象层级体系。

最顶层是 **API Layer**，负责接收外部请求并转换为内部 Engine 可处理的格式，这一层支持 OpenAI-compatible API、gRPC 以及 Python Engine API 等多种接入方式。往下是 **Scheduler Layer**，这是整个系统的决策中枢，负责决定哪些请求可以执行、如何分配计算资源，具体包括请求队列管理、抢占机制、优先级调度和 Block 级别的资源分配。再往下是 **KV Cache Management Layer**，这是 vLLM 的核心创新层，与 Scheduler 和 Execution 平级运作，专门负责 KV Cache 的分配、复用和释放，而不是仅仅作为 Model Layer 的附属。**Execution Layer** 负责实际执行模型推理，包括 Worker、Model Runner、Attention Kernel 和 CUDA Graph 优化等底层能力。最底层是 **Model Layer**，承载模型本身的前向传播，支持 Transformer、MoE、MLA 等多种模型架构以及量化和投机解码等技术。

这种分层架构的一个关键洞察在于：KV Cache Management 层的地位被提升到了与调度和执行同等重要的位置，这是 vLLM 区别于传统推理框架的核心特征。

![vLLM Engine Architecture](/assets/vllm-origin-evolution-community/vllm-anatomy-part1.jpg)

*图 1：vLLM Engine 组件架构与核心工作流程。该图展示了 vLLM 的五层架构：最顶层是 API Layer（支持 OpenAI API、gRPC、Python Engine API），其下是 Scheduler（负责请求队列管理、抢占、优先级调度），再往下是 KVCache Manager（核心创新层，负责 Block 分配与复用），以及 Model Executor（CPU 侧的任务分发与 GPU 侧的模型执行），最底层是 ICU/Kernels（FlashAttention、VLLMattention 等注意力核）。各层职责清晰，KV Cache 管理不再附属 Model Layer，而是与调度和执行平级的独立层次，体现了 vLLM 从系统层面重新定义推理运行时架构的设计思想（来源：vLLM 官方博客「Inside vLLM: Anatomy of a High-Throughput LLM Inference System」）<a href="https://blog.vllm.ai/blog/anatomy-of-vllm">[1]</a>*

### 1.2 核心组件协作

理解 vLLM 的另一个重要维度是把握核心组件之间的协作关系。当一个请求从外部进入系统时，首先到达 API Layer 进行参数验证和预处理；随后进入 Scheduler，Scheduler 基于当前资源状态做出调度决策，包括判断是否允许新请求执行、如何组织 batch、如何分配 slot；决策做出后，KVCacheManager 为请求分配 Blocks，这个过程可能涉及 Prefix Cache 命中检测；最后，Worker 调用 Model Runner 执行前向传播，Attention Kernel 使用 PagedAttention 访问 KV Blocks，生成的 Token 逐步返回客户端。

这个数据流中有一个关键的设计点：**Scheduler 和 KVCacheManager 是紧耦合的**。调度决策不能独立于 KV 资源状态——这是 vLLM 区别于传统推理框架的核心特征。

### 1.3 核心数据流

理解 vLLM 的关键在于理解数据的完整流转路径。当请求进入系统时，首先经过 HTTP/gRPC 接口到达 Engine 进行解析和验证；随后 Scheduler 基于可用 Block 数量和正在执行的请求数进行准入控制；通过准入控制后，KVCacheManager 为请求分配 Blocks，可能命中 Prefix Cache；接着 Worker 和 Attention Kernel 执行模型前向传播；最后生成的 Token 以流式方式返回。在这个完整链路中，Block Manager 与 KVCacheManager 协作完成底层的 Block 分配和释放。

---

## 二、vLLM 起源

### 2.1 诞生背景：传统推理的"显存碎片化"困境

今天大家谈大模型推理优化，往往会先想到算子融合、Kernel 优化、量化压缩这些技术点。但 vLLM 最早被提出的起点，实际上是对一个更基础问题的反思：**在线 serving 场景下，KV Cache 的内存管理方式是否合理？**

在 vLLM 出现之前，主流推理框架对 KV Cache 的处理方式可以概括为"连续内存分配"：每个请求的 prompt 和生成的 token 被视为一段连续的内存区域，系统在 prefill 阶段一次性为整个序列分配足够的 KV 空间。这种方式在离线推理或单请求场景下没有问题，因为内存布局简单、分配逻辑清晰。但一旦进入在线 serving、多个请求并发处理、请求长度又各不相同的情况，这种"连续分配"的思路会立刻暴露出一个致命缺陷：**显存碎片化**。

原因并不复杂。每个请求的长度不同、生成 token 数量不同、到达时间也不同。如果系统为每个请求都预留一段"可能用到的最长空间"，显存很快就会被撑满；但如果不做足够预留，每次 token 增长都可能触发重新分配和拷贝，结果不是性能剧烈波动，就是 OOM。UC Berkeley 的研究团队在 2023 年发表的 PagedAttention 论文，把这个问题描述得非常清楚：传统方法的本质是把 KV Cache 当成一段"连续数组"来管理，这就像操作系统在没有分页机制的情况下直接管理内存——不仅利用率低，而且极度脆弱。

### 2.2 PagedAttention：把操作系统的分页思想引入 KV Cache

vLLM 的核心创新，正是把操作系统里成熟的**分页（paging）机制**引入到 KV Cache 管理中。这就是 PagedAttention 的本质：不再为每个请求分配一段连续的 KV 空间，而是把 KV Cache 看成可以按需分配、按页管理的资源。

具体来说，PagedAttention 会把 KV Cache 划分为固定大小的 blocks（通常是 16 个 token 对应一个 block）。每个请求不再"拥有"一段连续内存，而是"映射"到一组 blocks 上。这些 blocks 可以是连续的，也可以是离散的，甚至可以部分重叠。当一段 KV 不再需要时，系统可以直接释放对应的 block，而不需要等待整个请求结束。这听起来很自然，但正是这种设计让它彻底改变了显存管理的游戏规则：**blocks 可以按需分配、动态释放、跨请求共享，而不再受困于"连续内存"的枷锁**。

传统方法与 PagedAttention 的本质差异体现在内存布局上。在传统连续内存分配方式下，每个请求需要一段连续空间，预留过长会造成浪费，预留过短又会导致 OOM；而 PagedAttention 采用分页管理方式，通过 Block Table 建立逻辑位置到物理 Block 的映射，Physical Memory 中的 Block 可以按需分配、动态扩展，不同请求之间可以共享相同的 Block，从而消除了碎片化问题。

![PagedAttention 原理图](/assets/vllm-origin-evolution-community/pagedattention-figure1-paged-attention-overview.png)

*图 4：PagedAttention 核心思想与操作系统分页机制的类比。左侧展示了操作系统中的虚拟内存分页机制：程序使用连续的虚拟地址空间，通过页表（Page Table）映射到离散的物理内存页；右侧展示了 PagedAttention 如何将这一思想引入 KV Cache 管理：逻辑上连续的序列被划分为多个 Block，通过 Block Table 建立到物理 Block 的映射。Physical Memory 中的 Block 不需要连续排列，可以按需动态分配，这从根本上消除了传统连续内存分配带来的显存碎片化问题。这一设计灵感直接来自操作系统的虚拟内存系统，是 PagedAttention 最核心的创新点（来源：PagedAttention 论文 Figure 1 <a href="https://arxiv.org/abs/2309.06180">[2]</a>）*

### 2.3 Block Table 机制详解

Block table 是 PagedAttention 的核心数据结构。它维护了逻辑序列位置到物理 block 的映射。当 Attention Kernel 执行时，它不再需要知道"完整的序列长度"，而只需要知道"哪些 blocks 包含这个请求的 KV"。这种解耦是 PagedAttention 能够实现上述优势的根本原因。

![Block Table 机制对比图](/assets/vllm-origin-evolution-community/pagedattention-figure2-block-table-comparison.png)

*图 5：传统连续内存分配与 PagedAttention 分页管理的详细对比。左侧展示了传统方法（以 Orca Continuous Batching 为例）：每个请求需要预先分配一段连续内存空间（灰色区域），这导致了严重的显存碎片化和内存浪费——当请求实际长度远小于预分配长度时，大量显存被浪费。右侧展示了 PagedAttention 的分页管理：KV Cache 被划分为固定大小的 Blocks（每个 Block 16 tokens），每个请求通过 Block Table 维护逻辑位置到物理 Block 的映射关系。Physical Memory 中的 Blocks 不需要连续，可以按需动态分配。图中 Request 1 的 Block 3 被释放后可以立即被 Request 2 复用，实现了极高的显存利用率。这种 Block Table 机制是 PagedAttention 能够实现高性能显存管理的关键数据结构（来源：PagedAttention 论文 Figure 2 <a href="https://arxiv.org/abs/2309.06180">[2]</a>）*

### 2.4 与 SGLang 的关系：同一赛道，不同出发点

很多人第一次接触 vLLM 和 SGLang 时，会觉得它们做的事情差不多——都是推理框架、都做 KV Cache 优化、都强调吞吐和延迟。但如果回到起点看，它们的出发点并不完全一样。

**SGLang 最早切入的，是"复杂 LLM 工作流如何表达"这个问题**。它的 RadixAttention 从一开始就是为"共享前缀复用"设计的，核心思路是把不同请求之间的共同前缀识别出来并复用计算。SGLang 的第一性原理更接近"共享前缀如何作为系统资产被长期保留"。

**而 vLLM 最早切入的，是"KV Cache 显存管理如何不被碎片化困扰"这个问题**。PagedAttention 的核心贡献是把操作系统的分页机制引入到推理引擎，关注的重点是"如何让离散分布的 KV blocks 被高效访问和管理"。vLLM 的第一性原理更接近"显存分页管理问题"。

这两条路线在后续发展中逐渐靠拢：vLLM 加入了 prefix caching 支持，SGLang 也强化了自己的 runtime 调度能力。但理解这个起点差异，仍然是理解两者设计哲学不同的关键。也正因为如此，vLLM 在很多开发者眼里更像是"把推理引擎的底座做好"，而 SGLang 则更像是"把复杂工作流的表达能力做好"。两条路线各有优势，也各自吸引了一批生态力量。

---

## 三、vLLM 发展历程

### 3.1 2023：PagedAttention 论文发布，vLLM 项目正式亮相

2023 年 6 月，UC Berkeley 的研究团队在 arxiv 上发布了 PagedAttention 论文，首次系统性地提出将操作系统分页思想引入大模型推理的 KV Cache 管理。这篇论文很快引发了社区关注，因为它第一次用工程化方式回答了一个长期被忽视的问题：在线 serving 场景下，KV Cache 的内存管理到底应该怎么做。

论文发布后不久，vLLM 团队就放出了开源实现。最开始的 vLLM 版本还不完善，但它已经展现出明显的性能优势：在多个基准测试上，vLLM 的吞吐量比当时主流的 HuggingFace Transformers 高出数倍。这种性能提升并不是来自某个单点优化，而是来自系统层面的设计创新。

这一年可以概括为 vLLM 的**奠基期**：核心机制（PagedAttention、Block Manager、Scheduler）已经成型，项目在性能上建立了明显优势，社区开始注意到这个新项目。

### 3.2 2024：持续迭代，逐步逼近 SOTA

2024 年是 vLLM 快速迭代的一年。这一年，它没有停下脚步，而是在多个维度上持续优化。

在模型支持方面，vLLM 从最初只支持 Llama、Mistral 等少数模型，逐步扩展到支持更多主流开源模型，包括 Qwen、DeepSeek、GLM 等。模型支持是推理框架的"生存基础"——无论底层优化多强，如果不能用最新模型，就很快会被社区遗忘。

在性能优化方面，vLLM 持续深化包括更高效的 attention kernel、更好的 CUDA graph 支持、更精细的显存管理策略，在这一年持续保持着推理性能上的领先地位。

在 Prefix Cache 能力补齐方面，虽然 PagedAttention 从第一天起就为 block 级别的共享提供了基础，但真正把 prefix caching 做成熟，还是在这一年完成的。vLLM 加入了更智能的前缀命中判断和缓存复用机制。

在部署模式方面，vLLM 从单一的 online serving，逐步支持了离线批量推理、OpenAI-compatible API、vLLM Server 等多种部署形态。

这一年可以概括为 vLLM 的**成熟期**：它不再只是一个"有创新想法的新项目"，而是一个在多个维度上都足够完善的工程系统。社区对它的认知也从"性能很强的新框架"变成了"可以放心在生产环境使用的推理引擎"。

### 3.3 2025：v1.0 发布，平台化能力全面增强

2025 年是 vLLM 发展史上一个重要的里程碑年份。这一年，vLLM 正式发布了 **v1.0** 版本，标志着项目从"快速迭代的实验性框架"走向"稳定成熟的推理平台"。

#### 3.3.1 v0 版本的局限性

在讨论 v1.0 之前，有必要先理解 v0 版本存在哪些不足。v0 系列在快速验证 PagedAttention 核心思想方面功不可没，但它也暴露了几个明显的问题。

首先是架构边界不够清晰。在 v0 时代，KV Cache 管理、Scheduler、Attention Kernel 之间的职责边界有时会模糊，模块之间的耦合度较高，导致后续功能扩展和维护成本上升。

其次是 Prefix Cache 能力相对基础。v0 的 prefix caching 主要是基于 block 粒度的简单命中判断，缺少更智能的缓存预取、分层存储和跨实例共享能力。这在简单场景下够用，但对于复杂工作流和大规模部署来说明显不够。

第三是生产级特性不够完善。v0 更偏向"能跑出好性能"的实验框架，在 OOM 处理的细腻度、调度策略的灵活性、日志监控的完整性等方面还有欠缺。对于真正要在生产环境部署的团队来说，这些是必须补齐的短板。

第四是与外部生态的集成深度不足。v0 更专注于内核优化，对云原生部署、Kubernetes 集成、量化工具链等外部生态的适配相对薄弱。

#### 3.3.2 v1.0 的核心增强

针对 v0 的这些问题，v1.0 做了系统性的增强和补齐。

在架构重构方面，v1.0 对内部架构做了大幅重构，引入了更清晰的模块边界和更规范的接口设计。最核心的变化是把 KV Cache 管理提升为真正的"一等公民"——它不再只是 Attention Kernel 的附属，而是被明确为需要 Scheduler、Block Manager、Worker Metadata 共同维护的运行时资源层。从源码结构上看，v1 引入了 KVCacheManager、BlockPool、KVCacheCoordinator 等更清晰的抽象层次，各组件之间的职责边界更加明确。

在 Prefix Cache 能力升级方面，v1.0 的 prefix caching 不再只是简单的"命中即复用"，而是发展出更复杂的策略体系。包括基于 block 粒度的 hash 匹配、智能的缓存淘汰策略（LRU、淘汰策略可配置）、与调度器的协同优化，以及对外部 KV 传输（如 NIXL connector）的支持。此外，v1.0 还在探索分层 KV Cache（类似 SGLang 的 HiCache）的能力，为大规模跨实例缓存共享做准备。

在生产级特性完善方面，v1.0 开始明确面向生产环境的需求，包括更可靠的 OOM 处理（graceful degradation）、更精细的调度策略（可配置优先级、公平调度）、更完善的日志和监控能力（结构化日志、指标导出）、以及更灵活的部署配置选项。

在 Prefill-Decode 分离的成熟化方面，虽然 PD 分离在 vLLM 中并不是从 v1.0 才开始的，但 v1.0 把这套机制做得更成熟，开始支持更灵活的部署拓扑，包括多节点 PD 分离和动态资源分配。

在外部生态集成加强方面，v1.0 强化了与 Kubernetes、Ray 等云原生生态的集成支持，提供了更规范的部署模板和更丰富的配置选项。

### 3.4 2026：持续演进，向更完整的推理平台靠近

到 2026 年，vLLM 仍然保持着活跃的开发节奏。这一阶段的重点不再是单点性能突破，而是**系统能力的整合和生态扩展**。

在多硬件支持方面，除了 NVIDIA GPU，vLLM 也在扩展对其他硬件平台的支持，这反映了它从"CUDA 优化方案"向"通用推理平台"的转型。

在推理服务治理方面，vLLM 正在完善更精细的流量调度、更灵活的部署配置、以及与 Kubernetes、Ray 等云原生生态的更好集成。

在 Prefix Cache 能力持续深化方面，vLLM 在 prefix caching 上的投入持续增加，包括更智能的缓存预取、更大规模的跨实例共享、以及与外部存储系统的协同。

在与训练后阶段的衔接方面，虽然 vLLM 在这一方向的步子比 SGLang 稍慢，但也开始探索与 RL/Post-Training 工作流的集成可能性。

从更高一层看，vLLM 的演进路径其实很清晰：**从一项突破性的 KV Cache 管理技术（PagedAttention），到一个高性能推理引擎，再到逐步扩展为一个面向生产环境的推理基础设施平台**。这条路线和 SGLang 形成了有趣的对照——两者最终都在向"平台化"靠近，但起点和路径并不相同。

---

## 四、核心组件深度解析

### 4.1 PagedAttention 原理与实现

#### 4.1.1 核心思想

PagedAttention 的核心创新在于**将操作系统的虚拟内存分页机制引入 KV Cache 管理**。在传统方法中，每个请求的 KV Cache 被视为一段连续数组；而在 PagedAttention 中，KV Cache 被划分为固定大小的 blocks，请求通过一个"页表"（block table）来访问这些离散分布的 blocks。

这种设计带来的改变是根本性的。由于 blocks 可以按需分配，系统不再需要为每个请求预留连续空间，从而消除了显存碎片化问题。同时，Decode 阶段可以动态添加新 blocks，不需要预先分配最大长度，这意味着更高效的显存利用。多个请求可以共享相同的 block，通过引用计数机制实现天然的前缀共享。此外，KV 释放可以按 block 粒度进行，而不需要等待整个请求结束。

从实现角度来看，传统 attention kernel 需要加载完整的 Q、K、V 矩阵，然后计算 attention scores；而 PagedAttention kernel 只需要加载请求实际占用的 blocks。具体来说，kernel 需要接收 Block Table 作为输入以告知它哪些 physical blocks 对应逻辑位置，然后按 block 粒度加载 KV，每个 block 独立加载以避免跨 block 的连续性假设，接着在每个 block 内部执行标准 attention 计算，最后将各 block 的 attention 输出拼接为完整结果。这种设计使得 **访存量与序列长度解耦**，而与实际占用的 block 数量成正比。当请求长度很长但实际 active tokens 有限时，性能提升尤为显著。

### 4.2 Block Manager 与 KVCacheManager

#### 4.2.1 Block Manager 职责

Block Manager 是 PagedAttention 的底层执行者，承担着四个核心职责：Block 分配为新请求分配空闲 blocks；Block 释放在请求结束时回收 blocks；引用计数管理跟踪哪些 blocks 被多个请求共享；Block 状态维护记录每个 block 的物理位置和使用状态。

#### 4.2.2 KVCacheManager 抽象

在 v1.0 中，KVCacheManager 是 Scheduler 和 Block Manager 之间的接口层。这个接口层的关键洞察在于：**Scheduler 不需要知道 Block Manager 的内部实现，只需要调用统一的资源管理 API**。

KVCacheManager 提供了几个核心接口：get_computed_blocks 用于查询请求已命中的前缀 blocks；allocate_slots 为请求分配新的 block slots；free 释放请求持有的所有 blocks；get_common_prefix_blocks 统计公共前缀 blocks 数量。

#### 4.2.3 BlockPool：统一资源池

在 v1.0 中，BlockPool 是 KV 资源的底层结构。它同时管理三类状态：Free blocks 队列存放可分配的空闲 blocks；Cached blocks 哈希表存储已缓存的前缀 blocks 用于 prefix matching；Active blocks 记录当前正在使用的 blocks。

这三类状态不是彼此隔离的，而是同一个 block 池的不同生命周期视图。一个 block 可能先在 free queue 中，被分配后成为 active block，填满后进入 cached blocks 哈希表，被命中后通过引用计数共享，最后所有引用释放后又回到 free queue。

#### 4.2.4 allocate_slots 流程深度解析

`allocate_slots` 是 KVCacheManager 最核心的接口之一。它的执行流程体现了 vLLM 的一个核心设计原则：**KV 资源管理不是一次性操作，而是一个持续的生命周期管理过程**。

整个流程包含五个步骤。首先是检查并清理，系统释放不再参与 attention 的旧 blocks，比如 sliding window 外的内容。其次是复用已有缓存，当有 prefix cache 命中时，系统将已计算的 blocks 接入请求的 block list。第三是外部 KV 接入，当有 external computed tokens 时（如 NIXL connector），系统为这些 tokens 分配 blocks 并接入。第四是分配新 slots，为本轮需要计算的新 tokens 分配新的 block slots。第五是提交缓存，将已满的 blocks 提交到 prefix cache。

### 4.3 Scheduler 调度策略

#### 4.3.1 调度器核心职责

vLLM 的 Scheduler 不仅仅决定"哪个请求先跑"，它还要处理四个方面的核心职责。

在 Admission Control 方面，Scheduler 需要判断当前资源是否足够执行新请求。在 Preemption 方面，当资源不足时，Scheduler 需要决定哪些请求应该等待或被抢占。在 Batch Composition 方面，Scheduler 需要决定哪些请求可以组成一个 batch 同时执行。在 Slot Allocation 方面，Scheduler 需要与 KVCacheManager 协作，为每个请求分配 block slots。

#### 4.3.2 调度策略

vLLM 支持多种调度策略。FCFS 策略按请求到达顺序进行调度。Priority 策略基于请求优先级进行调度。Custom 策略允许用户自定义调度策略。

调度决策的核心输入包括：当前可用 block 数量、各请求的 prompt length 和 max new tokens、Prefix cache 命中情况、以及已占用 GPU 显存比例。

#### 4.3.3 Preemption 机制

当显存不足时，vLLM 需要决定"谁应该等待"。Preemption 机制包含三种策略。Swap 策略将 KV blocks 从 GPU 换出到 CPU 内存。Eviction 策略从 prefix cache 中淘汰低价值 blocks。Waiting 策略让新请求等待，直到资源释放。

#### 4.3.4 Scheduler 与 KVCacheManager 的协作

这是 vLLM 架构中最关键的设计点之一：**Scheduler 的调度决策必须考虑 KV 资源状态**。

在调度器循环中，对于 waiting_queue 中的每个请求，系统首先查询 prefix cache 命中情况，获取 cached_blocks；然后估算本轮需要的新 blocks 数量；接着判断资源是否足够，如果可以分配则实际分配 blocks 并将请求加入 running 队列，否则跳过或等待。这种紧耦合设计确保了**调度决策的正确性**——不会调度一个无法执行的请求；同时也保证了**资源利用的最大化**——尽量让更多请求能够执行。

![vLLM Advanced Features](/assets/vllm-origin-evolution-community/vllm-anatomy-part2.jpg)

*图 2：Chunked Prefill、Prefix Caching 与分布式服务架构。该图展示了 vLLM 三大高级特性的协同工作方式：上图展示 Chunked Prefill 机制——将长 prompts 分解为多个 chunks（如图中 x-y-z 表示将 prompt P 分解为 3 个 chunks），避免单个长请求独占一个 engine step，允许调度器在同一 step 中交叉处理多个 prefill 请求，从而改善请求延迟。下图展示 Prefix Caching 与分布式服务——通过 Radix Tree 识别共享前缀并复用 KV Cache，支持 Prefix Attention（prompt 部分互相 attending，generation 只 attending prompt）；分布式层面支持 Tensor Parallelism、Pipeline Parallelism、Expert Parallelism 等多维度并行策略，实现跨 GPU、跨节点的大规模推理部署（来源：vLLM 官方博客「Inside vLLM: Anatomy of a High-Throughput LLM Inference System」）[1]*

### 4.4 Attention Kernel 实现

#### 4.4.1 Block-Sparse 访存

PagedAttention 需要专门的 Attention Kernel 来实现高效的 block-sparse 访存。传统 fused attention 需要扫描整个序列，而 PagedAttention 只需要访问请求实际占用的 blocks，访存量从 O(seq_len) 降低到 O(num_blocks * block_size)。

#### 4.4.2 CUDA Graph 优化

vLLM 使用 CUDA Graph 来减少 kernel launch 开销。系统将整个推理图（包括 attention、linear、activation）打包为单个 CUDA Graph，避免每个 kernel 的 launch overhead，这对于 batch size 稳定的场景效果显著。

#### 4.4.3 支持多种 Attention 变体

vLLM 的 Attention Kernel 需要支持多种 attention 语义。Full Attention 是标准 attention，所有 tokens 互相 attending。Sliding Window Attention 只关注 window 内的 tokens，window 外的历史自动忽略。Prefix Attention 中 prompt 部分互相 attending，generation 部分只 attending prompt。

这些不同的 attention 语义在 PagedAttention 框架下被统一抽象为不同的"block 使用策略"，这体现了 vLLM 在抽象层设计上的灵活性。

---

## 五、vLLM 与 SGLang：架构深度对比

### 5.1 设计哲学的根本差异

**vLLM：从显存管理出发**

vLLM 的核心抽象是 **Page**——它首先关注的是"如何高效管理离散分布的 KV blocks"。在这个视角下，Block 是第一抽象，Block Table 维护逻辑到物理的映射，所有其他能力（prefix cache、共享）都建立在分页机制之上。

**SGLang：从前缀复用出发**

SGLang 的核心抽象是 **Radix Tree**——它首先关注的是"如何识别和复用请求之间的共享前缀"。在这个视角下，Prefix/Radix Tree 是第一抽象，所有 blocks 最终映射到树上的节点，分页只是底层存储的实现细节。

### 5.2 性能差异的根源分析

当 workload 具有以下特征时，SGLang 通常表现更好：大量共享前缀的场景（如 few-shot、multi-turn、tree-of-thought）；长公共前缀的场景（系统 prompt、模板化的用户输入）；高缓存复用价值的场景（相同或相似的请求模式重复出现）。在这些场景下，Radix Tree 的前缀匹配能力可以直接转化为性能收益。

当 workload 具有以下特征时，vLLM 通常更稳定：请求多样性高，没有明显的共享模式；长尾请求，需要处理各种长度的请求；生产环境，更看重稳定性和可预测性。在这些场景下，vLLM 的分页管理机制提供了更好的显存控制和可预测性。

值得注意的是，双方正在互相学习，差距在缩小。vLLM 在强化 prefix caching 能力（block-level hash、淘汰策略），SGLang 在强化 runtime 调度能力（更精细的调度策略）。未来的推理框架可能会融合两种路线的优点。

![PagedAttention 性能对比图](/assets/vllm-origin-evolution-community/pagedattention-figure3-throughput-comparison.png)

*图 6：PagedAttention 与传统方法的端到端吞吐量对比。该图展示了在 ShareGPT 数据集上，使用 PagedAttention 的 vLLM 与使用传统连续内存管理方法的 Orca 在不同 batch size 下的吞吐量对比。结果显示，vLLM 在各种负载条件下都显著优于传统方法，尤其在 batch size 较大、序列长度较长的场景下优势更加明显。这验证了分页管理机制在实际 serving 场景中的有效性——通过消除显存碎片化、允许更激进的 batch 策略，PagedAttention 能够显著提升系统的吞吐能力（来源：PagedAttention 论文 Figure 3 <a href="https://arxiv.org/abs/2309.06180">[2]</a>）*

---

## 六、架构启示与芯片设计输入

### 6.1 软件定义硬件的关键洞察

从 vLLM 的架构演进中，可以提炼出对下一代系统与芯片设计的关键洞察。

**KV Cache 管理的范式转移**

vLLM 证明了**将操作系统虚拟内存管理思想引入推理引擎**的可行性。这一范式对硬件设计的启示包括以下几个方面。Block 粒度的硬件支持方面，当前 PagedAttention 的 block size 是软件可配置的（通常 16），但硬件可以有更优的固定块大小。硬件页表加速方面，block table 的查找可以由专用硬件加速。引用计数的原子操作方面，多请求共享 block 时的引用计数管理需要高效的硬件原语。

**调度与资源的紧耦合**

vLLM 的 Scheduler 和 KVCacheManager 是紧耦合的——调度决策不能独立于资源状态。这对系统设计的启示包括：资源感知的调度方面，芯片层面的任务调度需要考虑显存资源状态；软硬件协同设计方面，软件调度策略需要反映到硬件微架构中。

**分层缓存的必然趋势**

vLLM 正在探索的分层 KV Cache（GPU 显存 → CPU 内存 → NVMe SSD）对存储层次设计的启示包括：多级缓存层次方面，未来推理芯片可能需要更丰富的存储层次；缓存一致性协议方面，跨层次的 KV Cache 需要新的 coherence 协议；数据传输引擎方面，需要高效的跨层次数据移动能力。

![KV Offloading Architecture](/assets/vllm-origin-evolution-community/kv-offloading-connector.jpg)

*图 3：KV Offloading 架构与跨层次缓存管理。该图展示了 vLLM 的 Prefill-Decode 分离部署中，跨实例 KV Cache 传输的完整架构：Prefill Worker 负责处理 prompt 计算，将生成的 KV Cache 通过 NIXL Connector 上传到分布式 KV Cache Server；Decode Worker 则从 KV Cache Server 按需拉取已计算的 KV，避免重复计算。在传输层支持 Layer-by-Layer 模式（每层 attention 前后分别传输 KV）或 Store-Then-Fetch 模式（完整 prompt 计算后一次性传输）。这种架构实现了 Prefill 与 Decode 的资源隔离——Prefill 负责计算密集型的 prompt 处理，Decode 负责内存带宽密集型的 token 生成，两者通过独立的 KV 传输通道协同，从而实现更精细的延迟控制（TTFT 和 ITL 分别优化）和更好的系统吞吐（来源：vLLM 官方博客）[1]*

### 6.2 下一代系统的设计建议

基于 vLLM 的架构经验，以下是面向下一代 AI 推理系统与芯片的设计建议：支持灵活的 Block/Page 大小，允许软件配置最优的分页粒度；硬件级引用计数支持，为共享内存场景优化原子操作；资源状态的可观测性，提供硬件级别的资源状态接口供软件调度使用；多级缓存的原生支持，在芯片层面支持 GPU-CPU-NVMe 的分层缓存架构；KV 传输的专用通道，为跨实例 KV 共享设计高效传输机制。

---

## 七、社区生态

### 7.1 LMCache：专注 KV Cache 加速的中间层

**LMCache** 是近年来备受关注的一个开源项目，它的定位非常明确：**为大模型推理提供专门的 KV Cache 加速能力**。

LMCache 的核心思路是**把 KV Cache 的管理和复用从推理引擎中抽象出来，作为独立的中间层**。这种设计带来了几个关键优势。

首先是解耦。KV Cache 优化逻辑不再和特定推理引擎绑定，可以同时支持 vLLM、SGLang、TensorRT-LLM 等多种后端。

其次是更灵活的缓存策略。LMCache 可以实现更复杂的缓存逻辑，包括分布式缓存、跨实例共享、分层存储（GPU 显存 → CPU 内存 → NVMe SSD）等。

第三是更低的集成成本。对于已经使用某个推理引擎的团队来说，只需要接入 LMCache 中间层，就可以获得 KV Cache 加速能力，而不需要更换整个推理引擎。

从架构角度看，LMCache 正在成为推理基础设施中的"缓存加速层"，它填补了推理引擎和存储系统之间的空白。

### 7.2 生态组件的价值

这些生态组件的涌现说明了一个趋势：**推理基础设施正在从"单一框架"向"组件生态"演进**。没有一个框架可以独自解决所有问题，于是社区开始围绕核心引擎构建组件生态。

vLLM 在这个生态中的角色是**作为可靠的核心引擎**——它负责最核心的推理执行和 KV Cache 管理，而其他组件则负责补充特定能力。

---

## 八、参考资料

### 核心文献

[1] vLLM 官方博客：Inside vLLM: Anatomy of a High-Throughput LLM Inference System  
https://blog.vllm.ai/blog/anatomy-of-vllm

[2] PagedAttention 论文：Efficient Memory Management for Large Language Model Serving with PagedAttention  
https://arxiv.org/abs/2309.06180

[3] vLLM GitHub 仓库（用于观察项目整体结构、版本演进与目录组织）  
https://github.com/vllm-project/vllm

[4] SGLang 项目（用于对比分析）  
https://github.com/sgl-project/sglang

[5] vLLM 官方文档  
https://docs.vllm.ai/

[6] vLLM 官方文档：Prefill-Decode Disaggregation  
https://docs.vllm.ai/en/latest/design/arch_overview.html

[7] LMCache 项目（用于生态组件分析）  
https://github.com/LMCache/LMCache
