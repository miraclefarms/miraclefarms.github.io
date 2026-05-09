# 术语表

按字母顺序排列。

---

## A

**Attention Kernel**
实现 Attention 计算的 GPU kernel。在分页 KV 场景下，需要支持非连续 KVCache 的间接寻址访问。常见实现包括 FlashAttention、FlashInfer、Triton 自定义 kernel。

**Attention Sink**
StreamingLLM 提出的现象：模型在前几个 token 上集中了大量 attention 概率。简单丢弃这些 token 会破坏 attention 分布，因此实践中始终保留前 $k$ 个 token 的 K/V。

**Autoregressive Decoding（自回归解码）**
LLM 生成文本的基本方式：每次生成一个 token，将其追加到上下文后再生成下一个，直到结束符或达到长度限制。KVCache 是高效自回归解码的前提条件。

---

## B

**Block**
KVCache 的分配单位，存储固定数量（block_size）个 token 的 K/V 张量。等同于 Page。Block 是 Block Manager 分配、释放和共享的最小单元。

**Block Manager**
推理引擎中负责 KVCache 物理 Block 分配和释放的组件。维护空闲 Block 池，处理 Block 的引用计数、驱逐和换出逻辑。

**Block Size（块大小）**
每个 Block 存储的 token 数量。典型值为 16。block_size 影响内部碎片率（大 → 碎片多）和调度粒度（小 → 开销大）。

**Block Table**
每个 Sequence 维护的映射表，记录逻辑 Block 序号到物理 Block 地址的对应关系。在 Attention kernel 执行时传入 GPU，用于间接寻址。

---

## C

**Cache-Aware Routing**
路由层根据各副本上的 prefix cache 分布，把请求送到"最可能命中"的副本。在亲和性（命中率）和负载均衡之间需要权衡。

**Cache Hit Rate（缓存命中率）**
命中 Prefix Cache 的 KV token 数占总 Prompt token 数的比例。高命中率意味着更低的 TTFT 和更少的计算开销。

**CacheBlend / Prompt Cache**
针对 RAG 场景的"位置无关"prefix cache 研究方向。把文档块预先压缩或在加载时重算位置编码，以实现文档块在不同 prompt 位置下的 KV 复用。

**CLA（Cross-Layer Attention）**
跨层共享 KV 的方案：多层共用同一份 K/V，KVCache 体积按倍数下降。常与 YOCO 一起讨论。

**CP（Context Parallelism）**
见 [SP](#s)。在长上下文场景下沿 sequence 维度切分 KV。

**CXL（Compute Express Link）**
新一代设备互联标准，让 CPU、GPU、加速器以"内存语义"共享内存池。被认为可能重塑 KV 多级存储的层次。

**Causal Attention（因果注意力）**
LLM 使用的 Attention 变体：每个 token 只能关注自身及之前的 token，不能看到未来 token。这是 KVCache 可以按 token 顺序累积的前提。

**Chunked Prefill**
将长 Prompt 分成若干 chunk 分多次 Prefill，而不是一次性处理。允许 Prefill 和 Decode 在同一 batch 中混合执行（interleaved），兼顾 TTFT 和 TPOT。

**COW（Copy-on-Write，写时复制）**
当多个 Sequence 共享同一 Block（如 Prefix Cache 或 beam search 的分叉），对共享 Block 做写操作前必须先复制一份独立副本。保证各 Sequence 的 KV 不相互污染。

**CPU DRAM**
服务器 CPU 侧的内存，用于 KV Offload 时的二级存储。带宽远低于 GPU HBM，但容量更大（通常 512GB~2TB）。通过 PCIe 总线与 GPU 通信。

---

## D

**Decode 阶段**
LLM 推理的生成阶段。每步从 KVCache 读取历史 K/V，结合新 token 的 Q 做 Attention，输出下一个 token 的 logits，然后追加新 K/V 到 KVCache。Memory-bound。

**Disaggregated Prefill（PD 分离）**
见 [PD Disaggregation](#p)。

**DistServe**
PD 分离的代表性论文之一，强调"goodput"——SLO 内的有效吞吐——的优化。

---

## E

**EP（Expert Parallelism）**
MoE 模型中将 expert 分布到不同 GPU 的并行方式。与 KV 直接关系不大，但 all-to-all 通信会挤占 KV 传输的带宽预算。

**Eviction（驱逐）**
当 GPU KVCache 空间不足时，Block Manager 将部分无活跃引用的 Block 标记为可回收（换出或丢弃）。常见驱逐策略：LRU（最近最少使用）、优先级调度等。

---

## F

**FlashAttention**
一种 IO 感知的 Attention kernel 实现，通过分块计算减少 HBM 访问次数，提升计算效率。是 Prefill 阶段的主流 kernel 实现。

**FlashInfer**
针对 LLM 推理优化的 Attention kernel 库，支持分页 KVCache 的高效访问，被 vLLM、SGLang 等框架使用。

**Fragmentation（碎片化）**
KVCache 显存的浪费现象。内部碎片：Block 的最后若干 Slot 未被使用。外部碎片：连续显存被分配-释放后产生的不规则空洞（分页管理可消除外部碎片）。

---

## G

**GDS（GPUDirect Storage）**
让 GPU 可以零拷贝直接读写 NVMe，绕过 CPU DRAM 中转。对批量 KV 落盘 / 召回场景有意义。

**GQA（Grouped-Query Attention）**
一种 Attention 变体，多个 Query head 共享同一组 K/V head，减小 KVCache 体积。LLaMA-3、Mistral 等主流模型采用。

---

## H

**H2O（Heavy-Hitter Oracle）**
基于 attention 累积分数识别"高重要性"token，丢弃其他 token 的 KV 稀疏化方案。

**HBM（High Bandwidth Memory）**
GPU 显存类型，提供极高的带宽（如 H100 的 HBM3e 达 3.35 TB/s）。LLM 推理 Decode 阶段是 Memory-bound，HBM 带宽是核心限制。

**Head Dim（头维度）**
每个 Attention head 的 K/V 向量维度，通常为 128。影响 KVCache 大小：KV 大小正比于 num_kv_heads × head_dim。

---

## K

**KIVI**
2-bit KV 量化方法：K 用 per-channel、V 用 per-token，平衡精度与压缩比。

**KV Block**
见 [Block](#b)。

**KVCache**
Transformer 自回归解码过程中缓存的 Key 和 Value 张量。避免对历史 token 重复计算 Attention，是现代 LLM 推理的基础机制。

**KV Pool（KV 池）**
跨节点共享的"分布式 KV 存储"。代表实现：Mooncake Store、LMCache 远端层。被认为有可能成为类似 Redis 的基础设施层产品。

**KVQuant**
4-bit KV 量化方案，配合 outlier 处理与 RoPE 对齐，在长上下文上有较好质量。

**KV Transfer**
在 PD 分离架构中，将 Prefill 节点生成的 KVCache 传输到 Decode 节点的过程。传输带宽和延迟直接影响端到端 TTFT。

---

## L

**L1 / L2 / L3 / L4**
KVCache 的多级存储层级简记：L1 = GPU HBM，L2 = CPU DRAM，L3 = 本地 NVMe / SSD，L4 = 远端 / 分布式 KV 池。详见 [存储层级](storage-hierarchy.md)。

**LMCache**
专注于多级 KVCache 存储与跨实例共享的开源项目，可作为 vLLM 的 KV 后端。

---

## M

**Mamba / State Space Model（SSM）**
用状态空间模型替代 Transformer attention 的架构，序列状态恒定大小，无传统意义上的 KVCache。

**Memory-Bound**
计算受限于内存带宽而非计算能力。LLM Decode 阶段是 memory-bound：每步计算量小（单 token），但需要大量 HBM 带宽读取 KVCache。

**MHA（Multi-Head Attention）**
标准多头注意力，每个 Q/K/V head 独立，KVCache 最大。

**MLA（Multi-head Latent Attention）**
DeepSeek 提出的 Attention 变体，将 K/V 压缩到低维潜空间存储，显著减小 KVCache 体积。

**Mooncake**
Kimi 团队开源的以 KVCache 为中心的 PD 分离推理基础设施。Mooncake Store 是其 KV 池组件。

**MQA（Multi-Query Attention）**
所有 Q head 共享单组 K/V，KVCache 最小。极端的 GQA 特例。

---

## N

**NVLink / NVSwitch**
NVIDIA GPU 间的高带宽互联（NVLink 4.0 双向 900 GB/s+），同机柜内 KV 跨卡传输的主要通道。

---

## P

**Page**
见 [Block](#b)。

**PagedAttention**
vLLM 提出的 KVCache 分页管理方案。将 KVCache 划分为固定大小的 Block，支持非连续物理存储，消除外部碎片，显著提升显存利用率和并发能力。

**PCIe（Peripheral Component Interconnect Express）**
GPU 与 CPU 之间的总线接口。KV Offload 时，KV 数据通过 PCIe 在 GPU HBM 和 CPU DRAM 之间传输。PCIe 4.0 x16 双向带宽约 32 GB/s，远低于 HBM。

**PD Disaggregation（PD 分离）**
将 Prefill 和 Decode 部署到不同的 GPU/节点，分别优化各自的计算特征，解除相互干扰。核心工程挑战是 KV Transfer。

**Prefill 阶段**
LLM 推理的预填充阶段，处理输入 Prompt。所有 Prompt token 并行经过每层 Attention 和 FFN，产生初始 KVCache。Compute-bound。

**Prefix Cache（前缀缓存）**
缓存共享前缀的 KV Block，避免对相同前缀重复 Prefill。对于固定 System Prompt、多轮对话、RAG 等场景有显著的 TTFT 降低效果。

**Preemption（抢占）**
当 GPU KVCache 耗尽时，Scheduler 暂停部分 Sequence 并释放其 KVCache（换出或丢弃），为高优先级或新到来的 Sequence 腾出空间。

**Pre-warming（预热）**
新副本启动后，从 L4 KV 池或对等副本拉取热门 prefix，避免冷启动期的命中率塌陷。

---

## R

**RadixAttention**
SGLang 提出的 Prefix Cache 实现方案，使用 Radix Tree 组织 KV Block，支持细粒度的前缀匹配和高效的缓存管理。

**RDMA（Remote Direct Memory Access）**
允许网络设备直接访问远端内存，无需经过 CPU。在 PD 分离中，RDMA 用于高带宽、低延迟的跨节点 KV Transfer。

**Recompute（重算）**
被抢占的 Sequence 在恢复时，对已丢弃的 KVCache 重新执行 Prefill 以重建。计算成本高，适合短序列或 KVCache 体积小的场景。

**Remote Cache（远端缓存）**
存储在其他节点（通过 RDMA 访问）的 KVCache。支持跨推理实例的 Prefix Cache 命中。

**Ring Attention**
长上下文序列并行（SP/CP）的代表实现：各卡按 ring 拓扑轮转传递 K/V，每卡都看到所有 K/V。

**RWKV / RetNet**
线性注意力 / 类 RNN 架构，状态恒定大小，无传统意义上的 KVCache。

---

## S

**Scheduler（调度器）**
推理引擎中决定哪些请求进入当前 batch 的组件。需要考虑 KVCache 可用量、请求优先级、吞吐目标等约束。

**Sequence（序列）**
一个推理请求对应的完整 token 序列（Prompt + 已生成 token）。每个 Sequence 有独立的 Block Table 和 KVCache 分配。

**Session Affinity（会话粘性）**
路由层把同一 session 的所有请求送到同一副本，以最大化跨轮次 prefix cache 命中。

**Sliding Window Attention（SWA）**
每个 token 只关注最近 $w$ 个 token，KVCache 总量有上界。Mistral / Gemma 系列采用。

**Slot**
Block 内的最小存储单元，对应一个 token 的 K/V 数据。

**SnapKV**
Prefill 完成时利用末端 token 的 attention 模式预测重要 token，丢弃其余 K/V 的稀疏化方案。

**SP（Sequence Parallelism）/ CP（Context Parallelism）**
长上下文场景下沿 sequence 维度切分 KV 的并行方式。代表实现包括 Ring Attention、LoongTrain。

**Splitwise**
PD 分离的早期代表论文，强调 Prefill 和 Decode 计算特征不同因此应分开部署。

**StreamingLLM**
保留前 $k$ 个 attention sink token + 滑动窗口的稀疏化方案，让模型可以处理无限长流式输入。

**Swap（换入/换出）**
将 KVCache Block 在 GPU HBM 和 CPU DRAM 之间移动。换出（Swap-out）释放 GPU 显存，换入（Swap-in）恢复 Sequence 的 Decode 所需 KV。

---

## T

**TP（Tensor Parallelism，张量并行）**
将模型的 Attention 和 FFN 层按 head 或维度切分到多张 GPU 上并行计算。TP 配置影响 KVCache 的存储布局，PD 分离时 TP 不对称会引起 KV reshape 问题。

**TPOT（Time per Output Token）**
每个输出 token 的平均生成时间。衡量 Decode 阶段的速度。

**TTFT（Time to First Token）**
从请求到达到第一个 token 返回的时间。衡量 Prefill 阶段的速度。Prefix Cache 命中可以显著降低 TTFT。

---

## V

**vLLM**
最广泛使用的开源 LLM 推理引擎之一，PagedAttention 的原创提出者。

---

## Y

**YOCO（You Only Cache Once）**
跨层共享 KV 的架构思想：只在少数几层真正缓存 K/V，其余层复用，KVCache 体积可降一个数量级。
