---
title: "HierarchicalKV 论文解读：面向持续在线 Embedding 存储的 GPU 缓存语义哈希表"
date: 2026-04-03 09:30:00 +0800
author: Ethan
kind: essay
category: Essay
intro: "从系统架构视角解读 HierarchicalKV 如何以缓存语义重构 GPU 哈希表，用于持续在线 embedding 存储。"
---

本文基于论文 *HierarchicalKV: A GPU Hash Table with Cache Semantics for Continuous Online Embedding Storage* 撰写，尝试**从系统架构师的视角**梳理其问题背景、设计目标、关键机制、实验结果与适用边界。文中论述均以论文原文为依据，图片来自 arXiv HTML 页面原图资源<a href="https://arxiv.org/abs/2603.17168">[1]</a>。

## 一、引言

在推荐、搜索与广告系统中，embedding table 往往是模型中最主要的内存开销来源之一。随着特征规模持续增长，embedding 存储规模经常超出单张 GPU 高带宽显存（HBM）的容量上限。与此同时，在线训练与持续增量更新又要求系统在固定内存预算下不断接收新 key，并完成查询、写入和更新等操作。在这一背景下，如何为 embedding 构建能够长期运行、并在高负载下保持稳定吞吐的 GPU 存储结构，成为一个现实而具体的问题。

论文《HierarchicalKV》围绕这一问题提出了一种新的思路。作者认为，现有 GPU 哈希表大多采用“字典语义”，默认每个插入的 key 都应被保留；而在持续在线 embedding 存储场景下，这种设定会带来高负载退化、rehash 开销和容量失效等问题。基于此，论文提出 HKV（HierarchicalKV），尝试以**“缓存语义”重新组织 GPU 哈希表的工作方式**，使其在负载因子接近甚至达到 1.0 时仍能继续运行，并通过驱逐与准入控制优先保留更高价值的 embedding<a href="https://arxiv.org/abs/2603.17168">[1]</a>。

从系统架构角度看，这篇论文真正值得重视的地方，不只是做出一个更快的 GPU 哈希表，而是**重新定义了 embedding 存储系统的运行假设**：当“满载”不再是异常而是常态时，底层数据结构就不能继续沿用传统 dictionary semantics，而必须把容量压力、驱逐决策和并发控制直接纳入主路径设计。

## 二、研究背景

<figure>
  <img src="/assets/hierarchicalkv-paper-reading/fig-recsys-embedding.png" alt="推荐模型中的 embedding lookup 流程">
  <figcaption>图 1. 推荐模型中的 embedding lookup 流程。embedding table 是主要内存消耗来源，在线训练还要求系统持续接收新 key<a href="https://arxiv.org/html/2603.17168v1">[1]</a>。</figcaption>
</figure>

论文指出，推荐系统中的特征 ID 通常位于稀疏的 uint64 key 空间，其访问模式普遍具有幂律分布特征<a href="https://arxiv.org/abs/2603.17168">[1]</a>。这意味着，少量热点 embedding 会被频繁访问，而大量冷门 embedding 的命中次数较低。对于这类工作负载，存储系统真正关心的通常不是无条件保存所有 key，而是在有限容量下尽量维持高价值条目的保留率。

<figure>
  <img src="/assets/hierarchicalkv-paper-reading/fig-workload-analysis.png" alt="持续在线写入下的工作负载特征">
  <figcaption>图 2. 持续在线写入会推动哈希表逐渐逼近满载，并放大传统 dictionary-semantic 结构在高负载下的性能退化<a href="https://arxiv.org/html/2603.17168v1">[1]</a>。</figcaption>
</figure>

作者进一步分析了现有 GPU 哈希表的设计空间，并指出 WarpCore、WarpSpeed、cuCollections、BGHT、BP2HT 与 Hive 等方案都建立在 dictionary semantics 之上<a href="https://arxiv.org/abs/2603.17168">[1]</a>。在这种语义下，一旦哈希表接近满载，探测距离会增长，查找与插入吞吐开始下降，最终系统只能通过 rehash 或直接失败来处理持续增长的负载。

论文中的实验结果表明，这一问题在高负载场景下尤为明显。当负载因子从 0.25 上升到 1.00 时，WarpCore 的 find 吞吐下降 90%，BGHT 下降 31%，cuCollections 下降 100%<a href="https://arxiv.org/abs/2603.17168">[1]</a>。作者据此指出，在持续在线写入的 embedding 存储场景中，**满载更接近系统常态，而不是边缘情况**。也正因此，字典语义中“满表即异常”的前提并不适合作为该问题的基本建模方式<a href="https://arxiv.org/abs/2603.17168">[1]</a>。

从这一观察出发，论文提出应当将 embedding 存储视为缓存，而非传统字典。作者将这种新的系统约束概括为 cache semantics：当 bucket 满载时，插入应在表内通过驱逐或拒绝准入来完成；操作不应触发 rehash 或外部容量管理；查找成本也应与累计插入次数解耦<a href="https://arxiv.org/abs/2603.17168">[1]</a>。

**从架构视角看，这里的关键转变不是“加一个 eviction 策略”，而是把 eviction 变成系统语义的一部分。** 一旦接受这一点，查找路径、写入路径、bucket 组织方式以及并发协议都必须重新设计。

## 三、研究目标

在上述背景下，论文的目标是设计一个面向持续在线 embedding 存储的 GPU 哈希表库，使其能够在固定 HBM 预算下维持高负载运行，并把驱逐与准入控制直接纳入哈希表的正常操作语义中<a href="https://arxiv.org/abs/2603.17168">[1]</a>。

作者在论文中将这一目标进一步分解为四项挑战。首先，系统需要提供固定工作量的查找路径。对于 embedding cache 而言，操作必须在高负载条件下长期执行，因此查找开销应尽量保持稳定，尤其是 miss path 应限制在固定数量的 GPU 内存事务内。其次，系统需要支持满容量下的原地 upsert。持续到来的新 embedding 不能依赖 rehash 或外部维护流程进入系统，因此 bucket 满载时必须在表内完成插入、替换或拒绝。再次，系统需要在有限 bucket associativity 的前提下尽量提升高价值条目的保留能力。若驱逐了高价值 embedding，系统就需要承受额外的重新计算或远程拉取成本，因此驱逐决策本身会直接影响缓存质量。最后，系统需要支持混合负载下的并发访问。实际部署中，推理和训练会同时发起查询、更新与插入操作，若使用粗粒度锁，会把结构性修改与非结构性修改一并串行化，从而形成吞吐瓶颈<a href="https://arxiv.org/abs/2603.17168">[1]</a>。

因此，论文的研究目标并非单纯提高某一类操作的峰值性能，而是建立一种新的运行契约：在满载状态成为常态的前提下，使 GPU 哈希表仍能稳定执行查找、更新与插入，并通过策略化驱逐维持 embedding 存储质量<a href="https://arxiv.org/abs/2603.17168">[1]</a>。

**如果用系统架构语言概括，这四个目标分别对应四个核心指标：路径稳定性、满载可写性、容量利用效率，以及混合负载并发性。** 这也是判断 HKV 是否真正有工程价值的四条主线。

## 四、方法与创新

### 4.1 缓存语义下的 GPU 哈希表设计

HKV 的核心创新首先体现在语义层面。论文将其定义为“第一个通用型 GPU 哈希表库，其正常满容量运行契约是缓存语义的”<a href="https://arxiv.org/abs/2603.17168">[1]</a>。在这一设计下，哈希表满载不再被视为错误状态，而是系统的常规运行条件。每次满桶 upsert 都要在表内完成解决：要么驱逐低分条目，要么拒绝低价值新条目，而不是通过 rehash 将问题推迟到外部维护流程<a href="https://arxiv.org/abs/2603.17168">[1]</a>。

论文将 HKV 的设计归纳为四项核心机制：单桶约束的缓存行对齐 bucket、分数驱动的内联 upsert、基于分数的动态双桶选择，以及三组并发控制协议。此外，作者还将分层 key-value 分离作为超出 HBM 容量时的扩展机制加以实现<a href="https://arxiv.org/abs/2603.17168">[1]</a>。

**从架构上看，HKV 最重要的不是某一个局部优化，而是把“语义、数据布局、替换策略、并发协议”收束成一套统一设计。** 这使它更像一个为 embedding cache 场景定制的系统部件，而不是传统 GPU 哈希表上零散补丁的叠加。

### 4.2 单桶约束与缓存行对齐 bucket

<figure>
  <img src="/assets/hierarchicalkv-paper-reading/fig-bucket-layout.png" alt="HKV 单个 bucket 的内存布局">
  <figcaption>图 3. HKV 单个 bucket 的 digest 数组与 GPU L1 cache line 对齐，使一次 cache-line 读取即可覆盖整个 bucket 的候选范围<a href="https://arxiv.org/html/2603.17168v1">[1]</a>。</figcaption>
</figure>

HKV 在单桶模式下将每个 key 仅映射到一个 128-slot bucket，并把这 128 个 slot 的 one-byte digest 紧密排布为一个连续的 128B 数组，与 GPU L1 cache line 对齐<a href="https://arxiv.org/abs/2603.17168">[1]</a>。查找时，系统首先读取该 bucket 的 digest cache line，通过 SIMD 字节比较快速筛选潜在匹配项，仅对 digest 相同的 slot 再做完整 key 比较<a href="https://arxiv.org/abs/2603.17168">[1]</a>。

论文据此给出一个重要性质：在单桶模式下，若某个 key 不在表中，则查找操作只需检查该 bucket 一次，即可完成 definitive per-bucket miss；这一过程使用单次 128B 内存事务，并执行固定数量的 digest comparison<a href="https://arxiv.org/abs/2603.17168">[1]</a>。作者强调，这一设计的意义在于将 miss path 的成本固定下来。与既有 CPU/GPU 哈希表相比，HKV 的 miss path 不再随负载因子增长，且在负载因子达到 1.0 时仍可维持相同的 per-bucket 检查开销<a href="https://arxiv.org/abs/2603.17168">[1]</a>。

**这部分是整篇论文最有“系统味”的设计之一。** 从架构角度看，作者本质上是在用**单次 cache-line 访问换取确定性的 miss 成本上界**。这比单纯追求平均吞吐更重要，因为在线系统在高负载下往往首先死于 tail latency 和不可预测的探测路径，而不是均值性能不足。

### 4.3 内联分数驱动的 upsert 与驱逐机制

在插入路径上，HKV 引入了分数驱动的内联 upsert 机制。论文指出，当目标 bucket 中已存在目标 key 时，系统直接更新其 value 与 score；若 bucket 中仍有空槽，则直接写入新条目；若 bucket 已满，则扫描 bucket 中所有 score，选出最低分条目，并根据新条目的 score 执行准入判断<a href="https://arxiv.org/abs/2603.17168">[1]</a>。

若新条目的 score 低于当前 bucket 的最小 score，则系统直接拒绝插入；若新条目得分更高，则通过一次 CAS 将最低分条目替换为新条目<a href="https://arxiv.org/abs/2603.17168">[1]</a>。作者强调，这一机制把 score comparison、admission control 与 CAS commit 融合到同一条 insert path 中，从而避免了额外的 eviction workflow<a href="https://arxiv.org/abs/2603.17168">[1]</a>。

论文还将该机制与 MemC3、CacheLib、FBGEMM TBE 等系统进行了比较，并指出 HKV 的一个主要区别在于，它不依赖第二套 eviction metadata structure，也不需要 CPU 介入驱逐路径，而是直接将 score array 作为驱逐决策的基础<a href="https://arxiv.org/abs/2603.17168">[1]</a>。从实验角度看，论文报告满桶驱逐会带来 32%–41% 的 insert_or_assign 吞吐下降，但这一开销是有界的，因为每次 eviction scan 只处理固定大小的 128-slot bucket<a href="https://arxiv.org/abs/2603.17168">[1]</a>。

**这项设计的架构价值在于：把准入控制、替换决策和提交路径压缩进同一个 GPU 主路径。** 对在线系统而言，这意味着驱逐不再是旁路流程，而是可预测、可并行、可局部化的表内操作。这种“主路径内化”的思路，通常比事后补一个 eviction manager 更容易扩展到真实生产环境。

### 4.4 基于分数的动态双桶选择

<figure>
  <img src="/assets/hierarchicalkv-paper-reading/fig-dual-bucket.png" alt="HKV 双桶选择策略">
  <figcaption>图 4. HKV 的双桶策略分为两个阶段：早期优先提高容量利用率，满载后优先提高驱逐质量<a href="https://arxiv.org/html/2603.17168v1">[1]</a>。</figcaption>
</figure>

论文指出，单桶约束虽然带来了固定 miss 成本，但也会因碰撞过早触发驱逐。在 128-slot bucket 设置下，单桶模式会在负载因子约 0.66 时出现首次驱逐，从而导致部分 HBM 容量尚未被真正利用<a href="https://arxiv.org/abs/2603.17168">[1]</a>。

为解决这一问题，HKV 提出了动态双桶模式。该模式下，每个 key 映射到两个候选 bucket，并采用两阶段选择策略<a href="https://arxiv.org/abs/2603.17168">[1]</a>。在至少有一个候选 bucket 未满时，系统将新条目插入占用更低的 bucket，以提升整体内存利用率；当两个 bucket 都满时，系统比较两者的最小 score，并选择最小 score 更低的 bucket 执行驱逐，以提升替换质量<a href="https://arxiv.org/abs/2603.17168">[1]</a>。

论文认为，这一机制将传统 power-of-two-choices 的思路从“负载均衡”扩展到了“驱逐质量优化”<a href="https://arxiv.org/abs/2603.17168">[1]</a>。实验结果表明，双桶模式将首次驱逐的负载因子从 0.633 提高到 0.977，同时将 top-N score retention 从 95.39% 提升到 99.44%<a href="https://arxiv.org/abs/2603.17168">[1]</a>。

**从架构角度看，双桶策略解决的是“路径稳定性”和“容量利用率”之间的结构性冲突。** 单桶路径更稳定，但容量吃不满；多桶路径更灵活，但查找与插入路径可能变复杂。HKV 的做法不是简单在两者之间二选一，而是通过分阶段策略把“装填效率”和“驱逐质量”拆开优化，这是很典型的系统折中设计。

### 4.5 三组并发控制协议

针对混合工作负载下的同步问题，HKV 提出了 triple-group concurrency，将 GPU 侧操作划分为 reader、updater 和 inserter 三类<a href="https://arxiv.org/abs/2603.17168">[1]</a>。

论文定义如下：reader 包括 find、contains、size 等只读操作；updater 仅修改已有条目的 value 或 score，不改变 bucket 结构；inserter 则负责 insert_or_assign、find_or_insert、erase 等结构修改操作，包括 slot 分配、digest 更新与 eviction<a href="https://arxiv.org/abs/2603.17168">[1]</a>。

作者认为，这一机制的关键在于区分 structural writes 与 non-structural writes，并通过 CPU–GPU dual-layer lock 协调角色切换<a href="https://arxiv.org/abs/2603.17168">[1]</a>。相较传统 reader/writer lock，将 updater 单独拆出后，多个 updater 可以并行执行，而不必与所有写操作一起被串行化<a href="https://arxiv.org/abs/2603.17168">[1]</a>。

实验中，论文通过并发消融测试比较了 triple-group 与传统 R/W lock。结果显示，当 updater 从 1 增加到 10 时，triple-group 方案吞吐可达到 2.569 B-KV/s，而 R/W lock 降至 0.535 B-KV/s，最高形成 4.80× 的吞吐差距<a href="https://arxiv.org/abs/2603.17168">[1]</a>。

**这是我认为整篇文章里最容易被低估的一部分。** 很多系统在数据结构层面优化得很好，但一进入混合读写场景就被锁模型拖垮。HKV 把 updater 从 writer 中拆出来，本质上是在利用“结构写”和“非结构写”的差异重新设计并发域，这比传统 reader/writer lock 更贴近真实 workload。

### 4.6 分层键值分离与容量扩展

<figure>
  <img src="/assets/hierarchicalkv-paper-reading/fig-architecture.png" alt="HKV 架构图">
  <figcaption>图 5. HKV 将索引相关元数据保留在 HBM 中，并通过位置映射将 value 溢出到 HMEM，从而在不引入 CPU 数据路径的前提下扩展容量<a href="https://arxiv.org/html/2603.17168v1">[1]</a>。</figcaption>
</figure>

除上述四项核心机制外，HKV 还实现了分层 key-value 分离，以支持超出 HBM 容量的 embedding table<a href="https://arxiv.org/abs/2603.17168">[1]</a>。在该设计下，key、digest 与 score 固定存放在 HBM 中，而 value 可溢出到 pinned host memory（HMEM），通过 bucket 与 slot 位置计算直接访问，而非通过 per-entry pointer 间接寻址<a href="https://arxiv.org/abs/2603.17168">[1]</a>。

论文指出，这一设计的直接效果是保留 GPU 端 key-side processing 的高吞吐，同时减少额外指针开销<a href="https://arxiv.org/abs/2603.17168">[1]</a>。实验结果显示，在 HBM+HMEM 模式下，pointer-returning 的 find* 仍保留了纯 HBM 场景下 96.0% 的吞吐，而需要复制 value 的 find 受到 PCIe 带宽限制，吞吐显著下降<a href="https://arxiv.org/abs/2603.17168">[1]</a>。

**这部分的系统意义在于把“容量扩展”与“主路径性能”做了解耦。** 作者没有把所有内容都留在 HBM，也没有把访问退化成 CPU 参与的远程存取，而是把索引元数据尽量保留在 GPU 侧，把真正昂贵的 value 访问延后处理。对大规模 embedding 系统来说，这是一种很现实的分层思路。

## 五、实验结果与论文结论

<figure>
  <img src="/assets/hierarchicalkv-paper-reading/fig-throughput-vs-load-factor.png" alt="不同负载因子下的吞吐对比">
  <figcaption>图 6. 与多种 dictionary-semantic GPU 哈希表相比，HKV 的吞吐在高负载区间保持稳定，而基线系统在高负载下显著退化<a href="https://arxiv.org/html/2603.17168v1">[1]</a>。</figcaption>
</figure>

论文在 NVIDIA H100 NVL 上对 HKV 进行了系统评估，并与 WarpCore、BGHT、cuCollections 和 BP2HT 等 GPU 哈希表进行了比较<a href="https://arxiv.org/abs/2603.17168">[1]</a>。

在负载因子敏感性实验中，HKV 的 find 吞吐在负载因子 0.25 至 1.00 范围内基本保持稳定。论文报告，其 find 吞吐在这一范围内为 3.37–3.40 B-KV/s，变化不到 1%<a href="https://arxiv.org/abs/2603.17168">[1]</a>。相比之下，WarpCore、BGHT 和 cuCollections 都随负载上升出现明显退化，其中前两者在高负载下大幅下降，cuCollections 在负载逼近 1.0 时几乎失效<a href="https://arxiv.org/abs/2603.17168">[1]</a>。

<figure>
  <img src="/assets/hierarchicalkv-paper-reading/fig-api-throughput.png" alt="HKV 核心 API 吞吐">
  <figcaption>图 7. HKV 在纯 HBM 配置下的核心 API 吞吐均保持在 B-KV/s 量级，其中 find* 体现了较高的 key-side throughput<a href="https://arxiv.org/html/2603.17168v1">[1]</a>。</figcaption>
</figure>

在端到端吞吐实验中，HKV 的 find 吞吐达到 3.61–3.89 B-KV/s，pointer-returning 的 find* 约为 7.05 B-KV/s，表明在纯 HBM 模式下，key-side indexing throughput 保持在较高水平<a href="https://arxiv.org/abs/2603.17168">[1]</a>。

在组件消融方面，论文表明 digest pre-filtering 对性能提升具有直接作用。在配置 A–C 下，启用 digest filtering 后，find 吞吐在负载因子 0.50 时提升 1.65×–1.87×，在负载因子 1.00 时提升约 2.60×<a href="https://arxiv.org/abs/2603.17168">[1]</a>。此外，双桶策略显著提高了 top-N score retention，triple-group concurrency 也在 updater 密集负载下带来了较大吞吐优势<a href="https://arxiv.org/abs/2603.17168">[1]</a>。

据此，论文在结论部分指出，HKV 的主要贡献在于通过缓存语义、单桶约束 bucket、内联 upsert、双桶分数选择和三组并发协议的联合设计，使 GPU 哈希表能够在负载因子达到 1.0 时依然维持稳定查找、原地插入和混合负载执行能力<a href="https://arxiv.org/abs/2603.17168">[1]</a>。

**如果从系统架构师的标准来评价，这篇论文最强的证据不是“某个点上更快”，而是它在多个维度上同时建立了稳定性：高负载稳定、满载可写、混合负载可并发、容量扩展可分层。** 这类结果通常比单点 benchmark 更接近真实系统价值。

## 六、局限与未来展望

论文在结尾部分对 HKV 的边界和后续扩展方向作出了明确说明<a href="https://arxiv.org/abs/2603.17168">[1]</a>。

首先，reader 与 updater 仍然互斥。作者指出，这是在当前实现下为保证原子性和简化并发协议所作出的选择；若 value 能够通过单字原子操作进行安全更新，则这一限制未来可能放宽<a href="https://arxiv.org/abs/2603.17168">[1]</a>。

其次，双桶模式目前主要覆盖 insert_or_assign 与 find 等关键路径，而完整 API 仍主要由单桶路径支持，因此双桶设计的进一步推广仍有实现空间<a href="https://arxiv.org/abs/2603.17168">[1]</a>。

第三，论文将多 GPU sharding 明确交由应用层处理，说明 HKV 现阶段主要聚焦单 GPU 内的数据结构设计，而未直接解决跨 GPU 分布问题<a href="https://arxiv.org/abs/2603.17168">[1]</a>。

最后，作者提出，更宽的 GPU cache line 可能进一步支持更大的 bucket，SSD / GPUDirect Storage 分层以及 dynamic rehash 也仍是开放扩展方向<a href="https://arxiv.org/abs/2603.17168">[1]</a>。

因此，从原文表述来看，论文并未声称 HKV 已经覆盖所有 embedding 存储部署场景，而是将其定位为一种验证 cache semantics 可行性的 GPU 哈希表设计框架<a href="https://arxiv.org/abs/2603.17168">[1]</a>。

**这也意味着，HKV 当前更像一个很强的单 GPU 基础构件，而不是已经完成分布式封装的终局方案。** 如果后续要走向大规模生产系统，多 GPU 路由、分片均衡、跨设备一致性与更深层存储分层，仍然会是必须补上的系统层工作。

## 七、总结

《HierarchicalKV》这篇论文的核心价值，在于它没有将在线 embedding 存储继续视作一个传统“字典”问题，而是提出了更契合实际工作负载的“缓存语义”建模方式。围绕这一建模转变，作者设计了单桶缓存行对齐 bucket、内联分数驱动 upsert、动态双桶选择、三组并发控制和分层 key-value 分离等一整套机制，并在实验中展示了其在高负载、满容量和混合工作负载条件下的稳定性与可行性<a href="https://arxiv.org/abs/2603.17168">[1]</a>。

严格依据论文原文来看，HKV 的主要贡献并不只是获得了更高的 find 吞吐，而是证明了：在持续在线 embedding 存储这一问题设定下，将 GPU 哈希表从 dictionary semantics 转向 cache semantics，能够带来更稳定的满载运行能力与更贴近实际需求的系统行为<a href="https://arxiv.org/abs/2603.17168">[1]</a>。

**从系统架构师视角做一句总结：HKV 最重要的创新，不是把哈希表做得更快，而是把“高负载下可持续运行”本身定义成数据结构的第一原则。** 这一点对于未来的大规模 embedding 基础设施，可能比单次吞吐数字更值得关注。

## 参考文献

[1] Haoyu Rong, Jialin Yao, Max Langer, Shuchao Liu, Li Fan, Danyang Wang, Jian He, Junjie Chen, Junyi Rang, Jingjing Qian, Menglu Xu, Feng Yu, Michael Lee, Zheng Wang, Ed Oldridge. *HierarchicalKV: A GPU Hash Table with Cache Semantics for Continuous Online Embedding Storage*. arXiv, 2026. <a href="https://arxiv.org/abs/2603.17168">https://arxiv.org/abs/2603.17168</a>
