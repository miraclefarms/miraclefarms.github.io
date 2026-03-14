---
title: vLLM 如何管理 KV Cache：从 Block Pool 到调度器的运行时资源层
date: 2026-03-12 13:15:00 -0400
author: Ethan
kind: essay
category: Essay
intro: 基于 vLLM 当前主线源码，系统梳理 KV cache 的真实存储本体、block/page 抽象、调度接口、prefix cache 与执行映射之间的关系。
---

很多人第一次理解 vLLM，都会先记住一个关键词：**PagedAttention**。这当然没错。vLLM 之所以能在大模型推理系统里建立辨识度，一个重要原因就是它没有沿用“每个请求持有一段连续 KV cache”的传统思路，而是把 KV 存储拆成了按 block/page 管理的布局。但如果只把 vLLM 理解成一种 attention 访存优化，实际上会错过它在当前主线里更重要的一层：**KV cache 已经不再只是 attention 的附属缓存，而是被实现为一个由 scheduler、KV manager、block pool 和 worker metadata 共同维护的运行时资源层。**

这也是今天重新理解 vLLM 的必要性所在。在线推理系统面对的早已不是“要不要缓存 K/V”这种问题，而是：在变长请求、持续 decode、前缀复用、甚至远端 KV 回填的场景下，系统如何决定哪些请求可以继续推进，哪些 block 应该被复用，哪些 block 应该被释放，以及这些逻辑如何最终落到 attention backend 的执行路径上。基于当前 vLLM v1 主线源码来看，PagedAttention 当然仍然重要，但它更像是这套 runtime 抽象成立的前提，而不是终点。

本文尝试基于当前主线源码，把这套 KV cache 管理机制拆开来看。本文分析基于本地对齐的 vLLM 仓库主线版本：`main@48e376a007173910330a8c83f53474b21e4279c0`（本地最新提交时间：2026-03-05）。核心判断很简单：**vLLM 的关键创新，不只是把 KV cache 做成分页访问，而是把 KV cache 真正纳入了推理 runtime 的资源管理主链路。**

## 一、问题并不是“缓存 K/V”，而是如何在在线推理里管理它们

如果从离线推理或者单请求执行的视角看，KV cache 很容易被理解成模型执行过程中的中间状态：prompt prefill 阶段算出 K/V，decode 阶段把后续 token 追加进去，注意力计算时直接读取即可。但一旦进入在线 serving，这个看起来简单的对象会立刻暴露出系统层面的复杂性。

第一个问题是**请求长度不一致**。不同请求的 prompt 长度不同、生成长度不同，而且到达时间也不同。如果仍然用“每个请求分配一段连续 KV 区域”的方式来处理，就会很快遇到两个矛盾：要么为了安全而做较大预留，造成显存浪费；要么频繁地扩容和整理，带来碎片化和调度复杂度。

第二个问题是**请求不是一次性执行完成，而是在 decode 中持续增长**。在这种模式下，KV cache 不是“先分好再使用”的静态对象，而更像是一组需要不断追加、检查和回收的动态资源。系统每推进一步，都要重新面对一个问题：这个请求现在还能不能继续长出新的 token？如果能，所需的 KV blocks 从哪里来？

第三个问题是**前缀复用改变了 KV cache 的所有权语义**。一旦系统支持 prefix caching，那么一段已经算好的 KV block 就不再必然只属于某个 request。多个请求可能命中同一前缀，进而共享同一批 block。此时，KV cache 已不再只是“每个请求各自持有的一段内存”，而开始具备了共享资源的属性。

当前 vLLM 主线正是围绕这些问题组织起来的。一个直接的信号是，在 scheduler 的主循环中，请求是否能够被推进，并不是先定下来再让 KV cache 被动配合；相反，scheduler 会先查询本地 prefix cache 命中情况，再结合外部已计算 tokens（如果启用 connector/offload），最后调用 `allocate_slots(...)` 判断是否还能为该请求分配足够的 block。也就是说，**请求可调度性本身就包含了 KV 资源可分配性**。这已经说明，vLLM 当前主线处理的不是单纯的“缓存命中优化”，而是一个面向在线推理的动态资源管理问题。

## 二、KVCacheManager：调度器看到的不是底层 block，而是一套资源接口

从当前主线的接口设计来看，vLLM 并没有让 scheduler 直接操作底层 block 池，而是在中间放了一层清晰的 `KVCacheManager`。这层设计很关键，因为它表明 KV cache 在系统中的地位已经发生了变化：它不再只是底层内存细节，而是 scheduler 需要显式调用和理解的一组 runtime 操作。

这一点在源码中体现得很直接。`KVCacheManager` 文件开头定义了 `KVCacheBlocks`，注释里明确写到，它是 **Scheduler 和 KVCacheManager 之间的接口，用来隐藏 KV manager 的内部数据结构**。换句话说，scheduler 并不直接知道底层如何维护 block pool、hash 表和 ref count，它拿到的是一个更高层的资源视图：哪些 blocks 已经命中前缀缓存、哪些 blocks 是这次新分配的、当前有哪些公共前缀 blocks 可以暴露给后续执行路径。

更重要的是，`KVCacheManager` 对外暴露的方法本身就已经是“运行时资源管理 API”，而不是传统意义上的缓存容器接口。比如：

- `get_computed_blocks(request)`：用于查找该请求已经命中的前缀 blocks；
- `allocate_slots(...)`：用于为当前 step 中的新增 tokens、外部已计算 tokens、甚至 speculative lookahead tokens 申请 slot 和 block；
- `free(request)`：在请求结束时释放其持有的 blocks；
- `get_num_common_prefix_blocks(...)`：统计运行中的请求之间共有多少公共前缀 blocks。

这几组接口拼在一起时，含义其实很明确：**scheduler 面对的不是一块“KV 内存”，而是一套用于判断可执行性、申请资源、提交缓存和释放资源的 runtime service**。

其中最值得细读的是 `allocate_slots(...)`。它的注释已经非常接近一段系统设计说明：一次 slot 分配并不是简单“新增若干 block”，而是包含三个阶段。第一步，先释放那些已经不再参与注意力计算的 blocks，并检查当前 free blocks 是否足够；第二步，处理 prefix-cached tokens 和 external computed tokens，把已经算过的前缀部分接入当前 request 的 block 视图中；第三步，再为真正需要在本轮计算的新 tokens 和 lookahead tokens 分配新的 block slots。最后，只有那些已经成为 full block、且满足可提交条件的部分，才会进入 prefix cache。

这意味着，当前 vLLM 主线里的 KV slot 分配更像一套带有回收、复用、补分配和提交动作的运行时事务，而不是一次静态内存扩容。它面向的对象也不是单纯的“KV 张量”，而是一组随 request 生命周期流转的 block 引用。

## 三、BlockPool：prefix cache、共享 block 和 free blocks 共用同一个底层资源池

如果说 `KVCacheManager` 是 scheduler 看到的接口层，那么 `BlockPool` 才是当前主线中 KV 资源真正落地的底层结构。很多人讨论 vLLM 时，会把“prefix cache”“block allocator”“eviction 逻辑”分别看成不同模块，但从源码结构看，它们实际上共享着同一套底层资源池。

`BlockPool` 初始化时维护了四类核心状态：

1. 一组全局的 `blocks`；
2. 一个 `free_block_queue`，用于按可回收顺序维护空闲 blocks；
3. 一个 `cached_block_hash_to_block` 哈希映射，用于 prefix caching 命中查找；
4. 一个特殊的 `null_block`，用于滑窗等场景下占位。

这里最关键的理解是：**这些不是彼此隔离的资源，而是同一组 block 在不同生命周期阶段下的不同视图**。一个 block 最开始可能只是 free queue 里的空闲块；被分配后，它进入某个 request 的 block 列表并持有引用计数；当它被填满后，又可能获得 block hash 并进入 prefix cache 哈希索引；如果之后被其他 request 命中，这个 block 会通过 `touch()` 增加引用计数，成为共享 block；等所有引用都释放掉后，它又会重新回到 free queue，成为可被再次分配的资源。

`get_new_blocks()` 能很好体现这套机制的统一性。它不是简单从 free list 取块，而是先检查这些 block 是否仍带有缓存 hash；如果有，就先执行 `_maybe_evict_cached_block()`，把它从 prefix cache 哈希表中移除并清理 hash 元数据，然后再把它作为“新分配 block”交给请求使用。这说明“可被复用的 cached block”和“可被重新分配的 free block”并不是两套池子，而是同一个 block 池中的不同状态。

另一方面，`cache_full_blocks()` 又说明 prefix caching 的复用单位并不是任意 token 片段，而是 **full block**。只有当 block 被填满、且对应 hash 已经可用时，BlockPool 才会把它插入 `cached_block_hash_to_block`。这意味着 vLLM 当前主线里的 prefix cache 语义，是建立在 block-size 对齐和 full-block 完整性的基础上的，而不是任意 token 粒度的自由命中。事实上，`get_computed_blocks()` 的逻辑也明确体现了这一点：即使整个 prompt 理论上都能命中 cache，最后一个 token 仍可能需要重算以获取 logits，这让“prefix hit”从一开始就是一种带有工程约束的复用机制，而不是抽象层面无限细粒度的共享。

这种共享真正成立的关键，是 block 上的 `ref_cnt`。当某个 prefix hit 被另一请求复用时，BlockPool 会调用 `touch()`：如果该 block 当前处于 free queue 中，就先把它移出，再把引用计数加一；当请求结束或窗口外 block 被释放时，`free_blocks()` 则会减少 `ref_cnt`，只有 ref count 降为 0 且它不是 null block 时，block 才会真正重新进入 free queue。换句话说，**prefix reuse 的本质不是“复制一段 KV”，而是共享已有 block，并把共享关系纳入统一的生命周期管理**。

从这个角度看，当前主线里的 prefix cache 并不是附着在 KV cache 外部的一层额外优化，而是 BlockPool 生命周期管理的一种状态投影。它把复用、回收和重分配统一在同一套 block 状态机里，这正是 vLLM 作为 runtime 设计最有价值的地方之一。

## 四、为什么还要有 coordinator 和 single-type managers：KV 管理语义会随 attention 类型变化

如果 vLLM 面对的只是单一的 full attention 模型，那么理论上一个统一的 KV manager 也许就足够了。但当前主线并没有这么做，而是在 `KVCacheManager` 之下又引入了 `KVCacheCoordinator` 和一组 `SingleTypeKVCacheManager`。这说明另一个关键事实：**不同 attention 语义下，KV cache 的管理方式并不相同。**

`KVCacheCoordinator` 的职责是聚合多个 KV cache group，并为每个 group 创建对应的 single-type manager。它自己持有共享的 `BlockPool`，但具体到 prefix hit 的判断、skipped tokens 的处理、common prefix 的统计以及 block 释放策略，则由各自的 single-type manager 来决定。这层结构背后的设计意图很明确：vLLM 不再假设所有 KV cache 都服从一种统一语义，而是允许不同 attention spec 在共享 block 池上实现差异化管理。

这一点在 full attention 和 sliding window 的对比中最明显。对于 full attention，前缀命中基本可以按 block 顺序前向匹配，公共前缀 blocks 也可以相对直接地通过 ref count 判断：如果某个 block 的 `ref_cnt` 等于当前持有已分配 KV 的 request 数量，那么它就可以被视为“公共前缀块”。但在 sliding window 场景下，逻辑会明显变化。因为窗口外的历史 token 已不再参与注意力计算，系统不能继续无条件保留所有早期 blocks。于是 `remove_skipped_blocks()` 会根据窗口位置主动把不再需要的 blocks 替换成 `null_block`，并释放真实 blocks 回到池中。

这意味着，在当前主线里，KV cache 是否继续保留，不只取决于请求是否结束，还取决于 attention 语义本身。对于 full attention，请求未结束时一般不会主动回收前缀 blocks；而对于 sliding window，窗口滑出之后，历史 blocks 会被显式地标记为空位并释放。**KV 管理因此天然成为 attention-aware 的资源管理问题，而不是一个 attention 外部的通用缓存模块。**

从架构角度看，这也是 `KVCacheCoordinator + SingleTypeKVCacheManager` 最重要的意义：它把“统一资源池”和“差异化注意力语义”这两件事分层组织起来。统一的 block pool 提供底层资源基础，而 single-type managers 则定义不同 attention 类型下 block 生命周期的具体规则。

## 五、一个 request 的 KV 生命周期：它持有的不是连续内存，而是一组流动中的 block 引用

理解了接口层和 block pool 之后，再回头看一个 request 的生命周期，会更容易抓住当前主线的本质。对于 vLLM 来说，请求持有的并不是一段从头到尾连续存在的 KV 内存，而是一组会不断变化、可能共享、也可能被回收的 block 引用。

一个 request 初次进入 waiting 队列时，scheduler 首先会尝试通过 `get_computed_blocks()` 查询它的本地 prefix hits。如果启用了 connector 或外部 KV 传输机制，还会再加上 `external computed tokens`。这一步得到的并不是最终可执行状态，而只是当前 request 已有多少前缀可以直接复用，以及这些前缀对应哪些 blocks。

接下来进入 `allocate_slots(...)`。在这一阶段，系统会先清理那些已经不再需要参与注意力的旧 blocks，比如滑窗之外的部分；然后把 prefix hit 对应的 blocks 接入 request 的 block 列表，并在必要时通过 `touch()` 确保这些被命中的 blocks 不会在本轮被当作 eviction candidate；如果还存在外部已计算 tokens，则还要为它们补分配相应 blocks；最后，系统才为真正要在本轮计算的新 tokens 和 speculative lookahead tokens 分配新的 block slots。

当本轮执行结束后，已变成 full block 的部分才会被提交到 prefix cache。这里有两个值得强调的事实。第一，提交缓存是有时机约束的，不是 request 每多一个 token 就立刻进入 prefix cache；第二，prefix cache 的命中并不会把 block 从系统中“拷贝”一份给新请求，而是让新请求共享已有 block，并通过 ref count 纳入同一生命周期。

等到请求结束，或者在滑窗场景下部分历史 blocks 被移出窗口时，系统再通过 `free()` 或 `remove_skipped_blocks()` 释放它们。真正返回 free queue 的条件并不是“该请求不用了”，而是“所有引用这个 block 的 request 都已经释放了它”。所以从生命周期角度看，当前主线里的 request 并不是“拥有一段 KV 内存”，而是“持有一组随时间流转的 block 引用关系”。

这种视角很重要，因为它会直接改变我们对 KV cache 的理解方式。**它不是一个 per-request static buffer，而更像是 runtime 在每一步里不断重写的一份 block ownership graph。**

## 六、调度器为什么必须理解 KV：系统吞吐边界由 scheduler 与 KV manager 联合定义

到了这一步，可以看出为什么当前主线里的 scheduler 不能把 KV cache 当作透明实现细节。对于在线推理系统来说，请求能否进入本轮执行，并不只取决于 token budget、batch size 或 compute 预算，还取决于 KV resources 是否可满足。

在 `schedule()` 主循环里，这种耦合体现得非常直接。scheduler 会逐个检查 waiting requests，计算它们当前的本地和外部 computed tokens，再估算本轮需要新增多少 tokens，然后调用 `allocate_slots(...)`。如果这个调用返回 `None`，它并不表示程序错误，而是表示在当前资源状态下，这个请求本轮无法安全推进。此时 scheduler 只能选择跳过、等待、或在后续 step 再试。

这意味着，scheduler 决定的并不只是“谁更重要、先跑谁”，还包括“谁在当前 KV 资源状态下是可执行的”。从系统角度说，**scheduler 与 KV manager 共同定义了吞吐边界**：前者负责选择候选请求，后者负责判断这些候选请求在当前 block 资源和生命周期约束下能否真正进入执行。

这一层耦合还体现在 `common prefix blocks` 的输出上。当前主线里，scheduler 在完成一次调度后，会额外计算 running requests 之间共有多少公共前缀 blocks，并把它作为 `SchedulerOutput` 的一部分输出给下游。这说明 scheduler 输出的不再只是“本轮谁要跑、每个请求跑多少 token”，还包括一部分和 batch 内前缀结构有关的执行信息。KV cache 因此已经从“模型执行后的副产物”变成了调度器在组织 batch 时必须实时感知的系统状态。

从这个角度再看 vLLM，就会发现把它简单理解成一个 attention 优化库其实是远远不够的。它更接近一个推理 runtime：scheduler、KV manager、block pool 和 worker/backend 在同一套资源抽象下协同工作，而 KV cache 正是这套协同关系的核心载体之一。

## 七、分页式 KV 如何真正进入执行：block table 和 slot mapping 是关键桥梁

讨论 PagedAttention 时，很多文章只停留在“KV 被拆成 blocks，逻辑连续但物理不连续”的概念层面。但在当前主线源码里，这件事真正落到执行面，依赖的是 worker 侧对 `block_table` 和 `slot_mapping` 的维护。

从 request 视角看，它持有的是一组逻辑 block ids。真正进入 GPU 执行之前，worker 会先把这些 block ids 写入 per-request 的 block table；到了 batch 组织阶段，再根据本轮 batch 的 request 映射，把对应 block tables gather 成真正的输入视图。随后，`compute_slot_mappings()` 根据每个 token 在逻辑序列中的 position，先算出它落在哪个 block index 上，再通过 block table 查到这个 block 的物理 block id，最后进一步计算出对应的 slot id。

这一步的意义非常大。它说明对 attention backend 来说，请求并不是“携带一段连续 KV cache 张量”进入执行，而是“携带一份如何从 block table 中找到对应物理 KV slots 的映射说明”。换句话说，**分页式 KV 真正成立的关键，不是 block 的存在本身，而是 block table 和 slot mapping 把逻辑序列重新解释成了一套可执行的物理访问规则。**

当前 v1 的 `CommonAttentionMetadata` 也印证了这一点：backend 接收到的 batch 级元数据中，明确包含 `block_table_tensor` 和 `slot_mapping`。这表明分页化并不是 scheduler 侧的抽象游戏，而是 attention backend 的真实执行接口。正是因为有了这层映射，vLLM 才能在不要求每条序列物理连续存储的前提下，仍然为 backend 构造出可用的访问模式。

从这一步再回看，PagedAttention 的真正意义也会更清楚：它不是一个单独的 kernel 名词，而是一种让“逻辑连续”和“物理非连续”能够同时成立的系统级接口设计。

## 八、KV 存储的本体到底是什么：从物理页到请求视图的两层结构

理解 vLLM 的 KV cache，一个很容易混淆的地方是：调度层里看到的 `KVCacheBlocks`、`KVCacheBlock` 和 worker 执行时真正读写的 KV cache，并不是同一个层次的对象。前者更接近**运行时管理视图**，后者才是**真实的存储本体**。如果不先把这两层区分开，后面很容易把 block 管理误解成 KV 存储本身。

从当前 vLLM v1 主线源码看，真正的 KV 存储本体并不是 `KVCacheBlocks` 这样的 Python 容器，而是 worker 侧为 attention layer 分配的一组 **GPU Tensor**。在 `init_kv_cache()` 的路径里，系统会先按 `KVCacheConfig` 为每个 layer（或共享同一存储的一组 layers）分配原始字节缓冲区，再根据具体 attention backend 的 `get_kv_cache_shape()` 把这块连续内存重新解释成分页化的 KV cache 结构。对于常见的 FlashAttention backend，这个结构可以概括为：

```text
[2, num_blocks, block_size, num_kv_heads, head_size]
```

其中第一个维度的 `2` 对应 K 和 V，`num_blocks` 表示物理页/物理块数量，`block_size` 表示每个 block 容纳多少 token，后两维则是 KV head 数与 head size。也就是说，**KV cache 的真实本体，本质上是“按物理 block/page 切分的一块 layer 级 GPU Tensor”**。

和这层真实存储相对的，是 Python 层的 block 元数据对象。`KVCacheBlock` 本身并不保存任何 K/V 张量内容，它只维护：

- `block_id`
- `ref_cnt`
- `block_hash`
- free-list 链表指针
- `is_null`

这意味着 `KVCacheBlock` 的角色不是“存储一个 block 的 KV 数据”，而是**描述某个物理 block/page 当前的管理状态**。真正的数据仍然在底层 GPU tensor 中，而 `block_id` 才是二者之间最关键的关联键：一个 `KVCacheBlock(block_id=i)`，本质上对应着底层 KV tensor 的第 `i` 个物理页。

基于这层物理页元数据，vLLM 再在 request 维度上维护上层视图。真正持久的 request→block 映射保存在 `SingleTypeKVCacheManager.req_to_blocks` 中，也就是每个 request 当前关联着一串 `KVCacheBlock`。随后 `KVCacheCoordinator` 会把不同 KV cache group 的结果聚合起来，`KVCacheManager` 再将其包装为 `KVCacheBlocks`。因此，`KVCacheBlocks` 不是 KV 存储本体，而更像是：**某个 request 当前持有哪些 block 的 grouped view**。它服务的是 scheduler 和 manager 之间的接口边界，而不是执行层。

但 request 视图仍然不能直接驱动 attention backend。真正把“请求持有哪些 blocks”翻译成“模型该去哪里读写 KV”的，是 worker 侧的 `block_table` 和 `slot_mapping`。执行前，worker 会先把 request 的 block ids 写入 per-request 的 `block_table`；然后在 batch 组织阶段，将这些 block tables 按当前调度批次 gather 成本轮执行的视图。接下来，`compute_slot_mappings()` 会根据 token 在逻辑序列中的 position，先找到它所属的逻辑 block index，再通过 `block_table` 找到对应的 physical block id，最后计算出具体的 slot id。于是，一条完整的映射链就形成了：

```text
request
→ 持有一串 KVCacheBlock
→ 抽象成 KVCacheBlocks 视图
→ 提取 block ids 写入 block_table
→ token position 通过 block_table 映射成 slot_mapping
→ attention backend 依据 slot_mapping 读写底层 GPU kv_cache tensor
```

这条链路解释了一个非常关键的事实：**vLLM 的 block/page 抽象并不是独立于执行面的管理概念，而是连接 request 视图与物理 KV 存储的中间层。** `KVCacheBlocks` 让 scheduler 理解“当前请求关联了哪些 block”，而 `block_table + slot_mapping` 则进一步把这种逻辑关联落实成底层 GPU tensor 上的物理访问规则。

从这个角度看，PagedAttention 的意义也就更具体了。它不只是把 KV cache “分页化”这么简单，而是要求系统同时维护两套结构：一套是面向 request 和调度器的 block 生命周期视图，另一套是面向 backend 执行的物理页访问映射。前者回答“哪些 block 归谁管理、是否可复用、何时释放”，后者回答“这一批 token 的 K/V 应该写到哪一页、读自哪一页”。而真正的 KV 存储本体，则始终存在于 worker 侧那块被重新解释为 page/block layout 的 GPU tensor 中。

如果把这层关系说得更压缩一点，可以概括为：

> **在当前 vLLM 主线中，底层本体是 layer 级分页化 GPU KV tensor；`KVCacheBlock` 管理物理页元数据；`KVCacheBlocks` 提供 request 级 grouped view；`block_table + slot_mapping` 则把这种上层视图翻译成 attention backend 可执行的物理访问。**

## 九、从顶层管理到显存分配：KV cache 初始化、NIXL 注册与 zero-copy 边界

如果只看 `torch.zeros(...)` 那一行，很容易以为 vLLM 的 KV cache 初始化只是一个设备内分配动作。但把调用链往上追，会发现它实际上跨越了 **Engine、Worker、Runner 和 KV connector** 四层。这条链之所以重要，不只是因为它解释了“KV cache 什么时候分配”，还因为它同时揭示了：**为什么 vLLM 要做 engine / worker / runner 分层，NIXL 又是在哪个时间点接入这条链，并实现近似 zero-copy 的 KV 传输路径。**

从当前主线源码看，最上层入口在 `EngineCore.__init__()`。engine 启动后，会先进入 `_initialize_kv_caches()`：读取模型的 `kv_cache_specs`，profiling 可用显存，生成各 worker 使用的 `kv_cache_configs`，再生成 scheduler 看到的 `scheduler_kv_cache_config`。这一步的本质不是“分配显存”，而是**建立全局 KV 资源模型**：系统总共允许多少 blocks、按什么 group 组织、scheduler 后续该如何理解这些资源。也正因如此，scheduler 必须等这一步完成后才能创建，因为它要消费的不是抽象的“将来某处会有 KV cache”，而是一份已经定型的 KV 资源配置。

随后，engine 通过 executor 把 `kv_cache_configs` 下发给各 worker。到了 `WorkerBase.initialize_from_config()`，系统会按 `global_rank` 选出当前 worker 对应的那份配置，再进入 `GPUWorker.initialize_from_config()`。这一层的职责不是直接分配 tensor，而是完成**本地执行端的准备工作**。最关键的一步就是：在真正初始化 KV cache 之前，先调用 `ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)`。如果当前实例启用了 KV transfer，并且底层 connector 选的是 NIXL，那么这里会通过 `KVConnectorFactory.create_connector(...)` 创建 worker 侧的 NIXL connector，并最终进入 `NixlConnectorWorker.__init__()`，在其中构造 `self.nixl_wrapper = NixlWrapper(...)`。换句话说，**NIXL agent / wrapper 的建立发生在 worker 边界，而且先于 KV tensor 分配。**

真正的显存分配发生在 runner 层。`GPUWorker.initialize_from_config()` 接着会调用 `self.model_runner.initialize_kv_cache(kv_cache_config)`。在 `GPUModelRunner.initialize_kv_cache()` 中，runner 先构造 `BlockTables`，初始化 attention backend 和 metadata builder，然后进入 `init_kv_cache(...)`。再往下，`init_kv_cache()` 会调用 `_allocate_kv_cache(kv_cache_config, device)`，而 `_allocate_kv_cache()` 里的 `torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=device)`，才是原始 KV memory 的真实分配点。紧接着，`_reshape_kv_cache()` 又会根据各层的 `KVCacheSpec` 和 backend 的 `get_kv_cache_shape()`，把这些原始 byte buffer 重新解释成 layer 级、按 page/block 布局组织的 KV tensors。也就是说，**Engine 决定“怎么配”，Worker 负责“在哪个执行端准备好”，Runner 才真正把配置落成设备内的 KV tensor。**

NIXL 对 KV cache 的“注册”则发生在这个显存分配之后。`GPUModelRunner.initialize_kv_cache()` 在拿到 `kv_caches_dict` 后，会调用 `get_kv_connector(self.vllm_config, kv_caches_dict)`；如果当前进程已经有可用的 KV transfer group，就会构造 `ActiveKVConnector`，并立即执行 `self.kv_connector.register_kv_caches(kv_caches_dict)`。对于 NIXL，这一步会进入 `NixlConnectorWorker.register_kv_caches(...)`：遍历每个 layer 的 cache tensor，收集 `cache.data_ptr()`、tensor byte size、device id 等底层信息，调用 `self.nixl_wrapper.get_reg_descs(...)` 生成 memory descriptors，再通过 `self.nixl_wrapper.register_memory(...)` 把这些真实 KV tensors 注册给 NIXL。只有在这一步之后，NIXL 才真正获得了“这些 KV cache 位于哪些 device address、每个 region 有多大、后续该如何准备 xfer descriptor”的能力。

这也解释了 vLLM 在这里为什么可以做到接近 **zero-copy** 的传输路径。至少从当前 Python 调用链和 NIXL connector 的注册逻辑看，NIXL 不是把 KV block 先序列化成一块中间 Python buffer 再转发，而是直接围绕已分配好的 KV tensor 做 descriptor 注册：`register_kv_caches()` 里拿的是 `cache.data_ptr()` 和 tensor 大小，后续 `register_local_xfer_handler()`、`add_remote_agent()`、`prep_xfer_dlist()` 也都是围绕这些已注册内存区域组织块级传输描述。换句话说，**vLLM 当前 NIXL 路径的关键优势，不在于“少了一次函数调用”，而在于它把 KV cache 作为已注册的设备内存区域暴露给传输层，后续 block 读写直接围绕这些 region 展开，而不是先额外打包成独立复制缓冲区。**

需要强调的是，这里说的 zero-copy 更准确地应理解为：**在 vLLM 当前 NIXL connector 设计里，KV 传输建立在已注册的原始 KV memory region 之上，而不是建立在额外 staging buffer 之上。** 是否做到硬件层面绝对零拷贝，还会受后端、host buffer、memory type、平台能力等条件影响；但从 vLLM 的 Python 代码路径看，NIXL 设计追求的确实是“直接围绕原始 KV region 进行 descriptor 化和传输准备”。

那为什么不能直接用 NCCL 做同样的事？更准确地说，在 **vLLM 当前这类 KV connector 设计** 里，NCCL 并不提供一套等价的“已注册任意 KV 内存区域 + 远端 agent metadata + 按 block 准备传输描述符”的抽象。NIXL connector 这条链路依赖的不是单纯的 GPU 间数据搬运，而是：先用 `register_memory()` 注册本地 KV region，再通过 handshake 交换 agent metadata 和 base addresses，然后为远端 block 准备 xfer descriptors，最后按 request/block 粒度发起异步加载。NCCL 非常擅长做 rank 间 collective / point-to-point tensor 通信，但它在 vLLM 当前 Python 侧并没有被组织成这种“以注册内存区域为中心的 KV block connector”模型。因此，更稳妥的表述不是“NCCL 绝对做不到 zero-copy”，而是：**在 vLLM 当前主线里，NCCL 并没有提供与 NIXL connector 等价的 memory registration + remote descriptor + block-level KV pull 语义，所以这条 zero-copy 风格的 KV transfer 路径目前是围绕 NIXL 建起来的。**

如果把这整条链压缩成一句话，可以概括为：

> **Engine 先决定全局 KV 资源模型，Worker 先把 NIXL 这类 transfer connector 建起来，Runner 再真正分配并 reshape KV tensors，随后 connector 才把这些已分配的 KV memory region 注册出去。正是这种“先配置、再建环境、再分配 tensor、最后注册 region”的分层顺序，使 vLLM 能把 KV cache 同时纳入调度系统、显存布局和远端传输路径。**

## 十、Prefill 与 Decode 阶段里，block_table 和 slot_mapping 分别怎么工作

把 KV 存储本体和 request 视图区分清楚之后，下一个自然问题就是：这些映射结构究竟在执行时怎么被用到？从当前主线实现看，`block_table` 和 `slot_mapping` 虽然总是成对出现，但它们在 prefill 与 decode 阶段承担的角色并不完全相同。前者更偏向“描述一个 request 目前持有哪些物理 blocks”，后者则更接近“本轮参与计算的这些 token 应该写到哪些物理位置”。

先看 prefill。对于进入 prefill 的 request，worker 会先根据当前 `num_computed_tokens` 和本轮 `query_len` 构造输入 token、position 和 `seq_lens`。在这个阶段，新 token 的 position 是连续增长的：如果一个 request 之前已经有若干 computed tokens，那么本轮 query token 的位置就会从 `num_computed_tokens` 开始顺延。随后，worker 根据这些 position、request 当前的 block ids 以及 block size，计算每个 token 应该落在哪个逻辑 block，再通过 `block_table` 找到对应的 physical block id，最后得到具体的 `slot_mapping`。这一步的结果是：虽然请求视角上只是“给这个 request 追加了一段新 token”，但执行侧已经把这段逻辑追加翻译成了“把这些 key/value 写进哪些物理 page 的哪些 slot”。

这也是为什么 cache update 路径真正依赖的是 `slot_mapping`。在 FlashAttention 等 backend 中，写入 KV cache 时不会去遍历 request 的 Python block 列表，而是直接拿到 `kv_cache` 和 `slot_mapping`，调用底层 reshape-and-cache 类算子，把本轮生成出的 key/value 写进对应 slot。换句话说，**prefill 阶段的核心不是“先生成一个连续 KV 再拷贝进去”，而是边生成边按 slot_mapping 直接写入分页化 KV 存储本体。**

再看 decode。decode 阶段通常每个 request 每步只新增很少的 token，常见情况下甚至就是 1 个 token，但它需要读取的是“历史 KV + 当前 query”的组合。因此在 decode 中，`block_table` 的作用会更突出：backend 需要知道当前 request 逻辑上的整段上下文，对应到底层有哪些 physical blocks。与此同时，新增 token 仍然需要通过 `slot_mapping` 被写入正确位置。也就是说，decode 阶段实际上同时依赖两种映射：一方面通过 `block_table` 读取已有历史页，另一方面通过 `slot_mapping` 把新 token 的 K/V 追加进新的物理 slot。

从 `FlashAttentionMetadataBuilder.build()` 的结构也能看出这一点。构建 attention metadata 时，`CommonAttentionMetadata` 会同时携带 `block_table_tensor` 和 `slot_mapping`，然后 backend 再据此生成具体的执行 metadata。对于普通情况，backend 直接基于 `seq_lens`、`query_start_loc`、`block_table` 等信息组织 attention；而当存在公共前缀时，又会进一步把 `common_prefix_len`、`prefix_kv_lens` 等信息纳入调度元数据，用于 cascade / prefix-aware 的执行路径。这意味着 `block_table` 不只是 cache update 的辅助索引，而是 decode 阶段 attention 读取历史 KV 的关键输入。

如果把 prefill 和 decode 的差异压缩成一句话，可以这样理解：

> **prefill 更强调“本轮一批新 token 如何通过 slot_mapping 写入 paged KV cache”，而 decode 更强调“如何基于 block_table 读取已有历史页，同时再用 slot_mapping 追加新 token”。**

这也是 vLLM 当前分页化 KV 抽象真正成立的关键。系统并不是把“历史 KV 存储”和“新 token 写入”拆成两套完全不同的数据结构，而是让二者都围绕同一套物理 page/block 布局组织：`block_table` 描述逻辑上下文到物理页的映射，`slot_mapping` 描述当前批次 token 到物理写入位置的映射。一个偏读路径，一个偏写路径，但最终都落在同一个底层 KV tensor 上。

## 十一、Prefix cache 命中是如何查找的：为什么它是 block 级复用，而不是 token 级复用

到这里，其实还剩下一个很关键的问题：当我们说一个 request “命中了 prefix cache” 时，vLLM 当前主线到底在查什么？如果把这一步想象成“拿 prompt 字符串做最长前缀匹配”，就会低估它的工程约束。实际实现里，prefix cache 查找不是 token 级任意命中，而是围绕 **block hash、block size 对齐以及 attention-type 特定规则** 展开的。

从 `KVCacheManager.get_computed_blocks()` 出发，系统会先为 request 拿到预先计算好的 `block_hashes`，然后把“最大可命中长度”设为 `prompt_length - 1`。这个 `-1` 并不是随意的保守处理，而是因为即使所有 prompt token 都命中缓存，系统仍然通常需要重新计算最后一个 token 来获得 logits。接下来，真正的查找工作会进入 `KVCacheCoordinator.find_longest_cache_hit()`，再由不同的 single-type manager 分别判断在各自 attention 语义下最长可接受的 cache hit 前缀。

对于 full attention，这个过程相对直接。`FullAttentionManager.find_longest_cache_hit()` 会沿着 block hash 链从左到右扫描：只要当前 block hash 能在 `BlockPool` 的 cached-block 映射中找到，对应 block 就被视为可复用；一旦某个 block hash miss，后续更长的 block 链也就不再可能命中，于是扫描停止。这个逻辑很重要，因为它说明当前主线的 prefix cache 命中遵循的是一种“按 block 链前缀增长”的规则，而不是 token 粒度上的稀疏匹配。

更进一步地说，命中的最小复用单位不是 token，而是 **full block**。这是因为只有 full block 才会在 `cache_full_blocks()` 中生成 hash 并进入 prefix cache；而查找时 `find_longest_cache_hit()` 也只在 block hash 链上工作。这意味着，当前主线里的 prefix reuse 语义天然受到 block size 的约束：如果一个前缀还没有填满一个完整 block，它通常就不会作为稳定可复用单元进入 prefix cache。对于混合 attention 组，`KVCacheCoordinator` 还会进一步要求 cache hit 长度满足各组 block size 的最小公倍数对齐，以确保所有组都不会出现 partial block hit。这一点也再次说明，**当前主线并不支持真正意义上的 partial-block cache hit。**

当 attention 类型变复杂时，prefix hit 的判定也会变化。以 sliding window 为例，它不是简单地从左到右找最长共同前缀，而是要考虑窗口内仍然有效的连续块。因此，cache hit 的查找方式会受到窗口语义约束，而不再等价于 full attention 下的普通前缀链匹配。这也是为什么 vLLM 需要让不同 attention type 通过各自的 manager 来实现 `find_longest_cache_hit()`：prefix caching 并不是一个独立于 attention 语义的纯缓存层。

从文章的角度，这一实现细节值得专门强调，因为它能帮读者避免一个常见误解：vLLM 的 prefix cache 不是“对任意历史 token 做细粒度重用”，而是**对已经稳定成 full block 的前缀页做可验证、可管理、可共享的复用**。换句话说，当前主线真正缓存的不是“前缀本身”，而是“前缀对应的完整物理页及其 hash 身份”。这也解释了为什么前面一再强调 block/page 才是 vLLM KV 管理的基本资源单位——因为连 prefix cache 命中这件事，本质上也是围绕 block 在运转。

## 十二、Prefix caching 的真正意义：它改变了 block 的生命周期语义

在很多介绍里，prefix caching 被讲成一种“提升命中率、减少 prefill 重算”的优化功能。这种说法当然没有错，但如果放在当前主线的上下文中，它其实太浅了。更准确地说，**prefix caching 的意义不只是少算几段 prompt，而是让一部分 KV blocks 从请求私有状态变成了可共享的系统资源。**

首先，prefix caching 不是任意 token 级别的自由复用，而是建立在 full block 命中和 block-size 对齐约束上的。这意味着，能否命中、命中多少、哪些部分需要重算，都受当前 block 布局和对齐语义约束。vLLM 并不是把“整个前缀字符串”抽象地判成可复用，而是把已经稳定成 full block 的那部分 KV 纳入共享语义。

其次，一旦命中发生，新的 request 并不会复制一份已有 KV，而是通过 `touch()` 共享现有 block。这会直接改变 block 的生命周期：它不再只跟随单个 request 的开始和结束流转，而是可能跨多个 request 存活、被多次引用、在某些请求结束后依旧保留。

最后，当前主线里的“公共前缀”已经不只是 cache hit 的统计项。scheduler 会把 `num_common_prefix_blocks` 传给后续执行路径，而 attention backend（至少在部分 backend 中）已经开始基于 `common_prefix_len` 等信息组织更进一步的 batch 优化。这说明 prefix reuse 已经不只是“少算一点 prefill”，而正在进入 execution-aware reuse 的阶段。

因此，真正值得强调的不是“vLLM 支持 prefix caching”，而是：**当前 vLLM 主线通过 prefix caching 引入了共享 block 生命周期，并开始把这层共享关系继续传导到 batch 级执行优化。** 一旦从这个视角出发，ref count、common prefix blocks、cache commit 时机等设计就不再是零散技巧，而会显得内在连贯。

## 十三、当前主线的边界：本地 block runtime 已经成型，但不是终点

尽管当前 vLLM 主线已经把单机本地的 KV cache 管理做成了一套系统级 runtime，但这并不意味着问题已经被彻底解决。更准确的说法是：**vLLM 已经把“本地显存中的 block 级 KV 管理”系统化了，但更复杂的跨介质和远端管理仍在继续演进。**

一方面，这套设计本身就带来了额外复杂度。比如，系统需要维护 block hash、free queue、引用计数、公共前缀统计，还要让 backend 理解 block table 和 slot mapping。这些都是为了换取动态请求下更好的复用与调度能力，但它们也意味着实现不再是一个简单的 contiguous KV buffer。

另一方面，当前主线其实已经显式暴露出向更复杂 KV 体系延展的接口痕迹。`allocate_slots(...)` 里存在 `num_external_computed_tokens`、`delay_cache_blocks` 等参数；scheduler 中也存在 connector、remote KV、异步接收完成后再更新 request 状态的逻辑。这说明 vLLM 的 KV 管理抽象已经不再局限于“本地 GPU prefix cache”，而是在逐步吸纳远端已计算 KV、异步传输、offload 等更大范围的资源流动。

但对这篇文章来说，最重要的是不要把这些边界问题和主线混为一谈。当前最值得讲透的，仍然是本地 block/page runtime：scheduler 如何依赖 KV manager 判断可执行性，block pool 如何统一管理复用与回收，worker 如何把 block ids 转成执行 metadata。至于更远端的 KV offload 或跨节点复用，那更适合作为下一篇文章展开。

## 结语：vLLM 的重要性，在于它把 KV cache 推进到了 runtime 中枢

如果只把 vLLM 看成一种 PagedAttention 实现，那么它的意义主要停留在 attention 访存和显存利用率层面。但从当前主线源码来看，这个理解已经过于狭窄。vLLM 真正更有代表性的地方在于，它把 KV cache 从“模型执行中的中间状态”推进成了**推理 runtime 的中枢资源之一**。

在这套系统里，scheduler 不再只管谁先跑，而必须理解哪些请求在当前 KV 资源约束下可执行；KVCacheManager 不再只是缓存封装，而是调度器的资源接口；BlockPool 不只负责分配显存块，还统一维护了 block 的缓存、共享和回收生命周期；worker 则通过 block table 和 slot mapping，把这种分页式资源抽象真正落到了 attention backend 的执行面上。

换句话说，当前 vLLM 主线给出的最重要答案，不是“如何用 page 存 KV”，而是“如何把 KV cache 纳入一个面向动态请求、前缀复用和调度约束的运行时系统”。如果说早期人们把 KV cache 看成 Transformer 推理中的必要副产物，那么 vLLM 至少在当前主线里，已经把它明确提升成了推理系统设计必须正面处理的核心对象。

---

## 参考来源

### 版本对齐信息

- 仓库：`vllm-project/vllm`
- 分支：`main`
- 本文对齐 commit：`48e376a007173910330a8c83f53474b21e4279c0`
- commit 链接：[48e376a](https://github.com/vllm-project/vllm/commit/48e376a007173910330a8c83f53474b21e4279c0)

### 源码文件

[1] `scheduler.py` 展示 vLLM 将 KV 管理放入调度热路径：相对路径 `vllm/v1/core/sched/scheduler.py`  
GitHub：[scheduler.py](https://github.com/vllm-project/vllm/blob/48e376a007173910330a8c83f53474b21e4279c0/vllm/v1/core/sched/scheduler.py)  
重点关注 `Scheduler.__init__` 与 `schedule()`，其中 waiting request 在进入 running 前会先查询 computed blocks，并调用 `allocate_slots(...)` 判断当前步是否可调度。

[2] `kv_cache_manager.py` 定义调度器视角的 KV 资源接口：相对路径 `vllm/v1/core/kv_cache_manager.py`  
GitHub：[kv_cache_manager.py](https://github.com/vllm-project/vllm/blob/48e376a007173910330a8c83f53474b21e4279c0/vllm/v1/core/kv_cache_manager.py)  
重点关注 `KVCacheBlocks`、`get_computed_blocks()`、`allocate_slots()`、`free()` 与 `get_num_common_prefix_blocks()`。其中 `allocate_slots()` 的注释已明确给出 prefix tokens / new tokens / lookahead / external computed tokens 的处理布局。

[3] `block_pool.py` 统一管理 free blocks、cached blocks 与共享 block 生命周期：相对路径 `vllm/v1/core/block_pool.py`  
GitHub：[block_pool.py](https://github.com/vllm-project/vllm/blob/48e376a007173910330a8c83f53474b21e4279c0/vllm/v1/core/block_pool.py)  
重点关注 `get_new_blocks()`、`cache_full_blocks()`、`touch()`、`free_blocks()` 与 `reset_prefix_cache()`。该文件是理解 prefix cache 与 block 生命周期统一管理的关键。

[4] `kv_cache_coordinator.py` 展示多 KV cache group 的统一协调层：相对路径 `vllm/v1/core/kv_cache_coordinator.py`  
GitHub：[kv_cache_coordinator.py](https://github.com/vllm-project/vllm/blob/48e376a007173910330a8c83f53474b21e4279c0/vllm/v1/core/kv_cache_coordinator.py)  
重点关注 `KVCacheCoordinator` 的初始化和聚合接口，包括 `get_num_blocks_to_allocate()`、`allocate_new_blocks()`、`cache_blocks()`、`free()` 等。

[5] `single_type_kv_cache_manager.py` 展示不同 attention spec 下的差异化 KV 管理：相对路径 `vllm/v1/core/single_type_kv_cache_manager.py`  
GitHub：[single_type_kv_cache_manager.py](https://github.com/vllm-project/vllm/blob/48e376a007173910330a8c83f53474b21e4279c0/vllm/v1/core/single_type_kv_cache_manager.py)  
重点关注 `SingleTypeKVCacheManager` 抽象基类，以及 `FullAttentionManager`、`SlidingWindowManager` 对 `find_longest_cache_hit()`、`get_num_common_prefix_blocks()`、`remove_skipped_blocks()` 的不同实现。

[6] `block_table.py` 展示 worker 如何将 request blocks 转换为执行 metadata：相对路径 `vllm/v1/worker/gpu/block_table.py`  
GitHub：[block_table.py](https://github.com/vllm-project/vllm/blob/48e376a007173910330a8c83f53474b21e4279c0/vllm/v1/worker/gpu/block_table.py)  
重点关注 `append_block_ids()`、`gather_block_tables()` 与 `compute_slot_mappings()`。这是理解逻辑 block 到物理 slot 映射的关键。

[7] `attention/backend.py` 定义 attention backend 消费的 batch 级公共 metadata：相对路径 `vllm/v1/attention/backend.py`  
GitHub：[attention/backend.py](https://github.com/vllm-project/vllm/blob/48e376a007173910330a8c83f53474b21e4279c0/vllm/v1/attention/backend.py)  
重点关注 `CommonAttentionMetadata`，其中显式包含 `block_table_tensor` 与 `slot_mapping`。

[8] `flash_attn.py` 展示公共前缀信息已进入部分 backend 优化路径：相对路径 `vllm/v1/attention/backends/flash_attn.py`  
GitHub：[flash_attn.py](https://github.com/vllm-project/vllm/blob/48e376a007173910330a8c83f53474b21e4279c0/vllm/v1/attention/backends/flash_attn.py)  
重点关注 `common_prefix_len`、`prefix_kv_lens`、`prefix_scheduler_metadata` 和 `use_cascade = common_prefix_len > 0` 等逻辑。

[9] `engine/core.py` 展示 KV cache 初始化发生在 engine 启动阶段、scheduler 创建之前：相对路径 `vllm/v1/engine/core.py`  
GitHub：[engine/core.py](https://github.com/vllm-project/vllm/blob/48e376a007173910330a8c83f53474b21e4279c0/vllm/v1/engine/core.py)  
重点关注 `EngineCore.__init__()` 与 `_initialize_kv_caches()`，其中 engine 会先生成 `kv_cache_configs`，再调用 `model_executor.initialize_from_config(...)` 初始化各 worker 的 KV cache。

[10] `gpu_worker.py` 展示 worker 侧先初始化 KV transfer，再初始化 KV cache：相对路径 `vllm/v1/worker/gpu_worker.py`  
GitHub：[gpu_worker.py](https://github.com/vllm-project/vllm/blob/48e376a007173910330a8c83f53474b21e4279c0/vllm/v1/worker/gpu_worker.py)  
重点关注 `initialize_from_config()`，其中先调用 `ensure_kv_transfer_initialized(...)`，再进入 `model_runner.initialize_kv_cache(...)`。

[11] `model_runner.py` 展示 runner 侧如何把配置落成 BlockTables、attention backend 和 KV tensors：相对路径 `vllm/v1/worker/gpu/model_runner.py`  
GitHub：[model_runner.py](https://github.com/vllm-project/vllm/blob/48e376a007173910330a8c83f53474b21e4279c0/vllm/v1/worker/gpu/model_runner.py)  
重点关注 `initialize_kv_cache()`，其中依次创建 `BlockTables`、初始化 attention backend、调用 `init_kv_cache(...)`，最后通过 `get_kv_connector(...)` 把已分配好的 KV tensors 暴露给 connector。

[12] `attn_utils.py` 展示原始 KV memory 的真实分配与 reshape：相对路径 `vllm/v1/worker/gpu/attn_utils.py`  
GitHub：[attn_utils.py](https://github.com/vllm-project/vllm/blob/48e376a007173910330a8c83f53474b21e4279c0/vllm/v1/worker/gpu/attn_utils.py)  
重点关注 `_allocate_kv_cache()`、`_reshape_kv_cache()` 和 `init_kv_cache()`；其中 `_allocate_kv_cache()` 里的 `torch.zeros(...)` 是原始 KV tensor 的真实分配点。

[13] `kv_transfer_state.py` 展示 KV transfer connector 的初始化入口：相对路径 `vllm/distributed/kv_transfer/kv_transfer_state.py`  
GitHub：[kv_transfer_state.py](https://github.com/vllm-project/vllm/blob/48e376a007173910330a8c83f53474b21e4279c0/vllm/distributed/kv_transfer/kv_transfer_state.py)  
重点关注 `ensure_kv_transfer_initialized()`，它通过 `KVConnectorFactory.create_connector(...)` 创建 worker 侧的 transfer connector。

[14] `gpu/kv_connector.py` 展示已分配 KV tensors 如何被注册给 connector：相对路径 `vllm/v1/worker/gpu/kv_connector.py`  
GitHub：[gpu/kv_connector.py](https://github.com/vllm-project/vllm/blob/48e376a007173910330a8c83f53474b21e4279c0/vllm/v1/worker/gpu/kv_connector.py)  
重点关注 `ActiveKVConnector.__init__()`，其中直接调用 `self.kv_connector.register_kv_caches(kv_caches_dict)`。

[15] `nixl_connector.py` 展示 NIXL worker 初始化与 memory registration：相对路径 `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py`  
GitHub：[nixl_connector.py](https://github.com/vllm-project/vllm/blob/48e376a007173910330a8c83f53474b21e4279c0/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py)  
重点关注 `NixlConnectorWorker.__init__()`、`register_kv_caches()`、`register_local_xfer_handler()` 和 `add_remote_agent()`；其中 `register_kv_caches()` 通过 `cache.data_ptr()`、`get_reg_descs()` 与 `register_memory()` 把真实 KV memory region 注册给 NIXL。

### 相关 PR

[16] `Use block table apis for capture inputs`：PR `#35671`  
链接：[PR #35671](https://github.com/vllm-project/vllm/pull/35671)  
说明 block table API 仍是当前执行主链的一部分，并在演进中。

[17] `Avoid prefix cache hit in the same schedule step for mamba layers`：PR `#29387`  
链接：[PR #29387](https://github.com/vllm-project/vllm/pull/29387)  
说明 prefix caching 与 schedule step 的交互语义并非静态设计点，仍在持续修正。

[18] `Fix CPU memory leak from Request reference cycle in prefix caching`：PR `#34183`  
链接：[PR #34183](https://github.com/vllm-project/vllm/pull/34183)  
说明 prefix caching 不只是性能特性，也涉及 request 生命周期与引用管理复杂度。

[19] `Support multiple KV cache groups in Hybrid KV Coordinator`：PR `#31707`  
链接：[PR #31707](https://github.com/vllm-project/vllm/pull/31707)  
可用来支撑“当前主线已是多组 KV 协调架构”的判断。

### 官方博客 / 原文延伸阅读

[20] `Inside vLLM: Anatomy of a High-Throughput LLM Inference System`：vLLM 官方架构总览文  
链接：[Anatomy of vLLM](https://vllm.ai/blog/anatomy-of-vllm)  
这篇文章从整体系统角度覆盖 Engine、Scheduler、Prefix Caching、Disaggregated P/D 等主题，适合作为理解 vLLM 全局结构的官方背景材料；但它并不是一篇专门围绕 KV cache 内部运行时与显存布局的源码级剖析。

[21] `vLLM Router: A High-Performance and Prefill/Decode Aware Load Balancer for Large-scale Serving`：vLLM 官方对状态感知路由与 P/D 解耦的说明  
链接：[vLLM Router](https://vllm.ai/blog/vllm-router-release)  
文中明确指出大规模 serving 需要感知 KV cache 这一状态，并提到 router 支持 `NIXL` 与 `NCCL-based (with ZMQ discovery)` 的 disaggregation backends。本文对 NIXL / NCCL 的讨论更聚焦于 **当前 vLLM Python connector 路径中 memory registration 与 block-level KV transfer 的语义差异**，与 router 文中的部署层描述并不冲突。

[22] `Inside vLLM’s New KV Offloading Connector: Smarter Memory Transfer for Maximizing Inference Throughput`：vLLM 官方对 KV offloading connector 的原文说明  
链接：[KV Offloading Connector](https://vllm.ai/blog/kv-offloading-connector)  
这篇官方博客重点讨论 CPU KV offloading、异步 connector API、`cudaMemcpyAsync` / DMA 路径与吞吐优化，是理解“KV data 如何通过 connector API 在不同介质间搬运”的第一手材料。本文对 NIXL register / zero-copy 的讨论则进一步补充了另一条以已注册 device memory region 为中心的 KV transfer 路径。
