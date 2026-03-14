---
title: SGLang 如何管理 KVCache：从 RadixAttention 到 HiCache 的底层技术主线
date: 2026-03-14 12:40:00 +0800
author: Ethan
kind: essay
category: Essay
intro: 基于 LMSYS/官方 blog、SGLang 文档与源码，深入分析 SGLang 如何用 RadixAttention 组织共享前缀，及其向 HiCache 分层缓存体系的扩展。
---

很多系统在讨论 KVCache 时，默认会把问题定义成“如何管理一块不断膨胀的显存资源”：怎么分页、怎么回收、怎么复用、怎么避免碎片。这套问题定义当然成立，但如果直接把它套到 SGLang 上，往往会漏掉它最关键的一层。对 SGLang 来说，最早被抬到系统中心的，不是 page 或 block，而是**共享前缀**本身。

这也是本文要回答的核心问题：**SGLang 到底如何理解和管理 KVCache？** 从公开材料和当前源码看，更准确的说法不是“它支持 prefix cache”，而是：**它把共享前缀组织成 runtime 的第一抽象，再让底层 page/KV 存储、调度与分层缓存围绕这层抽象展开。** 这并不意味着 SGLang 没有 page、没有底层内存布局；恰恰相反，它同样高度依赖 page 粒度。真正的差异在于，SGLang 更早暴露给系统上层的是“哪些前缀可以共享、它们现在在哪一层、怎样继续被命中和保护”，而不是单纯“某个请求占了多少 page”。

## 一、问题不是“要不要缓存 KV”，而是如何把共享前缀变成系统资产

SGLang 最早切入的问题，就不是单请求推理，而是复杂 LLM workload 里的共享前缀。LMSYS 在早期文章里列出的 few-shot、多轮对话、self-consistency 和 tree-of-thought 等场景，本质上都在反复复用同一段 prompt 骨架；如果系统每次都从头做 prefill，那么大量计算其实是在重复消费已经出现过的上下文<a href="#ref-1">[1]</a>。

![SGLang 在早期 blog 中展示的共享前缀 workload 示例](../sglang-origin-evolution-community/assets/sglang-prefix-sharing-workloads.jpg)
*图 1：LMSYS 早期 blog 中的共享前缀 workload 示例。它的重点不是“某些请求刚好相似”，而是很多典型 LLM workload 天然具有可共享的前缀骨架<a href="#ref-1">[1]</a>。*

这一步很重要，因为它决定了系统抽象的起点。如果你先把 KV 看成 per-request buffer，那么自然会优先围绕 page、block、slot 组织运行时；但如果你先把问题定义成“哪些历史前缀值得长期保留，并在后续请求里再次命中”，那么系统首先要组织的就是**请求之间共享的前缀关系**。SGLang 的 KVCache 主线，正是从这里分叉出去的。

## 二、RadixAttention：为什么共享前缀会成为第一抽象

SGLang 对这个问题给出的第一个系统性答案，是 RadixAttention。LMSYS 最早对它的描述，并不是“一个更快的 attention kernel”，而是一种围绕共享前缀组织缓存与复用的机制<a href="#ref-1">[1]</a>。 这一点和当前源码的文件级定位是对得上的：`radix_cache.py` 的开头直接把 radix tree 定义为 **“用于管理 KV cache 的数据结构”**<a href="#ref-2">[2]</a>。 这个表述本身就很强，它说明 radix tree 在 SGLang 中不是辅助索引，而是管理 KV cache 的核心结构之一。

![SGLang 早期关于 RadixAttention 的概览图](../sglang-origin-evolution-community/assets/sglang-radixattention-overview.jpg)
*图 2：RadixAttention 的关键不是“树形结构好看”，而是它把共享前缀变成了可搜索、可插入、可淘汰、可保护的运行时对象<a href="#ref-1">[1]</a>。*

### 1. `RadixKey`：前缀不是裸 token 序列，而是带命名空间的匹配键

从实现看，SGLang 并不是直接把 token 列表塞进树里，而是先封装成 `RadixKey`。这个对象除了 `token_ids` 外，还有两个很重要的字段：`extra_key` 和 `is_bigram`<a href="#ref-2">[2]</a>。 `extra_key` 表示匹配命名空间，用来隔离不同 LoRA、不同 cache salt 或其他不应共享状态的请求；`is_bigram` 则用于 EAGLE 场景下的 bigram key 变换<a href="#ref-2">[2]</a>。

这意味着 SGLang 的 prefix reuse 不是“只要 token 前缀相同就共享”的无条件复用，而是一种**带命名空间边界的复用**。从工程角度看，这个约束很重要：共享前缀不是抽象语义上的“文本相似”，而是系统确认它们可以共享同一条缓存生命周期之后，才会真的进入同一棵树。

### 2. page 对齐并没有缺席，而是被压在底层

RadixAttention 虽然强调 prefix tree，但并不意味着它忽略底层 page 粒度。`page_align_keys()`、`_key_match_paged(...)` 和 `match_prefix()` 里的 page 对齐逻辑都表明：当 `page_size > 1` 时，key 会先被截断到 page 对齐长度，后续匹配也不是逐 token 进行，而是按 page 粒度比较<a href="#ref-2">[2]</a>。

这说明两个事实。第一，SGLang 的 prefix cache 不是无限细粒度的任意 token 命中；第二，它同样受 page 粒度约束。更准确的说法不是“SGLang 不讲 page”，而是：**page 在 SGLang 中被压在 prefix tree 的底层实现里，服务于上层的 prefix-first 组织方式。**

### 3. `TreeNode`：树节点已经是缓存生命周期对象，而非纯逻辑前缀

`TreeNode` 的字段很能说明问题。它不只维护 `children`、`parent`、`key`、`value`，还维护：
- `lock_ref`
- `last_access_time`
- `creation_time`
- `hit_count`
- `host_ref_counter`
- `host_value`
- `hash_value`
- `priority`
- pin 相关状态

这组字段意味着，树节点承载的不是“某段文本前缀”的纯逻辑语义，而是 prefix 对应 KV 的实际生命周期状态：它当前有没有 device value、是否有 host 备份、是否被请求锁住、最近是否被访问、在 eviction 里优先级如何、是否已按 page 计算 hash<a href="#ref-2">[2]</a>。 因此更稳妥的表述是：**TreeNode 已经是 prefix 语义与缓存生命周期的统一对象。**

### 4. `match_prefix()` 并不是纯读操作，它会主动精化树结构

`match_prefix()` 的实现和注释都很值得细读。它不仅返回最长缓存前缀，还明确说明：如果匹配结束在某个已存储 segment 的中间，系统会通过 `_split_node()` 把精确边界切出来，从而提高后续匹配效率<a href="#ref-2">[2]</a>。 也就是说，prefix match 在 SGLang 中不只是“查一查有没有命中”，它还会驱动树结构的进一步显式化。

`_split_node()` 的逻辑很直接：创建一个新的 `new_node`，把 child 被命中的前缀部分切出来归给 `new_node`，让 child 保留剩余 suffix，同时把 `hash_value` 也按 `split_len` 和 `page_size` 一起拆开<a href="#ref-2">[2]</a>。 这一步背后的意义很大：

> SGLang 并不是把整条请求路径原封不动地挂在树上，而是会在共享边界出现时，把“共享前缀”提炼成显式节点。

这正是共享前缀之所以能成为第一抽象的实现基础之一。

### 5. `insert()`：系统沿共享路径合并，只为新的 suffix 建增量节点

`_insert_helper()` 的行为也和普通“把 key 插入树里”的直觉不完全一样。它会沿已有路径不断前进，更新访问时间和 priority；如果中途只部分匹配，也会先 split；只有剩余 suffix 才会真正新建节点<a href="#ref-2">[2]</a>。 这说明插入过程不是“每个请求独占一条新路径”，而是“先最大化复用已有共享前缀，再把新的增量部分挂上去”。

因此，从 RadixAttention 的实现视角看，SGLang 管理的并不是一组彼此独立的 request-local buffers，而是一棵会不断显式化共享边界、并用节点生命周期表达复用关系的 prefix tree。

## 三、Prefix caching 不是旁路功能，而是请求生命周期和调度器的一部分

仅仅把 prefix 存进树里还不够，真正关键的是：**prefix caching 如何进入 request lifecycle 与 scheduler。** 这一点在 `BasePrefixCache` 和调度相关实现里体现得很清楚。

### 1. `BasePrefixCache`：prefix caching 被定义成 runtime 接口，而不是 feature API

`BasePrefixCache` 对外暴露的抽象包括：
- `match_prefix`
- `cache_finished_req`
- `cache_unfinished_req`
- `evict`
- `inc_lock_ref` / `dec_lock_ref`
- `init_load_back`
- `ready_to_load_host_cache`
- `check_hicache_events`

这组接口很说明问题。前四项代表 prefix cache 已经参与普通请求生命周期；后三项则说明它从一开始就为 host load-back 与 HiCache 活动预留了扩展位<a href="#ref-3">[3]</a>。 换句话说，**SGLang 的 prefix cache 不是“后面再加一层 HiCache”的临时补丁，而是一开始就被抽象成可继续扩展到分层缓存的 runtime 接口。**

### 2. `cache_finished_req()` / `cache_unfinished_req()`：请求会持续把自身状态提交回 prefix tree

`cache_finished_req()` 的流程是：从 request 中取出已提交的 KV 长度和对应 KV indices，对 token 做 bigram 转换和 page 对齐，构造 `RadixKey`，执行 `insert()`，然后把已经在树里存在的重复 KV 释放掉，并最终对 `req.last_node` 执行 `dec_lock_ref()`<a href="#ref-2">[2]</a>。 

`cache_unfinished_req()` 更能体现它与请求生命周期的耦合：它会对当前 `fill_ids` 对应的 KV 先做 insert，再重新调用 `match_prefix()` 拿回新的 `device_indices` 与 `last_node`，回写到 `req_to_token_pool`，更新 `req.cache_protected_len`、`req.prefix_indices` 和 `req.last_node`，同时完成旧锁释放与新锁获取<a href="#ref-2">[2]</a>。

这说明 prefix cache 不是请求结束后再做的“归档缓存”，而是请求进行中就会持续回流并改写 request 自己的 KV 视图。更压缩一点说：**SGLang 的 request 不只是在使用 prefix cache，它也在持续把自身执行结果提交回 prefix cache，再反过来复用更新后的树结构。**

### 3. `evict()` / `lock_ref`：prefix tree 直接承担缓存保护与淘汰语义

`evict()` 会从 `evictable_leaves` 中取叶子，基于 LRU/LFU/FIFO/MRU/FILO/Priority 等策略构建堆，free 掉对应 `value` 后删除叶子，并可能把父节点继续加入可淘汰集合<a href="#ref-2">[2]</a>。 `inc_lock_ref()` / `dec_lock_ref()` 则会沿路径向上更新节点的 `lock_ref`，并同步维护 `evictable_size_` 与 `protected_size_`<a href="#ref-2">[2]</a>。

这意味着 prefix tree 不只是记录“哪些前缀能命中”，它还直接决定哪些缓存当前可淘汰、哪些被请求保护。树结构本身就是 eviction bookkeeping 的核心载体。

## 四、Prefix 命中如何进入 scheduler：SGLang 的调度器会显式感知 tree cache

`SchedulePolicy` 的实现表明，SGLang 并不把 prefix hit 当作一个与调度器无关的底层细节。代码里明确区分了 `CacheAwarePolicy` 和 `CacheAgnosticPolicy` 两类策略；其中前者包括 `LPM` 和 `DFS_WEIGHT`，即“最长前缀命中优先”和“基于树权重的深度优先”<a href="#ref-4">[4]</a>。

`_compute_prefix_matches()` 的流程尤其关键。对于等待队列中的每个请求，系统会把 `origin_input_ids + output_ids` 拼成 `prefix_ids`，再用 `tree_cache.match_prefix()` 计算匹配结果，并把 `prefix_indices`、`last_node`、`last_host_node`、`host_hit_length` 这些信息回写到请求对象<a href="#ref-4">[4]</a>。 这意味着 prefix 命中结果并不是局部临时变量，而是调度器显式维护的 request state。

之后，不同的 cache-aware policy 会直接利用这些信息改变等待队列顺序：
- **LPM**：优先调度拥有更长已命中前缀的请求<a href="#ref-4">[4]</a>；
- **DFS_WEIGHT**：先按 `last_node` 对请求聚类，再沿 prefix tree 计算权重并做深度优先排序，优先跑共享路径收益更大的那簇请求<a href="#ref-4">[4]</a>。

更有意思的是，代码里还专门维护了一个 `waiting_queue_radix_tree = RadixCache.create_simulated()`，用来做 in-batch prefix caching：即便某个请求对已有 cache 命中不长，只要它和 waiting queue 里的其他请求共享前缀，系统也会倾向于先调度一部分请求，以便后续在批内形成更高的前缀复用收益<a href="#ref-4">[4]</a>。

![SGLang v0.4 的 cache-aware load balancer 示意图](../sglang-origin-evolution-community/assets/sglang-cache-aware-load-balancer.png)
*图 3：cache-aware load balancer 可以看作同一条 prefix-first 逻辑在多实例路由层的外推：请求不再只按平均负载分发，而要尽量落到最可能命中已有前缀的 worker 上<a href="#ref-5">[5]</a>。*

这也解释了为什么 LMSYS 在 v0.4 文章里会进一步把 cache-aware 思路扩展到 load balancer：一旦你承认“共享前缀是系统资产”，那么从单机 scheduler 到多 worker 路由，最自然的延伸就是继续围绕“哪里最可能命中已有前缀”来组织请求流<a href="#ref-5">[5]</a>。

## 五、底层 page / KV 存储并没有缺席，但它在 SGLang 中更像支撑层

到这里需要非常克制地说清楚一点：把 SGLang 概括为 prefix-first runtime，并不等于它没有 paged KV、没有 memory pool、没有底层 layout。恰恰相反，当前实现里 page 粒度非常明确：`match_prefix()` 会先做 page 对齐截断，HiCache 文档也专门讨论了 `--page-size`、`layer_first`、`page_first` 与 `page_first_direct` 的差异<a href="#ref-2">[2]</a><a href="#ref-6">[6]</a>。

但这些细节在 SGLang 中的角色，更像是在支撑上层的 prefix-first 组织方式。HiCache design 文档给出的叙事顺序很能说明这一点：它先定义的是 HiRadixTree，说明每个节点对应一段连续 token span 的 KV，并记录这段 KV 位于 GPU、CPU 还是 L3；然后才继续讨论 local match、prefetch、write-back、page-size 和 layout 等工程实现细节<a href="#ref-6">[6]</a>。

因此，更稳妥的结论不是“一个讲 page，一个不讲 page”，而是：**两者都依赖底层 page/block 粒度，但 vLLM 更直接地把 block/page 暴露为运行时主抽象；SGLang 则更倾向于先组织共享前缀，再让 page 粒度与内存布局去支撑这层组织。**

## 六、HiCache：从 RadixCache 到 HiRadixCache 的分层扩展

如果前面的判断成立，那么 HiCache 的位置就很清楚。HiCache design 文档开头直接把它定义为 RadixAttention 思路的延伸：RadixAttention 用空闲 GPU memory 缓存和复用共享 prefix KV，而 HiCache 则把这套思路扩展到 host memory 和 distributed storage，形成类似 CPU 三层缓存的 L1/L2/L3 结构<a href="#ref-6">[6]</a>。

### 1. HiRadixTree：节点不仅表示前缀，还表示“这段前缀现在在哪一层”

design doc 对 HiRadixTree 的定义很关键：每个节点仍然表示一段连续 token span 的 KV cache，但它不再只记录 device 侧信息，而是记录这段 KV 当前位于 GPU、CPU、L3 或其中多层；对本地层保留精确地址元数据，对 L3 则按需向 backend 查询而不是持续同步所有元数据<a href="#ref-6">[6]</a>。

这说明 HiCache 并没有推翻 prefix tree，而是把原来的 “device-only prefix tree” 升级成了 “multi-tier prefix metadata tree”。

### 2. `HiRadixCache(RadixCache)`：源码直接给出了继承关系

这一点在源码里更加直接：`HiRadixCache` 是 `RadixCache` 的子类<a href="#ref-7">[7]</a>。 它的初始化流程大致是：
1. 读取 `page_size` 与 device 侧 `kv_cache`；
2. 根据 MHA / NSA / MLA 类型构造 host 侧 KV pool；
3. 解析 storage backend 配置与 prefetch 阈值/超时参数；
4. 创建 `HiCacheController`；
5. 维护一组异步状态：`ongoing_write_through`、`ongoing_load_back`、`ongoing_prefetch`、`ongoing_backup`；
6. 最后再调用 `super().__init__(params)`，把普通 RadixCache 的逻辑接进来<a href="#ref-7">[7]</a>。

这组步骤很能说明问题：HiCache 不是一个和 radix cache 平行的新系统，而是在原有 prefix-first 主线之上，额外接入了 host pool、storage backend 和异步数据流控制器。

### 3. match 的结果从“命中 device 前缀”升级为“命中本地多层前缀”

`BasePrefixCache.MatchResult` 在 HiCache 场景下除了 `device_indices` 外，还包含 `last_host_node` 和 `host_hit_length`<a href="#ref-3">[3]</a>。 这表明一旦进入 HiCache，match 的语义就不再只是“device 上命中了多少 KV”，而是“在本地 L1/L2 上连续命中了多少前缀，其中哪一段还需要 load-back 或继续 prefetch”。

对应地，design doc 里的 workflow 也分成了三步：
- **local match**：先在 L1/L2 的 HiRadixTree 里找连续前缀；
- **prefetch from L3**：对本地未命中的连续部分再查 L3，并根据 threshold 与 stop policy 决定是否拉回；
- **write-back**：prefill 完成后再把新生成的高价值 prefix 逐层写回更低层<a href="#ref-6">[6]</a>。

因此，HiCache 下的 prefix match 已经从“查 GPU 里有没有现成前缀”升级成“查整个本地多层缓存里有哪些连续前缀、它们接下来应该怎样被拉回或继续沉淀”。

### 4. load-back / prefetch / write-back：HiCache 的三种核心动作

从接口与设计文档可以把 HiCache 的核心动作压缩为三类：

- **load-back**：本地 host 已有的 KV 如何重新进入 GPU；这也是为什么 `BasePrefixCache` 里需要 `init_load_back()` 与 `ready_to_load_host_cache()`<a href="#ref-3">[3]</a>。
- **prefetch**：L3 已有的数据如何在真正计算前提前拉近，设计文档中对应 `best_effort`、`wait_complete` 和 `timeout` 三种 stop policy<a href="#ref-6">[6]</a>。
- **write-back**：新产生的 prefix KV 何时、以何种策略向 L2/L3 写回，文档中对应 `write_through`、`write_through_selective` 和 `write_back` 三类模式<a href="#ref-6">[6]</a>。

这三类动作组合在一起之后，HiCache 管理的就不再只是“当前有没有命中 cache”，而是**一整套 prefix KV 在多层之间迁移、复用、沉淀与回收的生命周期。**

![HiCache 总体 workflow 示意图](../sglang-origin-evolution-community/assets/sglang-hicache-overview.png)
*图 4：HiCache 的关键不是多了几层存储，而是它仍然沿着“先匹配共享前缀，再决定从哪一层取回和写回数据”的主线在扩展<a href="#ref-6">[6]</a>。*

## 七、为什么 HiCache 要显式讨论 page size、layout 和 I/O 优化

HiCache 最容易被低估的地方，是很多人会把它理解成“把 GPU cache 搬去 CPU/L3”。但从 design doc 看，它做的远不止数据分层，还重新设计了跨层数据组织与传输路径<a href="#ref-6">[6]</a>。

### 1. `page_size` 直接决定命中粒度与 I/O 粒度的权衡

文档明确指出：较大的 page size 可以减少 metadata 开销、提高 I/O 效率，但会降低部分前缀匹配时的命中细粒度；较小的 page size 则相反<a href="#ref-6">[6]</a>。 这说明 page 在 HiCache 中既不是纯执行细节，也不是单纯调参项，而是连接“命中语义”与“传输效率”的底层粒度选择。

### 2. `layer_first` / `page_first` / `page_first_direct`

文档还专门解释了为什么需要三种 memory layout：GPU 计算天然偏 `layer_first`，而 L3/host I/O 又更适合 `page_first`；`page_first_direct` 则试图在两者之间做折中<a href="#ref-6">[6]</a>。 这说明 HiCache 并不是简单把原有 GPU layout 原封不动搬去 host/L3，而是在为跨层数据移动重新设计内存组织方式。

### 3. zero-copy、batch-oriented organization、compute-transfer overlap

更进一步地，HiCache 还显式强调：
- 使用 zero-copy transfer 减少不必要的中间复制；
- 以 page 为单位做 batch-oriented data organization；
- 在 prefill 阶段让 CPU→GPU 的 KV 加载与 layer 计算 overlap；
- 使用 GPU-assisted I/O kernels 加速 CPU/GPU 之间的 KV 搬运<a href="#ref-6">[6]</a>。

这组设计说明：HiCache 的本质不只是“多一层存储”，而是**在 prefix-first cache 主线下，进一步把跨层数据移动本身做成优化对象。**

## 结语：SGLang 的关键在于先定义“共享前缀”，再组织整个 KV runtime

如果只从功能列表看，SGLang 当然也有 page、也有 memory pool、也有 eviction、也有 host/storage 分层；它并不是生活在另一个世界里的特殊系统。但它真正值得单独拿出来讲的地方，在于它最早被提升到系统主链路的，不是 page，而是**共享前缀**。

一旦这层起点成立，后面的设计就会变得连贯：RadixAttention 把共享前缀变成可搜索、可插入、可淘汰、可保护的树节点；请求生命周期会持续把自身状态提交回 prefix tree 并再次复用；scheduler 会显式感知 prefix hit 并改变等待队列顺序；HiCache 再把这条主线从 GPU 扩展到 host 与 L3，并为此重新设计 page size、layout 与跨层 I/O 路径。

因此，用一句话收束全文，当前最稳妥的结论是：**SGLang 的 KVCache 主线，不是从 page 出发寻找复用，而是从共享前缀出发组织 runtime；page、分层缓存与平台化能力，则是这条主线逐步长出来的工程形态。**



<a href="#ref-8">[8]</a> SGLang 源码：`python/sglang/srt/mem_cache/memory_pool.py`  
https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/memory_pool.py

<a href="#ref-9">[9]</a> SGLang 源码：`python/sglang/srt/mem_cache/memory_pool_host.py`  
https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/memory_pool_host.py


## 参考来源

<a href="#ref-1">[1]</a> LMSYS Blog, *Fast and Expressive LLM Inference with RadixAttention and SGLang*  
https://lmsys.org/blog/2024-01-17-sglang/

<a href="#ref-2">[2]</a> SGLang 源码：`python/sglang/srt/mem_cache/radix_cache.py`  
https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py

<a href="#ref-3">[3]</a> SGLang 源码：`python/sglang/srt/mem_cache/base_prefix_cache.py`  
https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/base_prefix_cache.py

<a href="#ref-4">[4]</a> SGLang 源码：`python/sglang/srt/managers/schedule_policy.py`  
https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_policy.py

<a href="#ref-5">[5]</a> LMSYS Blog, *SGLang v0.4*  
https://lmsys.org/blog/2024-12-04-sglang-v0-4/

<a href="#ref-6">[6]</a> SGLang 官方文档：*HiCache System Design and Optimization*  
https://docs.sglang.ai/advanced_features/hicache.html

<a href="#ref-7">[7]</a> SGLang 源码：`python/sglang/srt/mem_cache/hiradix_cache.py`  
https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/hiradix_cache.py
