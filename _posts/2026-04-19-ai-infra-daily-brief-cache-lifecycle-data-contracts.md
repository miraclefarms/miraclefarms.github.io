---
title: AI Infra 早报｜缓存生命周期开始长出正式数据面 contract
date: 2026-04-19 05:30:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: 过去三天，LMCache 把持久化、隔离和 trace 补进缓存主干，TensorRT-LLM 重写了批量 onboarding 与复用计费逻辑，Mooncake 和 vLLM 则继续把跨机传输、多模态特征与 SSD offload 从“能接上”推进到可正式依赖的数据面 contract。
---

过去三天，更值得写的变化不是哪家框架又多支持了一个模型，而是缓存和拆分式 serving 这条链路，开始被项目方当成一套必须写清 contract 的正式数据面。缓存不再只是“命中就快一些”的黑盒层，而是要回答 key 如何隔离、状态能否持久化、请求是否能被完整回放；调度器也不再把 KV reuse 当作事后减法，而是开始围绕 claim、onboard 和 token accounting 重新定义批量请求的生命周期。

这也是今天最值得记住的判断。AI Infra 现在正在从“把优化能力挂上去”转向“把数据路径的边界写严”。如果缓存 key 不能表达租户边界，调度器不能正确计算 chunking 和 reuse，跨机链路不能明确它到底搬运什么对象、走什么介质、落到哪里，那么这些能力都还停留在 demo 阶段。最近三天的更新说明，几条最关键的缓存与 serving 路径，已经开始越过这条线。

## 一、缓存系统不再只是共享池，而开始具备恢复、隔离和回放语义

**LMCache 新增 persistence interface 和 `nixl_store_dynamic` 适配器[[1]](https://github.com/LMCache/LMCache/pull/2938)**，这是这几天最值得写的一条缓存更新。它不只是多了一个 L2 backend，而是把“缓存重启之后还在不在”正式写进了接口层：每个 key 可以映射到确定性的文件路径，cache miss 时还能做二次磁盘查找，adapter 自己也带 persist/recover 语义。换句话说，LMCache 开始承认缓存不只是易失性的性能层，而可能是需要在服务重启、逐步回暖和容量管理里被长期维护的数据层。

**同一批改动里，LMCache 又把 `cache_salt` 做成 `ObjectKey` 和 IPC key 的一等公民[[2]](https://github.com/LMCache/LMCache/pull/3042)**。这条 PR 真正重要的地方，不在于多加了一个字段，而在于它明确切断了“同内容不同用户”共享一份缓存 key 的默认假设。未加 salt 的流量仍保持旧格式兼容，但一旦需要按租户或用户做隔离，key identity、文件名和多进程 wire format 都已经能稳定表达这层边界。缓存系统一旦开始把“谁的 key 算谁的”写进数据模型，而不是靠上层约定兜底，它就从共享池迈向了真正的基础设施。

**LMCache 随后又给 StorageManager 加上了二进制 trace 记录能力[[3]](https://github.com/LMCache/LMCache/pull/3063)**。这里新增的不是普通日志，而是可以把公开 API 调用写入 trace 文件、离线读取和回放的调用轨迹。它指向的是另一类成熟度：当缓存层被放进生产主链路后，团队已经不满足于“看到 hit ratio”，而是想知道某个请求到底经过了哪些 storage 操作、在哪一段拖慢、能不能把真实工作负载原样拿回来复盘。恢复、隔离、回放这三件事放在一起看，LMCache 正在把缓存从“加速附件”改造成带状态边界的数据系统。

## 二、调度器开始围绕复用语义重写请求生命周期，而不是事后补公式

**TensorRT-LLM 新增 `addSequenceBatch` 的两阶段 claim/onboard 机制，并统一 VSWA 与 non-reuse 路径[[4]](https://github.com/NVIDIA/TensorRT-LLM/pull/13029)**，这条 PR 的信号非常强。项目方已经不再满足于逐条 `addSequence` 地把请求塞进 KV 管理器，而是要先在单锁阶段把可复用 block 认领清楚，再批量 onboard 和分配，避免 host offloading 把原本可复用的块先赶走。更关键的是，它把先前分裂的 VSWA、non-reuse、star attention 和 dummy request 这些分支都拉回到同一套 batch path 下。调度器一旦开始把“请求进入系统的第一步”重写成可批处理、可协调部分匹配 ownership 的过程，说明 reuse 已经从优化项变成了生命周期设计前提。

**另一条同步合并的修复，则补上了 KV reuse 与 context chunking 组合下最容易失真的 token accounting[[5]](https://github.com/NVIDIA/TensorRT-LLM/pull/12976)**。PR 直接指出旧逻辑把 chunk window 错算成“reusable 越多，compute token 越少”，但真实情况是 `setPrepopulatedPromptLen` 会把窗口右移，非最后一个 chunk 仍然需要处理接近 `chunkSize` 的 token。这个修复之所以重要，不只是少了一处 off-by-some 的统计问题，而是调度器终于开始按真实执行语义给计算量记账，而不是把 reuse 当作一把统一折扣。只有这层计费逻辑算对，后面的调度、容量和吞吐预期才站得住。

把这两条更新放在一起看，TensorRT-LLM 现在处理的已经不是“是否支持复用”，而是“复用如何影响 onboarding、批量锁竞争、chunk 窗口和 token 成本模型”。这比单纯再加一个 reuse feature 更关键，因为它决定的是系统能否长期稳定地把 reuse 当作默认前提。

## 三、拆分式 serving 的数据面开始覆盖跨机传输、多模态对象和本地落盘介质

**Mooncake 在 EFA transport 里补上了 `fi_read`、endpoint LRU eviction 和 multi-NIC striping[[6]](https://github.com/kvcache-ai/Mooncake/pull/1821)**，这几项放在同一个 PR 里非常说明方向。`fi_read` 让 consumer 侧可以主动拉取 KV 数据，LRU eviction 解决长时间运行时 AV 容量被瞬时 peer 吃满的问题，multi-NIC striping 则把大块传输直接摊到所有活跃网卡上。它们共同指向一个事实：跨机缓存传输不再只是“链路能通”，而是要明确由谁发起、端点如何复用、拥塞和容量怎么管、超大对象如何跨多 NIC 运输。这已经是一套正式数据面的思路，而不是单机 demo 的延长线。

**Mooncake 同时又把 SSD offload path 暴露到了 Python `setup()` 接口里[[7]](https://github.com/kvcache-ai/Mooncake/pull/1884)**。过去这条路径只能靠进程级环境变量配置，多 TP 场景下所有 worker 共享同一个目录，实际上很难按 GPU 或实例分配独立 SSD 空间。现在 `ssd_offload_path` 变成实例级参数之后，offload 才真正具备“每个 worker 落到哪里”这层显式 contract。别小看这类接口改动，很多所谓支持 SSD offload 的系统，最后正是死在了路径作用域仍然是全局变量这种细节上。

**vLLM 则把多模态特征正式接进了拆分式 `/inference/v1/generate` 入口[[8]](https://github.com/vllm-project/vllm/pull/38405)**。这条 PR 允许 coordinator 把 `pixel_values`、`image_grid_thw` 这类预处理后的 feature 直接通过协议对象传给 worker，而不是继续把多模态输入留在渲染端内部消化。它真正说明的是，拆分式 serving 的 HTTP contract 正在从“只会搬 token”扩展到“可以搬运经过上游处理的结构化 feature”。一旦协议层明确知道自己传输的对象是什么，跨服务拓扑下的多模态推理才谈得上稳定落地。

## 四、今天真正值得记住的判断

今天真正值得记住的，不是谁又把某条优化路径跑快了一点，而是谁开始把缓存和 serving 里的隐式假设改写成正式 contract。LMCache 在补缓存的恢复、隔离和回放；TensorRT-LLM 在重写请求 onboarding 与 reuse 记账；Mooncake 和 vLLM 则把跨机传输、SSD 落盘和多模态对象传递这些过去容易靠经验兜底的环节，逐步写成明确的数据面接口。

如果这个方向继续下去，下一阶段 AI Infra 的差异就不会主要体现在“谁支持更多 headline feature”，而会体现在“谁更早把缓存和 serving 的生命周期写成可以恢复、可以隔离、可以批量调度、也可以被观测和回放的正式系统 contract”。

---

## 参考来源

[1] [LMCache 增加 persistence interface 与 nixl 动态持久化适配器](https://github.com/LMCache/LMCache/pull/2938)

[2] [LMCache 将 cache_salt 写入 ObjectKey 与 IPC key，实现缓存隔离](https://github.com/LMCache/LMCache/pull/3042)

[3] [LMCache 为 StorageManager 增加二进制 trace 记录能力](https://github.com/LMCache/LMCache/pull/3063)

[4] [TensorRT-LLM 用两阶段 claim/onboard 重写批量 addSequence](https://github.com/NVIDIA/TensorRT-LLM/pull/13029)

[5] [TensorRT-LLM 修复 KV reuse 与 context chunking 下的 token accounting](https://github.com/NVIDIA/TensorRT-LLM/pull/12976)

[6] [Mooncake 为 EFA transport 增加 fi_read、LRU eviction 与 multi-NIC striping](https://github.com/kvcache-ai/Mooncake/pull/1821)

[7] [Mooncake 将 SSD offload path 暴露为 Python setup 参数](https://github.com/kvcache-ai/Mooncake/pull/1884)

[8] [vLLM 为拆分式 /inference/v1/generate 增加多模态特征支持](https://github.com/vllm-project/vllm/pull/38405)
