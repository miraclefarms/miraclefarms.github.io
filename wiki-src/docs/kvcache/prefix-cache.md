# Prefix Cache

## 目标

Prefix Cache 的核心目标是：**对于具有相同前缀的多个请求，避免重复计算其 KVCache**。

在许多生产场景中，大量请求共享相同的前缀——最典型的是 System Prompt。例如，一个 API 服务的系统提示词可能有数百乃至数千 token，如果每个请求都从头 Prefill 这段文本，将消耗大量计算资源和首 token 延迟。Prefix Cache 让这段共享前缀的 KV 只需计算一次，之后可以被所有后续请求直接读取复用。

## 为什么前缀可以复用

Transformer Attention 是因果的（causal）：每个 token 的 K/V 只取决于它自身及其之前的 token，与之后的 token 无关。因此，如果两个请求的前 $n$ 个 token 完全相同，它们在每一层产生的前 $n$ 个 token 的 K/V 张量也完全相同。

这个性质允许引擎将这些 K/V Block 存储在共享的物理 Block 中，多个 Sequence 的 Block Table 可以指向相同的物理 Block（只读引用，引用计数管理）。

## 精确前缀匹配

当前主流引擎（vLLM、SGLang 等）实现的 Prefix Cache 基于**精确前缀匹配**：

1. 对 Block 进行哈希（通常以 Block 的 token ID 序列为输入）
2. 维护哈希表：`hash(tokens) → physical_block_id`
3. 新请求到来时，从请求头部开始逐 Block 做哈希查找
4. 命中则共享该物理 Block，未命中则需要计算

```
请求 A: [system_prompt | user_msg_A]
请求 B: [system_prompt | user_msg_B]

system_prompt → Block 0, Block 1, Block 2（已缓存）
user_msg_A   → Block 3（需要计算）
user_msg_B   → Block 3（需要计算，不同内容）
```

### 限制

- **仅精确匹配**：token 序列必须完全相同，一字之差就无法命中
- **Block 边界对齐**：命中粒度是 Block（如 16 tokens），Prompt 若不是 block_size 的整数倍，最后一个部分块无法参与共享（直到该 Block 被填满并"封住"）
- **tokenizer 一致性**：不同 tokenizer 版本或参数设置可能导致相同文本产生不同 token 序列，破坏缓存复用

## 命中与失效

### 命中

Prefix Cache 命中时：

1. Block Manager 将对应物理 Block 标记为被新 Sequence 引用（引用计数 +1）
2. 新 Sequence 的 Block Table 中，命中部分指向共享的物理 Block
3. 引擎跳过已命中的 Block 的 Prefill 计算，直接从命中边界开始 Prefill 剩余部分

**收益**：节省 Prefill 计算时间（正比于命中 token 数），降低 TTFT。

### 失效

共享 Block 有以下失效情形：

- **显存压力驱逐**：Block Manager 在显存不足时，按 LRU 或其他策略驱逐无活跃引用的共享 Block
- **无引用**：当所有引用该 Block 的 Sequence 完成后，引用计数归零，Block 进入可驱逐状态（但不一定立即释放）
- **主动 invalidate**：某些场景下（如缓存安全隔离），显式清除缓存

## 对不同 Workload 的影响

| Workload | Prefix Cache 收益 |
|----------|-------------------|
| 固定 System Prompt + 不同 user message | 高，System Prompt 命中率接近 100% |
| 多轮对话（同一 session 内） | 中高，历史对话轮次可复用 |
| RAG（固定文档集 + 不同问题） | 中，文档部分可命中，问题部分不可 |
| Agent 任务（反复工具调用，同类 Prompt） | 中，视 Prompt 结构而定 |
| 完全随机请求 | 低，几乎无法命中 |

## 工程设计的边界

Prefix Cache 是一个以空间换时间的机制，引入了以下工程复杂性：

- **引用计数**：Block 需要知道有多少活跃 Sequence 引用它，才能安全回收
- **COW（Copy-on-Write）**：当 beam search 或推测解码需要分叉一个 Sequence 时，共享 Block 必须先复制
- **哈希冲突处理**：Block 哈希可能冲突，需要二次校验（待补充具体方案）
- **跨请求安全隔离**：多租户场景下，不同 tenant 的 KV 不应共享，需要在哈希 key 中加入 tenant 维度

## 三大引擎的 Prefix Cache 实现对比

SGLang、vLLM 和 TensorRT-LLM 各自采用了不同的 Prefix Cache 实现方案，在匹配粒度、并发安全和工程复杂度上存在显著差异。

### SGLang：Radix Tree（基数树）

SGLang 的 RadixAttention 使用压缩 Radix Tree 组织 KV Block：

- **任意 token 边界命中**：通过 `child_key` 路由 + node splitting，可以在任意 token 位置匹配，不需要对齐 block_size 边界。这比 vLLM 的 block 级匹配精确得多
- **LRU 驱逐**：叶子节点优先驱逐，但树结构本身保持完整（非叶子节点只在前缀完全不可达时才回收）
- **cache-aware 负载均衡**：调度器将请求路由到已有前缀缓存的副本，最大化跨请求命中

来源：主站 essay [SGLang KVCache Runtime 架构](/notes/2026/03/14/sglang-kvcache-runtime-architecture/)

### vLLM：链式哈希 + 二分搜索

vLLM 的 Automatic Prefix Caching (APC) 采用哈希表 + lazy 匹配：

- **仅 block 级匹配**：每个 Block（默认 16 tokens）进行哈希，只在 block 边界检查匹配。这限制了匹配的最小粒度为 16 tokens
- **链式哈希**：`hash(tokens[0:block_size]) → block_0 → hash(tokens[block_size:2*block_size]) → block_1 → ...`，形成链式查找
- **lazy 驱逐**：Block 的引用计数管理由 BlockPool 统一处理，驱逐策略基于 LRU

来源：主站 essay [vLLM KVCache Runtime 架构](/notes/2026/03/12/vllm-kvcache-runtime-architecture/)

### TensorRT-LLM：两阶段 Claim + BlockKey

TensorRT-LLM 的实现更贴近底层 C++ 并发模型：

- **两阶段 Claim**：先 `claimUnusedBlocks()` 锁定可复用 Block → 再批量 onboard Sequence。单锁 + 批量操作避免了 Python 的 GIL 依赖，解决了 C++ 环境下的 TOCTOU 竞态条件
- **BlockKey 多维编码**：`BlockKey` 编码了 LoRA ID、多模态哈希、cache_salt 等参数，实现多维度的缓存隔离
- **priority-based LRU**：高优先级 Block 获得保留槽位，相比纯 LRU 命中率提升约 20%

来源：主站 essay [KV Cache 前缀匹配的设计分野](/notes/2026/05/13/kvcache-prefix-matching-design/)

### 三框架对比

| 特性 | SGLang RadixAttention | vLLM APC | TRT-LLM KVCacheManager |
|------|----------------------|----------|------------------------|
| 匹配粒度 | 任意 token 边界 | Block 边界（16 tokens） | Block 边界（可配） |
| 数据结构 | 压缩 Radix Tree | 链式哈希表 | 哈希表 + BlockKey |
| 并发安全 | Python GIL | Python GIL | C++ 两阶段 Claim |
| 缓存隔离 | token 序列 | token 序列 + cache_salt | BlockKey（LoRA/多模态/salt） |
| 驱逐策略 | 叶子节点 LRU | Block LRU | Priority-based LRU |

## 分布式 Prefix Cache

在 PD 分离或多副本部署中，Prefix Cache 需要跨实例共享。

### Mooncake Store：以 KV Cache 为中心的分布式池

Mooncake Store 将 KV Block 存储在 RDMA 可访问的分布式 DRAM 池中：

- **block-hash 寻址**：每个 KV Block 有全局唯一的哈希 key，调度器在分配 Block 前先查询 Store
- **零拷贝 GPU-to-RDMA**：KV 通过 GPUDirect RDMA 直接写入 Store，无需 CPU bounce buffer
- **Agentic workload 实测**（610 条 agent trace）：cache hit rate 从 1.7% 提升至 92.2%，吞吐 3.8×，P50 TTFT 降低 46×，端到端延迟降低 8.6×

来源：主站 reading [vLLM × Mooncake Store](/notes/2026/05/07/vllm-mooncake-store-distributed-kv-cache/)

### LMCache：多级存储 + 多后端

LMCache 提供可插拔的 KV 后端：
- Valkey 连接器：集群模式 / TLS / GLIDE 优化，企业级分布式缓存
- 原生 FS 连接器：零依赖持久化
- S3 L2 Adapter：对象命名 / 容量统计 / DeleteObject 驱逐 / circuit breaker
- cache_salt 隔离：将 salt 写入 ObjectKey / IPC key，切断同内容不同用户的默认共享假设

来源：主站 briefs 2026-03-25, 2026-04-19, 2026-04-25

## HiCache：SGLang 的分层 Prefix Cache

SGLang 的 HiCache 将单层 RadixAttention 扩展为三层存储（GPU/CPU/NVMe）：

- **生产数据**（Novita AI）：TTFT 降低 56%，吞吐 2×
- **生产数据**（Ant Group, DeepSeek-R1-671B）：TTFT 降低 84%
- **HiSparse**：在 GLM-5.1-FP8, 256 并发下实现 3–5× 吞吐，通过 LRU 将不活跃 KV 卸载到 host
- **ShadowRadix**：在 DeepSeek-V4 上实现 4K → 900K 上下文扩展，decode 吞吐仅从 266 → 240 tok/s（<10% 下降）

来源：主站 essay [SGLang KVCache Runtime 架构](/notes/2026/03/14/sglang-kvcache-runtime-architecture/) v1.1

## 模糊前缀匹配（待补充）

精确匹配在 RAG 等场景中存在局限：文档内容相同但 token 化后因上下文不同而略有差异，导致本可复用的 KV 无法命中。模糊 Prefix Cache（approximate/semantic prefix matching）是一个活跃的研究方向，待补充。

## 关联章节

- Prefix Cache 的物理基础：[Paged KV](paged-kv.md)
- 分布式 KV 池与 PD 分离的关系：[PD 分离](pd-disaggregation.md)
- Agent 场景的 Prefix Cache 表现：[工作负载维度](workloads.md)
- 各框架实现差异：[框架对比](frameworks.md)

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 框架搭建 |
| v0.2 | 2026-05-14 | 纳入三框架 Prefix Cache 实现对比、分布式 Prefix Cache（Mooncake Store / LMCache）、HiCache 生产数据
