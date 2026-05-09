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

## 模糊前缀匹配（待补充）

精确匹配在 RAG 等场景中存在局限：文档内容相同但 token 化后因上下文不同而略有差异，导致本可复用的 KV 无法命中。模糊 Prefix Cache（approximate/semantic prefix matching）是一个活跃的研究方向，待补充。
