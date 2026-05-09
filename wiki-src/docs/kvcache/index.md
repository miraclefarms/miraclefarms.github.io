# KVCache

KVCache 是 LLM 自回归解码过程中产生的核心中间状态。它缓存每一层 Transformer attention 计算所产生的 Key 和 Value 张量，使得 Decode 阶段每步生成新 token 时，不必对已有的上下文重复计算。

这个机制听起来简单，但它几乎影响推理系统的每一个关键指标：

- **首 token 延迟（TTFT）**：Prefill 阶段产生初始 KV Cache 的速度
- **生成吞吐（tokens/s）**：Decode 阶段访问和写入 KVCache 的带宽
- **显存占用**：KVCache 的大小直接影响可并发的请求数（批量大小上限）
- **长上下文能力**：序列越长，KVCache 越大，显存压力越高
- **Prefix Cache 命中率**：共享前缀的 KV 是否可以复用
- **PD 分离架构的传输成本**：Prefill 节点产生的 KV 需要传输给 Decode 节点

理解 KVCache，是理解现代 LLM 推理系统工程的核心入口。

## 阅读路径

本主题按照从基础到进阶的顺序组织：

1. **[基础概念](basics.md)** — 什么是 KVCache，为什么需要它，与 Attention 的关系
2. **[运行时架构](runtime-architecture.md)** — 推理引擎中 KVCache 的位置，Block Manager、Allocator 与 Scheduler 的关系
3. **[Prefix Cache](prefix-cache.md)** — 共享前缀的 KV 复用，精确匹配的边界与局限
4. **[Paged KV](paged-kv.md)** — 分页管理如何解决显存碎片问题，PagedAttention 的直觉
5. **[KV Offload](offload.md)** — 多级存储层次，Offload 的收益与代价
6. **[PD 分离中的 KVCache](pd-disaggregation.md)** — 跨节点 KV 传输的系统影响
7. **[评估方法](evaluation.md)** — 如何测量 KVCache 系统的性能
8. **[框架对比](frameworks.md)** — 主流推理引擎的 KVCache 实现概览
9. **[术语表](glossary.md)** — 快速查询关键术语
10. **[参考资料](references.md)** — 论文、文档、Issue 与博客文章

## 快速参考：KVCache 内存估算

对于一个模型，单个 token 的 KV Cache 大小可以用以下公式估算：

```
每 token KV 大小 = 2 × num_layers × num_kv_heads × head_dim × dtype_bytes
```

以 LLaMA-3 70B（BF16）为例：
- num_layers = 80，num_kv_heads = 8（GQA），head_dim = 128
- 每 token KV = 2 × 80 × 8 × 128 × 2 = **327 KB**

一个 80GB HBM 的 GPU，在模型权重之外约剩余 20-30GB 用于 KVCache，大约能缓存 **60,000–90,000 个 token** 的上下文（不含其他运行时开销）。实际可用量受批量大小、调度策略和碎片率影响。
