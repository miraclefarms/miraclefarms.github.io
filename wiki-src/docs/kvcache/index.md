# KVCache 综述

KVCache 早已不再是 Transformer 推理里"避免重复计算的小 trick"。在今天的生产推理集群里，它是与算力、带宽并列的**一等公民资源**——决定着成本结构、调度策略、SLO 是否可达。从 CPU 寄存器里的临时变量，演化成了类似数据库 Buffer Pool 的系统级资源池。

本综述把 KVCache 视为一个被四股力量同时拉扯的复合问题，按"四个正交维度 + 维度交叉"的方式组织：

1. **算法维度**——attention 变体、稀疏、压缩、量化在不断改写 KV 张量本身的形状和语义
2. **系统维度**——KV 在 HBM → DRAM → SSD → 远端的多级存储里流动
3. **部署维度**——PD 分离、并行切分、路由、扩缩容让 KV 跨进程跨节点存在
4. **工作负载维度**——多轮、Agent、Coding、RAG、Reasoning 让 KV 的访问模式高度异质

讨论 KVCache 时之所以容易"鸡同鸭讲"，是因为不同的人从不同维度切入。本综述的目标就是把"骨架"立起来，让算法、系统、部署、工作负载这四类讨论可以在一张统一的导航图上挂载具体技术点、论文和系统。

## 范围与非范围

- ✅ 覆盖：以**推理时**的 KVCache 为中心的算法 + 系统 + 部署 + 负载视角
- ❌ 不覆盖（或仅作背景）：训练时的 attention 优化（FlashAttention 类）、KVCache 之外的 prefill/decode 通用优化、模型架构本身的设计动机

## 阅读路径

| 路径 | 推荐章节 |
|------|----------|
| **新人路径** | [第一性原理](first-principles.md) → [基础概念](basics.md) → [Attention 变体](attention-variants.md) → [运行时架构](runtime-architecture.md) → [存储层级](storage-hierarchy.md) → [PD 分离](pd-disaggregation.md) → [评估方法](evaluation.md) |
| **算法路径** | [Attention 变体](attention-variants.md) → [稀疏化](sparsity.md) → [压缩与量化](compression-quantization.md) → [维度交叉](crossings.md) → [未来方向](future.md) |
| **系统路径** | [运行时架构](runtime-architecture.md) → [Paged KV](paged-kv.md) → [Prefix Cache](prefix-cache.md) → [KV Offload](offload.md) → [生命周期与淘汰](lifecycle.md) → [维度交叉](crossings.md) |
| **部署路径** | [PD 分离](pd-disaggregation.md) → [并行切分](parallelism.md) → [路由与亲和性](routing.md) → [弹性与故障](elasticity.md) → [维度交叉](crossings.md) |
| **应用路径** | [工作负载维度](workloads.md) → [维度交叉](crossings.md) → [评估方法](evaluation.md) |

## 章节结构

**导论**

- [第一性原理](first-principles.md) — 为什么 KVCache 是一等公民资源
- [基础概念](basics.md) — KVCache 是什么、与 Attention 的关系、规模直觉

**四个维度**

- 算法（横向）：[Attention 变体](attention-variants.md)、[稀疏化](sparsity.md)、[压缩与量化](compression-quantization.md)
- 系统（纵向）：[运行时架构](runtime-architecture.md)、[存储层级](storage-hierarchy.md)、[Paged KV](paged-kv.md)、[Prefix Cache](prefix-cache.md)、[KV Offload](offload.md)、[生命周期与淘汰](lifecycle.md)
- 部署：[PD 分离](pd-disaggregation.md)、[并行切分](parallelism.md)、[路由与亲和性](routing.md)、[弹性与故障](elasticity.md)
- 工作负载：[多轮 / Agent / Coding / RAG / Reasoning 画像](workloads.md)

**横向**

- [维度交叉](crossings.md) — 真实生产系统是四维空间里的点
- [评估方法](evaluation.md) — 指标体系、Workload 建模、可观测性
- [未来方向](future.md) — 算法 / 系统 / 部署 / 负载演进

**附录**

- [框架对比](frameworks.md) — vLLM / SGLang / TensorRT-LLM / LMCache / Mooncake
- [术语表](glossary.md)
- [参考资料](references.md)

## 快速参考：KVCache 内存估算

```
每 token KV 大小 = 2 × num_layers × num_kv_heads × head_dim × dtype_bytes
```

以 LLaMA-3 70B（BF16）为例：num_layers = 80，num_kv_heads = 8（GQA），head_dim = 128 →
**每 token KV ≈ 327 KB**。

一个 80GB HBM 的 GPU，在模型权重之外约剩 20-30 GB 用于 KVCache，大约能缓存 **60,000–90,000 个 token** 的上下文（不含其他运行时开销）。实际可用量受批量大小、调度策略和碎片率影响。
