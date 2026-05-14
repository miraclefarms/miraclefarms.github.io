# 生命周期与淘汰

> **系统维度（纵向）之六**：KV Block 从分配到回收的完整生命周期、淘汰策略、一致性问题。

[Paged KV](paged-kv.md) 解决了"KV 怎么放"，本章解决"KV 什么时候走、谁决定它走、走了之后怎么办"。

## 1. KV Block 的状态机

一个物理 Block 在引擎中的典型状态：

```
       ┌──────────┐     allocate     ┌──────────┐
       │   Free   │ ───────────────→ │  Active  │
       └──────────┘                  └─────┬────┘
            ▲                              │
            │ release                      │ seal (Block 写满)
            │                              ▼
       ┌──────────┐  evict (refcount=0) ┌──────────┐
       │  Cached  │ ◄────────────────── │  Sealed  │
       │ (可命中) │                      │ (引用中) │
       └──────────┘                      └──────────┘
            │ swap-out / drop                   │ refcount=0
            ▼                                   ▼
       ┌──────────┐                       ┌──────────┐
       │ Swapped  │                       │  Cached  │
       └──────────┘                       └──────────┘
```

关键转移：

- **Free → Active**：Sequence 申请新 Block 时分配
- **Active → Sealed**：Block 写满 block_size 个 token，可参与 Prefix Cache
- **Sealed → Cached**：所有引用 Sequence 都结束，但 Block 内容仍可被未来请求命中
- **Cached → Swapped / Free**：显存压力下被淘汰

## 2. 引用计数

物理 Block 是否能回收，由**引用计数（refcount）**决定：

- 每个引用此 Block 的 Sequence 占一个计数
- Sequence 完成或显式释放时计数 -1
- 计数归零后 Block 进入"可回收候选"，但不一定立即释放

**Copy-on-Write（COW）** 的触发：

- 多个 Sequence 共享一个 Sealed Block（典型场景：Prefix Cache、beam search 分叉）
- 其中一个要写入这个 Block 的下一个 slot 时，必须先复制出独立副本，把 refcount 从被写的副本上拆开

## 3. 淘汰策略

当 Free Block 不足时，触发淘汰。常见策略：

### LRU（最近最少使用）

最简单也最常用。维护一个按"最后访问时间"排序的链表，淘汰链表尾部。

- 优点：实现简单，对高命中场景友好
- 缺点：扫描类负载（一次性长 prompt）会污染缓存，把高价值 prefix 挤出去

### LFU（最近最常使用）

按访问频次排序。
- 优点：长期高频前缀更稳定地保留
- 缺点：新加入的高价值 prefix 难以"上位"，需 aging 机制

### Cost-Aware Eviction

不同 KV 的"重建成本"不同：

- 长 prefix block：重新 prefill 成本高，应优先保留
- 短或孤立 block：重建便宜，可以先丢
- 量化后的 KV：占用小但精度恢复成本高

公式形式：$\text{score} = w_1 \cdot \text{recency} + w_2 \cdot \text{rebuild\_cost} + w_3 \cdot \text{hit\_rate}$。

工业实现里 SGLang RadixAttention、LMCache 等都引入了类似 cost-aware 的考量。

### 引擎实践对比

| 引擎 | 默认策略 | 备注 |
|------|---------|------|
| vLLM (APC) | LRU on sealed blocks | 简单稳定 |
| SGLang RadixAttention | tree 结构 + LRU 叶子节点 | 拓扑感知淘汰 |
| TensorRT-LLM | 可配置（LRU/LFU） | 提供调参接口 |
| LMCache | 多级（per-tier）淘汰 | L1→L2→L3 逐层下沉 |

## 4. TTL、租约、显式 pin

某些场景需要更细的控制：

| 机制 | 用途 |
|------|------|
| **TTL** | 系统 prompt 等高价值 prefix 设置较长 TTL，避免被偶发流量挤掉 |
| **租约** | 多租户场景下，按 tenant 配额分配 KV 容量 |
| **显式 pin** | 业务侧显式声明"这段 prefix 必须保留"，例如热门 RAG 文档块 |
| **跨会话 anchor** | 同一用户多次请求间锚定 prefix，跨 session 复用 |

## 5. 多副本一致性

当 KV 在多节点共享（典型：[PD 分离](pd-disaggregation.md) 中跨 P/D 节点的 prefix cache，[L4 远端 KV 池](storage-hierarchy.md)），出现一致性问题：

| 问题 | 描述 | 常见处理 |
|------|------|---------|
| **写入竞争** | 两个节点同时为同一 prefix 计算 KV | 第一写入者 wins，其他丢弃 |
| **失效广播** | tokenizer 升级、量化方案切换、模型更新 | 通过 epoch 号或 cache version 失效 |
| **跨节点驱逐协调** | 多节点 LRU 信息不一致 | 通常容忍（命中率轻微下降可接受） |

工业上多采取**最终一致 + 失败重算**的设计：跨节点协议尽量简单，cache miss 的代价是重新 prefill，可接受。

## 6. 与模型权重热更新的交互

当模型权重切换时（升级、A/B 测试、tenant 隔离），所有 KV 必须失效：

- KV 是关于"特定模型在特定 token 序列下的中间状态"，模型变了 KV 就毫无意义
- 切换通常通过**蓝绿部署**：新副本加载新权重，路由切流，旧副本 drain 后回收
- 跨实例 KV 池要在 cache key 中纳入**模型版本 hash**

类似问题也出现在：

- LoRA / adapter 切换——不同 adapter 的 KV 不可混用
- 量化方案切换——KV 字节内容不同（详见 [压缩与量化](compression-quantization.md)）

## 7. 安全与隔离

多租户场景的隔离要求：

- **数据隔离**：tenant A 的 KV 绝不能被 tenant B 命中
- **侧信道**：通过命中率差异推断他人 prefix 的攻击，研究阶段
- **配额**：单租户的 KV 占用不能耗尽全局池

工程实践：cache key 中纳入 tenant_id，按 tenant 分桶维护 LRU，独立配额。

## 8. SCBench 的生命周期视角

SCBench（arxiv 2412.10319）从 KV 生命周期的角度重新理解长上下文方法，将 KV 的一生拆解为四个阶段：

| 阶段 | SCBench 场景 | 工程含义 |
|------|-------------|----------|
| **Generation** | KV 产生与写入 | Prefill/Decode 时的 KV 分配和填充 |
| **Compression** | 压缩/剪枝/摘要 | KV 不再增长的收敛过程 |
| **Retrieval** | 从 KV 中检索信息 | 对远距离历史 token 的注意力查询 |
| **Loading** | KV 被新请求复用 | 跨请求 Prefix Cache 命中 |

**核心发现：**
- Sub-O(n) 内存方法在多轮场景下系统性退化——丢弃的 KV 在新的 query 下可能变得关键
- O(n) 稀疏注意力方法在跨请求复用下保持或提升性能，说明"保留所有 KV 但稀疏读取"比"丢弃 KV"更稳健
- 这意味着：在多轮和 Agent 场景下，**精确前缀缓存的价值被低估了**

来源：主站 reading [SCBench](/notes/2026/04/21/scbench-kv-cache-lifecycle-analysis/)，主站 essay [KV Cache Agent Benchmark](/notes/2026/04/08/kvcache-agent-long-context-benchmark/)

## 关联章节

- Block / refcount / COW 的基础机制：[运行时架构](runtime-architecture.md)
- 不同 workload 对淘汰策略的不同敏感度：[工作负载维度](workloads.md)
- 跨节点一致性问题在 [PD 分离](pd-disaggregation.md) 中的具体表现
- SCBench 的评估方法论：[评估方法](evaluation.md) §SCBench

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 框架搭建 |
| v0.2 | 2026-05-14 | 新增 SCBench 生命周期四阶段视角及核心发现（Sub-O(n) 多轮退化、O(n) 稀疏注意力稳健性）；更新关联章节
