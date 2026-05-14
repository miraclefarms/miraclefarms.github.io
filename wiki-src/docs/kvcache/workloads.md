# 工作负载维度

> **维度四**：KV 的访问模式因场景而异。同一套引擎、同一个模型，在不同 workload 下的最优配置可以截然不同。

本章用统一的"画像维度"刻画五类典型工作负载，每种场景的 prefix 复用度、上下文长度、decode 长度、并发模式各不相同。

## 统一画像维度

四个变量描述任意工作负载：

| 维度 | 范围 |
|------|------|
| **Prefix 复用度** | 0%（完全独立）→ 100%（System Prompt 全员共享） |
| **上下文长度** | 1K → 1M token |
| **Decode 长度** | 几十 token → 数十万 token（reasoning） |
| **并发模式** | 短稳定流 → 突发 burst → 长持续连接 |

落到这个 4D 空间里，KV 关注点完全不同。

---

## 五类工作负载

| 工作负载 | Prefix 复用度 | 上下文长度 | Decode 长度 | 主要 KV 痛点 |
|---------|--------------|------------|------------|------------|
| [多轮对话](workload-multiturn.md) | 高（历史轮次） | 增长式 | 中 | session 亲和、swap |
| [Agent 协作](workload-agent.md) | 中高（task 级） | 中长 | 中 | tool 注入续接、跨实例共享 |
| [Coding / 长上下文](workload-coding.md) | 极高（文件不变） | 极长 | 中 | SP/CP + L3/L4 持久 |
| [RAG / 检索增强](workload-rag.md) | 中（system 高、docs 低） | 中 | 短 | 位置无关 prefix |
| [Reasoning / Long CoT](workload-reasoning.md) | 低 | 短 | 极长 | 容量、抢占代价 |

---

## 不同场景对算法 / 系统 / 部署的诉求

| 场景 | 算法首选 | 系统首选 | 部署首选 |
|------|---------|---------|---------|
| 多轮 | GQA、StreamingLLM 风格 | prefix cache + L2 swap | session affinity 路由 |
| Agent | GQA + 块对齐 prompt | 细 block size + 分布式 KV 池 | cache-aware 多副本 |
| Coding | GQA + KV 量化 | SP/CP + L4 KV 池 | 持久 prefix cache |
| RAG | 位置无关压缩（研究中） | 文档块级 prefix tree | tenant-aware 路由 |
| Reasoning | KV 量化（FP8） | 大 L2 容量 + swap | 大 batch decode 池，PD 不分 |

---

## 关联章节

- 各算法路线的细节：[Attention 变体](attention-variants.md)、[稀疏化](sparsity.md)、[压缩与量化](compression-quantization.md)
- 路由的具体机制：[路由与亲和性](routing.md)
- 工作负载与维度交叉的进一步讨论：[维度交叉](crossings.md)

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 框架搭建 |
| v0.2 | 2026-05-14 | Agent 章节纳入 Claude Code 缓存工程实证及 Mooncake Store agentic trace 实测数据 |
| v0.3 | 2026-05-14 | 拆分为子页面，本页保留总览与对比表 |
