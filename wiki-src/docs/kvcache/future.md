# 开放问题与未来方向

> KVCache 已经是系统级资源，但它还在快速演化。本章梳理目前看得到的几条主要趋势，按四个维度分组。这一章的内容时效性最强，会随技术进展持续更新。

---

## 本章结构

| 子页面 | 内容 |
|--------|------|
| [算法侧](future-algorithm.md) | 新一代 Attention（MLA / SSM）对 KV 语义的冲击；自适应稀疏 / 量化；学习型 KV 压缩；Position-Invariant Prefix Cache |
| [系统侧](future-system.md) | CXL 与近存计算；KV 池作为独立分布式存储产品；KV / 权重 / 激活的统一内存抽象；可观测性 |
| [部署侧](future-deployment.md) | Attention / MLP 分离；多 LoRA 共享 KV；Serverless LLM 冷启动；多租户公平性 |
| [工作负载演进](future-workloads.md) | Agent 长程规划的 KV 树状结构；多模态 KV；Reasoning long-decode 主导下的新权衡；用户 KV 粘性演化 |

---

## 跨维度的开放问题

把各维度的分散方向汇总成几个高价值的"未解之题"：

| 问题 | 难点 |
|------|------|
| Position-invariant prefix cache | 既要复用又要保持位置正确 |
| 标准化的 KV cache 协议 | 各引擎的实现差异大，统一接口难定 |
| 真正可移植的 KV 量化 | 不同硬件、不同 kernel 都能直接用 |
| KV cache 命中率的可预测性 | 当前难以提前估计某种部署下的命中率 |
| 多租户的公平性保证 | 共享 cache 池下的 SLO 隔离 |

---

## 时间线参考（2025 视角）

| 状态 | 技术 |
|------|------|
| **已成熟** | PagedAttention、Prefix Cache、GQA、FP8 KV、PD 分离（部分场景） |
| **2024–2025 进行时** | 跨实例 KV 池、cache-aware routing、MLA 深度部署、长上下文 SP/CP |
| **早期研究** | position-invariant cache、unified memory、CXL 落地、agent-aware 调度 |

KVCache 还远不是 settled science——它仍处在"还在发明基础概念"的阶段。

---

## 关联章节

- 当前实现的边界：[框架对比](frameworks.md)
- 已经被讨论的具体问题：[维度交叉](crossings.md)

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 框架搭建 |
| v0.2 | 2026-05-14 | 拆分为子页面，本页保留总览、开放问题汇总与时间线 |
