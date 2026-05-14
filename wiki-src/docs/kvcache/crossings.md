# 维度交叉

> 真实生产系统不在任何单一维度内闭合，而是四维空间里的一个点。本章列出最重要的交叉问题，作为深入调研的"高价值地带"。

四个维度（[算法](attention-variants.md) × [系统](storage-hierarchy.md) × [部署](pd-disaggregation.md) × [工作负载](workloads.md)）两两交叉，得到六组组合。下面挑出每组里最值得关注的几个具体问题。

---

## 本章结构

| 子页面 | 核心问题 |
|--------|---------|
| [算法 × 系统](crossing-algo-system.md) | KV 量化在传输中的角色；稀疏 KV 与 Paged 管理的兼容性；MLA / SSM 下的 prefix cache 语义 |
| [系统 × 部署](crossing-system-deployment.md) | PD 分离下 prefix cache 归 P 还是归 D；L4 KV 池与本地 cache 的层级协议；cache-aware routing 接口标准化 |
| [部署 × 工作负载](crossing-deployment-workload.md) | Agent 与 PD 分离的契合度；Coding 下 SP 与 prefix cache 的取舍；Reasoning 是否需要 PD 分离 |
| [算法 × 工作负载](crossing-algo-workload.md) | 多轮对话适合哪类稀疏；RAG 的位置不变性压缩可行性；Coding 下增量 KV + 跨层共享的组合 |
| [四维联动范式](crossing-paradigms.md) | Mooncake / SGLang / vLLM 生态 / TRT-LLM 四种工业范式的四维分解与选型参考 |

---

## 为什么维度交叉是高价值问题

单维度的优化（更好的稀疏算法、更快的 Paged 内存、更精细的路由）往往有清晰的论文和工程实现。
真正困难的是**跨维度的组合**：

- 算法优化 A 在系统 B 上是否仍然有效？
- 部署策略 C 对工作负载 D 是收益还是负担？

这类问题的答案是"视情况而定"——而"情况"本身就是四维空间里的坐标。

---

## 关联章节

- 各范式背后的具体框架：[框架对比](frameworks.md)
- 评估跨维度组合的方法学：[评估方法](evaluation.md)
- 未来可能的新范式：[未来方向](future.md)

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 框架搭建 |
| v0.2 | 2026-05-14 | 拆分为子页面，本页保留总览与导航 |
