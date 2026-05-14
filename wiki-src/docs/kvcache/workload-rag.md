# RAG / 检索增强

> **工作负载画像**：Prefix 复用度混合（system 高、docs 低）｜上下文中等｜Decode 短｜文档块位置不固定

## 特征

- Prompt 结构固定：`system prompt` + `检索文档块` + `用户问题`
- 文档块组合每次不同：top-k 检索结果随 query 变化，组合方式几乎无法预测
- 单个文档块高度重复：同一段文档在不同请求中反复出现，是潜在的 prefix cache 目标
- System prompt 全员共享：100% 命中——大多数 RAG 应用的唯一确定收益

---

## 关键 KV 问题

### 1. 文档块级 KV 复用的位置编码障碍

这是 RAG 工作负载的核心难题：

同一文档块在不同 prompt 里的**位置（token offset）不同**，而 KV 中的 Key 与 RoPE（Rotary Position Embedding）强绑定，位置不同则 K 值不同，精确 prefix cache **不可能**在文档块级别命中。

```
请求 A：[system(0-99)] [doc1(100-299)] [doc2(300-499)] [query(500-...)]
请求 B：[system(0-99)] [doc2(100-299)] [doc1(300-499)] [query(500-...)]
                           ↑ 同一个 doc2，但 token 位置不同，KV 完全不同
```

### 2. 当前的解决方向

**方向一：固定 docs 在 prompt 中的位置（工程可行）**

- 把 RAG prompt 结构设计为 `system + docs_fixed_order + query`
- 检索文档按固定规则排序（如文档 ID 字母序），确保同一文档块每次出现在相同 offset
- 代价：top-k 的排名信息丢失；文档数量变化时顺序改变，命中率下降

**方向二：位置无关 KV 压缩（研究阶段）**

- [CacheBlend (2024)](https://arxiv.org/abs/2405.16444)：把文档块 KV 预计算为"位置无关"格式，加载时按当前 offset 调整 RoPE
- [Prompt Cache (2023)](https://arxiv.org/abs/2311.04934)：通过模型训练让部分 attention 头产生位置无关的 KV
- 代价：需要修改模型推理 kernel，或引入专用训练；目前工业部署罕见

**方向三：退而求其次，只做 system prompt 级复用**

- System prompt 全员共享、完全固定：100% 命中
- docs 部分接受 miss，只有 system prompt 带来收益
- 简单可靠，是目前大多数生产 RAG 系统的实际做法

### 3. 检索文档的 KV 预计算（预热策略）

对于文档库固定的场景（知识库问答、企业文档检索），可以**提前 prefill 所有文档**，把 KV 存入 L3/L4 持久层：

- 文档块 KV 以固定 offset（如文档块在 prompt 中的第 k 个位置）预计算
- 新请求来时直接从持久层加载，跳过 prefill
- 限制：文档块位置必须固定（top-1 或固定排序），文档库变更时需要失效和重建

---

## 工程配置建议

| 配置项 | 推荐值 / 策略 | 原因 |
|--------|-------------|------|
| Prefix Cache | **开启**（仅 system prompt 受益） | System prompt 100% 命中，docs 部分大概率 miss |
| Prompt 结构 | system → docs（固定排序）→ query | 最大化 prefix 命中深度 |
| 检索 top-k | 限制为 top-1 或 top-3 + 固定排序 | 减少文档组合空间，提升命中率 |
| 文档块 KV 预热 | 固定文档库场景可用 | 绕过 prefill 直接加载 KV |
| 分布式 KV 池 | 推荐（跨实例共享 system prompt KV） | System prompt 全员共享，分布式命中收益高 |

---

## 关键指标

- **System prompt 命中率**：理论上应 100%，否则说明路由或 cache 管理有问题
- **文档块命中率**：区分 system prompt 与 docs 两部分单独统计，才能诊断位置编码问题
- **TTFT**：RAG 场景 decode 短，TTFT 主导用户感知延迟

---

## 关联章节

- Prefix Cache 的精确匹配语义与 RoPE 约束：[Prefix Cache](prefix-cache.md)
- 持久化文档 KV 的分层存储：[存储层级](storage-hierarchy.md)
- 位置无关压缩的研究方向：[压缩与量化](compression-quantization.md)、[未来方向 — 算法侧](future-algorithm.md)

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 从工作负载总览拆分，补充位置编码障碍分析与三种解决方向 |
