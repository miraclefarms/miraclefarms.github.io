# 未来方向 — 算法侧

> 算法侧的变化正在从根本上重写 KVCache 的语义假设——不是优化 KV，而是改变 KV 是什么。

## 1. 新一代 Attention 与 KV 复用语义

当前的 prefix cache、跨节点 KV 池等基础设施都建立在一个核心假设上：

> **K/V 是关于 token 序列的可拼接结果**：给定相同的 token prefix，不论是谁的请求，KV 必然相同。

但新一代 attention 正在动摇这个假设：

### MLA（Multi-head Latent Attention）

DeepSeek 提出的 MLA 将 K/V 压缩为低秩潜在向量（latent vector），而非标准的多头 K/V：

- 存储的不是 K/V，而是压缩的 latent
- 在每次 attention 计算时才从 latent 解压出 K/V
- **影响**：prefix cache 需要改为缓存 latent 而非 K/V；radix tree 的 block 结构仍适用，但 block 内容的语义不同

### 线性注意力 / 状态空间模型（SSM）

Mamba、RetNet 等模型用固定大小的 "state" 替代历史 KV：

- 没有传统意义上的 KV cache——历史被压缩进一个固定维度的 state
- **影响**：prefix cache 的整套概念不再适用；新的问题变成 "state 如何在请求间复用"
- 混合架构（Attention + SSM 层交替）同时有两套状态管理，更复杂

### 设计挑战

当模型不再满足"可拼接 KV"的假设，整套基础设施需要新的语义抽象：

- "共享 prefix" 的概念需要泛化为 "共享模型状态"
- Block 粒度的复用需要与新的状态压缩格式对齐
- 跨框架的 KV 协议（如 LMCache 的接口）需要支持多种后端格式

---

## 2. 自适应稀疏 / 量化的"按需精度"

当前稀疏化和量化方案的共同限制：**参数是全局的**——决定后所有 token 一视同仁。

**研究方向：根据 attention 模式动态调整每个 token 的精度**

- 识别"重要" token（attention score 高、语义关键）→ 保留 FP16 精度
- 识别"边缘" token（attention score 低、远距离历史）→ 降为 INT4/FP8
- 决策粒度：per-token、per-head、per-layer

**挑战：**

- 重要性估计本身有计算开销，需要高效的 online 决策机制
- Query-dependent 的重要性：同一个 token 对不同 query 的重要性不同——Quest（2024）等研究的核心问题
- 与 block-based prefix cache 的兼容性：部分 block 内的 token 精度不同，会破坏 block 的可复用性

**与现有工作的关系：**

- H2O、SnapKV、PyramidKV 是静态规则的稀疏化——迈向动态的第一步
- Quest 是 query-aware 的页面选择——更接近"按需精度"的方向
- 端到端训练的稀疏感知模型是终极形态

---

## 3. 学习型 KV 压缩的端到端训练

把"KV 压缩器"作为模型的一部分参与训练，而非事后压缩：

**优势：**

- 模型知道哪些信息需要保留，压缩质量远超事后规则
- 精度-体积曲线优于 SnapKV / H2O 等 training-free 方法

**代价：**

- 模型架构与压缩方案绑定：换一个压缩方案需要重训模型
- 部署侧失去"调整压缩率"的灵活性
- 压缩后的 KV 可能不再满足标准 attention kernel 的接口假设

**代表方向：**

- KVSharer、CLA（Cross-Layer Attention）：让模型学习跨层共享 KV
- 压缩感知训练（Compression-Aware Training）：在训练中模拟 KV 量化的误差
- 端到端的 KV token 剪枝训练

---

## 4. Position-Invariant Prefix Cache（研究前沿）

RAG 和 Agent 场景的核心痛点：同一段文本在不同 prompt 位置产生不同的 KV，无法精确复用。

**研究方向：**

- [CacheBlend (2024)](https://arxiv.org/abs/2405.16444)：预计算文档 KV，加载时按当前 offset 修正 RoPE
- [Prompt Cache (2023)](https://arxiv.org/abs/2311.04934)：训练模型让部分 attention 头对位置不敏感
- 绝对位置编码的替代：不依赖绝对位置的 encoding 方案从根本上消除这个问题

**难点：**

- RoPE 是当前最优的位置编码方案，与 KV 深度耦合
- 修正 RoPE 的近似方法会引入误差，对精度影响需要系统性评估
- 工业部署需要修改 attention kernel，工程代价高

---

## 关联章节

- 当前稀疏化方法：[稀疏化](sparsity.md)
- 当前量化方法：[压缩与量化](compression-quantization.md)
- 算法变化对系统的影响：[维度交叉](crossings.md)
- 系统侧的未来方向：[未来方向 — 系统侧](future-system.md)

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 从未来方向总览拆分，细化 MLA/SSM/动态精度/端到端训练四条路线 |
