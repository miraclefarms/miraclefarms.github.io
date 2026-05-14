# 算法 × 系统

> 算法改变 KV 的形状和语义，系统负责 KV 的存储和传输——两者的接口不对齐时，算法收益会在系统层被抵消。

## KV 量化在 Offload / 跨节点传输中的角色

量化后的 KV 在传输路径上有直接的带宽收益：

- FP8 = BF16 的 2×（每 token 字节数减半）
- INT4 = BF16 的 4×

但带宽收益能否落地，取决于**接收端的 attention kernel 是否原生支持量化 KV**：

- 如果接收端需要 dequantize 才能计算，节省的传输带宽部分被 dequantize 计算抵消
- **干净的设计**：传输路径与计算路径用同一套量化格式，避免格式转换

**实践陷阱：量化方案版本需要纳入 cache key**

如果系统在某次部署中从 BF16 切换到 FP8 KV，但 prefix cache 里仍然存有旧格式的 block，会出现字节内容不匹配——旧 block 命中后直接用，attention 计算结果错误。

正确做法：把量化格式（精度、量化粒度）的版本号编码进 block hash，切换格式时整个 cache 自动失效。详见 [生命周期与淘汰](lifecycle.md)。

---

## 稀疏 KV 与 Paged 内存管理的兼容性

Paged KV 的核心假设：**每 block 的 token 是连续的、固定大小的**。

大多数稀疏方案（H2O、SnapKV、PyramidKV）剪掉的是序列中**任意位置**的 token，剪完之后 block 内有"空洞"，破坏了 Paged 的假设。

**三种应对方案：**

| 方案 | 做法 | 优点 | 缺点 |
|------|------|------|------|
| 保留空洞 | block 内 invalid slot 标记为 0 | 实现最简单 | 浪费 HBM slot，碎片增加 |
| 压实（Compaction） | 幸存 token 移到 block 头部 | 无碎片浪费 | 需要更新 block table 的 token-id 映射，开销大 |
| 稀疏感知 kernel | 接受 block 内非连续，靠 mask 过滤 | 最灵活 | kernel 复杂度高，工程成本大 |

**实际部署的选择：**

- SWA（Sliding Window Attention）+ Attention Sink：保留的 token 在 block 边界对齐，与 paged 兼容性最好，工业部署最广泛
- Token 级重要性剪枝（H2O、SnapKV）：通常用压实方案，但需要 kernel 支持变长有效 block

---

## MLA / 线性注意力下的 Prefix Cache 语义

### MLA（Multi-head Latent Attention）

MLA 缓存的不是原始 K/V，而是压缩的 **latent 向量**（低秩表示），每次 attention 时才解压：

- 理论上 prefix cache 仍然可用：给定相同的 token prefix，latent 结果相同，按 token 序列哈希仍然有效
- 但现有的 paged kernel 和 prefix tree 都建在"block 存 K/V"的假设上——需要重写存储结构，把 block 内容改为 latent 格式
- **当前进展**：vLLM / SGLang 已适配 MLA 推理，但 MLA 的 prefix cache 语义尚未完全标准化

### 线性注意力 / 状态空间模型（Mamba、RetNet）

这类模型用**累计状态（state）**替代所有历史 KV：

- state 是对全部历史 token 的**不可逆压缩**，无法分段取 prefix
- 给定"前 N 个 token 的 state"，无法分离出"前 K 个 token 的 state"（K < N）
- **结论：线性注意力 / SSM 没有传统意义上的 prefix cache**

混合架构（Attention 层 + SSM 层交替）则同时有两套状态管理，prefix cache 仅作用于 Attention 层，SSM 层的 state 仍是不可分段的。

---

## 关联章节

- KV 量化的具体方法：[压缩与量化](compression-quantization.md)
- Paged KV 的 block 管理机制：[Paged KV](paged-kv.md)
- cache key 与生命周期的设计：[生命周期与淘汰](lifecycle.md)
- MLA / 线性注意力的算法细节：[Attention 变体](attention-variants.md)
- 稀疏化方法：[稀疏化](sparsity.md)

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 从维度交叉总览拆分，补充三种稀疏兼容方案对比表与 MLA/SSM 语义分析 |
