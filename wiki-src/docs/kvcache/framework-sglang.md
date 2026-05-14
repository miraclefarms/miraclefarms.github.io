# SGLang

[SGLang](https://github.com/sgl-project/sglang) 以高吞吐和 Prefix Cache 效率著称，是 RadixAttention 和 HiCache 的原始提出者。

!!! warning "内容时效性"
    SGLang 更新极快，下述特性可能已迭代。请以项目的最新文档和 Release Notes 为准。

---

## KVCache 架构

### RadixAttention

SGLang 的核心 KVCache 创新：使用压缩 Radix Tree 组织 KV Block。

- **任意 token 边界匹配**：不需要像 vLLM APC 那样对齐 block_size，可以在任意 token 边界命中 prefix
- **Cache-aware 负载均衡**：将请求路由到已有最长共同 prefix 的副本，最大化命中率
- **HiRadixCache**：HiCache 的 Radix Tree 扩展，支持多级 tier 和 SLRU 淘汰策略

### Block Manager

与 vLLM 类似的分页管理设计，但在以下方面有差异：

- Block 命中的边界更灵活（RadixAttention 的特性）
- SLRU（Segmented LRU）淘汰：比纯 LRU 更好地保护热门 prefix block 不被驱逐

---

## 主要特性

### HiCache（分层 KVCache）

HiCache 是 SGLang 的分层 KV offload 解决方案：

- 三层存储：GPU HBM → Host DRAM → NVMe
- 自适应 prefill delayer：延迟 prefill 执行以批量复用 offloaded KV

**生产实测数据：**

| 部署场景 | 指标 | 效果 |
|---------|------|------|
| Novita AI | TTFT | −56% |
| Novita AI | 吞吐 | 2× |
| Ant Group (DeepSeek-R1-671B) | TTFT | −84% |

### HiSparse

稀疏 KV cache 与 offload 的结合：

- 在 GLM-5.1-FP8 256 并发下实现 3–5× 吞吐提升
- LRU offload 不活跃的 KV 到 host memory

### ShadowRadix

超长上下文扩展支持（DeepSeek-V4 实测）：

- 上下文从 4K 扩展到 900K
- decode 吞吐仅从 266 → 240 tok/s（< 10% 下降）

### PD 分离

SGLang 的 disaggregated serving 实现：

- **GPU Staging Buffer**（PR #19890）：将 scatter head slices gather 为 contiguous memory，再做 bulk RDMA transfer
- Qwen3.5 Prefill TP4 + Decode DEP4 下：TPS/GPU 提升约 5×
- 支持 TP 不对称的 Prefill/Decode 节点配置

### KV 量化

- FP8 KV Cache：原生支持
- TurboQuant 集成（PR #23135）：3.88× 压缩，保持 93–105% decode 吞吐
- Tokenspeed MLA（PR #24925）：支持 FP8 KV cache + MLA 架构

### Attention Backend

- 深度集成 FlashInfer，是 SGLang 最优化的 attention kernel 路径
- FlashAttention 作为备选

---

## 与 vLLM 的关键差异

| 维度 | vLLM | SGLang |
|------|------|--------|
| Prefix Cache 边界 | Block 对齐（16 token） | 任意 token 边界（RadixAttention） |
| 淘汰策略 | LRU | SLRU（更好保护热 block） |
| Offload | 多级 tier（PR #40020） | HiCache（三层，adaptive prefill delay） |
| Attention Backend | FlashAttention / FlashInfer / Triton | 主要 FlashInfer |
| PD 分离状态 | 进行中（MultiConnector） | 生产可用（GPU Staging Buffer） |

---

## 关联章节

- RadixAttention 的原理：[Prefix Cache §RadixAttention](prefix-cache.md)
- HiCache 与分层存储：[存储层级](storage-hierarchy.md)
- 框架总览与对比表：[框架对比](frameworks.md)

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 从框架对比总览拆分，整理 RadixAttention / HiCache / ShadowRadix / PD 等核心特性 |
