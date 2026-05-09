# 基础概念

## 什么是 KVCache

KVCache（Key-Value Cache）是 Transformer 模型在**自回归解码（autoregressive decoding）**过程中，对已计算的 Attention Key 和 Value 张量进行缓存的机制。

Transformer 的 Attention 计算公式为：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

在解码阶段，每生成一个新 token，都需要用新 token 的 Query 去关注**所有之前的 Key 和 Value**（包括 Prompt 以及之前生成的 token）。如果不做缓存，每步都要重新对整个上下文做矩阵运算，复杂度随序列长度平方增长。KVCache 将这些 K/V 张量存下来，每步只需计算当前 token 的 Q，然后与缓存中的 K/V 做 attention，将每步复杂度降为线性。

## Prefill 与 Decode 中的角色

LLM 推理分为两个阶段，KVCache 在两个阶段中的角色完全不同：

### Prefill（预填充）

Prefill 处理输入 Prompt。它是一次**大型矩阵运算**：所有 Prompt token 并行经过每一层的 Attention 和 FFN，计算量集中，GPU 利用率高（compute-bound）。

Prefill 结束后，每一层 Transformer 都产生了 Prompt 所有 token 对应的 K 和 V 张量，这些张量写入 KVCache，供后续 Decode 使用。

### Decode（解码）

Decode 每步生成一个 token。对每一层 Attention：

1. 用当前 token 的 embedding 计算新的 Q、K、V
2. 将新的 K、V **追加到 KVCache 中**
3. 用新的 Q 对 KVCache 中所有 K 做点积（计算 attention scores）
4. 用 attention scores 对 KVCache 中所有 V 做加权求和

Decode 是**memory-bound**：每步计算量小，但需要大量显存带宽来读取 KVCache。这也是为什么 KVCache 的大小和访问效率对解码吞吐影响巨大。

## KVCache 的数据结构

对于一个层数为 $L$、KV head 数为 $H_{kv}$、head 维度为 $d$、数据类型为 dtype 的模型，处理长度为 $T$ 的序列产生的 KVCache 大小为：

```
KVCache 大小 = 2 × L × H_kv × d × T × sizeof(dtype)
```

- 系数 `2`：K 和 V 各一份
- 对于使用 GQA（Grouped-Query Attention）的模型，$H_{kv}$ 小于 $H_q$（Query head 数），显著减少 KVCache 大小
- 对于 MQA（Multi-Query Attention），$H_{kv} = 1$

### 典型规模

| 模型 | 类型 | 每 token KV 大小（BF16） |
|------|------|--------------------------|
| LLaMA-3 8B | GQA（8 KV heads） | ~32 KB |
| LLaMA-3 70B | GQA（8 KV heads） | ~327 KB |
| LLaMA-2 70B | MHA（64 KV heads） | ~2.6 MB |
| Mistral 7B | GQA（8 KV heads） | ~28 KB |

GQA/MQA 的引入大幅缩小了 KVCache 体积，是近两年大模型工程化的重要趋势。

## 显存影响

KVCache 与模型权重共享 GPU HBM 显存。两者存在竞争关系：

- **模型权重**：固定大小，加载一次，长期占用
- **KVCache**：随请求数量和序列长度动态增长

在实际部署中，KVCache 的峰值大小往往是限制最大并发请求数（即批量大小）的主要瓶颈，而不是计算能力本身。这是为什么 KVCache 管理效率是推理引擎工程的核心命题。

## 与模型结构的关系

不同 Attention 变体对 KVCache 的影响：

- **MHA（Multi-Head Attention）**：每个 head 独立的 K/V，KVCache 最大
- **GQA（Grouped-Query Attention）**：多个 Q head 共享一组 K/V，KVCache 按比例缩小
- **MQA（Multi-Query Attention）**：所有 Q head 共享同一组 K/V，KVCache 最小
- **MLA（Multi-head Latent Attention，DeepSeek）**：将 K/V 压缩到低维潜空间后存储，再解压使用，KVCache 进一步减小

这些结构选择直接影响 KVCache 的体积，进而影响推理系统的吞吐和成本。
