# KV 压缩与量化

> **算法维度（横向）之三**：保留所有 K/V 条目，但**降低每个条目的位宽或秩**，让 KVCache 在字节数上变小。

与 [稀疏化](sparsity.md) 的差别：稀疏是"丢掉一些 K/V"，本章是"K/V 都还在，但每个变小了"。两者经常组合使用。

## A. 压缩

### 1. 低秩分解 / SVD 类

策略：把 KVCache 张量沿 head 或 channel 维度做 SVD，只保留 top-$r$ 个奇异值与对应向量。

- 离线 SVD：用校准集决定分解矩阵，推理时缓存的是低秩因子
- 在线 SVD：随生成动态调整，质量更稳但计算开销大

工程上更常用的是 **MLA**（[Attention 变体 §2](attention-variants.md)）——它把低秩思想直接做进了模型架构里训练，避免了"事后压缩"的精度损失。事后 SVD 压缩在公开工业系统里部署较少。

### 2. 学习型压缩 / 摘要式压缩

策略：用一个轻量"压缩头"把一段 KV 压成更短的 KV（信息聚合）。

代表方向：

- **Compressed Context Memory** 类工作——把超出窗口的历史 K/V 压成几个 summary token
- **Recurrent KV**——KV 不再是 token 级，而是"段级"摘要

适用：**多轮对话、长 Agent 链路**等"历史 token 多但单 token 重要性低"的场景。
代价：质量损失需要任务级评估，不是"通用免费午餐"。

### 3. CacheGen / Streaming 类

不是缩 KV 本身，而是在**跨节点传输 KV** 时做压缩+流式传输。
- 把 KV 量化 + 编码成 bitstream
- 接收端边收边解码，与计算 overlap

代表论文：[CacheGen (2024)](https://arxiv.org/abs/2310.07240)。

## B. 量化

量化是工业上最广泛部署的 KVCache 优化路径，原因：兼容性好、硬件友好、收益直接。

### 1. 数据类型谱

| 精度 | KV 体积（相对 BF16） | 工业部署成熟度 |
|------|---------------------|----------------|
| FP16 / BF16 | 1.0× | 默认 |
| FP8（E4M3 / E5M2） | 0.5× | H100/H200 一线方案 |
| INT8 | 0.5× | 成熟，质量略劣于 FP8 |
| INT4 | 0.25× | 研究阶段，部分场景可用 |
| 1-bit / KIVI 极致量化 | < 0.1× | 论文阶段 |

### 2. 量化粒度

| 粒度 | 描述 | 精度/速度权衡 |
|------|------|---------------|
| **Per-tensor** | 整个张量一个 scale | 最快，最不准 |
| **Per-channel** | 每个 channel（head_dim 维度）一个 scale | 中等 |
| **Per-token** | 每个 token 一个 scale | 较准，scale 表大 |
| **Per-group** | 每 g 个连续元素一组 | 主流折中（如 g=64） |

paged attention 下，**per-block** 也是常见粒度——每个 KV block 共享一组 scale。

### 3. 静态 vs 动态、对称 vs 非对称

- **静态量化**：scale 提前用校准集确定。简单、推理零额外开销，但 outlier 敏感。
- **动态量化**：scale 在写入时基于当前张量统计计算。适应性好，但 prefill / decode 路径都要插入计算。
- **对称量化**：零点固定为 0。硬件对乘加更友好。
- **非对称量化**：保留 zero point，对正负不对称分布更准（Key 张量常见这种分布）。

**经验**：K 比 V 更难量化——K 的分布更不均匀（attention sink 等极端值），常见做法是 K 用更细粒度、V 用更粗粒度。

### 4. 代表方法

| 方法 | 核心 idea | 备注 |
|------|-----------|------|
| **KIVI** | K per-channel quant + V per-token quant，2-bit 也能保精度 | [arXiv:2402.02750](https://arxiv.org/abs/2402.02750) |
| **KVQuant** | 4-bit per-channel + outlier 处理 + RoPE 对齐 | [arXiv:2401.18079](https://arxiv.org/abs/2401.18079) |
| **Atom** | 推理全栈低比特（含 KV）| [arXiv:2310.19102](https://arxiv.org/abs/2310.19102) |
| **QServe** | W4A8KV4 + Lookahead Search 等 | [arXiv:2405.04532](https://arxiv.org/abs/2405.04532) |

## C. 系统层面的协同

### 量化 KV 与 paged 管理

- block_size 与量化 group_size 的对齐：例如 16 token/block × per-token quant，scale 表是 16 个 fp16 值，与 block metadata 一起存
- copy-on-write 时要不要重新量化？通常不要，直接拷贝量化后的字节

### 量化 KV 与 offload / 跨节点传输

- 在 PCIe / RDMA 上传输的就是量化后的字节，带宽收益正比于压缩比
- 接收端要么直接用量化 KV 算 attention（kernel 支持），要么 dequantize 回来（计算开销）

详见 [维度交叉 §3.1](crossings.md)。

### 量化 KV 与 prefix cache

哈希以**量化后的字节**为输入还是**原始 token id** 为输入？工业实践基本都是后者（token id）——量化是引擎内部细节，缓存语义应稳定于 token 级。但要注意：换一个量化方案后，老的物理 KV block 字节内容变了但 token id 哈希不变，导致命中后读到的是错的字节——这要求**量化方案版本号纳入 cache key 或在切换时清空缓存**。

## D. 算法维度小结：三类操作的统一视角

把 [Attention 变体](attention-variants.md)、[稀疏化](sparsity.md) 和本章合在一起看：

| 类别 | 操作 | 代表 |
|------|------|------|
| **改形状** | 改变 KV 张量的维度 | MQA/GQA/MLA、SWA |
| **减条目** | 在 token/head/layer 维度上去掉 K/V | StreamingLLM、H2O、SnapKV、YOCO |
| **减位宽/秩** | K/V 都在但每个变小 | KIVI、KVQuant、SVD |

它们共同效应是**减少 KV 的"有效字节数"**，但作用通道、精度风险、kernel 友好度各不相同。在生产部署里这三类经常**叠加使用**：GQA 模型 + FP8 KV + SnapKV 剪枝是当前一线推理引擎的常见组合。

## 关联章节

- 量化与 paged 块管理：[Paged KV](paged-kv.md)
- 量化 KV 在跨节点传输中的角色：[PD 分离](pd-disaggregation.md)
- 量化对端到端任务精度的影响：[评估方法](evaluation.md)
