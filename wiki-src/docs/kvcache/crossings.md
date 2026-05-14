# 维度交叉

> 真实生产系统不在任何单一维度内闭合，而是四维空间里的一个点。本章列出最重要的交叉问题，作为深入调研的"高价值地带"。

四个维度（[算法](attention-variants.md) × [系统](storage-hierarchy.md) × [部署](pd-disaggregation.md) × [工作负载](workloads.md)）两两交叉，得到六组组合。下面挑出每组里最值得关注的几个具体问题。

## 3.1 算法 × 系统

### KV 量化在 offload / 跨节点传输中的角色

- 量化后的 KV 在 PCIe / RDMA 上等效带宽变大（FP8 = BF16 的 2×）
- 但接收端的 attention kernel 必须支持量化 KV，否则要 dequantize 浪费收益
- **干净的设计**：传输路径与计算路径都用同一套量化格式
- 实践陷阱：[量化方案版本号必须纳入 cache key](lifecycle.md)，否则换方案后字节内容不匹配

### 稀疏 KV 与 Paged 内存管理的兼容性

- [Paged](paged-kv.md) 的核心假设：每 block 的 token 是连续的、固定大小的
- 大多数稀疏方案（[H2O](sparsity.md)、SnapKV）剪掉的是序列中**任意位置**的 token，剪完之后 block 内有"空洞"
- 三种应对：
    1. **保留空洞**：浪费 slot，碎片增加
    2. **压实**：把幸存 token 移到 block 头部，但要更新 block table 的 token-id 映射
    3. **稀疏感知 kernel**：接受 block 内非连续，靠 mask 过滤——增加 kernel 复杂度
- 工业上目前 SWA + attention sink 是最容易兼容的稀疏方案，token 级重要性剪枝部署时多用方案 2 或 3

### MLA / 线性注意力下的 prefix cache 语义

- [MLA](attention-variants.md) 缓存的是 latent 向量而非原 K/V——理论上仍可按 token 序列哈希复用
- 但 paged kernel、prefix tree 都建在"token block 含 K/V"的假设上，需重写
- 线性注意力 / Mamba：状态是**累计的**，无法按 token 序列分段→ **无 prefix cache 可言**

## 3.2 系统 × 部署

### PD 分离下 prefix cache 应该归 P 还是归 D

三种放置：

| 方案 | 位置 | 优缺点 |
|------|------|--------|
| **归 P** | Prefill 节点维护 prefix cache | 命中后直接跳过 prefill；但 P 节点要维持 KV，资源占用增加 |
| **归 D** | Decode 节点维护 prefix cache | D 反正要 KV，自然落点；但 P 节点的算力可能浪费在重算 |
| **独立 KV 池（L4）** | 跨 P/D 共享的 cache | 灵活、可共享；但增加一跳网络延迟 |

主流路线：**归 D + 共享 L4** 的混合（Mooncake 风格）。

### 跨节点 KV 池（L4）与本地 cache（L1/L2）的层级关系

- L4 是"权威 cache"——跨副本共享，命中率最高
- L1/L2 是"本地 cache"——访问最快，但冷启动后空
- 层级协议：L1 miss → L2 miss → L4 fetch → 反向回填
- 设计取舍：回填策略（write-through vs write-back）、淘汰一致性

### Cache-aware routing 与 paged 内存管理的协同

- 路由层基于"副本上有什么 prefix"决策，需要 paged 元数据反向暴露
- 引擎：暴露 `(prefix_hash → block_count)` 给 router
- Router：用这些信息选副本
- 接口标准化是关键开放问题——目前各引擎暴露格式不一

## 3.3 部署 × 工作负载

### Agent 场景与 PD 分离的契合度

- Agent 是"短而多"的请求——单步 decode 通常 < 100 token
- PD 分离的固定开销（KV 传输、调度协议）摊薄不了，相对劣势
- **结论**：纯 Agent 负载不一定上 PD 分离，混部反而经济

### Coding 长上下文下 SP 与 prefix cache 的取舍

- SP/CP 把 KV 沿序列切到多卡，每卡只有部分 KV
- prefix cache 通常按"完整 prefix"组织，与 SP 的切分不对齐
- 当前主流方案：长上下文场景**优先保证 SP，prefix cache 命中率次要**
- 研究方向：SP-aware 的分布式 prefix cache

### Reasoning 长 decode 场景是否还需要 PD 分离

- reasoning 的特征：prefill 短、decode 极长（数十万 token）
- PD 分离主要解决"prefill 和 decode 互相挤占"的问题
- 当 decode 时间 ≫ prefill 时间，挤占的边际成本变小，PD 分离的必要性下降
- **结论**：纯 reasoning 负载下 PD 分离收益有限，混合负载下仍有价值

## 3.4 算法 × 工作负载

### 多轮对话适合哪类稀疏

- 历史轮次衰减：早期内容重要性逐渐下降
- 适配方案：**StreamingLLM 风格**——保留 attention sink + 滑动窗口，老内容自然丢弃
- 不适配：H2O 等"重要性评分"方案——多轮场景下评分窗口很难定义

### RAG 场景下位置不变性压缩 / 块级 KV 的可行性

- 同一文档块在不同 prompt 中位置不同——传统 K/V（含 RoPE）位置敏感
- 解法方向：
    - **预压缩**：[Prompt Cache](https://arxiv.org/abs/2311.04934) 把文档块压成位置无关的 KV summary
    - **位置再校准**：[CacheBlend](https://arxiv.org/abs/2405.16444) 在加载时重算 RoPE
    - **质量代价**：通常有 1-3 个 BLEU / EM 点的精度损失
- 实用前提：**业务能容忍质量损失**

### Coding 场景下增量 KV 与跨层共享 KV 的组合

- 编辑前的 prefix 完全可复用——传统 prefix cache 已经覆盖
- 编辑点之后必须重 prefill——每层都要算
- [YOCO / CLA](sparsity.md) 类 layer-shared KV 在这里有意外收益：**跨层共享意味着重 prefill 也只算一份**
- 这是个尚未被充分挖掘的组合方向

## 3.5 四维联动的真实落地范式

把所有维度拼起来，可以看到工业界的几种典型"全栈范式"：

### Mooncake 范式

**PD 分离 + 全局 KV 池 + 调度器统一感知**

- 算法侧：MLA / GQA + FP8 KV
- 系统侧：四级存储，KV 池作为独立产品
- 部署侧：PD 分离 + cache-aware 全局路由
- 工作负载：通用，多轮 + RAG + coding 都覆盖

### SGLang 范式

**RadixAttention + 结构化生成**

- 算法侧：标准 GQA，主要靠系统优化
- 系统侧：radix tree 组织的细粒度 prefix cache
- 部署侧：cache-aware router
- 工作负载：结构化输出（JSON、约束生成）+ 多轮

### vLLM 生态范式

**Paged + LMCache + 分布式扩展**

- 算法侧：GQA + 可选 FP8/INT8
- 系统侧：PagedAttention + LMCache 的多级存储
- 部署侧：vLLM Production Stack 提供路由 + 自动扩缩容
- 工作负载：通用 API 服务

### 其他范式（待补充）

- **TensorRT-LLM 范式**：NVIDIA 全栈优化
- **AIBrix 范式**：K8s 原生的弹性 + cache aware
- **Dynamo 范式**：NVIDIA 推出的分布式推理基础设施

## 关联章节

- 各范式背后的具体框架：[框架对比](frameworks.md)
- 评估这些跨维度组合的方法学：[评估方法](evaluation.md)
- 未来可能的新范式：[未来方向](future.md)

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 框架搭建 |
