# 开放问题与未来方向

> KVCache 已经是系统级资源，但它还在快速演化。本章梳理目前看得到的几条主要趋势，按四个维度分组。这一章的内容时效性最强，会随技术进展持续更新。

## 1. 算法侧

### 与新一代 attention 匹配的 KV 复用语义

- MLA、线性注意力、混合架构正在改变 KV 的形状或彻底取消 KV
- **prefix cache、跨节点 KV 池等基础设施都建在"K/V 是关于 token 序列的可拼接结果"的假设上**
- 当模型不再满足这个假设，整套基础设施需要新的语义抽象——这是个开放设计问题

### 自适应稀疏 / 量化的"按需精度"机制

- 当前稀疏 / 量化是**全局参数**：决定后所有 token 一视同仁
- 未来方向：根据 attention 模式动态调整每个 token 的精度
    - "重要"的 token 用 FP16，"边缘"的 token 用 INT4
    - 在线决策框架——但要避免决策本身的开销超过节省
- 与 [稀疏化](sparsity.md) 中 query-aware 路线（Quest 等）的自然融合

### 学习型 KV 压缩的端到端训练

- 把"KV 压缩器"作为模型的一部分参与训练
- 比事后压缩有更好的精度-体积曲线
- 代价：模型架构与压缩方案绑定，部署侧不能"换压缩器"

## 2. 系统侧

### CXL、近存计算等新硬件对 L2/L3 的重塑

- **CXL（Compute Express Link）**：让 CPU、GPU、加速器以"内存语义"共享更大池子的内存
- 一旦 CXL 普及，L2/L3 边界模糊——KV 可以放在远端但访问像本地 DRAM
- 影响：[存储层级](storage-hierarchy.md) 的层数可能从 4 变 3，offload 决策简化
- 时间表：硬件 2025-2026 大规模可用，软件栈成熟还要更久

### KV 池作为独立的"分布式存储产品"

- 类比：Redis、Memcached 在 web 时代的位置
- 已有早期形态：[LMCache](https://github.com/LMCache/LMCache)、Mooncake Store
- 未来可能出现：
    - 标准 KV cache 协议（类似 Redis Protocol）
    - 多引擎兼容的 KV 后端
    - 公有云 KV cache 产品（按 GB-hour 计费）
- 与对象存储、向量数据库一样，**变成"基础设施层"的可能性很高**

### KV 与权重、激活的统一内存抽象

- 当前 GPU 内存里 KV / 权重 / 激活是分开管理的
- 三者都是"显存中的张量"，可以有统一的分配器、统一的 tier 管理
- 研究方向：unified GPU memory manager，类似 OS 的 page-level 统一管理
- 工程难点：三者的访问模式差异巨大（权重静态、KV 增长式、激活短暂）

## 3. 部署侧

### 比 PD 分离更细粒度的资源解耦

PD 分离把"prefill"和"decode"分开。下一步是：

- **Attention / MLP 分离**：attention 是 memory-bound，MLP 是 compute-bound，可以分别部署在不同硬件
- **逐层流水**：把不同层放在不同硬件类型
- 代价：通信开销的进一步增加，KV 在节点间穿梭更频繁

研究阶段，工业部署罕见。

### 多模型共享 KV 基础设施

场景：同一基础模型 + 多个 LoRA / adapter / fine-tune 版本

- 共享的 backbone KV（来自相同 token）可以共用
- adapter 层差异部分独立维护
- 节省：LoRA 服务场景下 KV 占用可降一个数量级

代表方向：S-LoRA、Punica。

### Serverless LLM 与 KV 冷启动

Serverless 的核心矛盾：**冷启动慢**。对 LLM：

- 模型权重加载：分钟级
- KV cache 是空的，命中率为 0

未来可能的方案：

- 预热的副本池（"温暖" pre-warmed instances）
- 跨实例 KV 池让新副本能"立刻"看到热门 prefix
- 模型权重的快速 lazy loading

## 4. 工作负载演进

### Agent 长程规划带来的 KV 访问模式变化

- 传统 LLM 服务：请求-响应模式，KV 短寿
- Agent：单 task 持续数分钟到数小时，KV 长寿且有复杂的"分支-合并"模式
- 影响：KV 生命周期管理需要更长视野，调度器要从"批次级"转向"任务级"

### 多模态 KV

- 图像 / 视频 token 的 KV 与文本 KV 的特性差异
    - 图像 KV 通常更"冗余"（局部相似性高）
    - 视频 KV 数量巨大（每秒几百到几千 token）
- 多模态稀疏 / 压缩有更大空间——但研究尚浅

### Reasoning 模型 long-decode 主导下的新权衡

- o1 / r1 类模型让 decode 长度变成主流瓶颈
- KV 单调增长 + 单请求长持续，对资源池的要求与传统 chat 不同
- 可能的趋势：
    - **大 batch decode 池**——专门服务 reasoning 的副本组
    - **更激进的稀疏 / 量化**——decode 时段的 KV 容量必须激进压缩
    - **抢占代价的重新评估**——已生成几万 token 的请求几乎不能被抢占

### 用户会话粘性的演化

- 早期 LLM：无状态请求
- Chat：单 session 短粘性
- Agent / Coding Copilot：跨 session 长粘性，用户与"自己的 KV"绑定
- 这一趋势让 KV 越来越像"用户的私有数据"——隔离、配额、计费的复杂度上升

## 5. 跨维度的开放问题

把上面分散的方向汇总成几个高价值的"未解之题"：

| 问题 | 难点 |
|------|------|
| Position-invariant prefix cache | 既要复用又要保持位置正确 |
| 标准化的 KV cache 协议 | 各引擎的实现差异大，统一接口难定 |
| 真正可移植的 KV 量化 | 不同硬件、不同 kernel 都能直接用 |
| KV cache 命中率的可预测性 | 当前难以提前估计某种部署下的命中率 |
| 多租户的公平性保证 | 共享 cache 池下的 SLO 隔离 |

## 6. 时间线参考（2025）

- **已成熟**：PagedAttention、Prefix Cache、GQA、FP8 KV、PD 分离（部分场景）
- **2024-2025 进行时**：跨实例 KV 池、cache-aware routing、MLA 深度部署、长上下文 SP/CP
- **早期研究**：position-invariant cache、unified memory、CXL 落地、agent-aware 调度

KVCache 还远不是 settled science——它仍处在"还在发明基础概念"的阶段。

## 关联章节

- 当前实现的边界：[框架对比](frameworks.md)
- 已经被讨论的具体问题：[维度交叉](crossings.md)

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 框架搭建 |
