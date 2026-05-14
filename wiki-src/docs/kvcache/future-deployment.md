# 未来方向 — 部署侧

> 部署侧的演化方向是：比 PD 分离更细粒度的解耦，以及让 KV 基础设施适配新的部署形态（Serverless、多 LoRA）。

## 1. 比 PD 分离更细粒度的资源解耦

PD 分离把 Prefill 和 Decode 分开，是第一层解耦。下一步可能的方向：

### Attention / MLP 分离

Transformer 层由两个子模块组成，资源特性截然不同：

| 子模块 | 计算特性 | 瓶颈 |
|--------|---------|------|
| Attention | Memory-bound（KV 读写密集） | HBM 带宽 |
| MLP / FFN | Compute-bound（矩阵乘法密集） | FLOPS |

**理论方向：** 把 Attention 层和 MLP 层分别部署在不同类型的硬件上（带宽优化 GPU vs. 算力优化 GPU），各自做最适合自己的优化。

**工程代价：**
- 每层的 attention 输出需要传输到 MLP 节点，再传回——通信量巨大
- KV 仅存在于 attention 节点，MLP 不需要；但每层传输的激活值增加
- 端到端延迟可能因通信开销而上升

**当前状态：** 研究阶段，工业部署几乎没有。

### 逐层流水（Layer-wise Pipeline）

不同层分布在不同硬件类型上，以流水线方式执行：

- 理论上可以把不同层放在不同显存/算力比的设备
- 实践上流水阶段不均衡（prefill vs. decode 的层计算时间差异）导致 bubble
- 目前更多用于超大模型的权重 offload（层权重分布在 CPU + GPU），而非 KV 优化

---

## 2. 多模型共享 KV 基础设施（LoRA 服务）

场景：同一基础模型 + 多个 LoRA / adapter / fine-tune 版本

**核心观察：**

- LoRA 只修改模型权重，不修改 attention 的计算逻辑
- 对于相同的输入 token，不同 LoRA 版本产生**不同**的 KV——但 backbone 的贡献可以分解

**理论方向：**

- 基础模型的 KV（不含 LoRA 修改）可以被所有 LoRA 版本共享
- 每个 LoRA 版本维护自己的 delta KV（仅 LoRA 层的增量）
- 合并时：KV_final = KV_base + KV_delta

**实践约束：**

- LoRA 的 adapter 通常作用于 K/V projection，delta 不是简单的加法
- 实现需要修改 attention kernel，支持 base + delta 的分离存储
- S-LoRA、Punica 探索了多 LoRA 服务的基础，但 KV 共享仍是开放问题

**潜在收益：**

- 若 LoRA 服务场景 100 个 LoRA 版本共享同一 prefix cache，KV 占用可降一个数量级
- 路由层可以"先查 base KV，再补 delta KV"，大幅提升命中率

---

## 3. Serverless LLM 与 KV 冷启动

**Serverless 推理的核心矛盾：冷启动慢**

| 冷启动阶段 | 典型延迟 | 瓶颈 |
|-----------|---------|------|
| 实例启动（容器/VM） | 5–30s | 基础设施 |
| 模型权重加载（70B BF16） | 30–120s | 存储 I/O |
| KV cache 为空，命中率 0% | 前几分钟 | Workload 预热 |

**缓解方向：**

### 预热实例池（Pre-warmed Instances）

- 维护一批"温暖"实例，权重已加载，随时可接受请求
- 代价：闲置实例的资源成本
- 适合流量可预测的场景

### 跨实例 KV 池

- 新冷启动的实例连接分布式 KV 池，立刻可以命中热门 prefix
- 不需要自己"预热"——直接复用其他实例积累的 KV
- Mooncake Store / LMCache 提供的能力

### 模型权重 Lazy Loading

- 把权重分块，只加载被请求的层（按需加载）
- 早期的 token 生成可以用已加载的层先运行，后续层在后台继续加载
- ServerlessLLM 项目探索了这个方向

---

## 4. 多租户下的 KV 公平性与隔离

随着 KV 池变成共享基础设施，公平性和隔离变得关键：

### 问题

- 高频用户积累了大量 hot KV，占用 pool 容量，导致低频用户 miss 率高
- 共享 KV pool 下，某租户的 KV 被另一租户命中（privacy 问题）
- 一个租户的大量写入可能驱逐其他租户的 KV（eviction 不公平）

### 当前解决方案

- **cache_salt 隔离**：每个租户/用户的 block hash 加入 salt，防止跨租户命中（vLLM PR #39837，LMCache PR #3042）
- **Per-tenant 容量配额**：限制每个租户在 pool 中的最大 KV 占用量
- **优先级 LRU**：TRT-LLM 引入 priority-based LRU，高优先级租户的 block 不容易被驱逐

### 开放问题

- 公平性指标：如何定义"公平的 KV 驱逐"？（按使用量？按付费？按 SLO？）
- 隔离与共享的边界：既要复用公共 prefix（节省计算），又要保证私有内容不泄露
- 多租户 SLO 保证：在共享 pool 下，如何承诺每个租户的 TTFT P99？

---

## 关联章节

- PD 分离的当前实现：[PD 分离](pd-disaggregation.md)
- 路由与亲和性在多租户下的扩展：[路由与亲和性](routing.md)
- 多租户 cache 隔离在框架中的实现：[框架对比](frameworks.md)
- 系统侧的未来方向：[未来方向 — 系统侧](future-system.md)

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 从未来方向总览拆分，补充 LoRA KV 共享分析、Serverless 冷启动路径、多租户公平性问题 |
