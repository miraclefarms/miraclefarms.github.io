# 并行切分下的 KV 形态

> **部署维度之二**：当模型被 TP / PP / SP / EP 切到多卡多节点时，KV 也跟着被切——切的方式不同，搬运代价、聚合代价、与 prefix cache 的兼容度都不同。

## 1. 四种主流切法

| 切分维度 | 缩写 | KV 沿哪个维度切 | 通信代价 |
|----------|------|----------------|---------|
| 张量并行 | **TP** | head 维度 | All-Reduce/All-Gather per layer |
| 流水并行 | **PP** | layer 维度 | 微批之间的激活传递 |
| 序列 / 上下文并行 | **SP / CP** | sequence 维度 | Attention 计算时的 K/V 交换 |
| 专家并行 | **EP** | （MoE）专家维度 | 与 KV 关系间接 |

## 2. TP（Tensor Parallelism）下的 KV

**怎么切**：每张 GPU 持有 $H_{kv}/N$ 个 KV head（$N$ 是 TP size）。
- TP=8 + GQA(8) → 每卡 1 个 KV head
- TP=4 + MHA(64) → 每卡 16 个 KV head

**对系统的影响**：

- KV 总量不变，但**每卡只有 KV 的 1/N**——整卡 HBM 压力下降
- decode 时每卡只算自己负责 head 的 attention，不需要跨卡通信 KV 本身（但有 q/o 张量的 All-Reduce）
- prefix cache 可以**逐卡独立维护**：每卡的 KV 段是自己负责 head 的，互不干扰

**与 GQA 的边界**：当 $\text{TP size} > H_{kv}$ 时，每卡分不到一个完整 head——必须复制 KV head 或调整 TP/GQA 配比。例如 GQA(8) 模型不能跑 TP=16（除非用其他切分方式辅助）。

## 3. PP（Pipeline Parallelism）下的 KV

**怎么切**：模型沿 layer 维度被切到 $S$ 个 stage，每个 stage 持有自己 stage 的 KV。

**对系统的影响**：

- 每个 stage 的 KV 是**完整 token 序列在该 stage 几层的 K/V**
- decode 时每步生成的 token 需要顺序穿过所有 stage，KV 写入也按 stage 顺序发生
- prefix cache 可以**逐 stage 独立维护**

**问题**：

- PP 的"气泡"问题在 decode 阶段尤其严重——每步只有一个 token 在流水
- 因此**纯 PP 几乎不用于 decode**，更多是 prefill 时配合大模型权重加载
- decode 阶段通常退回 TP

## 4. SP / CP（Sequence / Context Parallelism）下的 KV

**怎么切**：序列被切成 $C$ 段，每张卡负责一段 token 的 K/V。

**典型场景**：长上下文（128K+），单卡 KV 装不下时强制开启。

**两种主流实现**：

| 方案 | 思路 | 代表 |
|------|------|------|
| **Ring Attention** | 各卡按 ring 拓扑轮转传递 K/V，每卡看到所有 K/V | [arXiv:2310.01889](https://arxiv.org/abs/2310.01889) |
| **Striped/Loongtrain** | 不规则切分让负载更均衡 | LoongTrain 等 |

**对 KV 的影响**：

- 单卡只持有自己段的 K/V → 容量上限被切除
- decode 时**每生成一个 token 都需要看到所有段**——要么 ring 通信要么集中聚合
- prefix cache 复杂：跨段命中需要重新切分对齐

**何时启用**：单请求 KV 超过单卡容量上限。8K-32K 一般 TP 足够；超过 128K 通常上 SP/CP。

## 5. EP（Expert Parallelism）

MoE 模型中，专家被分布到不同 GPU。**EP 与 KV 的关系是间接的**：

- KV 来自 attention，与 expert 选择无关
- 但 expert 路由会把同一 token 在不同层路由到不同 GPU → 跨卡的激活传递增加
- KVCache 本身仍按 TP/SP/PP 规则切分

主要影响是 EP 增加了 all-to-all 通信，间接挤压了 KV 传输的带宽预算。

## 6. 切分组合：实际部署的拓扑

工业部署里几乎都是**多种切分组合**：

| 典型组合 | 适用场景 |
|---------|---------|
| TP=8 | 单机 8 卡，最常见的 70B 模型部署 |
| TP=8 × DP=4 | 多副本数据并行扩展吞吐 |
| TP=8 × PP=2 | 跨节点超大模型 |
| TP=8 × SP=4 | 长上下文推理（如 128K+） |
| TP=4 × EP=8 | MoE 模型部署 |

## 7. 各切分下 KV 的搬运、聚合、通信代价

| 切分 | KV 主要通信开销 | 频率 |
|------|----------------|------|
| TP | 不直接搬 KV，但每层 attention 后 q/o all-reduce | 每层每步 |
| PP | KV 不跨 stage 流动，但 token 激活跨 stage 传递 | 每步 |
| SP/CP | 每步要在 ring 上传一遍 KV（或 K/V 摘要） | 每步 |
| EP | 与 KV 间接（all-to-all 占带宽） | 每个 MoE 层 |

## 8. PD 分离下的 TP 不对称问题

[PD 分离](pd-disaggregation.md) 中，**Prefill 节点和 Decode 节点的 TP 配置可以不同**：

- Prefill 用 TP=8（高算力）
- Decode 用 TP=4 或 TP=2（节省资源）

KV 在两端的 head 切分布局不一致，直接传输无法用，必须做**KV reshape**：

```
Prefill TP=8: 每卡 1 个 KV head           Decode TP=4: 每卡 2 个 KV head
[H0] [H1] [H2] [H3] [H4] [H5] [H6] [H7] → [H0H1] [H2H3] [H4H5] [H6H7]
```

这一 reshape 在传输路径上做（pre-send 还是 post-receive），是各引擎实现差异较大的地方。

## 9. 切分对 prefix cache 的影响

不同切分下 prefix cache 的"独立性"：

| 切分 | 各卡 prefix cache | 跨卡命中需要 |
|------|------------------|--------------|
| TP | 独立 | 无（每卡是不同 head） |
| PP | 独立 | 无（每 stage 是不同 layer） |
| SP/CP | **不独立** | 需要跨段哈希对齐 |
| 多副本 DP | 独立但**重复** | 需要 cache-aware routing 才能跨副本命中（见 [路由](routing.md)） |

SP/CP 的 prefix cache 是开放问题——目前主流引擎对长上下文 + cache 复用的组合支持都不算成熟。

## 关联章节

- 路由与亲和性如何让多副本命中：[路由与亲和性](routing.md)
- TP 不对称的 PD 分离：[PD 分离](pd-disaggregation.md)
- SP/CP 与 prefix cache 的开放问题：[维度交叉 §3.3](crossings.md)
