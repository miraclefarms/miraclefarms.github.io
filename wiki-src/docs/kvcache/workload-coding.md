# Coding / 长上下文

> **工作负载画像**：Prefix 复用度极高（文件内容不变）｜上下文极长（100K–1M token）｜TTFT 是关键体验指标

## 特征

- 仓库级上下文：代码补全、多文件编辑、代码理解任务，prompt 可达数十万到 100 万+ token
- 小幅编辑后大量重用：用户改了一行代码，前后大段内容不变，理论上可精确复用未改动部分的 KV
- TTFT 是核心：代码补全是实时交互场景，首 token 延迟直接影响体验
- 仓库内容相对稳定：同一仓库的不同请求（不同文件、不同用户、不同时刻）共享大量公共 prefix

---

## 关键 KV 问题

### 1. 增量 KV 更新

用户编辑一行代码后，能否复用前后未改动部分的 KV？

- **编辑点之前的 KV**：完全可精确复用——prefix cache 可直接命中
- **编辑点之后的 KV**：必须重算。原因：K/V 与位置编码（RoPE）强绑定，编辑点改变了后续所有 token 的绝对位置，KV 全部失效
- **实践影响**：prompt 结构应尽量把稳定内容（文件 header、函数定义、不变的上下文）放在 prefix，把频繁变化的内容（用户编辑区、query）放在末尾，最大化命中量

### 2. KV 与代码检索的协同

仓库级 prompt 通常是检索拼接的：

```
system prompt + 文件A全文 + 文件B全文 + ... + 用户问题
```

- 每个文件块在不同请求里可能以不同顺序出现，导致位置编码变化，精确 prefix cache 无法命中文件块
- **解法**：固定文件块的拼接顺序（如按文件路径字母序），让 prefix 结构在同一仓库的请求间保持一致
- **跨用户共享**：同一仓库不同用户的请求，如果 prefix 相同（同样的文件块、同样的顺序），可以跨实例命中分布式 KV 池

### 3. 长上下文下的容量瓶颈

单个请求可能消耗巨大的 KV：

- LLaMA-3 70B（BF16），100K token prompt：KV ≈ 32.7 GB
- 单张 80GB HBM 的 GPU，去掉模型权重后约剩 20–30GB 用于 KVCache
- 单个请求就可能超出单卡 KV 容量，必须启用 SP/CP（Sequence Parallelism / Context Parallelism）

**SP/CP 的 KV 影响：**
- Context Parallelism 把 prompt 切分到多张卡，每张卡只存自己负责的那段 KV
- 全局 prefix cache 命中需要跨卡协调，命中判断逻辑更复杂
- KV 传输跨节点增加 TTFT（与 PD 分离类似的权衡）

---

## 工程配置建议

| 配置项 | 推荐值 / 策略 | 原因 |
|--------|-------------|------|
| Prefix Cache | **必开** | 仓库文件 prefix 极高复用度 |
| Block Size | 32–64 | 长上下文 block 数多，大 block 减少元数据开销 |
| SP / CP | 100K+ token 必开 | 单卡 KV 容量不足 |
| 分布式 KV 池 | **推荐**（L3/L4 持久层） | 仓库级文件 KV 可跨用户跨实例共享 |
| Prompt 结构 | 稳定内容在前，动态内容在后 | 最大化 prefix cache 命中 |

---

## 关键指标

- **Prefix Cache 命中率**：高仓库复用度场景应 > 70%，否则说明 prompt 结构或路由有问题
- **TTFT（P50 / P99）**：实时代码补全对 TTFT 最敏感，P99 应控制在用户感知阈值内
- **Per-request KV 大小**：超过单卡容量的请求比例——触发 SP/CP 的频率
- **SP/CP TTFT 增量**：开启 CP 后的 TTFT 相比单卡的增量，反映 KV 传输开销

---

## 关联章节

- Context Parallelism 与 KV 切分：[并行切分](parallelism.md)
- 持久化 KV 池的分层存储：[存储层级](storage-hierarchy.md)
- Prefix Cache 命中的精确语义：[Prefix Cache](prefix-cache.md)

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 从工作负载总览拆分，补充 SP/CP 容量分析与 prompt 结构建议 |
