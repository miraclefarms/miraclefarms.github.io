# 多轮对话

> **工作负载画像**：Prefix 复用度高（历史轮次）｜上下文增长式｜Decode 长度中等｜Session 长寿命

## 特征

- 单 session 长寿命：一次对话可能持续几小时，KV 总量随轮次线性累积
- prefix 单调增长：每轮新消息到来时，前 N-1 轮是完整且确定的 prefix
- 跨轮高度复用：第 N 轮的 prefill 理论命中率 = 前 N-1 轮 token 数 / 第 N 轮 prompt 总 token 数
- 空闲期天然存在：用户思考期间 KV 占用 HBM 但无计算需求，是换出的自然窗口

---

## 关键 KV 问题

### 1. Session 级 KV 持久化

用户思考期间（数秒到数分钟），前 N-1 轮的 KV 放在哪里？

| 策略 | TTFT 影响 | HBM 占用 | 适用场景 |
|------|----------|---------|---------|
| 保留在 HBM | 无额外延迟 | 持续占用 | 活跃 session、空闲 < 5s |
| Swap 到 DRAM | +100–500ms（PCIe） | 释放 HBM | 空闲 5s–5min |
| Swap 到 SSD | +1–5s | 释放 HBM | 空闲 > 5min，非实时对话 |
| Recompute | 正比于历史 token 数 | 完全释放 | 历史极短或极低优先级 |

实践上，多轮对话通常采用分层策略：活跃 session 留在 HBM，短暂空闲 swap 到 DRAM，长期空闲 swap 到 SSD 或丢弃重算。

### 2. 路由亲和性（Session Affinity）

同一 session 的所有轮次必须路由到**同一副本**，否则 prefix cache 强制 miss，前 N-1 轮的 KV 完全浪费。

**选项对比：**

- **Hash-based routing**：对 session ID 做一致性哈希，实现简单，但副本扩缩容时命中率波动
- **Cache-aware routing**：调度器查询每个副本的 radix tree，选择已有最长共同 prefix 的副本——SGLang 的默认策略，命中率最优
- **Sticky routing + 心跳**：负载均衡层记录 session→副本映射，副本故障时才迁移；额外引入状态存储

### 3. 历史压缩

当会话上下文超过 100K token 时，早期轮次的 KV 是否可以压缩？

- **StreamingLLM 风格**：保留最近 K 个 block + attention sink token，丢弃中间历史；TTFT/吞吐提升明显，但远距离信息永久丢失。SCBench 研究表明此类 sub-O(n) 方法在多轮场景下系统性退化。
- **KV 量化**：早期轮次降精度（FP16→INT8/INT4）；接受少量精度损失换取容量，需评估多轮后的退化程度
- **摘要式压缩（研究阶段）**：生成文本摘要后重新 prefill，从语义层压缩历史；与位置编码和 KV 语义的关系尚待探索

---

## 工程配置建议

| 配置项 | 推荐值 / 策略 | 原因 |
|--------|-------------|------|
| Prefix Cache | **必开** | 第 N 轮 prefill 中前 N-1 轮完全命中 |
| Block Size | 16–32 | 多轮 prefix 增长均匀，大 block 命中率高 |
| Session routing | Cache-aware 优先，Hash 次选 | 跨轮命中率最大化 |
| KV Offload 阈值 | 空闲 > 5s 触发 swap 到 DRAM | 平衡命中率与 HBM 利用率 |
| KV 量化 | FP8 可接受 | 历史轮次精度需求低 |

---

## 关键指标

- **Per-turn Prefix Cache 命中率**：稳定多轮对话应 > 80%（第 N 轮理论命中率 ≈ 前 N-1 轮 token 数 / 第 N 轮 prompt 总 token 数）
- **TTFT（P50 / P99）**：P99 受 swap-in 延迟影响，与 P50 分开监控；若两者差距大，说明换入频繁
- **Session HBM 占用时长**：session 空闲期间的 HBM 占用率，是 offload 策略是否生效的直接指标

---

## 关联章节

- KV 换入换出的机制与策略：[KV Offload](offload.md)
- Session affinity 路由的实现细节：[路由与亲和性](routing.md)
- 历史压缩的算法方案：[稀疏化](sparsity.md)、[压缩与量化](compression-quantization.md)
- 多轮评估方法与 SCBench 框架：[Benchmark 与工具](evaluation-benchmarks.md)

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 从工作负载总览拆分，补充工程配置与指标细节 |
