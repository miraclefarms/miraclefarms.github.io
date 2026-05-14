# Reasoning / Long CoT

> **工作负载画像**：Prefix 复用度低（推理链一次性）｜Prefill 短｜Decode 极长（数万 token）｜单请求资源占用大

## 特征

- Prefill 短，Decode 极长：思维链（Chain-of-Thought）输出可能达 10K–100K token
- KV 单调膨胀：decode 阶段每步增加一个 token 的 K/V，KV 池随时间线性增长
- 跨请求复用空间小：每次推理链是一次性的，很少有跨请求的公共前缀
- 长占用高代价：单个请求持续数十秒到数分钟，且中途被抢占代价极高

---

## 关键 KV 问题

### 1. Decode 阶段的容量瓶颈

以 DeepSeek-R1 671B（FP8）为例，生成 30K decode token 时：

```
每 token KV ≈ 2 × 61 × 8 × 128 × 1 字节 ≈ 125 KB（FP8，GQA）
30K token KV ≈ 125 KB × 30,000 ≈ 3.75 GB（单请求）
```

若同时运行 10 个这样的 reasoning 请求，KV 占用 > 37 GB，已超单卡 HBM 余量。

**应对策略：**

- 大 L2 容量（DRAM offload）：decode 途中 overflow 的 KV 自动 swap 到 DRAM
- 限制 max batch size：减少同时运行的 long-decode 请求数
- KV 量化（FP8）：减少每 token 的 KV 大小，在精度可接受范围内扩容

### 2. 抢占代价高

传统 preemption 策略（丢弃 KV，重算）在 reasoning 场景下代价极高：

- 已生成 20K token 的请求被抢占 → 下一次请求到来时需重新 prefill 这 20K token
- 这相当于一次巨大的 prefill 操作，严重冲击 TTFT 和整体吞吐

**倾向策略：Swap 而非 Recompute**

- 把被抢占请求的 KV swap 到 DRAM，等资源释放后换回
- 代价是换出/换入的 PCIe 带宽，但远小于重算代价
- 设计上应为 long reasoning 请求分配更高的抢占保护优先级

### 3. KV 单调增长 vs. 稀疏化

Reasoning 的 decode KV 能否稀疏化？

- **H2O / SnapKV 类方法**：基于 attention score 保留重要 token 的 KV，丢弃其余
- **问题**：思维链的每一步都可能引用"某个远距离的结论"，精确预测哪些 token 重要非常困难
- **当前实践**：生产中多采用保守量化（FP8）而非激进稀疏，避免推理链质量退化

### 4. 与推测解码的交互

Speculative decoding（draft model 生成候选，main model 验证）在 reasoning 场景下的 KV 影响：

- Draft model 有独立的 KV 池（通常小模型，KV 小）
- Main model 的 KV 在 accept 步骤后才正式写入
- 验证失败时 draft model 的 KV 需要丢弃回滚
- 长 decode 场景下，speculative decoding 的 accept rate 随上下文变长可能下降

### 5. PD 分离的边际收益降低

Reasoning 场景的 decode 长度远大于 prefill：

- 典型比例：prefill 1K token，decode 30K token → decode/prefill 比 ≈ 30×
- PD 分离的核心收益是让 decode GPU 专心服务，但 reasoning 本身就是 decode-dominant
- 额外的 KV 传输开销相对于本身极长的 decode 时间，边际节省很小
- 实践中：reasoning 专属 decode 池（不做 PD 分离）是常见配置

---

## 工程配置建议

| 配置项 | 推荐值 / 策略 | 原因 |
|--------|-------------|------|
| KV Offload | **大 L2**（DRAM），阈值低 | 单请求 decode KV 膨胀快，需早换出 |
| KV 量化 | FP8（decode 阶段）| 容量扩张，精度损失在 reasoning 中需评估 |
| Batch size | 较小（2–8） | 避免多个 long reasoning 请求同时耗尽 KV |
| 抢占策略 | Swap 优先，Recompute 最后 | 已生成万 token 的请求重算代价极高 |
| PD 分离 | **不推荐** | Decode-dominant workload，传输开销 > 收益 |
| Prefix Cache | 收益有限，但 system prompt 仍开 | 跨请求前缀复用空间小 |

---

## 关键指标

- **Peak KV per request**：decode 末尾时单请求的 KV 大小，决定最大可支持的并发数
- **Swap-out / Swap-in 频率**：KV overflow 到 DRAM 的触发率，反映容量是否足够
- **Recomputation Rate**：被迫重算的比例；reasoning 场景应控制在极低水平
- **TPOT**：decode 速度，受 HBM 带宽和 KV 大小共同影响

---

## 关联章节

- KV 容量与多级存储：[存储层级](storage-hierarchy.md)、[KV Offload](offload.md)
- 抢占与调度策略：[生命周期与淘汰](lifecycle.md)
- KV 量化对 reasoning 精度的影响：[压缩与量化](compression-quantization.md)
- Reasoning 场景的未来演化：[未来方向 — 工作负载演进](future-workloads.md)

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 从工作负载总览拆分，补充容量计算示例、抢占代价分析、PD 分离权衡 |
