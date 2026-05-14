# 系统 × 部署

> 系统决定 KV 存在哪里，部署决定 KV 在哪个进程里——两者的边界在 PD 分离和分布式 KV 池出现后变得模糊。

## PD 分离下 Prefix Cache 应该归 P 还是归 D

PD 分离把 Prefill（P）和 Decode（D）解耦到不同进程甚至不同节点，但 prefix cache 必须存在某个地方。三种放置方案：

| 方案 | Cache 位置 | 优点 | 缺点 |
|------|-----------|------|------|
| **归 P** | Prefill 节点维护 | 命中后直接跳过 prefill，P 节点负载降低 | P 节点要维持 KV 占用 HBM，资源利用率下降 |
| **归 D** | Decode 节点维护 | D 反正要用 KV，自然落点；无额外传输 | P 节点命中后仍需计算，再传给 D |
| **独立 KV 池（L4）** | 跨 P/D 共享的分布式池 | 命中率最高；P/D 扩缩容不影响 cache | 增加一跳网络延迟；需要独立维护 KV 池 |

**主流路线：归 D + 共享 L4 的混合**

- D 节点维护本地 prefix cache（L1/L2），处理高频热点
- L4 层（Mooncake Store / LMCache）跨实例共享，处理冷启动和跨副本场景
- P 节点命中 L4 后，跳过 prefill，直接把 KV 传给 D

---

## 跨节点 KV 池（L4）与本地 Cache（L1/L2）的层级关系

**四级存储的层级协议：**

```
L1: GPU HBM (本地，最快，最小)
  ↓ miss
L2: CPU DRAM (本地，快，中等)
  ↓ miss
L3: 本地 SSD (本地，慢，大)
  ↓ miss
L4: 分布式 KV 池 (跨节点，网络延迟，最大)
  ↓ miss
计算 prefill（最慢）
```

**回填策略（Cache Fill Policy）：**

- **Write-through**：KV 命中并使用后，立即回写到 L4——保证 L4 一致性，但增加写带宽压力
- **Write-back**：KV 在本地 L1/L2 积累到一定量后，批量写入 L4——降低写带宽，但 L4 可能滞后
- **Write-on-evict**：本地 block 被 LRU 驱逐时才写入 L4——最节省带宽，但驱逐时机不可控

**淘汰一致性问题：**

如果多个副本同时维护同一 prefix 的 KV，某个副本 evict 了一个 block，其他副本不感知——L4 需要作为"权威层"，而本地层只作为 L4 的 read cache，避免一致性混乱。

---

## Cache-aware Routing 与 Paged 内存管理的协同

路由决策需要知道"哪个副本已经有这个 prefix 的 KV"，这要求推理引擎的 paged 元数据能**反向暴露**给 router。

**当前实现方式：**

- **SGLang**：RadixTree 直接暴露 prefix 命中查询接口，router 可以 query 每个副本的最长匹配 prefix
- **vLLM**：通过 BlockManager 暴露 cache hit 统计，Production Stack 的 router 使用 prefix hash 查询
- **Mooncake**：全局 KV 池自带路由语义——block-hash 全局寻址意味着 router 天然知道哪个节点有哪个 block

**接口标准化是关键开放问题：**

目前各引擎暴露的格式不一致：

```
SGLang:  get_prefix_cache_hit(prefix_tokens) → hit_length
vLLM:    cache_hit_rate, prefix_hash → ?
TRT-LLM: 无公开标准接口
```

这导致路由层需要为每个引擎写适配代码。标准化的 cache metadata 协议（类似 HTTP Cache-Control）是 KV 基础设施成熟的必要条件。

---

## 关联章节

- PD 分离的架构与 KV 传输协议：[PD 分离](pd-disaggregation.md)
- 四级存储的层级设计：[存储层级](storage-hierarchy.md)
- 路由策略与 session affinity：[路由与亲和性](routing.md)
- 各框架的路由接口实现：[框架对比](frameworks.md)

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 从维度交叉总览拆分，补充回填策略对比、淘汰一致性问题、路由接口现状 |
