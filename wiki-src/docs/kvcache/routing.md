# 路由与亲和性

> **部署维度之三**：在多副本部署里，把哪个请求送给哪个副本——这个看似 LB 的问题，在 KVCache 视角下是**最大化 prefix 命中**与**避免热点**的权衡。

## 1. 问题描述

假设有 $N$ 个 Decode 副本，每个副本各自维护 prefix cache。一个新请求到来，路由器有两类策略：

- **Random / Round-robin**：均匀分布请求，负载平均，但 prefix 命中率约等于 $1/N$（同一前缀只命中"碰巧路由到的那个"副本上）
- **Cache-Aware**：根据请求前缀路由到"最可能命中"的副本，命中率高，但会引入热点

KVCache 视角下的核心矛盾：**亲和性（hit rate）vs 负载均衡（utilization）**。

## 2. 亲和性调度的几种实现

### 一致性哈希（Consistent Hashing）

把请求的"前缀指纹"哈希后映射到一个环，副本占据环上的若干位置。

- 优点：副本扩缩容时只重哈希局部
- 缺点：前缀不均匀分布时会有热点（System Prompt 全部命中同一副本）

### Prefix-Tree-Aware 路由（SGLang 路线）

路由器维护一个全局的 prefix tree：

- 每个节点记录"哪些副本上有这段前缀的 KV"
- 新请求查找最长公共前缀，路由到拥有该前缀的副本

代表实现：SGLang Router、AIBrix。

### Token-Level Hash 路由

在 block 粒度上做 hash：每个 16-token 块的 hash 决定副本。

- 优点：粒度细
- 缺点：实现复杂，需要预先 tokenize

## 3. 负载与命中率的折中

完全亲和会出问题：

| 场景 | 现象 |
|------|------|
| 热门 System Prompt | 所有用户请求都打到同一副本，副本饱和 |
| 长会话用户 | 单用户的请求集中到一个副本，长尾 |
| 流量倾斜 | 部分前缀流量是其他的 100 倍，副本利用不均 |

实践中的折中策略：

| 策略 | 思路 |
|------|------|
| **Two-choice with cache score** | 选两个候选副本，比较"prefix 命中长度 - 负载惩罚"，取得分高者 |
| **加权亲和** | 命中分 + 负载分加权，权重可调 |
| **Replicated hot prefix** | 检测到热点前缀时主动复制到多个副本 |
| **Cache-aware queue length** | 副本不仅看 cache 命中，也考虑当前队列长度 |

经验：**纯 cache-aware 通常比 RR 提升 30%+ TTFT**，但要在 50% 副本利用率以上避免热点。

## 4. 多租户的公平性

在多租户共享集群里，亲和性还要叠加**租户隔离与配额**：

- 大客户的固定 System Prompt 不应挤掉小客户的命中机会
- 单客户突发流量不应让其他客户长尾劣化
- 路由层可以按 tenant 做 sub-pool（"分层路由"）

工程上常见做法：

- **Tenant-aware hash**：在 hash key 中加入 tenant_id，避免跨租户串扰
- **Per-tenant quota**：限制单租户在各副本的 KV 占用
- **Tier-aware**：高优先级 tenant 使用专属副本池，普通流量用共享池

## 5. 与会话粘性的关系

路由要考虑的不只是"内容相同"，还有"用户连续请求"：

| 场景 | 推荐策略 |
|------|---------|
| 短无状态请求（API） | 内容驱动的亲和 |
| 多轮对话（同一 session） | 把同 session 的轮次路由到同副本（**session affinity**） |
| RAG 流水线 | 文档块层级的亲和 |
| Agent 多步骤 | 整个 agent run 的亲和 |

session affinity 是简单的 HTTP 层面就能做（cookie/header），cache-aware routing 是引擎层面感知 KV 分布——两者经常组合。

## 6. 路由层与引擎层的接口

路由器要做出好决策，需要从引擎暴露：

- 当前 prefix cache 的 token 列表（粗粒度即可）
- 当前队列长度、KV 占用率
- 预估的"如果路由这个请求"的边际收益

工业实现：

| 系统 | 接口 |
|------|------|
| **SGLang Router** | 引擎主动上报 cache 状态 |
| **AIBrix** | K8s 原生的 cache-aware autoscaler + router |
| **vLLM Production Stack** | 通过 metrics endpoint 暴露 cache |
| **Mooncake** | 全局调度器统一感知所有副本 |

## 7. 跨集群与多区域

更高层级的路由问题：

- 用户请求来自不同区域 → 走最近区域 vs 走有缓存的区域？
- 全球部署下的 prefix 局部性
- 跨数据中心的 KV 复制成本（一般不做，命中率不值得）

主流做法：**区域内 cache-aware，区域间靠近用户**。

## 关联章节

- 多副本扩缩容时的 cache 迁移：[弹性与故障](elasticity.md)
- prefix cache 的本地实现：[Prefix Cache](prefix-cache.md)
- 工作负载特征对路由策略的影响：[工作负载维度](workloads.md)

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 框架搭建 |
