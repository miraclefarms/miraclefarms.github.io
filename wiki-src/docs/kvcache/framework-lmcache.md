# LMCache

[LMCache](https://github.com/LMCache/LMCache) 是专注于 KVCache 分层存储和跨实例共享的项目，定位是推理引擎的 **KVCache 后端插件**，而非独立推理引擎。

!!! warning "内容时效性"
    LMCache 更新极快。请以项目的最新文档和 Release Notes 为准。

---

## 定位与架构

### 与其他框架的区别

LMCache 不是推理引擎，而是 KVCache 存储层的专用实现：

```
推理引擎（vLLM / SGLang）
        ↕  LMCache 接口
LMCache 存储层（多后端可插拔）
    ├── Valkey（Redis 兼容，集群/TLS/GLIDE）
    ├── 本地 FS 连接器（零依赖持久化）
    └── S3 L2 Adapter（对象存储后端）
```

**核心价值：** 多个推理引擎实例共享同一套 KVCache 存储，实现跨实例的 Prefix Cache 命中。

### 多级存储

KV 分布到四层：

1. **GPU HBM**：最快，由推理引擎本地管理
2. **CPU DRAM**：LMCache 管理的一级 offload 目标
3. **本地 SSD**：通过本地 FS 连接器持久化
4. **远端对象存储 / RDMA**：S3 Adapter 或分布式 DRAM 池

---

## 主要特性

### 可插拔后端

LMCache 的多后端设计：

- **Valkey 连接器**：支持集群模式 / TLS / GLIDE（AWS Valkey 托管）
- **原生 FS 连接器**：零外部依赖的本地文件持久化，适合单机部署
- **S3 L2 Adapter**：对象命名规范 / 容量统计 / DeleteObject 驱逐 / circuit breaker

### PD 后端优化

- **fire-and-forget `batched_submit_put_task`**（PR #3038）：worker 线程不阻塞等待 alloc + RDMA write，大幅降低 Prefill 节点的 PD 传输延迟
- **`batched_contains()`**（PR #2966）：批量查询替代逐 key 串行查询，减少 round-trip

### Cache 隔离

- `cache_salt`（PR #3042）：写入 ObjectKey / IPC key，按租户/用户隔离 KV，防止跨租户命中
- 与 vLLM 的 cache_salt 机制对齐，可以端到端保证隔离语义

### 可观测性

LMCache 在可观测性上投入较多：

- L0 subscriber（PR #2974）：追踪 GPU KV block 的生命周期、空闲时间、复用间隔
- StorageManager 二进制 trace 能力（PR #3063）：细粒度的 I/O 追踪
- BlendEngineV2 per-request root OTel span（PR #3062）：每个请求的完整 KV 操作链路

### MP 模式与跨节点传输

- block-id 级 KV 传输内核（PR #2838）
- HND KV 格式（PR #2826）：跨节点传输的序列化格式
- AMD hipFile GPU-direct storage 支持（PR #2799）

### 持久化与恢复

- persistence interface + nixl_store_dynamic 适配器（PR #2938）
- 支持 KV cache 的持久化到磁盘，服务重启后可以恢复 prefix cache 状态

---

## 使用场景

### 适合 LMCache 的场景

- **多副本 Prefix Cache 共享**：多个 vLLM 实例同时服务同一模型，公共 prefix（system prompt、文档库）跨实例共享
- **长期持久化 KV**：文档库、代码仓库的 KV 预计算后持久化到 SSD/S3，新实例启动即可命中
- **混合存储策略**：热 KV 在 GPU/DRAM，冷 KV 在 SSD/S3，按访问频率自动分层

### 不适合的场景

- 单实例、低并发场景：vLLM/SGLang 内置的 prefix cache 已足够，引入 LMCache 有额外复杂度
- 对延迟极度敏感的实时场景：额外的存储查询 round-trip 可能影响 TTFT

---

## 关联章节

- 分布式 KV 存储的系统设计：[存储层级](storage-hierarchy.md)
- 跨实例 Prefix Cache 的路由配合：[路由与亲和性](routing.md)
- 框架总览与对比表：[框架对比](frameworks.md)

来源：主站 briefs 2026-03-25 ~ 2026-05-14

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 从框架对比总览拆分，整理多后端架构、PD 优化、可观测性、持久化等核心特性 |
