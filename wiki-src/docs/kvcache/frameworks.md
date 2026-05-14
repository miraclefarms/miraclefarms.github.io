# 框架对比

本章梳理主流推理引擎在 KVCache 管理上的实现概览与横向对比。各框架的详细特性见对应子页面。

!!! warning "内容时效性"
    推理引擎更新极快，下表信息可能已过时。在做具体技术选型时，请以各项目的最新文档和 Release Notes 为准。

---

## 框架子页面

| 框架 | 定位 | 子页面 |
|------|------|--------|
| [vLLM](framework-vllm.md) | PagedAttention 原创；生态最广泛 | [详细特性 →](framework-vllm.md) |
| [SGLang](framework-sglang.md) | RadixAttention；高吞吐；HiCache 分层 offload | [详细特性 →](framework-sglang.md) |
| [TensorRT-LLM](framework-trtllm.md) | NVIDIA 官方；极致 GPU 性能；NIXL + NVLink-C2C | [详细特性 →](framework-trtllm.md) |
| [LMCache](framework-lmcache.md) | KVCache 存储插件；多后端可插拔；跨实例共享 | [详细特性 →](framework-lmcache.md) |
| [Mooncake](framework-mooncake.md) | 以 KV 为中心的调度；分布式 RDMA KV 池 | [详细特性 →](framework-mooncake.md) |

---

## 对比维度

评估一个推理引擎的 KVCache 实现，从以下维度考察：

| 维度 | 说明 |
|------|------|
| KV 分页管理 | 是否支持 Block/Page 粒度分配，block_size 是否可配置 |
| Prefix Cache | 是否支持跨请求的 KV 复用，命中粒度，哈希方案 |
| KV Offload | 是否支持 CPU DRAM/SSD/远端 offload，策略是否可定制 |
| PD 分离支持 | 是否支持 Prefill/Decode 分离部署，KV 传输协议 |
| 多节点 KV 传输 | 跨节点 KV 传输带宽、延迟、TP 不对称处理 |
| KV 量化 | 是否支持 FP8/INT8 KV 存储，量化粒度 |
| 监控与指标 | 是否暴露 KVCache 利用率、命中率、Offload 带宽等可观测指标 |

---

## 横向对比表

| 框架 | 分页管理 | Prefix Cache | KV Offload | PD 分离 | KV 量化 | 分布式 KV |
|------|---------|-------------|-----------|---------|---------|-----------|
| vLLM | ✅ PagedAttention | ✅ APC，block 级链式哈希 | ✅ 多级 tier | 进行中（MultiConnector） | ✅ FP8/NVFP4/TurboQuant | ✅ Mooncake Store |
| SGLang | ✅ HiRadixCache | ✅ Radix Tree，任意 token 边界 | ✅ HiCache 三层 | ✅ GPU Staging Buffer，5× TPS | ✅ FP8/TurboQuant/MLA | 进行中 |
| TensorRT-LLM | ✅ KVCacheManager，两阶段 Claim | ✅ BlockKey 多维编码，priority LRU | ✅ HBM→Host→NVMe+GDS | ✅ Disaggregated serving，最高 6.11× | ✅ FP8/NVFP4 | ✅ NIXL，Dynamo |
| LMCache | ✅ 多级存储 | ✅ 跨实例 Prefix Cache | ✅ HBM→DRAM→SSD→S3 | ✅ fire-and-forget PD 后端 | 待补充 | ✅ Valkey/FS/S3 后端 |
| Mooncake | ✅ block 级 | ✅ 全局 block-hash 寻址 | ✅ SSD offload | ✅ 以 KV 为中心的调度 | 待补充 | ✅ 分布式 RDMA DRAM 池 |

---

## 关联章节

- 各框架 Prefix Cache 实现对比：[Prefix Cache](prefix-cache.md)
- PD 分离中的 KV 传输：[PD 分离](pd-disaggregation.md)
- KV 量化的具体方法：[压缩与量化](compression-quantization.md)
- 分布式 KV 池的工程细节：[存储层级](storage-hierarchy.md)、[路由与亲和性](routing.md)

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 框架搭建 |
| v0.2 | 2026-05-14 | 纳入各框架实质性技术细节，更新对比表增加分布式 KV 列 |
| v0.3 | 2026-05-14 | 拆分为子页面，本页保留对比维度与横向对比表 |
