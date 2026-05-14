# 框架对比

本页梳理主流推理引擎在 KVCache 管理上的实现概览。目标是建立对比维度框架，而不是逐行比较代码实现——后者变化频繁，建议直接查阅各项目的官方文档和最新代码。

## 对比维度

评估一个推理引擎的 KVCache 实现，可以从以下维度考察：

| 维度 | 说明 |
|------|------|
| KV 分页管理 | 是否支持 Block/Page 粒度分配，block_size 是否可配置 |
| Prefix Cache | 是否支持跨请求的 KV 复用，命中粒度，哈希方案 |
| KV Offload | 是否支持 CPU DRAM/SSD/远端 offload，策略是否可定制 |
| PD 分离支持 | 是否支持 Prefill/Decode 分离部署，KV 传输协议 |
| 多节点 KV 传输 | 跨节点 KV 传输带宽、延迟、TP 不对称处理 |
| KV 量化 | 是否支持 FP8/INT8 KV 存储，量化粒度 |
| 监控与指标 | 是否暴露 KVCache 利用率、命中率、Offload 带宽等可观测指标 |

## 主要框架概述

### vLLM

[vLLM](https://github.com/vllm-project/vllm) 是目前生态最成熟的开源推理引擎之一。

**KVCache 相关特性：**

- **PagedAttention**：vLLM 是 PagedAttention 的原始提出者（Kwon et al., 2023），其 Block Manager 是业界实现的参考基准。BlockPool 统一管理 prefix cache、shared blocks、free blocks，通过单一 `ref_cnt` 生命周期状态机
- **Prefix Cache（Automatic Prefix Caching, APC）**：支持精确哈希匹配的 Prefix Cache，链式哈希 + lazy 匹配，以 Block（16 tokens）为粒度
- **Mooncake Store 集成**（PR #40900）：将分布式 KV Cache 池接入 vLLM，MooncakeStoreConnector 在 scheduler 查询 Store 后进行 block 分配。Agentic workload 实测：cache hit 1.7%→92.2%，吞吐 3.8×，P50 TTFT −46×
- **KV Offload**：多级 KV cache offloading 框架（PR #40020），支持链式二级 tier（GPU→CPU→NVMe）；OffloadingManager 注入 per-request 身份追踪（PR #42507）
- **PD 分离**：vLLM V1 架构在推进 disaggregated prefill 支持，MultiConnector 支持 PD + Store 拓扑
- **KV 量化**：支持 FP8 KV Cache（部分硬件），NVFP4 KV sliding window（PR #42464）；TurboQuant 集成 via PR #40396
- **Attention Backend**：支持 FlashAttention、FlashInfer、Triton，TOKENSPEED_MLA 覆盖 Blackwell MLA prefill/decode（含 fp8 KV cache, PR #41778）
- **cache_salt 隔离**：PR #39837 引入 cache_salt 参数，按租户/用户隔离共享 KV Block

### SGLang

[SGLang](https://github.com/sgl-project/sglang) 以高吞吐和 Prefix Cache 效率著称。

**KVCache 相关特性：**

- **RadixAttention**：使用压缩 Radix Tree 组织 KV Block，支持任意 token 边界匹配（不需对齐 block_size）。Cache-aware 负载均衡将请求路由到已有前缀的副本
- **HiCache**：分层 KVCache（GPU/CPU/NVMe），生产实测 Novita AI TTFT −56%、2× 吞吐；Ant Group DeepSeek-R1-671B TTFT −84%
- **HiSparse**：在 GLM-5.1-FP8 256 并发下 3–5× 吞吐，LRU offload 不活跃 KV 到 host
- **ShadowRadix**：DeepSeek-V4 上 4K→900K 上下文扩展，decode 吞吐仅 266→240 tok/s（<10% 下降）
- **Block Manager**：与 vLLM 类似的分页管理，HiRadixCache 支持 SLRU 淘汰策略和 adaptive prefill delayer
- **PD 分离**：GPU Staging Buffer（PR #19890）将 scatter head slices gather 为 contiguous memory 做 bulk RDMA transfer，Qwen3.5 Prefill TP4+Decode DEP4 下 TPS/GPU 提升约 5×
- **KV 量化**：支持 FP8 KV Cache，TurboQuant 集成 via PR #23135（3.88× 压缩，93–105% decode 吞吐）
- **Tokenspeed MLA**：PR #24925 支持 fp8 KV cache
- **Attention Backend**：深度集成 FlashInfer

### TensorRT-LLM

[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) 是 NVIDIA 官方推理框架，以极致的 GPU 性能优化为目标。

**KVCache 相关特性：**

- **Paged KV Cache**：KVCacheManager 支持分页管理，block_size 可配置。两阶段 Claim（`addSequenceBatch`）先锁定可复用 Block 再批量 onboard，解决 C++ 环境下的 TOCTOU 竞态
- **三层存储**：GPU HBM → Host DRAM → NVMe（via GDS），事件驱动路由自动迁移 Block
- **BlockKey 多维编码**：编码 LoRA ID、多模态哈希、cache_salt 等参数，实现多维缓存隔离
- **Priority-based LRU**：高优先级 Block 获得保留槽位，命中率 +20% vs 纯 LRU
- **Chunked Context**：长 Prompt 切分成 chunks 处理，与 KVCache 管理协同
- **PD 分离**：Disaggregated serving 支持，实测 DeepSeek R1 1.4–1.8× 吞吐（GB200），Qwen 3 最高 6.11×；+MTP 额外 1.6–2.5×
- **KV 传输**：NIXL 传输后端，GH200 NVLink-C2C 900GB/s = 7× PCIe Gen5
- **KV 量化**：支持 FP8 KV Cache，NVFP4 原生格式；MLA FP4 模型有 BF16 fallback pool
- **Prefix Reuse**：PR #13139 将每个 pending request 的 radix tree 遍历从 5 次收敛为单次 `analyzePrefixReuse()`

来源：主站 essay [TRT-LLM KVCache Runtime 架构](/notes/2026/05/09/trtllm-kvcache-runtime-architecture/)

### LMCache

[LMCache](https://github.com/LMCache/LMCache) 是专注于 KVCache 分层存储和跨实例共享的项目。

**定位：**作为推理引擎的 KVCache 后端插件，而非独立推理引擎。

**KVCache 相关特性：**

- **多级存储**：将 KVCache 分布到 GPU HBM、CPU DRAM、本地 SSD 和远端内存池（RDMA）
- **跨实例 Prefix Cache**：多个推理实例共享同一套 KVCache 存储，实现实例间 Prefix Cache 命中
- **可插拔后端**：Valkey 连接器（集群模式/TLS/GLIDE）、原生 FS 连接器（零依赖持久化）、S3 L2 Adapter（对象命名/容量统计/DeleteObject 驱逐/circuit breaker）
- **PD 后端**：fire-and-forget `batched_submit_put_task`（worker 线程不再阻塞等 alloc+RDMA write, PR #3038）；`batched_contains()` 替代逐 key 串行查询（PR #2966）
- **Cache 隔离**：cache_salt 写入 ObjectKey/IPC key，按租户/用户隔离（PR #3042）
- **可观测性**：L0 subscriber 追踪 GPU KV block 生命周期/空闲时间/复用间隔（PR #2974）；StorageManager 二进制 trace 能力（PR #3063）；BlendEngineV2 per-request root OTel span（PR #3062）
- **MP 模式**：block-id 级 KV 传输内核（PR #2838）；HND KV 格式（PR #2826）；AMD hipFile GPU-direct storage（PR #2799）
- **Persist/Recover**：persistence interface + nixl_store_dynamic 适配器（PR #2938）

来源：主站 briefs 2026-03-25 ~ 2026-05-14

### Mooncake

[Mooncake](https://github.com/kvcache-ai/Mooncake) 是 Kimi 团队开源的以 KVCache 为中心的调度系统。

**KVCache 相关特性：**

- **分布式 KV Store**：RDMA-capable 分布式 DRAM 池，block-hash 全局寻址，零拷贝 GPU-to-RDMA 路径
- **Store 元数据**：ObjectDataType 枚举让元数据层知道每块存储内容类型（PR #1719）；DSA 式分配策略（PR #2080）
- **HA 与恢复**：Redis HA 领导者选举后端（PR #1722）；client-based 三阶段恢复流水线——hot keys → DRAM entries → storage tier（PR #1876）
- **传输层**：PG 与 TENT 集成（PR #1676）+ P2P 内存区域本地注册（PR #1690）；EFA transport 补 fi_read/LRU eviction/multi-NIC striping（PR #1821）；Get 路径 batch route query（PR #1970）
- **SSD Offload**：暴露为 Python setup 参数（从全局环境变量→实例级参数, PR #1884）
- **与 vLLM 集成**：MooncakeStoreConnector（vLLM PR #40900），scheduler 查询 Store 后分配 block

来源：主站 reading [vLLM × Mooncake Store](/notes/2026/05/07/vllm-mooncake-store-distributed-kv-cache/)，主站 briefs 2026-03-19 ~ 2026-05-14

## 对比表

| 框架 | 分页管理 | Prefix Cache | KV Offload | PD 分离 | KV 量化 | 分布式 KV |
|------|---------|-------------|-----------|---------|---------|-----------|
| vLLM | ✅（原创 PagedAttention） | ✅（APC，block 级链式哈希） | ✅（多级 tier） | 进行中（MultiConnector） | ✅（FP8/NVFP4/TurboQuant） | ✅（Mooncake Store） |
| SGLang | ✅（HiRadixCache） | ✅（Radix Tree，任意 token 边界） | ✅（HiCache 三层） | ✅（GPU Staging Buffer, 5× TPS） | ✅（FP8/TurboQuant/Tokenspeed MLA） | 进行中 |
| TensorRT-LLM | ✅（KVCacheManager，两阶段 Claim） | ✅（BlockKey 多维编码，priority LRU） | ✅（HBM→Host→NVMe + GDS） | ✅（Disaggregated serving, 1.4–6.11×） | ✅（FP8/NVFP4） | ✅（NIXL, Dynamo） |
| LMCache | ✅（多级存储） | ✅（跨实例 Prefix Cache） | ✅（HBM→DRAM→SSD→S3） | ✅（fire-and-forget PD 后端） | 待补充 | ✅（Valkey/FS/S3 后端） |
| Mooncake | ✅（block 级） | ✅（全局 block-hash 寻址） | ✅（SSD offload） | ✅（以 KV 为中心的调度） | 待补充 | ✅（分布式 DRAM 池） |

!!! warning "内容时效性"
    推理引擎更新极快，上表信息可能已过时。在做具体技术选型时，请以各项目的最新文档和 Release Notes 为准。

## 关联章节

- 各框架 Prefix Cache 实现对比：[Prefix Cache](prefix-cache.md) §三大引擎
- PD 分离中的 KV 传输：[PD 分离](pd-disaggregation.md)
- KV 量化的具体方法：[压缩与量化](compression-quantization.md)
- 分布式 KV 池的工程细节：[存储层级](storage-hierarchy.md)、[路由与亲和性](routing.md)

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 框架搭建 |
| v0.2 | 2026-05-14 | 纳入各框架的实质性技术细节：vLLM Mooncake Store/多级 offload/TurboQuant、SGLang HiCache/HiSparse/ShadowRadix/PD Staging、TRT-LLM 三层存储/两阶段 Claim/Disaggregated serving、LMCache 多后端生态、Mooncake 分布式 Store 详情；更新对比表增加分布式 KV 列
