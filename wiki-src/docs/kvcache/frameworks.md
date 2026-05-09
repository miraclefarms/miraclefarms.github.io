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

- **PagedAttention**：vLLM 是 PagedAttention 的原始提出者（Kwon et al., 2023），其 Block Manager 是业界实现的参考基准
- **Prefix Cache（Automatic Prefix Caching, APC）**：支持精确哈希匹配的 Prefix Cache，以 Block 为粒度
- **KV Offload（CPU Swap）**：支持将 KV Block 换出到 CPU DRAM，策略为优先换出低优先级请求
- **PD 分离**：vLLM V1 架构在推进 disaggregated prefill 支持（详细状态见官方 roadmap）
- **KV 量化**：支持 FP8 KV Cache（部分硬件）
- **Attention Backend**：支持 FlashAttention、FlashInfer、Triton，KV 访问在 kernel 内部处理

### SGLang

[SGLang](https://github.com/sgl-project/sglang) 以高吞吐和 Prefix Cache 效率著称。

**KVCache 相关特性：**

- **RadixAttention**：SGLang 提出的 Prefix Cache 方案，使用 Radix Tree（基数树）组织 KV Block，支持更细粒度的前缀共享和更高效的缓存管理
- **Block Manager**：与 vLLM 类似的分页管理，但调度策略和缓存驱逐逻辑有所不同
- **KV Offload**：支持 CPU DRAM offload（具体策略待补充）
- **PD 分离**：SGLang 在积极开发 disaggregated prefill 支持（mooncake 集成等）
- **Attention Backend**：深度集成 FlashInfer，KV 管理和 Attention kernel 紧密协作

### TensorRT-LLM

[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) 是 NVIDIA 官方推理框架，以极致的 GPU 性能优化为目标。

**KVCache 相关特性：**

- **Paged KV Cache**：支持分页管理（KVCacheManager），block_size 可配置
- **Chunked Context**：支持将长 Prompt 切分成 chunks 处理，与 KVCache 管理协同
- **KV Cache Reuse**：支持跨请求的 KV 复用（类 Prefix Cache）
- **KV 量化**：支持 FP8 KV Cache，与 TensorRT 量化工具链集成
- **PD 分离**：待补充

### LMCache

[LMCache](https://github.com/LMCache/LMCache) 是专注于 KVCache 分层存储和跨实例共享的项目。

**定位：**作为推理引擎的 KVCache 后端插件，而非独立推理引擎。

**KVCache 相关特性：**

- **多级存储**：将 KVCache 分布到 GPU HBM、CPU DRAM、本地 SSD 和远端内存池（RDMA）
- **跨实例 Prefix Cache**：多个推理实例可以共享同一套 KVCache 存储，实现实例间的 Prefix Cache 命中
- **与 vLLM 集成**：可以作为 vLLM 的 KV 后端使用

### Mooncake（月饼，待补充）

[Mooncake](https://github.com/kvcache-ai/Mooncake) 是 Kimi 团队开源的以 KVCache 为中心的调度系统，支持 PD 分离和 KV 传输。详细实现待补充。

## 对比表（粗粒度）

| 框架 | 分页管理 | Prefix Cache | CPU Offload | PD 分离 | KV 量化 |
|------|---------|-------------|-------------|---------|---------|
| vLLM | ✅（原创） | ✅（APC） | ✅ | 进行中 | ✅（FP8） |
| SGLang | ✅ | ✅（RadixAttention） | ✅ | 进行中 | 待补充 |
| TensorRT-LLM | ✅ | ✅ | 待补充 | 待补充 | ✅（FP8） |
| LMCache | ✅（多级） | ✅（跨实例） | ✅ | 待补充 | 待补充 |

!!! warning "内容时效性"
    推理引擎更新极快，上表信息可能已过时。在做具体技术选型时，请以各项目的最新文档和 Release Notes 为准。

## 深度分析（待补充）

以下内容将在后续版本中逐步补充：

- vLLM V1 架构重构对 KVCache 管理的影响
- SGLang RadixAttention vs vLLM APC 的性能对比
- TensorRT-LLM KVCacheManager 的实现细节
- LMCache 的远端 KV Store 架构
- 各框架在 PD 分离场景下的 KV 传输实现
