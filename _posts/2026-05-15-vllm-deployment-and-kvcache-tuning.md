---
title: vLLM 部署方式、KV Cache 调参与最佳实践
date: 2026-05-15 12:00:00 +0800
author: MiracleFarms
kind: field-note
category: Field Note
intro: 基于 vLLM v0.20.2 源码阅读与官方文档，梳理部署方式全景、KV Cache 全部可调参数及其调优空间、调度与并行策略、硬件适配数据和常见生产陷阱。
tags: [vLLM, KV Cache, Inference]
---

vLLM 的生产调优比初看上去复杂得多——不是因为参数多，而是因为 batch、block、memory 三个维度的约束在 V1 架构中互相耦合。选错 `max_num_batched_tokens` 会让 chunked prefill 形同虚设，KV cache 的 FP8 量化在短上下文场景中反而拖慢 decode，`gpu_memory_utilization` 设到 0.92 在多模态模型上可能直接 OOM。这些配置的"正确值"高度依赖模型结构、硬件代际和负载特征。

本文基于 vLLM v0.20.2（commit `f3d536059`，2026-05-15）<a href="https://github.com/vllm-project/vllm">[1]</a> 的源码阅读，结合 vLLM 官方博客的性能数据与生产指南<a href="https://blog.vllm.ai/">[2]</a>，把部署链路和参数调优空间一次性梳理清楚。

## 一、部署方式全景

vLLM 的部署模式分三层：离线批量推理、在线 serving、跨实例 disaggregated serving。每一层对应不同的生产阶段和运维复杂度。

最轻量的入口是 **Python LLM API**（`vllm/entrypoints/llm.py:106`）<a href="https://docs.vllm.ai/en/latest/design/arch_overview.html">[3]</a>：

```python
from vllm import LLM
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
outputs = llm.generate(["What is vLLM?"])
```

这层 API 封装了 V1 Engine 的初始化、模型加载、KV cache 预分配全过程，适合离线评测、批量推理和快速原型。但 `LLM` 类不走网络协议，不能直接做在线 serving。

在线 serving 的主线是 `vllm serve`<a href="https://docs.vllm.ai/en/latest/configuration/engine_args/">[4]</a>，一行命令拉起 OpenAI 兼容的 HTTP 服务：

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.92 \
    --max-model-len 8192 \
    --max-num-batched-tokens 16384 \
    --enable-prefix-caching
```

V1 架构采用多进程模型：1 个 API server 进程（处理 HTTP、tokenization、streaming）、1 个 engine core 进程（调度器 + KV cache 管理、运行在 busy loop 中）、N 个 GPU worker 进程（每个 GPU 一个）。当 `--data-parallel-size > 1` 时，额外启动 1 个 DP coordinator 进程。这组进程通过 shared memory 和 IPC 通信，避免了 V0 中 Python GIL 和进程间序列化开销。

`vllm serve` 暴露 `/v1/chat/completions`、`/v1/completions`、`/v1/embeddings` 等标准端点，同时支持 Anthropic Messages API、SageMaker 集成和 gRPC 协议。Server 参数支持 YAML 配置文件，通过 `--config config.yaml` 加载。

**Disaggregated serving（P/D 分离）**<a href="https://docs.vllm.ai/en/latest/features/disagg_prefill.html">[5]</a> 通过 `KVTransferConfig`（`vllm/config/kv_transfer.py:23`）实现，将 prefill 和 decode 分配到不同的 vLLM 实例。当前支持的 KV 传输连接器有 9 种：

| 连接器 | 传输机制 | 适用场景 |
|--------|----------|----------|
| `P2pNcclConnector` | NCCL point-to-point | 单节点多 GPU，动态 xPyD 扩展 |
| `MooncakeConnector` | GPUDirect RDMA | 跨节点分布式 KV cache 池 |
| `NixlConnector` | 全异步 send/recv | 低延迟 P/D 传输 |
| `MORIIOConnector` | AMD RDMA | AMD MI300X，单节点 2.5x goodput |
| `LMCacheConnectorV1` | NIXL 传输 | 外部 KV cache 服务 |
| `OffloadingConnector` | CPU memory offload | 显存受限场景 |
| `FlexKVConnectorV1` | 分布式 KV store | 超大规模推理 |
| `MultiConnector` | 多个子连接器串联 | PD + offload 等组合场景 |
| `ExampleConnector` | 参考实现 | 开发和调试 |

典型用法——单机 2 卡 P/D 分离：

```bash
# Prefill 实例（GPU 0）
vllm serve MODEL --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}'

# Decode 实例（GPU 1）
vllm serve MODEL --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}' --gpu-memory-utilization 0.7
```

Decode 实例的 `gpu_memory_utilization` 建议降到 0.7，因为需要预留显存作为 NCCL 接收 buffer。P/D 分离的核心价值是独立调优 TTFT 和 ITL——prefill 实例可以用更大的 `max_num_batched_tokens` 提升 prefill 吞吐，decode 实例则用更保守的 batch 保证延迟稳定性。但它不会直接提升总吞吐，只是让延迟更可预测。

**多机多卡**场景下，通过标准分布式启动器配置 `--tensor-parallel-size`、`--pipeline-parallel-size` 和 `--data-parallel-size`，配合 `--master-addr`、`--master-port` 指定通信地址。Ray 集群集成通过 `--distributed-executor-backend ray` 启用，适合已有 Ray 基础设施的团队，但维护成本更高——vLLM 的主推路径是原生多进程执行器。

**选择原则**：离线推理用 Python LLM API，在线 serving 首选 `vllm serve`（V1 engine），需要延迟隔离时上 P/D 分离，多机分布式直接用 `--tensor-parallel-size` + `--data-parallel-size`，已有 Ray 基础设施则继续用 Ray executor。

## 二、KV Cache 参数体系

vLLM 的 KV cache 配置集中在 `CacheConfig` 类（`vllm/config/cache.py:42`），共 20+ 个可调参数。与 TRT-LLM 的层级化控制不同，vLLM 的 KV cache 设计更偏向"少而精"——只保留决策真正需要的参数，大量内部优化由 engine 自动处理。

### 2.1 容量控制：两条规则，两个参数

GPU KV cache 的可用空间由两条规则决定：`gpu_memory_utilization`（默认 0.92）控制总显存中给模型执行器的比例；`kv_cache_memory_bytes`（可选）提供精确的 per-GPU KV cache 字节数上限——设置后直接忽略 `gpu_memory_utilization`。

`gpu_memory_utilization` 是生产中最常动的参数之一。0.92 的默认值在纯文本模型上通常安全，但在以下场景需要下调：多模态模型（图像 encoder 和 vision token 占用额外显存）、MoE 模型（专家参数和 all-to-all buffer 占用显著空间）、P/D 分离的 decode 端（需预留 KV 传输 buffer）。8-GPU 部署 DeepSeek 类 MoE 时，0.70–0.80 是更安全的起点。

`kv_cache_memory_bytes` 适合对显存布局有精确预期的场景。与 TRT-LLM 的 `free_gpu_memory_fraction` 不同，vLLM 用的是一个**包含模型权重和运行时 buffer 的总池**比例——这意味着增大 TP 后，模型权重在各 GPU 上均摊，但 `gpu_memory_utilization` 仍然作用于每张 GPU 的剩余显存。

还有一个方式是通过环境变量 `VLLM_CPU_KVCACHE_SPACE` 为 CPU backend 预留 KV cache 空间，但这仅适用于 CPU 推理场景。

### 2.2 块大小与前缀复用

`block_size` 默认 16 token，是 KV cache 的物理分配粒度——也是 vLLM PagedAttention 论文<a href="https://arxiv.org/abs/2309.06180">[6]</a> 提出时的原始设定。与 TRT-LLM 的 32 相比，16 的粒度更细，理论上前缀复用的对齐损失更小（一个 1024 token 前缀如果恰好差几个 token 不够对齐，fine grain 比 coarse grain 浪费更少）。但 vLLM 在 V1 架构中引入了 `hash_block_size` 来进一步解耦哈希粒度和物理块粒度——前缀缓存的 key 可以在 8 token 粒度上计算哈希，物理块仍然维持在 16 token 上分配。这意味着你不必为了提升复用率而改 `block_size`（这会影响整个 attention kernel 的 tile 尺寸），只需设 `--block-size` 搭配 `--hash-block-size` 即可。

**前缀缓存（APC）是 V1 的默认开启功能**（`enable_prefix_caching` 默认为 `True`）。这比 TRT-LLM 激进得多——在 TRT-LLM 中你需要显式打开 `enable_block_reuse`，而 vLLM 认为现代 serving 场景（chat history 复用、system prompt 固定、multi-round conversation）天然适合 prefix caching，除非你明确知道负载没有共享前缀。APC 基于 SHA256 哈希（`prefix_caching_hash_algo` 默认为 `"sha256"`），对每个 block 计算 `(parent_hash, block_tokens, extra_hashes)` 的哈希值。Extra hashes 包含 LoRA ID、多模态输入哈希和 cache salt，用于多租户隔离。

可选哈希算法：`sha256_cbor`（跨语言可复现）、`xxhash`（更快但密码学上不安全，多租户场景有碰撞风险）、`xxhash_cbor`（快速 + 可复现）。`hash_block_size` 的引入是 V1 对 prefix caching 最重要的改进——当多个 KV cache group 有不同的物理 block size 时（如 hybrid attention 模型），vLLM 会自动选最小物理 block size 作为 `hash_block_size`，通过 `convert_block_hashes()` 在不同粒度间做哈希合并。

**Prefix caching 的适用边界很明确**：对长文档 QA（多次查询同一文档）和多轮对话（chat history 高度复用）收益最大；对完全随机的单轮请求完全无益；多模态请求的 vision token 哈希在不同请求间不易匹配，prefix caching 对多模态的收益有限。请求级可通过 `"cache_salt"` 字段做租户级隔离，防止不同租户的 system prompt 意外共享。

### 2.3 KV Cache 数据类型与量化

`cache_dtype` 控制 KV cache 存储精度（`vllm/config/cache.py:74`），支持 `"auto"`（跟随模型权重 dtype）、`"fp8"`、`"fp8_e4m3"`、`"fp8_e5m2"`、`"bfloat16"`、`"nvfp4"`（Blackwell GPU），以及实验性的 `"turboquant_k8v4"`、`"turboquant_4bit_nc"`、`"turboquant_k3v4_nc"`、`"turboquant_3bit_nc"`。

2026 年 4 月 22 日的博客文章 *The State of FP8 KV-Cache and Attention Quantization in vLLM*<a href="https://blog.vllm.ai/2026/04/22/fp8-kv-cache.html">[7]</a> 提供了全面的生产验证数据：**FP8 KV cache 已成为生产就绪功能**。在 H100 上，Llama-3.1-8B 的 FP8 decode ITL 斜率仅为 BF16 的 54%（几乎将 decode 成本减半），output throughput 提升 14.9%，break-even 约在 7K token 上下。关键改进包括：两级累加修复（恢复 NIAH 精度从 13% 到 89%）、`--kv-cache-dtype-skip-layers` 跳过 sliding window 层、per-head 量化 scales 静态计算、query 量化融合入 `torch.compile` 路径。

短上下文（< 7K token）场景中 BF16 可能更快，因为 FP8 的量化/反量化开销在没有足够 KV cache 压力时得不偿失。`head_dim=256` 的模型（如某些 MoE 架构）prefill 会慢约 1.6x，因为两级累加在寄存器压力下效率下降。对于 hybrid attention 模型（如 gpt-oss-20b），推荐 `--kv-cache-dtype fp8 --kv-cache-dtype-skip-layers sliding_window`——跳过 sliding window 层的量化，因为它们 KV cache 本就很小，FP8 收益不大却可能引入精度问题。

TurboQuant 系列（4-bit 和 3-bit KV cache）<a href="https://blog.vllm.ai/2026/05/11/turboquant.html">[8]</a> 提供了更大的内存容量（3.4x），但以吞吐为代价——3-bit 变体会降低 40–52% 的吞吐，且可能造成 20 分的推理 benchmark 精度下降。当前的共识是：**FP8 是首选默认值**，TurboQuant 仅适用于显存极度受限且精度容忍度更高的 batch 离线推理。

### 2.4 CPU 卸载

vLLM 的 KV cache CPU 卸载通过 `kv_offloading_size`（单位 GiB）和 `kv_offloading_backend`（`"native"` 或 `"lmcache"`）控制。当 TP > 1 时，`kv_offloading_size` 是所有 TP rank 的总 buffer 大小之和。

与 TRT-LLM 的 pinned memory 预分配不同，vLLM 的 native offload 使用了 V1 的 block pool 管理逻辑——复用率低的 block 被迁移到 CPU，需要时通过 UVA（Unified Virtual Addressing）或显式拷贝加载回来。但这仍然是 PCIe 带宽敏感的：如果 `vllm:iteration_latency_seconds` 随 offload 量增加而明显上升，说明 PCIe 不够快，优先考虑使用 Mooncake 等分布式 KV cache 池方案<a href="https://blog.vllm.ai/2026/05/06/mooncake.html">[9]</a> 替代本地 CPU offload。

实验性功能 `VLLM_USE_SIMPLE_KV_OFFLOAD` 环境变量提供了一个轻量级的 offload 路径，用于 debug 和性能对比。

V1 的一个重要变化是：**移除了 GPU ↔ CPU KV cache swap**（V0 中的 SWAP 预占模式），改为 RECOMPUTE 预占——当 KV cache 不足时，被驱逐的请求不会换出到 CPU，而是直接丢弃 KV cache，后续由 prefill 重新计算。这个决策减少了显存碎片和 swap 带宽开销，代价是被驱逐请求的延迟惩罚更大。

## 三、调度器策略与并行配置

### 3.1 批处理核心参数

`max_num_batched_tokens` 和 `max_num_seqs` 是 V1 调度器的两个核心约束（`vllm/config/scheduler.py:49`）。与 TRT-LLM 的 IFB 调度器不同，V1 的调度器对所有 token 一视同仁——不严格区分 prefill 和 decode 的阶段，而是通过 token budget 做统一调度。这种设计让 chunked prefill 成为 V1 的默认行为。

vLLM 对硬件做了感知式的默认值匹配（`vllm/engine/arg_utils.py` 中的 `get_batch_defaults()`）：

| 硬件 | max_num_batched_tokens (离线) | max_num_batched_tokens (在线) | max_num_seqs |
|------|------------------------------|------------------------------|-------------|
| **H100 / H200 / MI300x**（显存 ≥ 70GB） | 16384 | 8192 | 1024 |
| **A100 / 小显存 GPU**（默认） | 8192 | 2048 | 256 |
| **TPU V6E** | 2048 | 1024 | 默认 |
| **CPU** | 4096 × world_size | 2048 × world_size | 256 × world_size |

吞吐优先模式（`--performance-mode throughput`）下，这些默认值翻倍。对于小模型在大 GPU 上的部署，`max_num_batched_tokens` 应该设到更高（如 32768 或更大），否则 GPU 利用率无法填满。

`max_num_scheduled_tokens` 提供了一个额外的软上限——当 speculative decoding 等机制可能向 batch 中追加额外 token 时，scheduler 可以用比 `max_num_batched_tokens` 更小的值控制基础调度量，给 spec tokens 留出空间。

### 3.2 Chunked Prefill

V1 的 chunked prefill（`enable_chunked_prefill` 默认为 `True`）不需要单独开启——它已经内置在调度逻辑中。核心参数是 `max_num_partial_prefills`（默认 1），即同一迭代中可以部分 prefill 的最大序列数。设为 >1 时启用并发 partial prefill，允许更短的 prompt "跳过" 长 prompt 的队列，提升短请求的 TTFT。

`max_long_partial_prefills`（默认 1）和 `long_prefill_token_threshold`（默认 0，自动设为 `max_model_len × 0.04`）进一步区分长/短 prompt 的调度优先级。调度策略 `policy` 支持 `"fcfs"`（先来先服务，默认）和 `"priority"`（基于请求优先级的抢占调度）。

`scheduler_reserve_full_isl`（默认 `True`）是 V1 的一个关键保护机制——scheduler 在接收新请求前，先确认整个 input sequence 能否放入 KV cache，而不是只看第一个 chunk。这避免了过度接收请求导致的 KV cache 颠簸。

### 3.3 并行策略

vLLM 的并行配置集中在 `ParallelConfig`（`vllm/config/parallel.py:108`），支持六种并行维度的组合：

| 策略 | 参数 | 默认值 | 推荐场景 |
|------|------|--------|---------|
| Tensor Parallel | `tensor_parallel_size` | 1 | 模型超过单 GPU 显存 |
| Pipeline Parallel | `pipeline_parallel_size` | 1 | 超大模型叠加 PP |
| Data Parallel | `data_parallel_size` | 1 | 多副本 serving、MoE 专家分片 |
| Expert Parallel | `enable_expert_parallel` | `False` | MoE 专家分布 |
| Prefill Context Parallel | `prefill_context_parallel_size` | 1 | 超长 prefill 序列分片 |
| Decode Context Parallel | `decode_context_parallel_size` | 1 | 超长 decode 序列分片 |

**vLLM 的 DP 概念与传统的 data parallelism 不同**——它主要用于 MoE 模型的专家分片。`data_parallel_size` 与 `tensor_parallel_size` 相乘得到总的 TP + EP rank 数。`enable_expert_parallel` 开启后，MoE 层从 TP 切换到 EP。`enable_ep_weight_filter` 进一步跳过非本地专家的权重读取，大幅减少 MoE 模型的磁盘 I/O。EPLB（Expert Parallel Load Balancing）通过 `eplb_config` 配置 window size（默认 1000 step）和重排间隔（默认 3000 step），动态调整专家分布以平衡负载。

与 TRT-LLM 的固定 5 种并行方案不同，vLLM 的设计更灵活——`prefill_context_parallel_size` 和 `decode_context_parallel_size` 可以独立设置（即 PCP 和 DCP 分离），让超长上下文场景中的 prefill 和 decode 分别以不同的 CP 配置运行。DCP 的通信后端可选 `"ag_rs"`（allgather + reducescatter，默认）或 `"a2a"`（all-to-all，适合 MLA 模型）。

**Data Parallel 的三种 LB 模式**：`data_parallel_external_lb`（外部负载均衡，适合 Kubernetes "one-pod-per-rank" 的 wide-EP 部署）、`data_parallel_hybrid_lb`（混合模式，vLLM 负责节点内 DP rank 间负载均衡，外部 LB 负责节点间分发）。这对于非 MoE 模型的多副本 serving 不需要 `--data-parallel-*` 参数——应该启动独立 vLLM 实例。

### 3.4 CUDA Graph 与优化等级

vLLM 用四级优化等级（`-O0` 到 `-O3`）来控制 CUDA graph 捕获策略和编译融合程度（`vllm/config/vllm.py:282`）：

| 等级 | CUDA Graph | 编译 | 关键融合 | 适用场景 |
|------|-----------|------|---------|---------|
| O0 | NONE | NONE | 无 | 开发/调试，最快启动 |
| O1 | PIECEWISE | VLLM_COMPILE | norm_quant, act_quant, mla_dual_rms_norm | 中等性能 + 快启动 |
| O2（默认） | FULL_AND_PIECEWISE | VLLM_COMPILE | O1 + allreduce_rms, attn_quant, SP, gemm_comms, rope_kvcache | 生产环境 |
| O3 | FULL_AND_PIECEWISE | VLLM_COMPILE + FlashInfer autotune | 同 O2 | 极致性能（目前等同 O2） |

**O2 是生产默认值**。FULL cudagraph 对解码阶段的 kernel launch overhead 消除效果最好，PIECEWISE 处理 prefill 阶段的动态形状。Kernel fusion 是 vLLM 近期吞吐提升的核心驱动力——DeepSeek V3.2 上约 33 → 10 次 kernel launch per layer，带来 1.28x speedup（batch size 1）<a href="https://blog.vllm.ai/2026/05/11/artificial-analysis-leaderboard.html">[10]</a>。

`--enforce-eager` 完全关闭 CUDA graph，用于 kernel 调试和精度对比。`--compilation-config` 提供精细控制：`cudagraph_mode`（`NONE` / `PIECEWISE` / `FULL` / `FULL_AND_PIECEWISE`）、`cudagraph_capture_sizes`（手动指定捕获的 batch size 列表）、`fusions` 的逐项开关。

## 四、硬件适配与性能参考

不同 GPU 架构对 vLLM 配置的敏感度差异显著，以下是官方 benchmark 的关键数据：

**H100（Hopper）**<a href="https://blog.vllm.ai/2026/04/22/fp8-kv-cache.html">[7]</a>：
- Llama 3.1 8B, FP8 KV cache, TP=1：output throughput +14.9% vs BF16，decode ITL 成本降至 BF16 的 54%
- Llama 3.3 70B, FP8 KV cache：burst throughput 2.6x BF16（4×H100），burst TTFT 从 ~17s 降至 ~1.3s

**B200 / GB200（Blackwell）**：FlashInfer attention backend 优先级最高，FP8 KV cache 的 break-even 降至约 4K token。支持 NVFP4 weight + FP8 KV cache 的组合。

**MI300X（AMD）**<a href="https://blog.vllm.ai/2026/04/07/mori-io.html">[11]</a>：
- Qwen3-235B-A22B-FP8，MORI-IO P/D 分离：**2.5x goodput** vs 合设模式
- 消除所有 ITL 超标，仅剩 TTFT 超标（由 P/D 传输开销引起）

**GB300（Grace-Blackwell）**<a href="https://blog.vllm.ai/2026/02/13/deepseek-v3-2-gb300.html">[12]</a>：
- DeepSeek V3.2 NVFP4, TP=2：**7,360 tok/s per GPU**

**DeepSeek V3.2 综合数据**<a href="https://blog.vllm.ai/2026/05/11/artificial-analysis-leaderboard.html">[10]</a>：
- 单用户：230 tok/s（kernel fusion + MTP=1 + P/D disaggregation）
- 并发 256：7.33 req/s（TEP=8，比 baseline +10%）
- 注意：125 tok/s concurrency 1 no MTP, 234 tok/s with MTP=1, 262 tok/s with P/D disaggregation + MTP=3

量化选择上：H100 优先 BF16 weight + FP8 KV cache（当前最佳性价比组合），B200 优先 FP8 KV cache + 可选 NVFP4 weight，MI300X 推荐 FP8 weight + FP8 KV cache。**L40S 由于显存带宽较低，FP8 KV cache 几乎是必须的**。

## 五、部署监控

`vllm serve` 自动导出 Prometheus metrics 到 `/metrics` 端点。核心监控指标按用途分为三组<a href="https://docs.vllm.ai/en/latest/configuration/engine_args/">[4]</a>：

**KV cache 利用率**：`vllm:gpu_cache_usage_perc`（GPU KV cache 使用百分比）。持续 >95% 意味着 KV cache 是吞吐瓶颈，需要增大 `gpu_memory_utilization`、减小 `max_model_len` 或增加 GPU。监控 `vllm:num_preemptions_total`（累计预占次数）——频繁预占说明 KV cache 容量不足或并发过高。

**请求延迟**：`vllm:time_to_first_token_seconds`（TTFT）、`vllm:time_per_output_token_seconds`（TPOT / ITL）、`vllm:e2e_request_latency_seconds`（端到端延迟）、`vllm:request_success_total`（成功请求数）。TTFT 持续超标时检查 `max_num_batched_tokens` 是否过小或 chunked prefill 是否需要更大的 `max_num_partial_prefills`。ITL 超标通常指向 decode 阶段被 prefill 阻塞——考虑 P/D 分离。

**吞吐与排队**：`vllm:request_prompt_tokens`、`vllm:request_generation_tokens`、`vllm:request_queue_time_seconds`（排队时间）。排队时间持续增长意味着请求到达率超过处理能力——需要提升并行度或增加 data parallel 副本。

vLLM 的 V1 架构还通过 engine core 的 busy loop 暴露内部状态。`/stats` 端点提供更细粒度的 scheduler 和 KV cache 统计。`--enable-log-requests` 开启后，每个请求的生命周期事件（排队、scheduling、prefill 完成、decode 完成）都会被记录，用于追溯延迟瓶颈。

## 六、常见生产陷阱

以下是基于源码分析和社区讨论总结的高频问题：

**`gpu_memory_utilization` 设太高导致 OOM**。多模态模型和多 GPU MoE 模型的额外 buffer（vision encoder 输出、all-to-all 中间结果、CUDA graph 显存）通常不在 `gpu_memory_utilization` 的计算中。解法：多模态模型从 0.70 起步，纯文本 MoE 从 0.80 起步，确认稳定后再上调。

**FP8 KV cache 在短上下文场景中反向优化**。7K token 以下是 break-even 的大致分界线——低于这个值，FP8 的量化/反量化开销超过显存节省带来的收益。如果负载以短对话为主（平均 context < 4K），BF16 KV cache 更合适。

**`max_num_batched_tokens` 和 `max_num_seqs` 不匹配导致 batch 利用率低**。如果 `max_num_batched_tokens / max_num_seqs` 远小于平均 prompt 长度，scheduler 可能只能调度 1–2 个请求，其余 GPU 算力被浪费。规则：`max_num_batched_tokens` 应该至少是 `max_num_seqs × (平均 input 长度 / chunked prefill chunk 数)`。

**Prefix caching 在多模态场景中的隐性失效**。Vision token 的哈希在不同请求间难以匹配，即使文本 prompt 完全相同的两张不同图片也会产生不同的 KV cache 哈希。多模态 serving 不需要关闭 APC，但也不应期望高 cache hit rate。

**CPU 核心数不足导致 engine core 饥饿**。V1 的多进程架构需要至少 `2 + N` 个物理 CPU 核心（N = GPU 数量）。以 8 GPU 且 `--tp=2 --dp=4` 为例，共需 17 个进程（4 API server + 4 engine core + 8 GPU worker + 1 DP coordinator），CPU 核心不足时 engine core 无法及时向 GPU worker 下发调度结果，造成 GPU idle bubbles。

**`--block-size` 手工调大破坏了 prefix caching 粒度**。默认 16 已经是经过大量 benchmark 的平衡值。增大到 32 或 64 会减少 block 元数据开销，但前缀 hash 的粒度和复用率同步下降。除非模型 head_dim 很大且确认 block size 是瓶颈，否则不要动这个参数。

**多租户场景缺少 cache salt 隔离**。不同租户的 system prompt 可能通过 prefix caching 意外共享 KV cache，造成 prompt 泄漏。通过 API 请求中的 `"cache_salt"` 字段做租户级隔离是最简单有效的方案。

---

## 参考资料

[1] [vLLM GitHub Repository](https://github.com/vllm-project/vllm)

[2] [vLLM Official Blog](https://blog.vllm.ai/)

[3] [vLLM Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview.html)

[4] [vLLM Engine Arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/)

[5] [Disaggregated Prefilling in vLLM](https://docs.vllm.ai/en/latest/features/disagg_prefill.html)

[6] [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

[7] [The State of FP8 KV-Cache and Attention Quantization in vLLM](https://blog.vllm.ai/2026/04/22/fp8-kv-cache.html)

[8] [A First Comprehensive Study of TurboQuant: Accuracy and Performance](https://blog.vllm.ai/2026/05/11/turboquant.html)

[9] [Serving Agentic Workloads with vLLM × Mooncake](https://blog.vllm.ai/2026/05/06/mooncake.html)

[10] [vLLM Tops Artificial Analysis Leaderboard](https://blog.vllm.ai/2026/05/11/artificial-analysis-leaderboard.html)

[11] [MORI-IO: Disaggregated Serving on AMD MI300X](https://blog.vllm.ai/2026/04/07/mori-io.html)

[12] [DeepSeek-V3.2 on GB300: 7,360 tok/s per GPU](https://blog.vllm.ai/2026/02/13/deepseek-v3-2-gb300.html)

### 版本对齐信息

| 依赖 | 版本/Commit | 日期 |
|------|-----------|------|
| vLLM | `f3d536059` (v0.20.2) | 2026-05-15 |
| vLLM Blog / Docs | 最新发布 | 2026-05-15 |
| PagedAttention Paper | arXiv 2309.06180 | 2023-09 |
