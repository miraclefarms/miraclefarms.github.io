---
title: SGLang 部署方式、KV Cache 调参与最佳实践
date: 2026-05-14 12:00:00 +0800
author: MiracleFarms
kind: field-note
category: Field Note
intro: 基于 SGLang v0.5.11 源码阅读与官方文档，梳理部署方式全景、KV Cache 全部可调参数及其调优空间、调度与并行策略、监控指标和常见生产陷阱。
tags: [SGLang, KV Cache, Inference]
---

> **版本声明**：本文分析基于 SGLang commit `50f405816e`（v0.5.11，2026-05-14）；除非特别说明，以下描述均基于此版本。

SGLang 在生产环境的部署和调参复杂度高于第一眼的印象——它同时提供七种部署入口、20+ 个 KV cache 相关参数、十几种 attention backend 选择，以及五维并行的叠加能力。把 `mem_fraction_static`、`chunked_prefill_size` 和 `schedule_conservativeness` 三个参数的联动关系吃透，是吞吐量不掉的底线——这三个值看似正交，实际上共同决定了 KV cache 池大小、预填充分块粒度和调度器对新请求的拥抱程度。任何一个偏离最优值，吞吐可能掉 30% 以上。

本文基于 SGLang v0.5.11（commit `50f405816e`）<a href="https://github.com/sgl-project/sglang">[1]</a> 的源码阅读，结合官方调参指南<a href="https://docs.sglang.ai/advanced_features/hyperparameter_tuning.html">[2]</a> 和 HiCache 最佳实践<a href="https://docs.sglang.ai/advanced_features/hicache_best_practices.html">[3]</a>，把部署链路和参数调优空间一次性梳理清楚。

## 一、部署方式全景

SGLang 从轻量 Python API 到全分布式 disaggregated serving 覆盖七种部署模式，每种对应不同的生产阶段和运维复杂度。

最轻量的入口是 **Engine API**<a href="https://docs.sglang.ai/backend/engine_api.html">[4]</a>：

```python
import sglang as sgl
llm = sgl.Engine(model_path="meta-llama/Meta-Llama-3.1-8B-Instruct")
output = llm.generate(["What is the capital of France?"])
```

`Engine()` 初始化即完整封装了模型加载、Scheduler 进程启动、内存池分配全过程。适合离线推理、评测脚本和快速原型。调用 `llm.shutdown()` 释放资源。Engine 不走 HTTP 协议，不能直接做在线 serving。

在线 serving 的主线是 `sglang serve`（或等价旧命令 `python -m sglang.launch_server`），一行拉起 OpenAI 兼容的 HTTP 服务：

```bash
sglang serve meta-llama/Llama-3.1-8B-Instruct \
    --tp-size 1 \
    --mem-fraction-static 0.85 \
    --host 0.0.0.0 --port 30000
```

背后启动 FastAPI + Uvicorn，暴露 `/v1/chat/completions`、`/v1/completions`、`/v1/models` 标准端点，以及 `/health`、`/health_generate`、`/metrics` 等运维端点。加 `--grpc-mode` 切换为 gRPC 协议，用于 sgl-router 等外部路由器的 token 级高性能转发。SGLang 同时支持 OpenAI、Anthropic、Ollama 等多种 API 协议栈，通过检测请求格式自动路由。

与 TRT-LLM 不同的是，SGLang 采用多进程 Scheduler 架构：Scheduler 进程独立运行，通过 `Recv/Reply` 接口与 detokenizer 和 tokenizer 进程通信。每个 TP rank 上都有一个 Scheduler 进程，用 NCCL 同步调度决策。这层间接性使得 SGLang 在错误隔离和灵活度上有优势——Scheduler crash 不会影响已完成的请求——但也引入了额外的进程间通信开销。

**sgl-router**<a href="https://github.com/sgl-project/sglang/tree/main/sgl-router">[5]</a> 是 SGLang 推荐的多实例数据并行方案（替代直接用 `--dp-size`），路由策略包括 round-robin、follow-bootstrap-room、token-aware balancing。相比框架内 DP，router 的隔离性更强，一个实例崩溃不影响其他。

Disaggregated serving（P/D 分离）用 `--disaggregation-mode prefill` 或 `--disaggregation-mode decode` 分别启动 prefill 和 decode 服务器，KV cache 通过 Mooncake（默认）、NIXL 或 Ascend 做跨节点 RDMA 传输。Prefill 节点可以通过 `--enable-hierarchical-cache` 扩展 HiCache，decode 节点通过 `--disaggregation-decode-enable-offload-kvcache` 异步 offload KV cache 给 prefill 复用<a href="https://docs.sglang.ai/advanced_features/pd_disaggregation.html">[6]</a>。

Embedding 模式下使用 `--is-embedding`，配合 `--chunked-prefill-size -1` 和 `--disable-radix-cache`、`--prefill-only-disable-kv-cache`，彻底跳过 KV cache 物理分配，最大化吞吐。

**选择原则**：离线评测用 Engine API，在线 serving 首选 `sglang serve`，多实例 DP 用 sgl-router，多机 disaggregated 用 PD 部署模式，TP/PP/EP 组合用于超过单 GPU 显存的大型模型或长上下文场景。

## 二、KV Cache 参数体系

SGLang 的 KV cache 配置集中在 `ServerArgs`（`server_args.py`），共 20+ 个可调参数。这些参数从最粗粒度的显存上限，逐层下钻到前缀缓存策略、KV 量化精度和多级缓存架构。

### 2.1 容量控制

GPU KV cache 的可用空间由三条规则共同决定，取最小值：`mem_fraction_static` × GPU 总显存（减去 reserved memory）、`max_total_tokens`、`max_running_requests` 隐含的并发限制。

`mem_fraction_static` 默认自动计算（通常约 0.88），公式是 `(GPU 显存 - reserved_mem) / GPU 显存`<a href="https://docs.sglang.ai/advanced_features/hyperparameter_tuning.html">[2]</a>，reserved memory 包含 chunked prefill 的 activation 预留（`max(chunked_prefill_size, 2048) * 1.5`）、CUDA graph 占用（`cuda_graph_max_bs * 2`）、并行度调整项（`tp_size * pp_size / 8 * 1024`）。启动日志中的 `available_gpu_mem` 是判断依据：5–8 GB 说明 KV cache 池合理；>10 GB 说明空间浪费，逐步上调 `mem_fraction_static`（官方建议每次 +0.01）直到逼近 OOM 边界。

`max_total_tokens` 提供硬上限，以 token 数为单位直接控制内存池容量。当你知道精确的并发需求时（如最多同时服务 128 个 32K 上下文请求），用 `max_total_tokens` 比 fraction 更可预测。不设时默认由 fraction 决定。

`max_running_requests` 自动计算，限制同时处于运行状态的请求数量。一个容易被忽视的细节：若 `max_running_requests=2048` 但每个请求的 `max_new_tokens=4096`，KV cache pool 可能在并发 500 时就耗尽——调度器为每个请求预留的空间是 `new_token_ratio × max_new_tokens`，不是实际生成 token 数。

### 2.2 page_size：KV 缓存粒度的暗线

`page_size` 默认 1（token 级粒度），但会按 attention backend 自动调整：MLA 模型统一 64 或 128、FlashAttention-4 非 MLA 为 128、TRT-LLM MHA backend 为 16/32/64、MUSA 为 64、HiCache 固定 64。`page_size=1` 意味着前缀匹配可以精确到 token 级，不会出现差几个 token 凑不齐一个 block 而无法复用的问题。代价是更大的 page table 元数据开销和更频繁的 radix tree 操作。64 是 HiCache 的推荐值，在复用率和元数据开销之间取得平衡。

### 2.3 前缀缓存：RadixAttention

SGLang 的前缀缓存系统叫 RadixAttention<a href="https://docs.sglang.ai/advanced_features/server_arguments.html">[7]</a>，基于 radix tree 数据结构做 token 序列前缀匹配。与 TRT-LLM 基于 block hash 的复用不同，RadixAttention 用 sha256 哈希 token 序列后在 radix tree 中查找已缓存的 KV page 索引——匹配规则是前缀字节级精确匹配，任何字符差异都导致新的 radix tree 分支。

三个关键控制参数：

`--disable-radix-cache` 关闭整个 RadixAttention 前缀缓存。关闭后回退为 `ChunkCache`，只做 chunk 级粗粒度缓存。多模态场景（图片/视频输入）当前需要主动关闭，因为 RadixAttention 暂未兼容多模态内容哈希。

`--radix-eviction-policy` 控制淘汰策略，默认 `lru`。可选 `lfu`（按使用频率）、`slru`（分段 LRU）、`priority`（按请求优先级）。生产环境中若请求波次有明显冷热分层（如每天开始时 system prompt 命中率高、后续是零散的 one-shot 请求），`slru` 比 `lru` 更优——分段设计保证热前缀不会短时间被大量冷数据挤出。

`--schedule-policy lpm`（Longest Prefix Match）开启后，调度器会重新排序 waiting queue 中的请求，让前缀匹配长的请求先执行，最大化前缀复用率。代价是调度排序引入额外 CPU 开销，当 waiting queue 超过 128 时自动降级为 FCFS。

### 2.4 KV Cache 量化：FP8/FP4

`--kv-cache-dtype` 控制 KV cache 存储精度：`auto`（跟随模型）、`fp8_e5m2`、`fp8_e4m3`、`fp4_e2m1`<a href="https://docs.sglang.ai/advanced_features/quantized_kv_cache.html">[8]</a>。

FP8 E4M3 是推荐选项——精度损失最小（`gsm8k` 上 DeepSeek-R1-0528：BF16 0.9157 → FP8 0.9154）。FP8 E5M2 动态范围更大但精度更低。FP4 E2M1（MXFP4，需要 CUDA 12.8+ 和 PyTorch 2.8.0+）在大模型上泛化尚可——DeepSeek-R1-0528 在 `gsm8k` 上 FP4 保持 0.9124，但在 `aime25` 上从 0.5067 降到 0.4000；GPT-OSS-120B 在 `aime25` 上从 0.7533 掉到 0.3533。大模型 + 简单任务 FP4 可用，小模型 + 复杂推理 FP4 有明显退化。

FP8 KV cache 需要 scaling factors，可从 checkpoint 自动加载（如 ModelOpt 预量化模型）或通过 `--quantization-param-path` 指定 JSON 文件。不提供时默认 scaling factor = 1.0，可能导致精度异常。FP4 的 block scaling（block=16）在量化/反量化时在线计算，不需要额外文件。

量化选择对显存的影响：BF16 → FP4 支持约 3.56× 更长的上下文或更多并发（含 scale buffer 开销）；FP4 比 FP8 多存约 1.78× 的 token。

### 2.5 HiCache：三级层次缓存

`--enable-hierarchical-cache` 将 KV cache 从单一 GPU 池扩展到 GPU → CPU → Storage 三级架构<a href="https://docs.sglang.ai/advanced_features/hicache_best_practices.html">[3]</a>。

核心配置链：

```bash
sglang serve model-path \
    --page-size 64 \
    --enable-hierarchical-cache \
    --hicache-ratio 2 \
    --hicache-io-backend kernel \
    --hicache-mem-layout page_first \
    --hicache-write-policy write_through \
    --hicache-storage-backend mooncake \
    --hicache-storage-prefetch-policy timeout
```

`hicache_ratio` 默认 2.0，即 CPU 缓存 = GPU 缓存 × 2。也可以用 `--hicache-size` 指定绝对 GB 数覆盖 ratio。`write_through`（默认）保证 GPU 写入时同步写到 CPU，`write_back` 延迟写入以换取更低延迟但可能丢数据，`write_through_selective` 选择性穿透只写高频条目。

`hicache_mem_layout` 是容易被忽略但重要的选择：`page_first` 配合 `kernel` IO backend 实现 zero-copy 传输，`page_first_direct` 是针对 `direct` backend 优化的相同布局。`layer_first` 是传统布局，layer 数据连续存放，兼容性最广但 IO 效率不如 page_first。

Storage 层支持 Mooncake（RDMA 分布式存储）、HF3FS、NIXL、AIBrix 等。预取策略 `best_effort`（需要时及时终止）、`wait_complete`（完整加载、最高复用率）、`timeout`（超时平衡）。Prefetch timeout 的计算公式：`2.0 + 0.1 × (tokens / 1024)` 秒，最大 30 秒。

HiCache 运行时可动态挂载/卸载 storage backend（不需要重启），通过 HTTP admin 端点操作<a href="https://docs.sglang.ai/advanced_features/hicache_storage_runtime_attach_detach.html">[9]</a>。异构 TP 场景（prefill tp=4, decode tp=8）需要通过 `--hicache-storage-backend-extra-config '{"tp_lcm_size": 8}'` 设置 TP 的最小公倍数，让 head 分片在跨集群间可对齐。

### 2.6 SWA 与 Mixed Attention 模型

`--swa-full-tokens-ratio` 控制 sliding window attention（SWA）层 KV cache 占全 attention 层的比例，默认 0.8。这意味着每层 SWA 的 KV 分配是全 attention 层的 80%，用于减少非核心层对显存的占用。`--disable-hybrid-swa-memory` 关闭混合 SWA 内存池，分开分配独立池则更易于调试但内存利用率更低。

### 2.7 其他内存管理

`--enable-memory-saver` 启用 `TorchMemorySaverAdapter`，通过减少 PyTorch 分配器碎片降低 OOM 风险。`--cpu-offload-gb` 控制 CPU 卸载量，`--offload-group-size` 控制卸载的 layer 组粒度。`--enable-symm-mem` 启用 NCCL 对称内存用于快速集合通信，配合 `SGLANG_SYMM_MEM_PREALLOC_GB_SIZE` 指定预分配大小。

## 三、调度器策略与并行配置

### 3.1 调度核心参数

`max_prefill_tokens`（默认 16384）和 `chunked_prefill_size` 是 SGLang 调度器的两个核心约束。`max_prefill_tokens` 限定单次 prefill batch 的总 token 上限，`chunked_prefill_size` 将长 prompt 拆成多个 chunk 迭代。`chunked_prefill_size = -1` 关闭分块，整个 context 一次处理。自动计算值由 GPU 显存阶梯决定（见 4.2 节）。

`--schedule-conservativeness` 是 SGLang 调度器独有的调参杠杆，默认 1.0。本质是一个乘法因子作用于 `new_token_ratio`——这是调度器估算每个运行中请求还需要生成多少 token 的比例。内置衰减机制：初始值 `0.7 × schedule_conservativeness`，每步衰减到最小值 `0.098 × schedule_conservativeness`<a href="https://docs.sglang.ai/advanced_features/hyperparameter_tuning.html">[2]</a>。

调参口诀直接：日志中 `token usage < 0.9` 且 `#queue-req > 0` → 调度器太保守，降到 0.3；日志中频繁出现 `KV cache pool is full. Retract requests.` → 调度器太激进，升到 1.3。偶尔 retract（约 1 次/分钟）是正常的平衡态。

`--schedule-policy` 提供七种策略：

- **fcfs**（默认）：按到达顺序，适配最广泛场景
- **lof**（Longest Output First）：优先处理 `max_new_tokens` 最大的请求，适合离线批处理
- **lpm**（Longest Prefix Match）：重组请求最大化前缀复用，适合固定 system prompt 场景
- **dfs-weight**：基于 radix tree 深度优先搜索加权，LPM 的高阶替代
- **random**：随机化顺序
- **priority**：按请求优先级排序
- **routing-key**：优先匹配运行中 batch 的路由键

`--enable-priority-scheduling` 启用后，高优先级请求可以抢占低优先级的运行中请求——当优先级差值超过 `--priority-scheduling-preemption-threshold`（默认 10）且 KV cache 空间不足时触发。目前仅支持 fcfs 和 lof 策略组合。

### 3.2 Chunked Prefill

`--chunked-prefill-size` 启用后，长 prompt 被拆成多个 chunk 分别做 prefill——消除一个长 prompt 阻塞全部 decode 请求的队头阻塞。`--enable-mixed-chunk` 进一步允许解码和 chunked prefill 在同一个 batch 里混合执行，提升 batch 密度。对于 pipeline parallel（PP>1）场景，`--enable-dynamic-chunking` 动态预测 chunk 大小来减少 pipeline bubble，平滑因子由 `SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR`（默认 0.75）控制。

Chunked prefill 有几个自动行为值得关注：DP attention 模式下 `chunked_prefill_size` 自动除以 `dp_size`（避免 MoE kernel 问题）；多模态 + transformer backend 自动关闭；context parallel 启用时每个 prefill batch 只允许 1 个请求。

### 3.3 并行策略

SGLang 支持五种并行维度同时叠加：

| 策略 | 参数 | 推荐场景 |
|------|------|---------|
| Tensor Parallel | `--tp-size` | 模型超过单 GPU 显存时必用 |
| Pipeline Parallel | `--pp-size` | 超大模型 TP 不够用 + 长上下文场景 |
| Expert Parallel | `--ep-size` | MoE 专家分发（DeepSeek TP=8 推荐 EP=8） |
| Data Parallel | `--dp-size` | 多实例吞吐（推荐用 sgl-router 而非框架 DP） |
| Attention CP | `--attn-cp-size` | 超长序列 prefill（DeepSeek V3.2 专有） |

**Data Parallelism Attention (DPA)**<a href="https://docs.sglang.ai/advanced_features/dp_dpa_smg_guide.html">[10]</a> 是 SGLang 独有的并行策略：`--enable-dp-attention --dp-size 8` 在 attention 层使用 data parallelism（每个 DP rank 处理不同请求的 attention），在 FFN/MLP 层使用 tensor parallelism（复制所有请求）。这种混合策略对 DeepSeek 这类 MLA 模型特别有效——attention 计算量小而频繁，DP 减少同步开销；MoE 层计算量大，TP 确保被充分利用。开启 DPA 后 `schedule_conservativeness` 自动 × 0.3（因为实际并发等于 dp_size × 可见并发）。

**Pipeline Parallelism**<a href="https://docs.sglang.ai/advanced_features/pipeline_parallelism.html">[11]</a>：`--pp-size` 设置 pipeline stage 数。layer 分布默认自动均分，用 `SGLANG_PP_LAYER_PARTITION=15,15,15,16` 可手动指定每节点的 layer 数。`--pp-max-micro-batch-size` 自动设为 `max_running_requests // pp_size`。

**Expert Parallelism**<a href="https://docs.sglang.ai/advanced_features/expert_parallelism.html">[12]</a>：`--ep-size` + `--moe-a2a-backend` 控制 MoE 模型专家分发。A2A 后端选择直接影响性能档次——DeepEP（`deepep`）默认自动模式在 prefill 用 `normal`、decode 用 `low_latency`，是生产环境最成熟的方案。Mooncake、MORI、FlashInfer A2A 等均有独立性能特征。`--enable-eplb` 启用专家负载均衡，配合 `--eplb-rebalance-num-iterations`（默认 1000）自动调优。

### 3.4 CUDA Graph

SGLang 通过 `--cuda-graph-max-bs` 控制 CUDA graph 的最大 batch size（自动根据 GPU 显存阶梯设定，从 8 到 512）。`--cuda-graph-bs` 可以显式指定 batch size 列表。`--disable-cuda-graph-padding` 时生成连续的 `[1, 2, 3, ..., max_bs]`，无 padding 浪费但 graph 数量多、显存开销大。默认配置下 `[1,2,4,8,12,16,24,...,max_bs]` 的阶梯列表在 256+BS 时步进为 32，在显存效率和延迟覆盖之间折中。

Piecewise CUDA graph 是 SGLang 对 chunked prefill 的专有优化：将 prefill chunk 分成独立部分独立录制 CUDA graph，遇到超出录制范围的 token 数时才退出 graph。`--enable-breakable-cuda-graph` 启用，`--piecewise-cuda-graph-max-tokens` 控制每段大小（MLA 默认 2048，否则等于 `chunked_prefill_size`）。

## 四、硬件适配与性能参考

### 4.1 Attention Backend 选择

SGLang 的 attention backend 自动选择基于 GPU 架构 + CUDA 版本 + 模型架构的综合判断<a href="https://docs.sglang.ai/advanced_features/attention_backend.html">[13]</a>：Hopper MHA → `fa3`；Hopper MLA → `fa3`；Blackwell MHA → `trtllm_mha`；Blackwell MLA → `flashinfer`；AMD/HIP → `aiter`；FlashInfer 可用（无 attention sink）→ `flashinfer`；最终回退 → `triton`。

`--prefill-attention-backend` 和 `--decode-attention-backend` 可以分别为 prefill 和 decode 指定不同的 backend。一个常见的优化是 prefill 用 `flashinfer`（ragged prefill 优势大）、decode 用 `fa3`（decode 延迟低）。DeepSeek NSA 模型还需额外指定 `--nsa-prefill-backend` 和 `--nsa-decode-backend`。

### 4.2 GPU 阶梯自动配置

SGLang 根据 GPU 显存阶梯自动设定 `chunked_prefill_size` 和 `cuda_graph_max_bs`：

| GPU 显存 | chunked_prefill_size | cuda_graph_max_bs (tp<4) | cuda_graph_max_bs (tp≥4) |
|----------|---------------------|-------------------------|--------------------------|
| < 10 GB | 2048 | 8 | — |
| 10–35 GB | 2048 | 24 | 80 |
| 35–60 GB | 4096 | 32 | 160 |
| 60–90 GB | 8192 | 256 | 512 |
| 90–140 GB | 8192 | 256 | 512 |
| > 140 GB | 16384 | 512 | 512 |

自动配置对大多数非极端场景足够好，但在两类情况下需要手动干预：显存极其紧张时下调 `chunked_prefill_size` 和 `cuda_graph_max_bs` 留出更多 KV cache 空间；显存有大量余量时上调 `cuda_graph_max_bs` 提高大 batch 时的 GPU 利用率。

### 4.3 量化建议

H200（HBM3e）路径：FP8 KV cache + BF16 权重 → 模型权重 FP8 量化 → 进阶到 FP4 KV cache。B200（Blackwell HBM3e）：FP4 权重（NVFP4）+ FP8 KV cache → FP4 KV cache（实验性）。L40S：由于显存带宽低，INT4/FP8 权重量化几乎是必须的，KV cache 至少用 FP8 E4M3。

DeepSeek-R1-0528（TP=8, EP=8）在 FP8 KV cache 下实测吞吐约 11,000+ tok/s（ISL/OSL 1K/2K），FP4 KV cache 下 `gsm8k` 精度 0.9124 vs BF16 的 0.9157<a href="https://docs.sglang.ai/advanced_features/quantized_kv_cache.html">[8]</a>。Qwen3-235B-A22B 在 FP4 KV cache 下 `gsm8k` 甚至略高于 FP8（0.9186 vs 0.9181）。

## 五、部署监控

`--enable-metrics` 启用后，SGLang 在 `/metrics` 端点自动暴露 Prometheus metrics。关键指标按用途分为三组：

**KV cache 利用率**：调度器日志中的 `token usage` 直接反映 KV cache pool 填充率。`>0.9` 健康；持续 `<0.9` 且 `#queue-req > 0` 说明调度器过于保守，降低 `--schedule-conservativeness`。`available_gpu_mem` 的建议范围是 5–8 GB——太高则上调 `--mem-fraction-static`。

**前缀复用**：`--enable-cache-report` 启用后，`/cache-report` 端点输出 `PrefixCacheStats`，包括当前缓存条目数、总 token 数、命中率等。对于固定 system prompt 场景，命中率应稳定 >70%。调度器日志中通过 `token usage` 和 `#running-req` 的比率可间接推断——若 `#queue-req` 低但 `#running-req` 也低，很可能是前缀复用不足导致每个请求都独占大片 KV 空间。

**硬件异常**：`--watchdog-timeout`（默认 300 秒）是生产部署的安全阀——单次 forward 超过此时间，server 主动 crash 防止 hang。`--soft-watchdog-timeout` 不 crash 而是 dump 诊断信息。`--crash-dump-folder` 指定 crash dump 路径。`--log-requests`（默认关闭，level 1–3）在怀疑有请求异常时打开，按请求 ID 追踪全生命周期。

`--enable-metrics-for-all-schedulers` 让所有 TP rank（不只 TP 0）都独立输出指标，在 DPA 场景下尤其有用——否则所有请求的 metrics 都从 TP 0 报告，无法观测不同 DP rank 的负载不均。`--enable-mfu-metrics` 启用 MFU（Model FLOPs Utilization）估算，帮助判断当前配置是否在逼近硬件计算上限。

## 六、常见生产陷阱

以下基于源码分析和社区讨论<a href="https://github.com/sgl-project/sglang/issues?q=is%3Aissue+kv+cache+tuning">[14]</a> 总结的高频问题：

**`mem_fraction_static` 设太高导致 OOM**。尤其在 MoE 模型 + DP attention + CUDA graph 叠加时，实际显存需求可能比默认 reserved_mem 估算高 10–15%。解法：从 0.80 开始逐步上调，确认稳定后再推向 0.88 或更高。

**FP8 KV cache 未提供 scaling factors 导致精度异常**。FP8 KV cache 没有正确的 scaling factor 时默认用 1.0，在长 context 下 kv 值域偏移会导致生成质量下降。解法：使用 ModelOpt 预量化的 checkpoint（`k_scale`/`v_scale` 已内置），或通过 `--quantization-param-path` 提供 JSON 文件。

**`chunked_prefill_size = -1` 在高并发时导致队头阻塞**。一个 128K 的长 prompt 若在一次 iteration 中完整 prefill，会阻塞所有 decode 请求 10 秒以上。几乎总是应该启用 chunked prefill——仅 embedding 模式是合理禁用场景。

**Schedule conservativeness 在不同 workload 下需要完全不同的值**。同一个模型，system prompt 固定的 chatbot 场景需要 `schedule_conservativeness=0.3` 甚至更低；API 调用场景 `max_new_tokens` 方差大则需要 1.0；离线批处理用 1.0–1.3。用一个值服务所有场景是极常见的错误。

**DP attention 未配合 schedule_conservativeness 调整**。开启 DPA 后 `schedule_conservativeness` 自动 × 0.3，但如果后续手动覆盖了该值，容易导致调度器对并发容量估计严重偏差。

**HiCache `page_first` layout 用了 `direct` IO backend**。`page_first` 只兼容 `kernel` IO backend，用 `direct` 会自动降级为 `layer_first`，丢掉 zero-copy 优势。正确做法是 `page_first` 配 `kernel`，或改为 `page_first_direct` 配 `direct`。

**Attention backend 不支持 KV cache dtype**。`fa3` + `fp8_e5m2` 会自动降级为 `triton`（性能低一个数量级），`trtllm_mla` 只支持 `fp8_e4m3`、`fp4_e2m1`、`bf16`。启用量化 KV cache 前务必确认 attention backend 兼容性。

---

## 参考资料

[1] [SGLang GitHub Repository](https://github.com/sgl-project/sglang)

[2] [SGLang Hyperparameter Tuning Guide](https://docs.sglang.ai/advanced_features/hyperparameter_tuning.html)

[3] [SGLang HiCache Best Practices](https://docs.sglang.ai/advanced_features/hicache_best_practices.html)

[4] [SGLang Engine API](https://docs.sglang.ai/backend/engine_api.html)

[5] [SGLang Router (Model Gateway)](https://github.com/sgl-project/sglang/tree/main/sgl-router)

[6] [SGLang PD Disaggregation](https://docs.sglang.ai/advanced_features/pd_disaggregation.html)

[7] [SGLang Server Arguments Reference](https://docs.sglang.ai/advanced_features/server_arguments.html)

[8] [SGLang Quantized KV Cache](https://docs.sglang.ai/advanced_features/quantized_kv_cache.html)

[9] [SGLang Runtime Attach/Detach HiCache Storage Backend](https://docs.sglang.ai/advanced_features/hicache_storage_runtime_attach_detach.html)

[10] [SGLang DP/DPA/DP Router Guide](https://docs.sglang.ai/advanced_features/dp_dpa_smg_guide.html)

[11] [SGLang Pipeline Parallelism](https://docs.sglang.ai/advanced_features/pipeline_parallelism.html)

[12] [SGLang Expert Parallelism](https://docs.sglang.ai/advanced_features/expert_parallelism.html)

[13] [SGLang Attention Backend](https://docs.sglang.ai/advanced_features/attention_backend.html)

[14] [SGLang GitHub Issues: KV Cache Tuning](https://github.com/sgl-project/sglang/issues?q=is%3Aissue+kv+cache+tuning)

### 版本对齐信息

| 依赖 | 版本/Commit | 日期 |
|------|-----------|------|
| SGLang | `50f405816e` (v0.5.11) | 2026-05-14 |
