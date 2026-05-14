---
title: TensorRT-LLM 部署方式、KV Cache 调参与最佳实践
date: 2026-05-14 12:00:00 +0800
author: MiracleFarms
kind: field-note
category: Field Note
intro: 基于 TRT-LLM v1.3.0rc14 源码阅读与 NVIDIA 官方部署指南，梳理部署方式全景、KV Cache 全部可调参数及其调优空间、调度与并行策略、硬件适配数据和常见生产陷阱。
tags: [TRT-LLM, KV Cache, Inference]
---

在生产环境部署 TRT-LLM 时，最重要的"一次性决策"是选对部署模式，最重要的"持续调优"是吃透 KV cache 参数的层级关系。这两个维度出现偏差的代价差异很大——选错部署模式会让整个 serving 链路推倒重来，KV cache 配置不当则会让同一台机器少跑 30% 以上的吞吐。

本文基于 TRT-LLM v1.3.0rc14（commit `95204b7802`，2026-05-14）<a href="https://github.com/NVIDIA/TensorRT-LLM">[1]</a> 的源码阅读，结合 NVIDIA 官方性能指南和 DeepSeek-R1 部署最佳实践<a href="https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html">[2]</a>，把部署链路和参数调优空间一次性梳理清楚。

## 一、部署方式全景

TRT-LLM 从 Python API 到全分布式 disaggregated serving 共覆盖七种部署模式，每种对应不同的生产阶段和运维复杂度。

最轻量的入口是 **Python LLM API**<a href="https://nvidia.github.io/TensorRT-LLM/llm-api/">[3]</a>：

```python
from tensorrt_llm import LLM
llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", backend="pytorch")
output = llm.generate(["What is TensorRT-LLM?"])
```

这层 API 封装了模型加载、引擎构建（TRT backend）或 PyTorch 直接加载（pytorch backend）、executor 初始化的全过程，适合离线推理、评测脚本和快速原型。但它不走网络协议，不能直接做在线 serving。

在线 serving 的主线是 `trtllm-serve`<a href="https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html">[4]</a>，一行命令拉起 OpenAI 兼容的 HTTP 服务：

```bash
trtllm-serve nvidia/Llama-3.1-8B-Instruct-FP8 \
    --backend pytorch \
    --tp_size 1 \
    --max_batch_size 3840 \
    --max_num_tokens 7680
```

背后启动 FastAPI 服务，暴露 `/v1/chat/completions`、`/v1/completions`、`/v1/models` 标准端点，以及 `/health`、`/metrics`、`/stats`、`/iteration_stats` 等运维端点。加 `--grpc` 切换为 gRPC 协议，用于 sgl-router 等外部路由器的 token 级高性能转发。

`trtllm-serve` 同时支持三种 backend：`pytorch`（默认，基于 `_TorchLLM` 和 PyExecutor）、`tensorrt`（基于 TRT engine 的 `_TrtLLM`）、`_autodeploy`（自动部署实验性后端）。选 backend 的核心权衡是：PyTorch backend 启动快、迭代灵活，适合模型还在频繁切换的阶段；TensorRT backend 构建慢但推理性能上限更高，适合模型固定后的长期 serving。

多机多卡场景下，`trtllm-llmapi-launch` 通过 MPI 把 `trtllm-serve` 分发到集群节点，配合 Slurm 做调度。Triton Inference Server 集成位于 `triton_backend/` 目录，适合已有 Triton 基础设施的团队，但维护成本更高——NVIDIA 目前的主推路径是 `trtllm-serve`。

Disaggregated serving（P/D 分离）通过 `trtllm-serve disaggregated` 子命令实现，将 prefill 和 decode 分配到不同 GPU 组，消除 prefill 突发对 decode 延迟的干扰。配合 `CacheTransceiverConfig` 可选用 UCX、NIXL、Mooncake 或 MPI 作为 KV cache 跨节点传输后端。

最后是 **Ray Executor**，通过 `orchestrator_type="ray"` 启用，将 TRT-LLM 嵌入 Ray 集群做容错调度，适合多模型混部和弹性伸缩场景，但有额外的序列化开销。

**选择原则**：离线评测用 Python API，在线 serving 首选 `trtllm-serve`（pytorch backend），追求极致吞吐且模型稳定时切 TensorRT backend，多机 disaggregated 场景用 `trtllm-serve disaggregated`，已有 Triton 基建设施则继续用 Triton backend。

## 二、KV Cache 参数体系

TRT-LLM 的 KV cache 配置集中在 `KvCacheConfig` 类（`tensorrt_llm/llmapi/llm_args.py:2505`），共 20+ 个可调参数。这些参数构成了一个层级化控制体系：从最粗粒度的显存上限，到中等粒度的复用策略，再到逐 token 的注意力窗口。

### 2.1 容量控制：三条上限规则

GPU KV cache 的可用空间由三条规则共同决定，取最小值：`free_gpu_memory_fraction` × (空闲 GPU 显存)、`max_tokens`、`max_gpu_total_bytes`。

`free_gpu_memory_fraction` 默认 0.9，即在模型权重和运行时 buffer 之外，将剩余显存的 90% 分配给 KV cache。这个值是生产中最常调的参数之一：DeepSeek-R1-0528 在 B200 上推荐 0.95（吞吐优先）<a href="https://github.com/nvidia/TensorRT-LLM/blob/main/docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md">[5]</a>，但若 OOM 则退到 0.85（B200）或 0.80（H200）。8-GPU 部署的 MoE 模型通常要把 fraction 压到 0.70–0.80，因为专家参数和 CUDA graph padding 一起占用大量显存。

`max_tokens` 提供硬上限，以 token 数为单位直接控制 KV cache 容量。当你知道精确的并发需求时（如"最多同时服务 64 个 8K 上下文请求"），用 `max_tokens` 比用 fraction 更可预测。不设时默认由 fraction 决定。

### 2.2 块大小与复用策略

`tokens_per_block` 默认 32，是 KV cache 的物理分配粒度。较小的块（如 16）提高前缀复用的命中率——一个 1024 token 前缀如果恰好差几个 token 不够对齐 32 token block，最后一个 block 无法复用；块更小则对齐损失更小。但更小的块意味着更多的 block 元数据开销和更频繁的 radix tree 操作。32 是经过大量 benchmark 后的平衡值，一般不需要改。

三个布尔开关控制复用行为：`enable_block_reuse`（默认 true）是总开关，关闭后整个前缀复用系统不工作；`enable_partial_reuse`（默认 true）允许一个 block 内部 token 序列部分匹配时复用；`copy_on_partial_reuse`（默认 true）在部分匹配的 block 正被其他请求使用时，通过拷贝方式让新请求也能复用——代价是额外的显存拷贝，收益是更高的复用率。

多模态场景（图片/视频输入）需要 `enable_block_reuse: false`，因为 KV cache 复用系统尚未兼容多模态内容哈希。

### 2.3 注意力窗口控制

`max_attention_window` 是 per-layer 的滑动窗口大小列表。长度小于 layer 数时自动重复填充。典型用法：

```yaml
kv_cache_config:
  max_attention_window: [4096, 256]  # 交替：全注意力层 + SWA 层
```

这会交替设置 attention window——奇数层使用 4096 token 的全注意力窗口，偶数层使用 256 token 的滑动窗口，大幅减少 KV cache 显存占用。对于 Llama-4-Scout 这类混合注意力模型，这是实验性的调优方向。

`sink_token_length` 指定始终保留在 attention window 中的起始 token 数（sink tokens），用于需要"锚定"前缀的 attention 模式。

### 2.4 主机内存卸载（Host Offload）

`host_cache_size` 是 offload 的总开关。设置为 >0（单位字节）即启用：当 GPU 显存紧张时，复用率低的 block 被拷贝到 pinned CPU memory，需要时再加载回来<a href="https://nvidia.github.io/TensorRT-LLM/features/kvcache.html">[6]</a>。这在 GH200（GPU-CPU 高带宽）上收益最大，x86 + Hopper 仍有净收益，但 PCIe 带宽有限的旧架构可能得不偿失。

pinned memory 分配是一次性开销——45 GiB 的 host cache 在 x86 机器上可能耗时 10+ 秒。`secondary_offload_min_priority`（默认 35）控制只有优先级高于此阈值的 block 才被 offload，低优先级的直接丢弃，以控制 PCIe 流量。

实验性的 KV Cache Manager V2（`use_kv_cache_manager_v2: true`）采用 suspend/resume 机制，自动管理多层存储（GPU/HOST/DISK），无需手动设置 `host_cache_size`。

### 2.5 数据类型

`dtype` 控制 KV cache 存储精度，支持 `auto`（跟随模型配置）、`fp8`、`nvfp4`，以及任意 `torch.dtype` 字符串（如 `float16`、`bfloat16`）。DeepSeek-R1-0528 在 B200 上默认启用 FP8 KV cache，相比 BF16 有"无明显精度损失"的速度提升<a href="https://github.com/nvidia/TensorRT-LLM/blob/main/docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md">[5]</a>。Blackwell GPU 上可进一步使用 `nvfp4` 与 FP4 模型权重保持一致，进一步降低显存压力。

### 2.6 可观测性参数

`event_buffer_max_size`（默认 0，即关闭）启用后，`get_kv_cache_events()` API 可追踪 block 的创建、存储、移除和更新事件，是排查前缀复用异常的利器。`iteration_stats_interval`（默认 1）控制 KV cache 迭代统计的采样频率。

## 三、调度器策略与并行配置

### 3.1 批处理核心参数

`max_batch_size` 和 `max_num_tokens` 是 IFB（In-Flight Batching）调度器的两个核心约束。`max_num_tokens` 限制每次迭代的总 token 数（context + generation），默认 8192。调优方向明确：更大的 `max_num_tokens` 提高 GPU 利用率但增加 TTFT；更小则降低延迟但牺牲吞吐。

DeepSeek-R1 在不同硬件上的推荐配置差异很大：B200 最大吞吐场景用 `max_batch_size=896, max_num_tokens=2048`（FP8 KV cache），H200 用 `max_batch_size=128, max_num_tokens=1151`<a href="https://github.com/nvidia/TensorRT-LLM/blob/main/docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md">[5]</a>。差距来自 B200 更强的计算和内存带宽——更大的 batch 能填满更多 SM 而不会让延迟失控。

### 3.2 Chunked Prefill

`enable_chunked_prefill` 是一个几乎应该永远开启的开关。它将长 prompt 拆成多个迭代分别计算，消除"一个大 prompt 阻塞所有 decode 请求"的队头阻塞问题，同时允许更小的 `max_num_tokens` 设置以降低单次迭代延迟。

### 3.3 容量调度策略

`capacity_scheduler_policy` 提供三种模式：`GUARANTEED_NO_EVICT`（默认，永远不会驱逐已接受的请求——宁可拒绝新请求）、`MAX_UTILIZATION`（允许在请求间做重调度以最大化利用率）、`STATIC_BATCH`（固定批次大小，不做动态调整）。

`context_chunking_policy` 控制 chunked prefill 的公平性：`FIRST_COME_FIRST_SERVED` 按到达顺序分配 context 预算，`EQUAL_PROGRESS` 在各活跃请求间均分。`DynamicBatchConfig` 提供运行时的动态批大小调整（仅 TensorRT backend），默认开启 batch size 调优但关闭 `max_num_tokens` 调优。

### 3.4 并行策略

TRT-LLM 支持五种并行维度同时叠加：

| 策略 | 参数 | 推荐场景 |
|------|------|---------|
| Tensor Parallel | `tensor_parallel_size` | 模型超过单 GPU 显存时必用 |
| Pipeline Parallel | `pipeline_parallel_size` | 超大模型（如 405B+）放不下 TP 时叠加 |
| Expert Parallel | `moe_expert_parallel_size` | MoE 模型专家分发（DeepSeek-R1 推荐 EP=8） |
| Context Parallel | `context_parallel_size` | 超长上下文（>128K）序列拆分 |
| Attention DP | `enable_attention_dp` | 高吞吐场景 KV cache 按 DP rank 分区 |

对于 MoE 模型，`moe_tp_size × moe_ep_size` 必须等于总 `tensor_parallel_size`。以 8×B200 部署 DeepSeek-R1 为例：TP=8、EP=8 是最大吞吐配置（43,146 tok/s，ISL=1K/OSL=2K）；EP=2 是最低延迟配置（274 tok/s/user 单用户）<a href="https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html">[2]</a>。

### 3.5 CUDA Graph 调优

TRT-LLM 用 CUDA graph 将一系列 kernel launch 预录制为单次 dispatch，减少 CPU launch overhead。关键参数是 `cuda_graph_config.batch_sizes`（预录制的 batch size 列表）。2026 年 4 月起，默认配置从稀疏的 `{256, 512, 1024, 2048}`（23 个 graph）改为 64 步进列表 `{192, 256, ..., 2048}`（49 个 graph），聚合模式吞吐提升 1.3x，disagg 模式提升 1.5x<a href="https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog20_Tuning_CUDA_Graph_Batch_Sizes_for_Higher_Output_Throughput.html">[7]</a>。代价是额外 260 MB 显存，对于显存紧张的部署可以显式回退到旧配置。

## 四、硬件适配与性能参考

不同 GPU 架构对 TRT-LLM 配置的敏感度差异显著，以下是官方 benchmark 的关键数据：

**H200（Hopper）**<a href="https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html">[2]</a>：
- Llama 3.1 8B FP8, TP=1：27,027 tok/s（ISL/OSL 128/128）
- Llama 3.3 70B FP8, TP=2：7,467 tok/s（128/2048）
- DeepSeek-R1 FP8, TP=8, EP=8：11,489 tok/s（1K/2K）

**B200（Blackwell）**<a href="https://github.com/nvidia/TensorRT-LLM/blob/main/docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md">[5]</a>：
- DeepSeek-R1-0528 FP4, TP=8, EP=8, FP8 KV cache：**43,146 tok/s**（1K/2K）
- 单用户延迟：274 tok/s（EP=2 + MTP speculative decoding）

**GB200（Grace-Blackwell）**<a href="https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html">[2]</a>：
- Llama 3.3 70B FP4, TP=1：11,100 tok/s（128/128）

**H100 vs A100**：TRT-LLM 在 H100 上达到 A100 的 4.6x 吞吐（10,000 tok/s at 100ms TTFT）<a href="https://nvidia.github.io/TensorRT-LLM/blogs/H100vsA100.html">[8]</a>。

量化选择上，H200 优先 FP8，B200 优先 FP4（配合 NVFP4 权重和 FP8 KV cache），L40S 由于显存带宽较低，INT4/FP8 量化几乎是 mandatory 的。

## 五、部署监控

`trtllm-serve` 自动导出 Prometheus metrics 到 `/metrics` 端点。核心监控指标按用途分为三组<a href="https://nvidia.github.io/TensorRT-LLM/features/kvcache.html">[6]</a>：

**KV cache 利用率**：`trtllm_kv_cache_utilization`（GPU 池利用率）、`trtllm_kv_cache_host_utilization`（主机池利用率）。GPU 利用率持续 >95% 意味着 KV cache 是瓶颈，需要增大 `free_gpu_memory_fraction` 或增加 GPU。

**复用效率**：`trtllm_kv_cache_hit_rate`（累计命中率）、`trtllm_kv_cache_iter_reuse_rate`（单次迭代复用率）。对于固定 system prompt 场景，复用率应稳定 >70%；如果低且 `enable_block_reuse` 已开启，检查是否有 cache salt 隔离、多模态请求混入、或 prompt 在请求间有微小差异。

**Offload 开销**：`trtllm_kv_cache_offload_bytes_total` + `trtllm_kv_cache_onboard_bytes_total` 累计传输量。如果 `trtllm_iteration_latency_seconds` 随 offload 量增加而明显上升，说明 PCIe 带宽成为瓶颈，需要提高 `secondary_offload_min_priority` 以减少流量或切换到 GH200 架构。

`/stats` 端点的 `kvCacheIterationStats` 提供更详细的 per-window 数据，包括 `primaryMaxNumBlocks`、`secondaryUsedNumBlocks`、`iterOffloadBytes`、`iterFullReusedBlocks`、`iterCacheHitRate` 等，适合做细粒度性能分析。

## 六、常见生产陷阱

以下是基于源码分析和社区讨论<a href="https://github.com/NVIDIA/TensorRT-LLM/issues?q=is%3Aissue+kv+cache+tuning">[9]</a> 总结的高频问题：

**KV cache fraction 设太高导致 OOM**。尤其在 MoE 模型 + CUDA graph padding 的场景下，实际显存需求可能比估算高 15-20%。解法：从 0.80 开始逐步上调，确认稳定后再推向 0.90。

**`max_num_tokens` 未设置导致请求串行执行**。默认值 8192 对大多数场景足够，但如果同时设置了很小的 `max_tokens`（KV cache 容量以 token 计），会人为制造 bottleneck。

**CUDA graph batch size 太稀疏导致 padding 浪费**。旧的默认配置（256/512/1024/2048）在小批次时 padding 废 token 可达 50%。新版 +64 配置已将 padding 控制在 ≤63 tokens，如果不确定用哪个，直接用新版默认。

**多模态请求未关闭 block reuse**。混合文本和图像/视频请求时，如果未设 `enable_block_reuse: false`，可能导致 KV cache 状态错误。当前版本的解决方式是整体关闭复用——这是功能 gap 而非设计选择。

**KV cache 隔离缺失**。多租户场景下，如果未通过 `cache_salt`（per-request salt string）做租户隔离，不同租户的 system prompt 可能意外共享 KV cache，造成 prompt 泄露。TRT-LLM 在 `BlockKey` 中内建了 `cacheSaltID` 字段，需在 application 层显式传入。

**Offload 在旧架构上得不偿失**。CPU-GPU 链路过慢时，onboard 的开销可能超过重新计算 prefill 的代价。判断标准：监控 `trtllm_iteration_latency_seconds`，如果 onboard 后延迟不降反升，就关掉 offload。

---

## 参考资料

[1] [TensorRT-LLM GitHub Repository](https://github.com/NVIDIA/TensorRT-LLM)

[2] [TRT-LLM Performance Overview](https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html)

[3] [TRT-LLM LLM API Documentation](https://nvidia.github.io/TensorRT-LLM/llm-api/)

[4] [trtllm-serve CLI Reference](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html)

[5] [Best Performance Practice on DeepSeek-R1 in TensorRT-LLM](https://github.com/nvidia/TensorRT-LLM/blob/main/docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md)

[6] [TRT-LLM KV Cache System](https://nvidia.github.io/TensorRT-LLM/features/kvcache.html)

[7] [Tuning CUDA Graph Batch Sizes for Higher Output Throughput](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog20_Tuning_CUDA_Graph_Batch_Sizes_for_Higher_Output_Throughput.html)

[8] [NVIDIA H100 vs A100: TensorRT-LLM Performance](https://nvidia.github.io/TensorRT-LLM/blogs/H100vsA100.html)

[9] [TRT-LLM GitHub Issues: KV Cache Tuning](https://github.com/NVIDIA/TensorRT-LLM/issues?q=is%3Aissue+kv+cache+tuning)

[10] [TRT-LLM Paged Attention + IFB Scheduler](https://nvidia.github.io/TensorRT-LLM/features/paged-attention-ifb-scheduler.html)

[11] [TRT-LLM Parallelism Strategies](https://nvidia.github.io/TensorRT-LLM/features/parallel-strategy.html)

### 版本对齐信息

| 依赖 | 版本/Commit | 日期 |
|------|-----------|------|
| TensorRT-LLM | `95204b7802` (v1.3.0rc14) | 2026-05-14 |
| NVIDIA Performance Overview | main branch docs | 2026-05-14 |
| DeepSeek-R1 Best Practice Guide | main branch docs | 2026-05-14 |
