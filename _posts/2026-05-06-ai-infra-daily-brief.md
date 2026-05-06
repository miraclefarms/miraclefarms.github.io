---
title: AI Infra 早报｜推理投产前的集中加固：TRT-LLM 砍 6s JIT 开销、Mooncake 五连补多节点故障、llama.cpp 算法级优化
date: 2026-05-06 08:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: 推理基础设施正在做投产前的集中加固——TRT-LLM 一波性能 PR 砍掉 FMHA JIT 6 秒开销，Mooncake 连补五个多节点传输故障，llama.cpp 用 FWHT 把 KV rotation 从 O(N²) 降到 O(N log N)。与此同时，LMCache 引入多租户隔离，KV cache 正从单租户工具向共享基础设施演进。
---

![题图](/assets/2026-05-06-ai-infra-daily-brief/cover.png)


今天的信号很明确：推理基础设施正在做**投产前的集中加固**。TRT-LLM 一天内合并六七个性能 PR，砍掉 eager generation 下约 6 秒的 FMHA JIT 重编译开销，并为 Qwen3.5 GDN 层定制 Triton kernel fusion；Mooncake 在 24 小时内修了五个多节点传输引擎故障——RDMA QP 并发销毁崩溃、EFA 因 API 版本过低被强制降级到软件路径、P2P 握手环形死锁；llama.cpp 则直接在算法层动手，用 Fast Walsh-Hadamard Transform 替换 KV cache rotation 的矩阵乘法，复杂度从 O(N²) 降到 O(N log N)。这些都不是探索性工作，而是"让现有路径真正跑稳、跑快"的工程行为。第二个趋势同样值得关注：LMCache 引入了 IsolatedLRU 多租户隔离策略和 HuggingFace Buckets 内置后端，标志着 KV cache 正从单租户优化工具向共享基础设施演进。

## 一、TRT-LLM：推理部署前的最后一轮性能打磨

TRT-LLM 今天合并了一波高密度性能 PR，覆盖面从 eager generation 的 JIT 开销到 MoE cubins 再到逐模型定制 kernel。

最有直接影响的是 **#13505**——eager generation 路径下 FMHA kernel 的选型与 CUDA graph warmup 不对齐，导致每次冷启动多出约 6 秒的 JIT 重编译。修复方案是导出 cubin 并对齐 kernel 选型逻辑，直接砍掉这笔开销，对 TTFT 有立竿见影的改善[[1]](https://github.com/NVIDIA/TensorRT-LLM/pull/13505)。

针对 MoE 模型，**#12440** 更新了 MoE 推理 kernel 的 cubins，影响所有 MoE 模型在 TRT-LLM 上的推理吞吐[[2]](https://github.com/NVIDIA/TensorRT-LLM/pull/12440)。**#12966** 则为 Qwen3.5 的 GDN 层引入三个 Triton kernel——sigmoid 融入 gating kernel、split QKV + transpose 融合、rotary embed split 融合[[3]](https://github.com/NVIDIA/TensorRT-LLM/pull/12966)。逐模型定制 kernel fusion 是投产前的典型操作：通用路径已经够快了，现在要把特定模型的瓶颈也压干净。

此外，**#13748** 优化了 beam search 从 prefill 切换到 decode 的 handoff 开销[[4]](https://github.com/NVIDIA/TensorRT-LLM/pull/13748)；**#13012** 通过四个定向改动减少 decode step 的 Python/C++ API 调用开销，直接贡献 decode throughput[[5]](https://github.com/NVIDIA/TensorRT-LLM/pull/13012)；**#13574** 修复了 piecewise CUDA graph capture 漏掉 `num_tokens` 值的静默失败[[6]](https://github.com/NVIDIA/TensorRT-LLM/pull/13574)。

## 二、Mooncake：从"能跑"到"能在生产拓扑下稳定跑"

Mooncake 在 24 小时内合并了五个以上的传输引擎修复，每一个都是多节点真实部署才会触发的故障。

**#1903** 修复了 RDMA 传输路径下的 use-after-free 崩溃——QP 并发销毁导致 `ibv_post_send` 段错误，生产环境必现[[7]](https://github.com/kvcache-ai/Mooncake/pull/1903)。**#2041** 解决了一个更隐蔽的问题：Mooncake 请求 libfabric API 1.14 时，EFA provider 对低于 1.18 的调用者强制关闭 device RDMA，所有 EFA 代际都降级到软件路径而不报错。升级到 1.18 后硬件 RDMA 才真正启用[[8]](https://github.com/kvcache-ai/Mooncake/pull/2041)。

**#1959** 修复了 P2P 握手场景下的环形死锁——三个 TransferEngine 实例可以形成 T0 等 T1、T1 等 T2、T2 等 T0 的等待环[[9]](https://github.com/kvcache-ai/Mooncake/pull/1959)。**#2035** 修正了 dmabuf 注册的地址边界问题——`cuMemGetHandleForAddressRange` 要求精确的 allocation 边界，而当前路径传入了用户地址导致注册失败[[10]](https://github.com/kvcache-ai/Mooncake/pull/2035)。**#2034** 则确保 dmabuf 注册前 CUDA primary context 已初始化[[11]](https://github.com/kvcache-ai/Mooncake/pull/2034)。

另外，**#2040** 修复了 `extend_group_size_to()` 后 `dist.get_world_size()` 不反映更新值的问题——根因是 `MooncakeBackend` 继承的 `Backend::getSize()` 是非虚方法[[12]](https://github.com/kvcache-ai/Mooncake/pull/2040)。**#2028** 统一了 nvlink 和 ubshmem allocator 的入口，是传输引擎基础设施的整理工作[[13]](https://github.com/kvcache-ai/Mooncake/pull/2028)。

## 三、llama.cpp：边缘推理框架的算法级优化

llama.cpp 今天同时推进了三个维度，最值得关注的是算法层面的改动。

**#22631** 用 **Fast Walsh-Hadamard Transform (FWHT)** 替换了 KV cache rotation 的矩阵乘法实现，复杂度从 O(N²) 降到 O(N log N)。实现上巧妙地利用了 `ggml_unary` 中 `GGML_UNARY_OP_SILU_BACK` 的槽位[[14]](https://github.com/ggml-org/llama.cpp/pull/22631)。对于一个以边缘推理为定位的框架来说，直接在算法层做复杂度降级，比逐 kernel 调参的收益上限更高。

在 kernel 层面，**#22423** 在 CPU 后端新增了 RMS_NORM + MUL 融合 kernel，单 pass 完成计算，避免中间结果的 materialization。这对 Apple Silicon 等内存带宽受限平台有直接收益[[15]](https://github.com/ggml-org/llama.cpp/pull/22423)。

新模型方面，**#22101** 加入了 IBM granite-4.0-1b-speech 语音模型支持，采用 Conformer encoder + QFormer projector 架构[[16]](https://github.com/ggml-org/llama.cpp/pull/22101)。此外 **#22290** 修复了延迟加载 backend 的问题，减少启动时不必要的库加载[[17]](https://github.com/ggml-org/llama.cpp/pull/22290)。

## 四、LMCache：KV cache 从单租户工具走向共享基础设施

LMCache 今天合并的 PR 指向同一个方向——**KV cache 正在变成需要多租户隔离和可插拔存储的基础设施**。

**#3137** 引入了 `IsolatedLRU` 驱逐策略，不同 `cache_salt`（用户）之间互不驱逐，并支持配额配置[[18]](https://github.com/LMCache/LMCache/pull/3137)。**#3060** 将 HuggingFace Buckets 作为内置远程存储后端，降低了对外部 S3/MinIO 的依赖门槛[[19]](https://github.com/LMCache/LMCache/pull/3060)。**#3119** 提取了共享 `RawBlockCore`，在多进程场景下复用 raw-block L2 适配器[[20]](https://github.com/LMCache/LMCache/pull/3119)。

值得注意的是 **#3179**——修复了 Blend KV cache 的 CB 查找正确性问题：`cb_store_final` 从未在指纹表中注册 chunk，导致 CB 命中率恒为 0%。这是一个存在已久的静默 bug[[21]](https://github.com/LMCache/LMCache/pull/3179)。

## 五、Megatron 推理栈持续补齐能力

Megatron 在推理侧继续推进，引入 **vLLM 的 grouped gemm kernel 作为 MoE 推理后端**[[22]](https://github.com/NVIDIA/Megatron-LM/pull/4566)；**#4570** 将 allgatherv 和 MoE 预处理隐藏在 shared expert 计算背后，用通信-计算重叠来隐藏开销[[23]](https://github.com/NVIDIA/Megatron-LM/pull/4570)；**#4306** 将推理上下文的逐请求 bookkeeping 从 GPU 搬到 pinned CPU，引入 `ContextGPUView` 减少 GPU 显存占用[[24]](https://github.com/NVIDIA/Megatron-LM/pull/4306)。这些是 Megatron 从训练框架走向独立推理服务的关键步骤。

此外 **#4609** 修复了 `--overlap-param-gather` + layerwise distributed optimizer 下的梯度损坏 bug[[25]](https://github.com/NVIDIA/Megatron-LM/pull/4609)；**#3656** 将 chunked MLP 扩展到训练阶段[[26]](https://github.com/NVIDIA/Megatron-LM/pull/3656)；**#4587** 为 legacy A2A dispatcher 重新启用 EP sync 并禁用其 decode 阶段 cudagraph[[27]](https://github.com/NVIDIA/Megatron-LM/pull/4587)。

## 今天真正值得记住的判断

推理基础设施正在经历一轮"投产前集中加固"。TRT-LLM 砍 JIT 开销、逐模型定制 kernel，Mooncake 补多节点传输故障，llama.cpp 做 FWHT 算法降级——这些行为有一个共同特征：都不是在探索新方向，而是在让已经选定的路径真正达到生产可用。与此同时，LMCache 的多租户隔离和 HuggingFace 内置后端指向了另一个趋势：KV cache 不再只是 vLLM 的一个内部优化，它正在变成可以被多个推理服务共享、需要租户隔离和可插拔存储的独立基础设施层。

---

## 参考来源

[1] [Drop cubin and eliminate ~6s FMHA JIT recompile in eager generation](https://github.com/NVIDIA/TensorRT-LLM/pull/13505)

[2] [Update TRTLLM MoE cubins](https://github.com/NVIDIA/TensorRT-LLM/pull/12440)

[3] [Fuse GDN elementwise ops and split/transpose kernels](https://github.com/NVIDIA/TensorRT-LLM/pull/12966)

[4] [Reduce beam-search prefill->decode handoff cost](https://github.com/NVIDIA/TensorRT-LLM/pull/13748)

[5] [AutoDeploy: reduce C++ dispatch overhead in decode scheduling loop](https://github.com/NVIDIA/TensorRT-LLM/pull/13012)

[6] [Broader capture of piecewise cudagraph](https://github.com/NVIDIA/TensorRT-LLM/pull/13574)

[7] [rdma: fix use-after-free crash in ibv_post_send](https://github.com/kvcache-ai/Mooncake/pull/1903)

[8] [fix(efa): request libfabric API 1.18 for device RDMA](https://github.com/kvcache-ai/Mooncake/pull/2041)

[9] [Fix possible deadlock in RDMA transport connection setup](https://github.com/kvcache-ai/Mooncake/pull/1959)

[10] [Use allocation base addr for dmabuf-based mem registration](https://github.com/kvcache-ai/Mooncake/pull/2035)

[11] [Init CUDA primary context before dmabuf-based mem registration](https://github.com/kvcache-ai/Mooncake/pull/2034)

[12] [Inherit ProcessGroup to fix dynamic getSize() after extend_group_size_to](https://github.com/kvcache-ai/Mooncake/pull/2040)

[13] [Unify fabric allocator plumbing](https://github.com/kvcache-ai/Mooncake/pull/2028)

[14] [ggml: implement fast walsh-hadamard transform for kv rotation](https://github.com/ggml-org/llama.cpp/pull/22631)

[15] [ggml-cpu: fuse RMS_NORM + MUL on CPU backend](https://github.com/ggml-org/llama.cpp/pull/22423)

[16] [mtmd: add granite-speech support](https://github.com/ggml-org/llama.cpp/pull/22101)

[17] [common: only load backends when required](https://github.com/ggml-org/llama.cpp/pull/22290)

[18] [Add IsolatedLRU eviction policy + per-cache_salt quotas](https://github.com/LMCache/LMCache/pull/3137)

[19] [Add Hugging Face Buckets as a built-in remote storage backend](https://github.com/LMCache/LMCache/pull/3060)

[20] [Add raw_block MP L2 adapter support via shared RawBlockCore](https://github.com/LMCache/LMCache/pull/3119)

[21] [Fix CB lookup correctness, thread safety, and store-complete race](https://github.com/LMCache/LMCache/pull/3179)

[22] [Add vLLM grouped gemm backend for MoE inference](https://github.com/NVIDIA/Megatron-LM/pull/4566)

[23] [Enable shared expert overlap with allgatherv in inference](https://github.com/NVIDIA/Megatron-LM/pull/4570)

[24] [Move inference context bookkeeping to CPU with ContextGPUView](https://github.com/NVIDIA/Megatron-LM/pull/4306)

[25] [Fix gradient corruption with layerwise param all-gather overlap](https://github.com/NVIDIA/Megatron-LM/pull/4609)

[26] [Add logic to enable chunked MLP during training](https://github.com/NVIDIA/Megatron-LM/pull/3656)

[27] [Re-enable EP syncs for legacy A2A dispatcher + simplify ep_sync](https://github.com/NVIDIA/Megatron-LM/pull/4587)