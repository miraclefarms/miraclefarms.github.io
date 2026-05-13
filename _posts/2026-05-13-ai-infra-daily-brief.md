---
title: AI Infra 早报｜KV Cache Offload 跨框架打通，推测解码升格为系统组件
date: 2026-05-13 08:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: vLLM 合入 MooncakeStoreConnector、LMCache 接入 3FS、NIXL 发布 v1.1.0、TRT-LLM 修复 KV 页面泄漏——KV offload 从各家独立实验进入互通协作的生产化阶段；同一窗口内，推测解码多框架同步修 bug，升格为常规维护对象。
tags: [KV-Cache, Speculative-Decoding, Quantization, vLLM, SGLang, TRT-LLM, llama.cpp]
---

![题图](/assets/2026-05-13-ai-infra-daily-brief/cover.jpg)


三天内，vLLM 把 Mooncake 分布式 KV pool 接入通用 KV connector、LMCache 把字节 3FS 文件系统拉进 offload 后端、NIXL 发了 1.1.0 正式版、TRT-LLM 修了 KV 页面泄漏和 token offset 错误——这些看似分散的改动拼在一起指向一件事：KV cache 跨节点、跨层级的 offload 链路正从各家独立实验，变为可互通、可依赖的生产基础设施。同窗口内推测解码密集修 bug 的信号同样值得重视：能被修的东西，说明它已经被认真用起来了。

## 一、KV Cache Offload 链路：从各自实验到协作生产

vLLM 合入 MooncakeStoreConnector[[1]](https://github.com/vllm-project/vllm/pull/40900)，把 Mooncake 分布式 KV pool 首次作为通用 KV connector 对接。这不是沿用 Mooncake 自身 P/D 协议，而是让任何走 vLLM KV connector 的场景都能访问共享 KV pool——扩展的是 offload 接入面，不是单点优化。同窗口内，`wait_for_save=False` 路径下 store 被静默丢弃的可靠性 bug 被修复[[2]](https://github.com/vllm-project/vllm/pull/41945)；hybrid 模型多 KV cache group 新增带语义标签的 metadata 事件[[3]](https://github.com/vllm-project/vllm/pull/40984)，为 LMCache 等外部消费者提供稳定接口；NIXL connector 依赖同步升级到 1.x[[4]](https://github.com/vllm-project/vllm/pull/42364)。从连接器打通、到 bug 修复、到接口协议化、再到上游依赖锁定，一条可生产的 offload 调用链正在闭合。

LMCache 侧也在同步补位：3FS 分布式文件系统后端合入[[5]](https://github.com/LMCache/LMCache/pull/3120)，AI native 高性能文件系统首次进入 offload 存储 tier；PD backend 变为 fully async[[6]](https://github.com/LMCache/LMCache/pull/3038)，`batched_submit_put_task` 不再阻塞 vLLM worker 线程等 RDMA 完成，PD 路径延迟直接改善。TRT-LLM 修了两个正确性问题——V2 delay batching 路径下 context 请求关闭不释放 KV 页面的内存泄漏[[7]](https://github.com/NVIDIA/TensorRT-LLM/pull/13805)，以及分布式 KV 跨节点传输中 token offset 计算有误[[8]](https://github.com/NVIDIA/TensorRT-LLM/pull/13937)。这类补丁不性感，但直接影响长时间运行的稳定性。

**整套链路的关键转变：offload 正从单框架实验选项变为跨框架、跨存储层级的协作基础设施，每个补丁都在消除"能跑"和"可靠跑"之间的缝隙。**

## 二、推测解码：生产 bug 密度说明它已被认真使用

PEagle（Parallel Eagle）接入 vLLM speculators 路径[[9]](https://github.com/vllm-project/vllm/pull/41826)，parallel_drafting 开箱即用；MRV2 的 rejection sampler 切到 synthetic mode 与 MRV1 对齐[[10]](https://github.com/vllm-project/vllm/pull/41035)，消除路径分歧。TRT-LLM 发现 Eagle3 每次 CUDA graph capture 都重新分配 `max_num_tokens × hidden_size × num_capture_layers` 的 hidden_states buffer[[11]](https://github.com/NVIDIA/TensorRT-LLM/pull/13920)，改为预分配复用，warmup 内存开销大幅下降；同时修了 Blackwell SM120 上 allReduce cluster launch 未生效导致 Eagle3 四卡测试不稳的问题[[12]](https://github.com/NVIDIA/TensorRT-LLM/pull/13169)。

SGLang 修了 Eagle draft decode 步骤的 RoPE position IDs 计算错误[[13]](https://github.com/sgl-project/sglang/pull/25015)——`positions.add_(1)` 在 forward 前执行导致 position 偏移；STANDALONE draft worker 不需要 hidden_states 却仍在 capture 和 copy[[14]](https://github.com/sgl-project/sglang/pull/25037)，现在 optional schema 让整条路径跳过。llama.cpp 则合入 parallel drafting[[15]](https://github.com/ggml-org/llama.cpp/pull/22838)，draft context 可为多 sequence 并行生成推测。

**四个框架在同一窗口内修 Eagle 系列的生产 bug——这种修复密度本身就说明推测解码已从探索特性转为需要常规维护的系统组件。**

## 三、MXFP4 / NVFP4 量化跨后端对齐

vLLM 合入 MXFP4 linear layers 对接 compressed-tensors 量化路径[[16]](https://github.com/vllm-project/vllm/pull/41664)；W8W8 group quant kernel 用 2D grid 消除 divmod 开销[[17]](https://github.com/vllm-project/vllm/pull/42153)。llama.cpp WebGPU backend 首次支持 MXFP4 MUL_MAT[[18]](https://github.com/ggml-org/llama.cpp/pull/22906)，gpt-oss-20b 成为可运行目标。SGLang 释放 NVFP4 权重处理后不再读取的 source scale tensors[[19]](https://github.com/sgl-project/sglang/pull/25107)，直接降低显存占用。TRT-LLM 修了 MoE FP8+FP4 混合精度 autotune 性能问题[[20]](https://github.com/NVIDIA/TensorRT-LLM/pull/13971)。

**同一个量化格式在 CUDA 服务框架和 WebGPU edge runtime 同步落地，MXFP4 已越过 Blackwell 专属特性的阶段。**

## 四、vLLM MoE 重构：EP 大改动的地基工程

五个 MoE refactor PR 集中合入：`eplb_state | None` 替换 `enable_eplb` flag[[21]](https://github.com/vllm-project/vllm/pull/41055)；RoutedExperts alias 引入并移除 SharedExperts[[22]](https://github.com/vllm-project/vllm/pull/40735)；expert map 逻辑独立为 ExpertMapManager[[23]](https://github.com/vllm-project/vllm/pull/41046)；experts 类收口到独立目录[[24]](https://github.com/vllm-project/vllm/pull/42334)；sequence parallel 测试覆盖[[25]](https://github.com/vllm-project/vllm/pull/41299)。这些全部指向 PR #38590——预计是大规模 Expert Parallelism 重构。目录拆分、API 清理、测试覆盖同步推进，EP 方向的架构变动已进入不可逆阶段。

## 五、llama.cpp 边缘推理：三个异构后端同期提速

Hexagon HVX 用 splat helper 消除 scalar VTCM load[[26]](https://github.com/ggml-org/llama.cpp/pull/22993)，matmul 和 flash attention 均受益；Adreno OpenCL 新增 xmem F16xF32 GEMM prefill 路径[[27]](https://github.com/ggml-org/llama.cpp/pull/22755)；Metal 把 batch divisor 升为 function constant[[28]](https://github.com/ggml-org/llama.cpp/pull/22711)，减少每个 dispatch 开销。此外，CUDA provider 内部 AllReduce kernel[[29]](https://github.com/ggml-org/llama.cpp/pull/22299) 让 tensor parallelism 在无 NCCL 的 Windows 环境下性能大幅改善；Vulkan FA 支持不对称 K/V 类型[[30]](https://github.com/ggml-org/llama.cpp/pull/22589)。五个 release（b9113–b9128）密集发出，edge/mobile 推理已从"能跑"进入"要跑快"的阶段。

## 今天真正值得记住的判断

KV cache offload 链路在同一窗口内跨四个项目闭合了从连接器到存储后端到正确性修复的拼图——这种跨框架同步并不是协调出来的，而是各家的需求独立收敛到了同一个方向：推理部署对 KV cache 的生命周期管理已经不能用单机内存一揽子解决，分层 offload 是必然路径，区别只在于谁先跑通。同周期推测解码的 bug 修复密度，则说明另一个趋势已经不动声色地发生了：能被修的东西，才是真正在被用的东西。

---

## 参考来源

[1] [vLLM: Add MooncakeStoreConnector for KV cache offloading](https://github.com/vllm-project/vllm/pull/40900)

[2] [vLLM: Fix store deferral in kv_offload](https://github.com/vllm-project/vllm/pull/41945)

[3] [vLLM: feat(kv-events): emit KV cache metadata](https://github.com/vllm-project/vllm/pull/40984)

[4] [vLLM: [PD] Bump NIXL connector dependency to 1.x](https://github.com/vllm-project/vllm/pull/42364)

[5] [LMCache: Support 3FS storage backend](https://github.com/LMCache/LMCache/pull/3120)

[6] [LMCache: fully async PD backend](https://github.com/LMCache/LMCache/pull/3038)

[7] [TRT-LLM: Release deferred ctx KV pages in V2 delay batching](https://github.com/NVIDIA/TensorRT-LLM/pull/13805)

[8] [TRT-LLM: Decouple cached prefix from KVSlice token_range](https://github.com/NVIDIA/TensorRT-LLM/pull/13937)

[9] [vLLM: Added peagle speculators support](https://github.com/vllm-project/vllm/pull/41826)

[10] [vLLM: Apply synthetic mode to probabilistic rejection sampler](https://github.com/vllm-project/vllm/pull/41035)

[11] [TRT-LLM: Reuse hidden_states buffer across CUDA graph captures in Eagle3](https://github.com/NVIDIA/TensorRT-LLM/pull/13920)

[12] [TRT-LLM: Fix cluster launch enablement for SM120 GPUs](https://github.com/NVIDIA/TensorRT-LLM/pull/13169)

[13] [SGLang: Fix Eagle draft decode positions](https://github.com/sgl-project/sglang/pull/25015)

[14] [SGLang: spec: STANDALONE skips hidden_states end-to-end](https://github.com/sgl-project/sglang/pull/25037)

[15] [llama.cpp: spec: parallel drafting support](https://github.com/ggml-org/llama.cpp/pull/22838)

[16] [vLLM: MXFP4 Support for linear layers + compressed-tensors integration](https://github.com/vllm-project/vllm/pull/41664)

[17] [vLLM: Use 2D-grid to eliminate divmod in W8W8 group quant](https://github.com/vllm-project/vllm/pull/42153)

[18] [llama.cpp: ggml-webgpu enables running gpt-oss-20b](https://github.com/ggml-org/llama.cpp/pull/22906)

[19] [SGLang: perf(nvfp4): free unused source scales after weight processing](https://github.com/sgl-project/sglang/pull/25107)

[20] [TRT-LLM: Follow-up patch for MoE autotune in DEP](https://github.com/NVIDIA/TensorRT-LLM/pull/13971)

[21] [vLLM: MoE Refactor - EPLB refactoring for FusedMoE](https://github.com/vllm-project/vllm/pull/41055)

[22] [vLLM: MoE Refactor - Introduce RoutedExperts alias](https://github.com/vllm-project/vllm/pull/40735)

[23] [vLLM: MoE Refactor - Move expert map code into ExpertMapManager](https://github.com/vllm-project/vllm/pull/41046)

[24] [vLLM: MoE Refactor - Move remaining experts classes to experts directory](https://github.com/vllm-project/vllm/pull/42334)

[25] [vLLM: MoE Refactor - Add sequence parallel tests](https://github.com/vllm-project/vllm/pull/41299)

[26] [llama.cpp: hexagon: eliminate scalar VTCM loads via HVX splat helpers](https://github.com/ggml-org/llama.cpp/pull/22993)

[27] [llama.cpp: ggml-opencl: add opt-in Adreno xmem GEMM for prefill](https://github.com/ggml-org/llama.cpp/pull/22755)

[28] [llama.cpp: metal: promote mul_mv/mul_mm batch divisors to function constants](https://github.com/ggml-org/llama.cpp/pull/22711)

[29] [llama.cpp: internal AllReduce kernel for CUDA provider](https://github.com/ggml-org/llama.cpp/pull/22299)

[30] [llama.cpp: vulkan: Support asymmetric FA in scalar/mmq/coopmat1 paths](https://github.com/ggml-org/llama.cpp/pull/22589)