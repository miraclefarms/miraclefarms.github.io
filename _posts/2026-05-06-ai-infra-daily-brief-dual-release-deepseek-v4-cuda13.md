---
title: AI Infra 早报｜双版本日：DeepSeek V4 稳定化与硬件栈基线上移并排落地
date: 2026-05-06 08:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: vLLM v0.20.1 与 SGLang v0.5.11 同天发布，前者收拢 DeepSeek V4 稳定化补丁，后者将 CUDA 13.0 与 PyTorch 2.11 设为默认；与此同时，分离部署路径上的 RDMA 错误边界被 SGLang 与 Mooncake 集中补建，四个漏洞都是多节点真实部署才会触发的那种。
---

今天的 AI Infra 动态，本质上是一次集中的升级决策窗口开启。**vLLM v0.20.1 与 SGLang v0.5.11 同天发布**，方向各自清晰：前者收拢 DeepSeek V4 的稳定化补丁，后者把硬件栈基线整体上移——CUDA 13.0 成为默认，PyTorch 从 2.9 升至 2.11。两框架同步推进 DeepSeek 支持，生产用户这周同时面对两张升级决策单，需要主动判断要不要一并跟进。

紧跟其后的是分离部署路径上的一轮集中容错修复。SGLang 和 Mooncake 在同一时间窗口内各自处理了 RDMA/NIXL 层面的错误边界漏洞：断连未捕获、指针整数溢出、dmabuf 注册时序错误、三节点环形死锁——共同特征是单机路径完全不会触发，多节点真实部署才会暴露。版本发布是台面上的事，错误边界的补建才是背后更现实的工程进展。

## 一、双版本日：DeepSeek V4 稳定化与硬件栈基线上移

**vLLM v0.20.1[[1]](https://github.com/vllm-project/vllm/releases/tag/v0.20.1) 的主题是稳定化**。这个版本的改动集中在 DeepSeek V4 支持的可靠性上：multi-stream pre-attention GEMM 的优化与 base model support 的补入，都是把上一个主版本留下的边缘问题收拢，而不是进一步探索新能力边界。从节奏上看，v0.20.1 是一个明确的工程信号——框架决定先把 V4 支持打实再推进。

**SGLang v0.5.11[[2]](https://github.com/sgl-project/sglang/releases/tag/v0.5.11) 的方向截然不同**，这次升级是硬件栈基线整体上移。CUDA 13.0 成为新默认值，PyTorch 从 2.9 升至 2.11，sgl-kernel 与 Docker 镜像同步更新。对依赖预构建镜像的团队，下一次拉取镜像就会进入新基线，需要提前验证下游依赖的兼容性。与此同时，**AMD 方向的 DeepSeek V4 内核融合[[3]](https://github.com/sgl-project/sglang/pull/24424) 也在同天合并**——SGLang AMD 集成系列第 11 期，compressor element-wise kernel fusion，说明 SGLang 的 DeepSeek 覆盖面已经延伸到 AMD 硬件栈。

## 二、vLLM CPU 量化路径向 AVX2 与 FP8 延伸

两个 PR 同天合并，把 vLLM CPU 推理的量化覆盖范围从"AVX512 专属"向下延伸到更普通的部署环境。

**AVX2 INT8 量化[[4]](https://github.com/vllm-project/vllm/pull/41318) 打开了更大的 CPU 部署空间**。之前 static/dynamic scaled int8 量化受 `__AVX512F__` 门控限制，现在这道门被打开，AVX2 平台也能跑。这是可用性提升，不是性能里程碑——它的意义在于把量化路径从高端服务器扩展到更普通的 CPU 平台。紧跟其后的是 **Intel CPU 上的 FP8 W8A16 block-quantized 线性层支持[[5]](https://github.com/vllm-project/vllm/pull/41186)**，填补了 Intel 部署场景此前的量化空缺。两个改动叠在一起，CPU 推理的量化选项显著增多，Intel 平台的覆盖路径趋于完整。

## 三、分离部署路径的 RDMA 错误边界被集中补建

prefill-decode 分离路径上的容错逻辑，被 SGLang 和 Mooncake 同时推进，四个修复分属四类只有多节点真实部署才会暴露的错误。

**SGLang 处理了两个 NIXL/UCX 层的边界漏洞。** [#24296][[6]](https://github.com/sgl-project/sglang/pull/24296) 在 NixlKVSender 侧补上了对 `UCS_ERR_NOT_CONNECTED` / `CONNECTION_RESET` 等断连错误的捕获——这类错误之前会直接导致未处理异常，而不是进入重试或降级路径。[#24188][[7]](https://github.com/sgl-project/sglang/pull/24188) 把指针与长度数组改为 `np.uint64`，防止 XPU 设备地址超出 int64 上界时触发溢出，导致指针被截断打到错误位置。

**Mooncake 的两个修复更底层。** [#2035][[8]](https://github.com/kvcache-ai/Mooncake/pull/2035) 解决了 dmabuf-based 内存注册时 `cuMemGetHandleForAddressRange` 要求精确 allocation boundary 而传入用户地址导致的时序错误；[#1959][[9]](https://github.com/kvcache-ai/Mooncake/pull/1959) 修复了 T0/T1/T2 三节点 P2P handshake 形成环形死锁的问题——三节点同时互相等待对方先完成握手，导致整个建链过程僵住。

这四个修复的共同轮廓清楚：**disaggregation 路径的生产成熟度仍在追赶单机路径**，正在把真实多节点部署中暴露的基本容错行为逐条补入主路径。

## 四、TRT-LLM 消除 JIT 重编译，llama.cpp 换掉 KV 旋转算法

TRT-LLM 侧，**约 6 秒的 FMHA JIT 重编译延迟被消除[[10]](https://github.com/NVIDIA/TensorRT-LLM/pull/13505)**。根本原因是 CUDA graph warmup 与 eager generation 路径之间的 kernel 选择不一致，导致 eager generation 时重新触发 FMHA 编译，直接影响 TTFT 的可感知延迟。修复通过对齐 warmup 的 kernel 选择、drop cubin 路径来解决。同日，**Helix Parallelism 官方博客[[11]](https://github.com/NVIDIA/TensorRT-LLM/pull/13547) 合并**，NVIDIA 首次在文档层面公开这套并行机制——对希望理解 TRT-LLM 长期路线的用户，这份文档比一个新版本更有参考价值。

llama.cpp 这边，**KV 旋转从矩阵乘替换为 Fast Walsh-Hadamard Transform[[12]](https://github.com/ggml-org/llama.cpp/pull/22631)**，复杂度从 O(N²) 降到 O(N log N)。这个替换没有引入新的 ggml op，直接复用 `ggml_map_custom2` 路径，现有 backend 无需改动。同天的 **b9041 版本[[13]](https://github.com/ggml-org/llama.cpp/releases/tag/b9041)** 还包含 **CPU backend RMS_NORM + MUL 单 pass 融合[[14]](https://github.com/ggml-org/llama.cpp/pull/22423)**，避免中间张量分配。两个改动都是"对现有路径做算法替换"，代表 llama.cpp 本地推理核心算子层正在进入精细化阶段。

## 五、今天真正值得记住的判断

今天最值得记住的一件事，是**双版本日把升级决策压力集中释放了**。vLLM 在稳定，SGLang 在提基线，两框架的 DeepSeek 支持也都在同步向前推。对同时运行这两个框架的团队，这周是一个需要主动做决策的节点，而不是可以等等看的时机。

另一条值得带走的判断是关于 disaggregation 路径的现状：**单机路径上永远不会触发的错误，在多节点真实部署中正在被一条一条地暴露和修复**。这不是在质疑分离部署的方向，而是在说明它的成熟度仍有差距。谁把这道差距先填完，谁才真正具备在多节点规模上可信的分离部署能力。

---

## 参考来源

[1] [vLLM v0.20.1 release notes](https://github.com/vllm-project/vllm/releases/tag/v0.20.1)

[2] [SGLang v0.5.11 release notes](https://github.com/sgl-project/sglang/releases/tag/v0.5.11)

[3] [SGLang #24424: AMD/DeepSeek V4 compressor element-wise kernel fusion](https://github.com/sgl-project/sglang/pull/24424)

[4] [vLLM #41318: dnnl build for AVX2 W8A8 Int8 量化支持](https://github.com/vllm-project/vllm/pull/41318)

[5] [vLLM #41186: CPU FP8 W8A16 linear support](https://github.com/vllm-project/vllm/pull/41186)

[6] [SGLang #24296: Handle nixlRemoteDisconnectError in NixlKVSender](https://github.com/sgl-project/sglang/pull/24296)

[7] [SGLang #24188: 修复 disaggregation KV transfer 中 XPU 指针 int64 溢出](https://github.com/sgl-project/sglang/pull/24188)

[8] [Mooncake #2035: 修复 dmabuf 内存注册时 allocation base addr 问题](https://github.com/kvcache-ai/Mooncake/pull/2035)

[9] [Mooncake #1959: 修复 RDMA P2P handshake 三节点环形死锁](https://github.com/kvcache-ai/Mooncake/pull/1959)

[10] [TRT-LLM #13505: 消除 eager generation 中约 6 秒 FMHA JIT 重编译](https://github.com/NVIDIA/TensorRT-LLM/pull/13505)

[11] [TRT-LLM #13547: Helix Parallelism 官方博客文档](https://github.com/NVIDIA/TensorRT-LLM/pull/13547)

[12] [llama.cpp #22631: Fast Walsh-Hadamard Transform 替换 KV 旋转矩阵乘](https://github.com/ggml-org/llama.cpp/pull/22631)

[13] [llama.cpp b9041 release](https://github.com/ggml-org/llama.cpp/releases/tag/b9041)

[14] [llama.cpp #22423: CPU backend RMS_NORM + MUL 单 pass 融合](https://github.com/ggml-org/llama.cpp/pull/22423)
