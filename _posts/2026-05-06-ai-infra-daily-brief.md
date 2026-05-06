---
title: AI Infra 早报｜非 CUDA 推理栈集中补齐，KV 缓存进入跨介质治理阶段
date: 2026-05-06 08:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: 过去三天 vLLM、LMCache、llama.cpp 在 Intel CPU、ROCm、Hexagon 侧同步推进量化推理，非 CUDA 路径从"能跑"迈向"可用"。LMCache 连推多租户隔离、HF 远程后端和序列化，KV 治理复杂度陡升。TRT-LLM、Megatron 在 decode 路径密集优化，TRL 向 RL 工程化收口。
---

过去三天的 AI Infra 生态呈现出两条并行加速的主线。一条是推理框架对非 NVIDIA 硬件栈的系统性补齐——vLLM 在 Intel CPU 上连推 FP8 量化和 AVX2 INT8 算子下沉，LMCache 补齐 MI300X/MI325X/MI350X 的完整 ROCm 构建镜像，llama.cpp 在 CPU/OpenCL/Hexagon 三线同时打磨 fused kernel 和设备适配。这轮动作的一致性很高：各框架在非 CUDA 路径上同步把"能跑"到"跑得可用"之间的差距逐项填平。另一条主线是 KV 缓存系统从传输层跨入治理层——多租户隔离配额、跨介质序列化、远程 HF Bucket 后端、稀疏注意力的 FP8 KV 接通，KV 层正在成为推理系统里工程密度最高的子系统。

## 一、非 CUDA 推理路径的系统性补齐

**vLLM 在 Intel CPU 上首次支持 FP8 W8A16 block-quantized linear[[1]](https://github.com/vllm-project/vllm/pull/41186)**，同时将 CPU 后端的 INT8 量化算子从 AVX-512 下沉到 AVX2[[2]](https://github.com/vllm-project/vllm/pull/41318)。FP8 扩大了 Intel 侧的模型精度覆盖面，AVX2 下沉则让更多消费级和服务器级 CPU 跑得起 INT8 推理。RISC-V 侧的 OMP 线程绑定也在同步推进[[3]](https://github.com/vllm-project/vllm/pull/40569)，XPU 平台修复了 CUDA graph 内存估算的误触[[4]](https://github.com/vllm-project/vllm/pull/41344)。vLLM 这组更新的信号很明确：让非 CUDA 平台从"实验性支持"变成"日常可用"。

llama.cpp 的步伐同样密集。**CPU 后端实现了 RMS_NORM + MUL 的 fused kernel[[5]](https://github.com/ggml-org/llama.cpp/pull/22423)**，单 pass 完成，消除了中间张量的物化开销。OpenCL 设备的内存估算改用 `CL_DEVICE_GLOBAL_MEM_SIZE` 作为 `--fit` 的依据[[6]](https://github.com/ggml-org/llama.cpp/pull/22688)。Hexagon 端将 matmul 的 tail 行处理从 HVX 迁移到 HMX[[7]](https://github.com/ggml-org/llama.cpp/pull/22724)，端侧推理的计算效率直接受益。加上 KV rotation 引入快速 Walsh-Hadamard 变换，复杂度从 O(N²) 压缩到 O(N log N)[[8]](https://github.com/ggml-org/llama.cpp/pull/22631)，llama.cpp 在非 GPU 路径上的每一个热点都在被逐一打磨。

LMCache 补上了 AMD Instinct GPU 全系（MI300X/MI325X/MI350X）的 ROCm Docker 镜像[[9]](https://github.com/LMCache/LMCache/pull/3101)，降低了在 AMD 加速卡上部署 KV 缓存服务的门槛。

## 二、KV 缓存从传输层跨入治理层

KV 缓存系统正在经历一次治理能力的集体跃迁。**LMCache 引入了 IsolatedLRU 驱逐策略和 per-cache_salt 配额[[10]](https://github.com/LMCache/LMCache/pull/3137)**——不同租户之间的 KV 缓存互不可驱逐，解决的是多租户推理服务里高优先级请求的 KV 被低优先级流量挤掉的问题。同一批更新中，Hugging Face Bucket 被原生接入为 L2 远程存储后端[[11]](https://github.com/LMCache/LMCache/pull/3060)，raw_block 路径完成了多进程共享核心的适配[[12]](https://github.com/LMCache/LMCache/pull/3119)。而 KV cache 的序列化/反序列化支持[[13]](https://github.com/LMCache/LMCache/pull/3140)则为快照、迁移和跨实例恢复铺路。LMCache 正在从一个 KV 缓存传输库演变为一个完整的 KV 治理平台。

SGLang 侧，**稀疏注意力框架 HiSparse 通过 flashmla_kv 后端获得了原生 FP8 KV 缓存支持[[14]](https://github.com/sgl-project/sglang/pull/23013)**，稀疏注意力场景下的 KV 精度和显存效率同步提升。

Mooncake 在传输引擎层面修补了几个关键缺陷：3 节点 P2P 握手场景下的环形死锁[[15]](https://github.com/kvcache-ai/Mooncake/pull/1959)、dmabuf 注册必须匹配 CUDA allocation 边界的问题[[16]](https://github.com/kvcache-ai/Mooncake/pull/2035)，以及统一 nvlink 和 ubshmem allocator 的 ABI 入口[[17]](https://github.com/kvcache-ai/Mooncake/pull/2028)。这些修复解决的都是多节点 KV 传输在生产部署中才会暴露的边界条件。

## 三、TRT-LLM 与 Megatron 在 decode 路径同步压缩开销

TRT-LLM 本周的更新几乎全部围绕 decode 路径的单步执行效率。**Qwen3.5 的 GDN 层将 3 个 Triton kernel 融合为 1 个[[18]](https://github.com/NVIDIA/TensorRT-LLM/pull/12966)**，MoE 预编译 cubin 也得到更新[[19]](https://github.com/NVIDIA/TensorRT-LLM/pull/12440)。beam-search 的 prefill→decode handoff 开销被专门优化[[20]](https://github.com/NVIDIA/TensorRT-LLM/pull/13748)，AutoDeploy 在 decode 调度循环里通过四项改动缩减了 C++/Python 交互开销[[21]](https://github.com/NVIDIA/TensorRT-LLM/pull/13012)。MLA 架构的 cache-hit chunked prefill 也已支持[[22]](https://github.com/NVIDIA/TensorRT-LLM/pull/13677)，AutoDeploy 的 Model Onboarding 移除了 Llama 4、Qwen3 Next/MoE 的手动 patches[[23]](https://github.com/NVIDIA/TensorRT-LLM/pull/13247)。**同时 Helix Parallelism 博文正式发布[[24]](https://github.com/NVIDIA/TensorRT-LLM/pull/13547)**——这是 NVIDIA 为 MoE + 长序列场景准备的新并行策略，意味着 TRT-LLM 在执行层和策略层同时发力。

Megatron 在推理侧的动作同样密集。**每个 request/token 的记账张量从 GPU 迁到 pinned CPU（ContextGPUView）[[25]](https://github.com/NVIDIA/Megatron-LM/pull/4306)**，释放了可观的 GPU 显存。MoE 推理引入了 vLLM 的 grouped gemm 后端[[26]](https://github.com/NVIDIA/Megatron-LM/pull/4566)，shared expert 与 allgatherv 的重叠机制也实现了[[27]](https://github.com/NVIDIA/Megatron-LM/pull/4570)。训练侧，chunked MLP 从推理扩展到训练[[28]](https://github.com/NVIDIA/Megatron-LM/pull/3656)，layerwise distributed optimizer 的梯度损坏问题被定位修复[[29]](https://github.com/NVIDIA/Megatron-LM/pull/4609)，采样操作也下沉到 FlashInfer[[30]](https://github.com/NVIDIA/Megatron-LM/pull/2456)。Megatron 正在把推理优化从训练框架的附属品升级为一条独立演进的路径。

## 四、RL 训练栈从 demo 向生产收口

TRL 本周的更新反映出 RL 训练工具链正在把生产环境的工程需求逐一接住。**OpenReward Standard 环境适配器[[31]](https://github.com/huggingface/trl/pull/5696)**让任何实现了 ORS 协议的 reward 环境直接接入 TRL trainer。chunked NLL loss 与 PEFT 的兼容[[32]](https://github.com/huggingface/trl/pull/5676)让 SFT 阶段的显存峰值进一步可控。**Liger-kernel GRPO loss 暴露了 v0.8.0 的全部新参数（delta、vespo、KL bias correction）[[33]](https://github.com/huggingface/trl/pull/5690)**，length-normalized sigmoid DPO loss 也补上了 Tulu-3/OLMo 路线需要的实现[[34]](https://github.com/huggingface/trl/pull/5406)。SGLang 侧为 RL 场景的 expert 捕获接通了 DeepEP all-to-all 支持[[35]](https://github.com/sgl-project/sglang/pull/16859)。这些改动单看都不大，叠加在一起却指向一个判断：RL 训练栈正在从"能跑 demo"切换到"能接生产负载"。

## 五、今天真正值得记住的判断

推理民主化的节奏在加快。当 vLLM 同时在 Intel FP8、RISC-V 线程绑定和 XPU graph 修复三条线上推进，当 llama.cpp 在 CPU fused kernel、OpenCL 内存估算和 Hexagon HMX 三线同时打磨，这些动作背后的一致性指向一个结论——非 CUDA 路径已经不是"社区的业余爱好"，而是各框架认真对待的生产级目标。与此同时，KV 缓存治理的复杂度正在陡峭上升。多租户隔离、跨介质序列化、远程存储后端、稀疏注意力 FP8 适配——每一项都在把 KV 层从"推理引擎的一个模块"推向"需要独立治理策略的基础设施"。当推理民主化和 KV 治理同时加速，推理系统的工程重心正在发生一次结构性转移。

---

## 参考来源

[1] [Add FP8 W8A16 linear support for Intel CPUs](https://github.com/vllm-project/vllm/pull/41186)

[2] [[Feat] dnnl build for AVX2 W8A8 Int8](https://github.com/vllm-project/vllm/pull/41318)

[3] [[CPU][RISC-V] Auto-bind OMP threads and harden nobind path](https://github.com/vllm-project/vllm/pull/40569)

[4] [[XPU] Disable CUDA graph memory estimate on XPU platform](https://github.com/vllm-project/vllm/pull/41344)

[5] [ggml-cpu: fuse RMS_NORM + MUL on CPU backend](https://github.com/ggml-org/llama.cpp/pull/22423)

[6] [ggml: use CL_DEVICE_GLOBAL_MEM_SIZE as estimate for OpenCL --fit](https://github.com/ggml-org/llama.cpp/pull/22688)

[7] [Hexagon: Process M-tail rows on HMX instead of HVX](https://github.com/ggml-org/llama.cpp/pull/22724)

[8] [ggml: implement fast Walsh-Hadamard transform for kv rotation](https://github.com/ggml-org/llama.cpp/pull/22631)

[9] [[ROCm] Add Dockerfiles for AMD Instinct GPUs](https://github.com/LMCache/LMCache/pull/3101)

[10] [[MP][Feat] Add IsolatedLRU eviction policy + per-cache_salt quotas](https://github.com/LMCache/LMCache/pull/3137)

[11] [[Remote Backend] Add Hugging Face Buckets as a built-in remote storage backend](https://github.com/LMCache/LMCache/pull/3060)

[12] [[MP] Add raw_block MP L2 adapter support via shared RawBlockCore](https://github.com/LMCache/LMCache/pull/3119)

[13] [[Core] Implement serialization / deserialization](https://github.com/LMCache/LMCache/pull/3140)

[14] [[HiSparse] Support FP8 KV cache by routing to flashmla_kv backend](https://github.com/sgl-project/sglang/pull/23013)

[15] [[TE]: Fix possible dead lock in RDMA transport connection setup](https://github.com/kvcache-ai/Mooncake/pull/1959)

[16] [[TransferEngine] Use allocation base addr for dmabuf-based mem registration](https://github.com/kvcache-ai/Mooncake/pull/2035)

[17] [[TransferEngine] Unify fabric allocator plumbing](https://github.com/kvcache-ai/Mooncake/pull/2028)

[18] [Fuse GDN elementwise ops and split/transpose kernels](https://github.com/NVIDIA/TensorRT-LLM/pull/12966)

[19] [Update TRTLLM MoE cubins](https://github.com/NVIDIA/TensorRT-LLM/pull/12440)

[20] [Reduce beam-search prefill->decode handoff cost](https://github.com/NVIDIA/TensorRT-LLM/pull/13748)

[21] [AutoDeploy: reduce C++ dispatch overhead in decode scheduling loop](https://github.com/NVIDIA/TensorRT-LLM/pull/13012)

[22] [AutoDeploy's MLA chunked prefill loop support](https://github.com/NVIDIA/TensorRT-LLM/pull/13677)

[23] [AutoDeploy Model Onboarding](https://github.com/NVIDIA/TensorRT-LLM/pull/13247)

[24] [Blogpost for Helix Parallelism](https://github.com/NVIDIA/TensorRT-LLM/pull/13547)

[25] [Move inference context bookkeeping to CPU with ContextGPUView](https://github.com/NVIDIA/Megatron-LM/pull/4306)

[26] [Add vLLM grouped gemm backend for MoE inference](https://github.com/NVIDIA/Megatron-LM/pull/4566)

[27] [Enable shared expert overlap with allgatherv in inference](https://github.com/NVIDIA/Megatron-LM/pull/4570)

[28] [Add logic to enable chunked MLP during training](https://github.com/NVIDIA/Megatron-LM/pull/3656)

[29] [Fix gradient corruption with layerwise param all-gather overlap](https://github.com/NVIDIA/Megatron-LM/pull/4609)

[30] [FlashInfer sampling](https://github.com/NVIDIA/Megatron-LM/pull/2456)

[31] [[experimental] Add OpenReward Standard environment adapter](https://github.com/huggingface/trl/pull/5696)

[32] [Enable chunked NLL loss with PEFT in SFT](https://github.com/huggingface/trl/pull/5676)

[33] [[GRPO] update Liger-kernel GRPO loss](https://github.com/huggingface/trl/pull/5690)

[34] [Add length-normalized sigmoid loss type to DPO trainer](https://github.com/huggingface/trl/pull/5406)

[35] [[RL] DeepEP support for --enable-return-routed-experts](https://github.com/sgl-project/sglang/pull/16859)
