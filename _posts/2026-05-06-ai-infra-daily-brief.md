---
title: AI Infra 早报｜推理框架的决胜点从架构声明下沉到内核层真实收口
date: 2026-05-06 08:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: TRT-LLM 砍掉 6 秒 FMHA JIT 重编译、Megatron 为 DSv4 HybridModel 搭建完整推理链路、vLLM 解除 speculative decoding 对多模态模型的封锁——框架竞争的主战场已经从架构层宣言下沉到单步执行路径。KV 管理同时进入治理阶段，从"能不能传"演进到"怎么管得住、怎么不丢数据"。

过去三天，推理框架密集补齐了架构层承诺过的执行能力，把蓝图变成了生产可用的事实。TRT-LLM、Megatron-LM、vLLM 和 llama.cpp 各自从不同角度证明，性能竞争的决胜点已经下沉到内核层——谁能在单步执行路径上压掉更多延迟，谁才能拿到真实的吞吐优势。

## 一、TRT-LLM 内核层收口：从架构声明到执行路径真实提速

TRT-LLM 的更新几乎全部围绕一个目标：压降每一步推理的固定开销。**TRT-LLM 消除了 eager generation 场景下约 6 秒的 FMHA JIT 重编译**[[1]](https://github.com/NVIDIA/TensorRT-LLM/pull/13505)，方法是把 kernel selection 与 CUDA graph warmup 对齐，直接 drop cubin。这比任何架构层优化都更直接——TTFT 立刻受益。

同一思路延续到其他执行瓶颈：**Qwen3.5 GDN 层拿到了 3 个 fused Triton kernel**[[2]](https://github.com/NVIDIA/TensorRT-LLM/pull/12966)，每个 GDN 层省掉一次独立 elementwise kernel；**beam-search 的 prefill→decode 切换开销被压降**[[3]](https://github.com/NVIDIA/TensorRT-LLM/pull/13748)，同样直接改善 TTFT；**piecewise cudagraph capture 之前漏掉 `num_tokens` 变量范围**[[4]](https://github.com/NVIDIA/TensorRT-LLM/pull/13574)，修回来后 cudagraph 命中率提升。

调度层的优化同样针对固定开销。**AutoDeploy 的 decode 调度循环压缩了 C++ dispatch 开销**[[5]](https://github.com/NVIDIA/TensorRT-LLM/pull/13012)，包括用 `get_last_tokens` 替代 `get_token` + `get_num_tokens` 等 4 处针对性修改，decode 每一步都受益。此外，**FlashMLA 的 `tokens_per_block` 被重新绑定为 64 但未传播到 `kv_cache_config`**[[6]](https://github.com/NVIDIA/TensorRT-LLM/pull/13752)，这条修复确保 MLA 路径配置一致性；**cache-hit prefill 场景从 KV cache 读回前缀而非重算**[[7]](https://github.com/NVIDIA/TensorRT-LLM/pull/13677)，避免重复计算。

这批更新的共同特点：不谈架构创新，只盯执行瓶颈。JIT 重编译砍掉的是启动延迟，kernel fusion 压降的是调用开销，调度循环优化省掉的是每步的固定成本。TRT-LLM 正在用内核层面的真实收口，兑现架构层的性能承诺。

## 二、Megatron 为 DSv4 HybridModel 搭建完整推理链路

Megatron-LM 在三天内密集补齐了 DSv4 HybridModel 的推理与训练支撑。**CSA/HCA 原型首次接入 HybridModel**[[8]](https://github.com/NVIDIA/Megatron-LM/pull/4569)，当前支持 CP=1、TP=1 的 attention 变体；**mHC（multi-head Causal）支持同步补进**[[9]](https://github.com/NVIDIA/Megatron-LM/pull/4469)，覆盖 HybridModel/HybridStack 的另一个核心 attention 模式。

推理链路的关键组件同期就位：**GPT↔Hybrid 权重转换打通**[[10]](https://github.com/NVIDIA/Megatron-LM/pull/4482)，意味着已有 GPT checkpoint 可以迁移到 HybridModel；**vLLM 的 grouped gemm kernel 被移植到 Megatron 推理 MoE 后端**[[11]](https://github.com/NVIDIA/Megatron-LM/pull/4566)；**shared expert 计算与 allgatherv 通信重叠**[[12]](https://github.com/NVIDIA/Megatron-LM/pull/4570)，隐藏 MoE 推理的通信开销。内存管理方面，**per-request/per-token bookkeeping 从 GPU 搬到 pinned CPU**[[13]](https://github.com/NVIDIA/Megatron-LM/pull/4306)，只留 `ContextGPUView` 作为 forward 单 GPU 接口，减少显存占用。

训练相关的修复同样指向生产可用：**layerwise distributed optimizer 与 overlap-param-gather 的交互导致梯度损坏**[[14]](https://github.com/NVIDIA/Megatron-LM/pull/4609)，这是训练正确性修复；**chunked MLP 从 inference prefill 扩展到 training phase**[[15]](https://github.com/NVIDIA/Megatron-LM/pull/3656)，内存优化路径一致化；**集成测试覆盖 CSA/HCA attention + hash MoE routing + clamped SwiGLU**[[16]](https://github.com/NVIDIA/Megatron-LM/pull/4596)，确保链路可跑。

HybridModel 不再是架构声明，而是一条可执行的推理链路。从 attention 变体到 MoE 后端、从权重转换到显存管理，三天内补齐的组件足以支撑一次完整的推理调用。

## 三、Speculative decoding 打开多模态大门，KV 管理进入治理阶段

Speculative decoding 之前被限制在纯文本模型上，vLLM 解除了这层封锁。**vLLM 允许多模态模型参与 speculative decoding**[[17]](https://github.com/vllm-project/vllm/pull/41752)，之前 `parallel drafting` 的 `_raise_if_multimodal()` 无条件阻止多模态模型启动，现在改为 warning 允许运行。这标志着推理加速不再是文本模型的特权。llama.cpp 同期在 spec decode 路径上做了优化：**speculative checkpoint 保留在 device memory**[[18]](https://github.com/ggml-org/llama.cpp/pull/22679)，消除 D2H 拷贝开销；**Hadamard KV-cache rotation 从 O(N²) 矩阵乘法变为 O(N log N) FWHT**[[19]](https://github.com/ggml-org/llama.cpp/pull/22631)，对长上下文场景收益显著。

KV 管理正在从"能不能传"演进到"怎么管得住、怎么不丢数据"。SGLang 方面，**HiSparse 接入 FP8 KV cache**[[20]](https://github.com/sgl-project/sglang/pull/23013)，之前锁死 BF16，现在路由到 `flashmla_kv` kernel 实现 FP8 + sparse attention。LMCache 方面，**KV cache 远端存储新增 HuggingFace Buckets 原生支持**[[21]](https://github.com/LMCache/LMCache/pull/3060)；**序列化/反序列化能力补齐**[[22]](https://github.com/LMCache/LMCache/pull/3140)，是跨引擎、跨进程传递的前提；**CB blend 命中率从创生以来一直是 0%**[[23]](https://github.com/LMCache/LMCache/pull/3179)，因为 `cb_store_final` 从未在 fingerprint table 注册 chunk，同时修复线程安全和 store-complete 竞态。此外，**S3 L2 适配器的 store listener 在 store 完成 signal 之后才触发**[[24]](https://github.com/LMCache/LMCache/pull/3188)，导致竞态；**磁盘后端在 load 确认成功之前就更新 cache-policy hit**[[25]](https://github.com/LMCache/LMCache/pull/3149)，导致无效条目被标记为命中。这些修复指向同一个问题：KV 层的竞争从传输速度扩展到数据完整性和存储生态。

## 四、今天真正值得记住的判断

推理框架的竞争逻辑已经完成一轮下沉：从架构层声明"我们支持这种能力"，转向内核层证明"我们能在单步执行路径上做得更好"。TRT-LLM 砍掉 6 秒 JIT 重编译、Megatron 为 HybridModel 补齐推理链路、vLLM 打开多模态 speculative decoding、llama.cpp 用 FWHT 替代 O(N²) 矩阵乘法——这些更新没有一个在谈架构创新，全部在盯执行瓶颈。

KV 管理同时进入治理阶段。框架开始追问：数据会不会丢？命中率是不是真实的？远端存储接入成本够不够低？LMCache 修复 CB blend 命中率长期为 0 的问题，意味着 KV 治理从"能不能传"演进到"怎么管得住"。性能竞争到了这个阶段，架构创新的红利已经吃尽，决胜变量下沉到每一微秒的延迟、每一个 byte 的正确性。

---

## 参考来源

[1] [Drop cubin and eliminate ~6s FMHA JIT recompile in eager generation](https://github.com/NVIDIA/TensorRT-LLM/pull/13505)

[2] [Fuse GDN elementwise ops and split/transpose kernels for Qwen3.5](https://github.com/NVIDIA/TensorRT-LLM/pull/12966)

[3] [Reduce beam-search prefill→decode handoff cost](https://github.com/NVIDIA/TensorRT-LLM/pull/13748)

[4] [Broader capture of piecewise cudagraph](https://github.com/NVIDIA/TensorRT-LLM/pull/13574)

[5] [AutoDeploy: reduce C++ dispatch overhead in decode scheduling loop](https://github.com/NVIDIA/TensorRT-LLM/pull/13012)

[6] [Propagate FlashMLA tokens_per_block override onto kv_cache_config](https://github.com/NVIDIA/TensorRT-LLM/pull/13752)

[7] [AutoDeploy MLA chunked prefill loop support](https://github.com/NVIDIA/TensorRT-LLM/pull/13677)

[8] [Add the CSA/HCA prototype to HybridModel](https://github.com/NVIDIA/Megatron-LM/pull/4569)

[9] [Add mHC support for HybridModel on dsv4](https://github.com/NVIDIA/Megatron-LM/pull/4469)

[10] [Checkpoint conversion between GPT_model and Hybrid_model](https://github.com/NVIDIA/Megatron-LM/pull/4482)

[11] [Add vLLM grouped gemm backend for MoE inference](https://github.com/NVIDIA/Megatron-LM/pull/4566)

[12] [Enable shared expert overlap with allgatherv in inference](https://github.com/NVIDIA/Megatron-LM/pull/4570)

[13] [Move inference context bookkeeping to CPU with ContextGPUView](https://github.com/NVIDIA/Megatron-LM/pull/4306)

[14] [Fix gradient corruption with layerwise param all-gather overlap](https://github.com/NVIDIA/Megatron-LM/pull/4609)

[15] [Enable chunked MLP during training](https://github.com/NVIDIA/Megatron-LM/pull/3656)

[16] [DSv4 hybrid hash MoE integration coverage](https://github.com/NVIDIA/Megatron-LM/pull/4596)

[17] [[Spec Decode] Allow multimodal models with a warning](https://github.com/vllm-project/vllm/pull/41752)

[18] [llama : add option to save memory in device buffers (speculative decoding)](https://github.com/ggml-org/llama.cpp/pull/22679)

[19] [ggml : implement fast Walsh-Hadamard transform for KV rotation](https://github.com/ggml-org/llama.cpp/pull/22631)

[20] [[HiSparse] Support FP8 KV cache by routing to flashmla_kv backend](https://github.com/sgl-project/sglang/pull/23013)

[21] [[Remote Backend] Add Hugging Face Buckets as a built-in remote storage backend](https://github.com/LMCache/LMCache/pull/3060)

[22] [[Core] Implement serialization / deserialization](https://github.com/LMCache/LMCache/pull/3140)

[23] [[CB] Fix CB lookup correctness, thread safety, and store-complete race](https://github.com/LMCache/LMCache/pull/3179)

[24] [[Bugfix] Fix S3 L2 adapter listener race](https://github.com/LMCache/LMCache/pull/3188)

[25] [fix(disk): defer cache-policy hit update until load succeeds](https://github.com/LMCache/LMCache/pull/3149)