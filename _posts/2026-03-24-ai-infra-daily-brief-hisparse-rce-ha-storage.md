---
title: AI Infra 早报｜安全补丁领衔，推理框架在稀疏注意力与 HA 存储上同步破题
date: 2026-03-24 05:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: llama.cpp RPC 路径 RCE 漏洞紧急修复，所有使用 RPC 功能的部署应立即升级；SGLang 引入 HiSparse 层次化稀疏注意力，vLLM 落地零气泡异步调度+投机解码组合；Mooncake 引入 HA 存储后端抽象，LMCache 新增 Device-DAX 近内存持久化；TRL 新增 SDPO 训练器，Megatron-LM 同日合并 Muon μP 缩放与 Mamba GDN 支持。
---

今天有一条消息需要先说清楚：**llama.cpp 修复了 RPC 路径上的远程代码执行（RCE）漏洞**。RCE 不是普通的崩溃或功能异常——它意味着攻击者可能借助这个入口在你的推理服务器上执行任意代码。如果你的部署用到了 llama.cpp 的 RPC 功能（多机推理、远程模型服务），请优先升级到 b8492 或更高版本[[10]](https://github.com/ggml-org/llama.cpp/releases/tag/b8492)，其他内容可以之后再看。

## 一、推理侧：稀疏注意力、零气泡调度与多硬件补全

**SGLang 合并了 HiSparse 层次化稀疏注意力**[[5]](https://github.com/sgl-project/sglang/pull/20343)，这是本窗口推理方向最有想象力的进展。稀疏注意力的思路不新鲜，但层次化的实现方式——通过多级稀疏模式识别并跳过对输出贡献极小的注意力块——在工程上要比简单的 top-k 注意力更难落地。HiSparse 进入 SGLang 主分支意味着它通过了基本的工程验证，能不能在实际 workload 里做到精度与速度的双赢，值得持续跟踪。

**vLLM 这次落地的是零气泡异步调度与投机解码的组合**[[1]](https://github.com/vllm-project/vllm/pull/32951)。异步调度把 CPU 侧调度逻辑从 GPU 推理的关键路径上移走，投机解码减少每个 token 的生成轮次，两者各自能压缩延迟，组合后理论上可以同时消除两者的等待气泡。这是一个酝酿很久的大 PR，代码量和测试复杂度都不小，落地本身就是一个工程里程碑。

配合这个方向，vLLM 还修复了 MRV2 warmup 阶段不考虑 spec decode 内存分配的问题[[2]](https://github.com/vllm-project/vllm/pull/37812)——之前 warmup 完成不代表推理时内存够用，修复后两者对齐。图编译延迟化[[4]](https://github.com/vllm-project/vllm/pull/37609)则解决了 split_module 触发多余 recompile 的问题，含图编译的模型加载明显加快。

MoE 方向，vLLM 修复了 TRTLLM NVFP4 路由内核的精度 bug[[3]](https://github.com/vllm-project/vllm/pull/36725)，这个问题会静默地影响 NVFP4 量化 MoE 模型的推理准确性，是实际部署中比较隐蔽的坑。

**SGLang 在多硬件方向的推进也很活跃**。AMD GPU 获得了针对 Qwen3.5 的融合 GemmaRMSNorm HIP 内核[[9]](https://github.com/sgl-project/sglang/pull/21188)，消除了 AMD 上跑 Qwen3.5 的性能瓶颈；昇腾 NPU 上 MiniMax M2 的推理精度从 16.5% 大幅修复到 95.5%[[8]](https://github.com/sgl-project/sglang/pull/17695)——16.5% 基本是不可用的状态，这次修复让 NPU 侧的 MiniMax M2 真正可用。SGLang 的 Ngram 投机解码重构系列本窗口继续推进，修复了 C++ 实现中的同步 bug[[6]](https://github.com/sgl-project/sglang/pull/21186)（系列第三步）。

**llama.cpp 在安全补丁之外还有几个值得关注的进展**。Adreno GPU 新增 Q6_K 量化的 GEMM/GEMV OpenCL 内核[[11]](https://github.com/ggml-org/llama.cpp/releases/tag/b8493)，把移动端可用的量化精度从 Q4_K 扩展到 Q6_K；HTTP server 改用动态线程[[12]](https://github.com/ggml-org/llama.cpp/releases/tag/b8485)，高并发下不再受固定线程数限制；InternVL 获得动态高分辨率图像预处理支持[[13]](https://github.com/ggml-org/llama.cpp/releases/tag/b8477)，OCR 等场景的图像理解能力提升。

**TensorRT-LLM 本窗口** 修复了 Eagle 投机解码在 MLA Target + GQA Draft 组合下的 bug[[14]](https://github.com/NVIDIA/TensorRT-LLM/pull/12171)，改进了 micro batch 调度器对可复用 KV cache 块的感知[[15]](https://github.com/NVIDIA/TensorRT-LLM/pull/11637)，并新增了 allgather 的安全分块版本[[16]](https://github.com/NVIDIA/TensorRT-LLM/pull/12174)以应对大规模分布式场景下的通信稳定性问题。

## 二、训练侧：SDPO、μP 缩放与 Mamba 扩展

TRL 合并了 SDPO（Self-Distillation Policy Optimization）训练器[[17]](https://github.com/huggingface/trl/pull/4935)。SDPO 的核心思路是用当前策略的历史版本作为参考模型，省去了维护独立参考模型的推理开销——在大模型 RLHF 训练中，参考模型的推理开销不容忽视，SDPO 提供了一个成本更低的替代路径。同期修复了 AsyncGRPOTrainer 中 bfloat16 精度不足导致训练不稳定的问题[[18]](https://github.com/huggingface/trl/pull/5333)，改为 float32。

Megatron-LM 在同一窗口合并了两个方向各异但都值得关注的 PR。**Muon 优化器的 μP 缩放支持**[[19]](https://github.com/NVIDIA/Megatron-LM/pull/3715)：μP（maximal update parametrization）的价值在于超参数可迁移性——在小模型上调好的学习率等参数可以直接用于大模型，大幅降低大规模预训练的调参成本。Muon 最近在大模型训练中表现出色，加上 μP 缩放后工程实用性进一步提升。**Mamba SSM 引入 GDN 模块**[[20]](https://github.com/NVIDIA/Megatron-LM/pull/3535)：GDN（Gated Delta Networks）为 Mamba 的状态更新机制引入门控增量，扩展了 SSM 对长距离依赖的建模能力，也为 Mamba-Transformer 混合架构的训练打开了新配置空间。

## 三、生产部署：KV cache 走向高可用与多样化存储

**Mooncake 这次的进展是引入 HA 存储后端抽象**[[21]](https://github.com/kvcache-ai/Mooncake/pull/1678)。此前 Mooncake 的 KV cache 存储以高性能为核心设计目标，但在生产环境中，单点故障导致 KV cache 全丢、服务降级的场景同样不可忽视。HA 抽象层支持配置主备后端，主后端故障时自动切换，是 Mooncake 从"够快"迈向"够可靠"的关键一步。同期，分层调度器的增量化改造[[22]](https://github.com/kvcache-ai/Mooncake/pull/1675)解决了大规模部署下全量重算调度开销过高的问题，SSD 层空间管理的加固则防止了磁盘配额超限的运维风险。

**LMCache 的 Device-DAX 后端**[[23]](https://github.com/LMCache/LMCache/pull/2788)是一个相对小众但技术含量很高的进展。/dev/dax（Device DAX）是 Linux 下直接访问持久内存（如 CXL 内存、Intel Optane）的设备接口，提供接近 DRAM 的访问延迟同时具备持久化能力。对于配备 CXL 扩展内存或 Optane PMEM 的服务器，这意味着 KV cache 可以存在"比 DRAM 便宜、比 SSD 快、还能持久化"的存储层。LMCache 同期新增的 `describe kvcache` CLI 子命令[[24]](https://github.com/LMCache/LMCache/pull/2825)则解决了运维可观测性问题——之前要查 KV cache 状态只能翻代码或看日志，现在一条命令即可。

## 四、OpenClaw 稳定性修复

OpenClaw 本窗口修复了两个影响实际使用的问题。**auth store 竞态修复**[[25]](https://github.com/openclaw/openclaw/pull/53211)：并发写入时旧 token 覆盖新刷新的 token，导致认证状态意外失效——这种问题在高频操作下概率不低，且现象（频繁要求重新认证）容易误导排查方向。**doctor 命令清理插件配置**[[26]](https://github.com/openclaw/openclaw/pull/53187)：历史插件的白名单和入口引用如果不清理，会干扰插件系统的正常运行，现在 `openclaw doctor` 会主动处理这部分垃圾数据。

---

今天内容密度颇高，但有一个明显的结构：**安全优先，稳定其次，性能再次**。llama.cpp 的 RCE 修复没有什么好说的，就是立刻升级。稳定性方向，vLLM/SGLang/TRT-LLM 的多个 bugfix 指向 MoE、投机解码、并行模式这三个复杂度最高、也最容易出问题的交叉地带。性能方向，HiSparse 和零气泡调度是两个不同量级的进展——前者在开辟新路，后者在把已知路线跑通。存储方向，HA 和 Device-DAX 都在回答同一个问题：KV cache 基础设施如何从"能用"变成"生产级可靠"。

## 参考来源

[1] [vLLM 零气泡异步调度+投机解码](https://github.com/vllm-project/vllm/pull/32951)

[2] [vLLM MRV2 warmup 纳入 spec decode 内存](https://github.com/vllm-project/vllm/pull/37812)

[3] [vLLM TRTLLM NVFP4 MoE 路由精度修复](https://github.com/vllm-project/vllm/pull/36725)

[4] [vLLM 图编译延迟化](https://github.com/vllm-project/vllm/pull/37609)

[5] [SGLang HiSparse 层次化稀疏注意力](https://github.com/sgl-project/sglang/pull/20343)

[6] [SGLang Ngram 投机解码重构第三步](https://github.com/sgl-project/sglang/pull/21186)

[7] [SGLang DeepSeek V3/V2 CP in-seq-split 修复](https://github.com/sgl-project/sglang/pull/21192)

[8] [SGLang NPU MiniMax M2 精度修复](https://github.com/sgl-project/sglang/pull/17695)

[9] [SGLang AMD Qwen3.5 融合 GemmaRMSNorm HIP 内核](https://github.com/sgl-project/sglang/pull/21188)

[10] [llama.cpp RPC RCE 漏洞修复 b8492](https://github.com/ggml-org/llama.cpp/releases/tag/b8492)

[11] [llama.cpp Adreno Q6_K GEMM/GEMV b8493](https://github.com/ggml-org/llama.cpp/releases/tag/b8493)

[12] [llama.cpp server 动态线程 b8485](https://github.com/ggml-org/llama.cpp/releases/tag/b8485)

[13] [llama.cpp InternVL 动态高分辨率 b8477](https://github.com/ggml-org/llama.cpp/releases/tag/b8477)

[14] [TRT-LLM Eagle MLA+GQA 修复](https://github.com/NVIDIA/TensorRT-LLM/pull/12171)

[15] [TRT-LLM micro batch 调度纳入可复用 KV cache](https://github.com/NVIDIA/TensorRT-LLM/pull/11637)

[16] [TRT-LLM allgather 安全分块版本](https://github.com/NVIDIA/TensorRT-LLM/pull/12174)

[17] [TRL SDPO 训练器](https://github.com/huggingface/trl/pull/4935)

[18] [TRL AsyncGRPOTrainer float32 精度修复](https://github.com/huggingface/trl/pull/5333)

[19] [Megatron-LM Muon 优化器 μP 缩放](https://github.com/NVIDIA/Megatron-LM/pull/3715)

[20] [Megatron-LM Mamba GDN 模块](https://github.com/NVIDIA/Megatron-LM/pull/3535)

[21] [Mooncake HA 存储后端抽象](https://github.com/kvcache-ai/Mooncake/pull/1678)

[22] [Mooncake 分层调度器增量化与 SSD 加固](https://github.com/kvcache-ai/Mooncake/pull/1675)

[23] [LMCache Device-DAX 存储后端](https://github.com/LMCache/LMCache/pull/2788)

[24] [LMCache CLI describe kvcache](https://github.com/LMCache/LMCache/pull/2825)

[25] [OpenClaw auth store token 竞态修复](https://github.com/openclaw/openclaw/pull/53211)

[26] [OpenClaw doctor 清理失效插件配置](https://github.com/openclaw/openclaw/pull/53187)
