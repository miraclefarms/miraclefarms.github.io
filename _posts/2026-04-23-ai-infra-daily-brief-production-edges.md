---
title: AI Infra 早报｜生产边缘路径开始进入默认治理
date: 2026-04-23 05:30:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: 过去一天，多模态视频输入、KV/offload 状态保存、异构后端内存控制和训练/Serve 资源调度都在补默认路径。AI Infra 的新竞争点，正在从单点加速扩展到生产边缘条件的可治理性。
---

今天这批更新最有意思的地方，是几个主干项目都在处理“边缘路径”的默认化。视频输入不再只是把帧喂给模型，长视频并发解码、预抽帧元数据、ASR hotwords 都开始进入 OpenAI-compatible 服务接口；KV/offload 也不再只是把缓存搬到别处，而是要保存 Mamba 状态、区分热冷数据、支持 master 不可用时的降级启动；异构后端则在 WebGPU、SYCL、CPU INT4、FlexAttention 这些过去常被视作补充路径的地方继续补能力。

这说明 AI Infra 的生产竞争正在换重心。成熟框架要回答的问题，除了主流 GPU 上某个 kernel 能跑多快，还包括视频、语音、混合状态、浏览器 GPU、Intel/AMD/CPU、Train/Serve 恢复这些“真正上线后一定会遇到”的条件，能不能被写成清楚的默认行为。

## 一、多模态输入开始从模型能力变成服务路径能力

**vLLM 把视频解码默认切到 PyAV 后端，用于并发多模态视频 serving[[1]](https://github.com/vllm-project/vllm/pull/39986)**。这条 PR 的关键信号超过了“换一个 decoder”本身，它直接处理了服务端并发下的 Python GIL 和长视频扫描成本。原 OpenCV 路径在 `grab()` / `retrieve()` 时会持有 GIL，多个请求一并发就容易把视频解码串行化；新的 PyAV 路径用按帧 seek 和 slice threading，让并发请求可以继续前进。PR 里的 Video-MME 长视频测试显示，在并发 1 到 16 的区间里，PyAV 的请求吞吐从约 1.5 倍到 3 倍不等，P99 TTFT 也有大幅下降。

**vLLM 随后支持给预抽帧序列传入原视频 metadata[[2]](https://github.com/vllm-project/vllm/pull/40133)**。这看起来是接口小改，但它解决的是视频理解里很实际的语义损失：客户端已经抽好了 32 帧并用 `video/jpeg` 发给服务器时，服务端如果不知道这些帧来自原视频的哪些位置、总帧数是多少、时长和 FPS 是多少，就很难判断时间顺序和采样密度。现在 `frames_indices`、`total_num_frames`、`duration`、`fps` 可以跟着请求进入模型输入路径，多模态 serving 开始保留“视频作为时间对象”的上下文。

同一条线上，**vLLM 为 FunASR 增加 hotwords 支持[[3]](https://github.com/vllm-project/vllm/pull/39674)**，**llama.cpp 则根据模型能力打开 parallel tool calls 默认行为，并补了结构化输出测试[[4]](https://github.com/ggml-org/llama.cpp/pull/22217)**。这些变化都指向同一个事实：生产接口正在变厚。模型能力要真正可用，服务层必须把音频热词、视频时间信息、工具调用能力这些上下文一起纳入默认协议，不能只把它们留在模型内部。

## 二、KV 与 offload 路径开始认真处理“状态到底是什么”

缓存系统今天最有分量的更新，集中在状态语义。**SGLang 为 Mamba-hybrid 模型在 request retraction 时保存和恢复 MambaPool 状态[[5]](https://github.com/sgl-project/sglang/pull/22493)**。此前 retraction 只会把 attention KV cache offload 到 CPU，Mamba 的 conv / temporal buffer 会丢失，Qwen3.5-397B-A17B 这类混合模型在请求被收回后继续生成就可能被污染。现在 `HybridLinearKVPool` 会把 KV 与 Mamba state 一起 snapshot / restore，PR 还用 GPQA repeat 测试验证了这条路径。

**SGLang 另一条 HiCache 修复则把 `v_head_dim` 明确传给 MHA KV pool，并对 MiMo 的 cache geometry 做启动时校验[[6]](https://github.com/sgl-project/sglang/pull/23173)**。如果模型的 `v_head_dim != head_dim`，按旧逻辑给 V-cache 分配 buffer 会造成静默内存破坏或错误 attention 结果。这个修复的重要性在于，它没有把问题留到运行时崩溃或结果漂移，而是在缓存池的几何假设上直接加了 contract。

Mooncake 也在同一方向推进。**Mooncake Store 新增 offload-on-evict 模式，把 LOCAL_DISK offload 从 PutEnd 推迟到 BatchEvict，形成 DRAM 到本地盘的层级存储[[7]](https://github.com/kvcache-ai/Mooncake/pull/1899)**。过去 `enable_ssd_offload=true` 时，每次 put 都会排队写盘，热 key 即使永远留在 DRAM 里也要承担磁盘 I/O。新模式让 key 先待在 DRAM，只有水位压力触发 eviction 时才写入本地盘；作者的端到端验证里，100 个 1MB key 在 50MB DRAM 段上溢出后，96 个 key 通过 eviction 路径落盘，且全部可 bit-exact 读回。

**Mooncake 还支持 P2P client 在 master 不可用时降级启动[[8]](https://github.com/kvcache-ai/Mooncake/pull/1930)**，把 HA 状态从启动阶段就分成 FULL / DEGRADED，并跳过需要 master metadata 的 segment mount，等待后续恢复。这类改动没有 headline 性能数字，却很接近生产真实需求：缓存数据面不能因为控制面一时不可达就把整个应用启动卡死。

## 三、异构后端的优化开始围绕内存边界展开

异构路径今天也有一组很值得放在一起看的更新。**llama.cpp 的 WebGPU 后端实现 async tensor 和 event API，用四个 1MB buffer 支持异步模型加载[[9]](https://github.com/ggml-org/llama.cpp/pull/22099)**。PR 明确说目标是 Safari 和移动端这类 WebGPU 内存约束更严格的环境，异步 tensor API 能把 wllama 的内存占用降低约 20% 到 25%。浏览器 GPU 的问题从来不只是“有没有算子”，更现实的是能不能在很窄的内存预算下完成加载。

**llama.cpp 的 SYCL 后端则为 `mul_mat_id` 减少 staging buffer 过量分配，并增加 BF16 fast path[[10]](https://github.com/ggml-org/llama.cpp/pull/22119)**。它针对的是 Level Zero 上大词表和 MoE 模型容易触发的 host memory exhaustion：BF16 权重过去会被 cast 到 F32，带来多 GB 额外分配；MoE 的 `mul_mat_id` buffer 也会按总元素数分配，没有贴合实际 routed rows。新的路径让 Intel GPU / SYCL 侧更接近“能承载大模型结构”的现实要求。

SGLang 的 CPU 路径也在补能力。**SGLang 为 CPU 平台增加 GPTQ / AWQ 4-bit 量化支持，并把权重重排到 AMX 格式调用 4-bit sgl-kernel[[11]](https://github.com/sgl-project/sglang/pull/22685)**。这类更新的意义并非 CPU 会替代 GPU serving；更现实的变化是，框架开始把 CPU 当成有正式量化路径的 backend，纯 fallback 的定位正在变弱。

## 四、训练与 Serve 的资源调度开始承认恢复和压力边界

训练侧的变化也在围绕资源边界展开。**Megatron-LM 用新的 `--step-batch-size-schedule` 取代 rampup batch size，支持按 step 或 token 阈值定义任意 batch size 阶梯[[12]](https://github.com/NVIDIA/Megatron-LM/pull/4411)**。这比线性 rampup 更贴近大训练的实际节奏：不同阶段的 batch size 变化往往来自吞吐、稳定性、数据阶段和集群状态的共同约束，简单 rampup 很难描述。

**Megatron-LM 还在 RL 路径里把 optimizer onload 推迟到 logprobs 计算之后[[13]](https://github.com/NVIDIA/Megatron-LM/pull/4235)**，避免 logprob 计算和 optimizer 同时物化占内存。它是很小的内存时序调整，但恰好说明 RLHF / RLOO / GRPO 这类流程里，资源峰值常常来自不同阶段的对象重叠，单个算子本身反而未必是问题来源。

Ray Serve 今天的两条更新也很典型。**Ray Serve 把 node available resources 和 replica requested resources 拆成两个类型，避免调度器把节点容量语义和请求需求语义混在一起[[14]](https://github.com/ray-project/ray/pull/62778)**；**同时在 controller recovery 期间，如果还有 replica 处于 RECOVERING 状态，就延迟广播 `DEPLOYMENT_TARGETS`，避免代理短暂路由到不完整 replica 集合[[15]](https://github.com/ray-project/ray/pull/62751)**。这两条合起来看，服务编排正在把“资源能不能放下”和“恢复期间能不能安全路由”拆成更明确的状态机。

## 五、今天真正值得记住的判断

今天真正值得记住的，是生产系统的边缘条件正在被拉回主路径。多模态输入要保留时间和音频上下文，KV/offload 要保存混合模型的完整状态，WebGPU / SYCL / CPU 要在各自的内存和算子约束下继续前进，训练和 Serve 也要把资源、恢复、调度语义写清楚。

这类更新看起来分散，却共同说明一个趋势：下一阶段 AI Infra 的可靠性不会只来自更快的默认 GPU kernel，而来自框架能不能把那些过去靠部署经验兜底的边缘条件，变成可配置、可恢复、可测试的默认行为。

---

## 参考来源

[1] [vLLM 增加 PyAV 视频解码后端用于并发多模态 serving](https://github.com/vllm-project/vllm/pull/39986)

[2] [vLLM 支持为预抽帧视频输入传入原视频 metadata](https://github.com/vllm-project/vllm/pull/40133)

[3] [vLLM 为 FunASR 增加 hotwords 支持](https://github.com/vllm-project/vllm/pull/39674)

[4] [llama.cpp 根据模型能力启用 parallel tool calls 默认行为](https://github.com/ggml-org/llama.cpp/pull/22217)

[5] [SGLang 在 retraction 时 offload MambaPool 状态](https://github.com/sgl-project/sglang/pull/22493)

[6] [SGLang 校验 MiMo HiCache geometry 并修正 MHA KV pool 的 v_head_dim](https://github.com/sgl-project/sglang/pull/23173)

[7] [Mooncake Store 支持 offload-on-evict 层级存储模式](https://github.com/kvcache-ai/Mooncake/pull/1899)

[8] [Mooncake Store 支持 master 不可用时的降级启动](https://github.com/kvcache-ai/Mooncake/pull/1930)

[9] [llama.cpp WebGPU 后端实现 async tensor 和 event API](https://github.com/ggml-org/llama.cpp/pull/22099)

[10] [llama.cpp SYCL 后端优化 mul_mat_id 内存并增加 BF16 fast path](https://github.com/ggml-org/llama.cpp/pull/22119)

[11] [SGLang CPU 后端增加 GPTQ / AWQ 4-bit 量化支持](https://github.com/sgl-project/sglang/pull/22685)

[12] [Megatron-LM 用 step batch size schedule 替代 rampup batch size](https://github.com/NVIDIA/Megatron-LM/pull/4411)

[13] [Megatron-LM 在 RL 中推迟 optimizer onload 以降低内存重叠](https://github.com/NVIDIA/Megatron-LM/pull/4235)

[14] [Ray Serve 拆分节点可用资源和 replica 请求资源](https://github.com/ray-project/ray/pull/62778)

[15] [Ray Serve 在 replica recovery 期间延迟 DEPLOYMENT_TARGETS 广播](https://github.com/ray-project/ray/pull/62751)
