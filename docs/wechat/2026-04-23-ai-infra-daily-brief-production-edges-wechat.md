# 今日焦点：生产边缘路径进入默认治理

**📅 2026-04-23**

> 多模态输入、KV/offload、异构后端和资源调度，都在从“能跑”走向“默认可治理”。

---

## 推理侧

**vLLM 增加 PyAV 视频解码后端用于并发多模态 serving[1]** - 这次变化解决的是长视频并发解码里的服务端瓶颈。OpenCV 路径容易在 `grab()` / `retrieve()` 上持有 GIL，让多请求解码变成串行；PyAV 路径用按帧 seek 和 slice threading，让并发请求可以继续推进。Video-MME 长视频测试里，请求吞吐提升约 1.5 到 3 倍，P99 TTFT 也明显下降。

**vLLM 支持为预抽帧视频输入传入原视频 metadata[2]** - 客户端把视频抽成帧之后，服务端过去会丢掉原视频的时间信息。现在 `frames_indices`、`total_num_frames`、`duration`、`fps` 可以随请求进入服务路径，模型更容易判断这些帧来自视频的哪个位置。多模态输入开始保留“视频是时间对象”这个上下文。

**vLLM 为 FunASR 增加 hotwords 支持[3]** 与 **llama.cpp 根据模型能力启用 parallel tool calls 默认行为[4]** - 前者把 ASR 热词纳入 OpenAI-compatible transcription 路径，后者把工具调用能力和结构化输出测试接入 server 默认行为。这组变化说明生产接口正在变厚：模型能力要可用，服务层必须一起承载音频热词、视频时间信息和工具调用边界。

---

## KV 与缓存

**SGLang 在 retraction 时 offload MambaPool 状态[5]** - Mamba-hybrid 模型不只有 attention KV cache，还有 conv / temporal buffer 这类 SSM 状态。过去 request retraction 只保存 KV，Mamba 状态会丢，继续生成就可能被污染。现在 SGLang 在 CPU offload 时同时保存 KV 与 Mamba state，混合模型的请求状态语义被补完整。

**SGLang 校验 MiMo HiCache geometry 并修正 MHA KV pool 的 v_head_dim[6]** - 如果模型的 `v_head_dim != head_dim`，旧路径会按错误尺寸分配 V-cache buffer，可能导致静默内存破坏或错误 attention 结果。新的校验把这类 cache geometry 问题提前暴露在启动阶段。

**Mooncake Store 支持 offload-on-evict 层级存储模式[7]** - 过去开启 SSD offload 后，每个 put 都会排队写盘，热 key 即使从不离开 DRAM 也要承担磁盘 I/O。新模式把写盘推迟到 eviction，形成 DRAM 到 LOCAL_DISK 的层级路径。热数据留在内存里，冷数据在水位压力下再落盘。

**Mooncake Store 支持 master 不可用时的降级启动[8]** - P2P client 启动时如果连不上 master，现在可以进入 DEGRADED 状态，跳过需要 master metadata 的 segment mount，等待后续恢复。缓存数据面不再因为控制面一时不可达就把整个应用卡死。

---

## 异构后端

**llama.cpp WebGPU 后端实现 async tensor 和 event API[9]** - 这条更新面向 Safari 和移动端这类 WebGPU 内存更紧的环境。异步 tensor API 用四个 1MB buffer 支持模型加载，wllama 内存占用可降低约 20% 到 25%。浏览器 GPU 的重点除了算子覆盖，还包括能不能在很窄的内存预算下完成加载。

**llama.cpp SYCL 后端优化 mul_mat_id 内存并增加 BF16 fast path[10]** - 这解决的是 Intel GPU / Level Zero 上大词表和 MoE 模型容易 OOM 的问题。BF16 权重不再强行 cast 成 F32，MoE staging buffer 也按实际 routed rows 分配，异构后端开始正面处理大模型结构带来的内存边界。

**SGLang CPU 后端增加 GPTQ / AWQ 4-bit 量化支持[11]** - CPU 路径开始支持 GPTQ / AWQ unpack、AMX 格式重排和 4-bit sgl-kernel。CPU 不会因此替代 GPU serving，但它正在从 fallback 变成有正式量化路径的 backend。

---

## 训练与调度

**Megatron-LM 用 step batch size schedule 替代 rampup batch size[12]** - 新参数可以按 step 或 token 阈值定义任意 batch size 阶梯，比线性 rampup 更适合大训练里的阶段性资源变化。训练系统开始把 batch size 调整写成更明确的计划，固定 rampup 的表达能力正在变得不够用。

**Megatron-LM 在 RL 中推迟 optimizer onload 以降低内存重叠[13]** - 这次调整避免 logprobs 计算和 optimizer 同时物化占用内存。RLHF / RLOO / GRPO 这类流程里，资源峰值常常来自多个阶段对象重叠，小的时序调整也会影响能跑多大的任务。

**Ray Serve 拆分节点可用资源和 replica 请求资源[14]** 与 **Ray Serve 在 replica recovery 期间延迟 DEPLOYMENT_TARGETS 广播[15]** - 前者让调度器区分“节点还有什么资源”和“副本请求什么资源”，后者避免 controller recovery 时把不完整 replica 集合广播给代理。Serve 调度正在把资源语义和恢复语义写得更清楚。

---

> 一句话结论：**AI Infra 的下一阶段竞争，来自把生产边缘条件写成默认可治理路径。**

---

## 参考

[1] vLLM 增加 PyAV 视频解码后端用于并发多模态 serving：https://github.com/vllm-project/vllm/pull/39986

[2] vLLM 支持为预抽帧视频输入传入原视频 metadata：https://github.com/vllm-project/vllm/pull/40133

[3] vLLM 为 FunASR 增加 hotwords 支持：https://github.com/vllm-project/vllm/pull/39674

[4] llama.cpp 根据模型能力启用 parallel tool calls 默认行为：https://github.com/ggml-org/llama.cpp/pull/22217

[5] SGLang 在 retraction 时 offload MambaPool 状态：https://github.com/sgl-project/sglang/pull/22493

[6] SGLang 校验 MiMo HiCache geometry 并修正 MHA KV pool 的 v_head_dim：https://github.com/sgl-project/sglang/pull/23173

[7] Mooncake Store 支持 offload-on-evict 层级存储模式：https://github.com/kvcache-ai/Mooncake/pull/1899

[8] Mooncake Store 支持 master 不可用时的降级启动：https://github.com/kvcache-ai/Mooncake/pull/1930

[9] llama.cpp WebGPU 后端实现 async tensor 和 event API：https://github.com/ggml-org/llama.cpp/pull/22099

[10] llama.cpp SYCL 后端优化 mul_mat_id 内存并增加 BF16 fast path：https://github.com/ggml-org/llama.cpp/pull/22119

[11] SGLang CPU 后端增加 GPTQ / AWQ 4-bit 量化支持：https://github.com/sgl-project/sglang/pull/22685

[12] Megatron-LM 用 step batch size schedule 替代 rampup batch size：https://github.com/NVIDIA/Megatron-LM/pull/4411

[13] Megatron-LM 在 RL 中推迟 optimizer onload 以降低内存重叠：https://github.com/NVIDIA/Megatron-LM/pull/4235

[14] Ray Serve 拆分节点可用资源和 replica 请求资源：https://github.com/ray-project/ray/pull/62778

[15] Ray Serve 在 replica recovery 期间延迟 DEPLOYMENT_TARGETS 广播：https://github.com/ray-project/ray/pull/62751
