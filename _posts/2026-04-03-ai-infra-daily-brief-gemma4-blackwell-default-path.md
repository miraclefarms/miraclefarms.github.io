---
title: AI Infra 早报｜Gemma 4 跨框架同步落地，Blackwell 推理默认路径加速收紧
date: 2026-04-03 05:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: Gemma 4 在 24 小时内同步进入 vLLM 与 llama.cpp，是近期最密集的一次跨框架型号协调上线。SGLang 将 TRT-LLM 稀疏注意力内核升格为 Blackwell DSA 预填充默认路径，Mooncake 落地无锁 P2P 路由缓存。TRT-LLM 修复分离式推理卡死与 KV 缓存计数错误，Megatron-LM 解决 FSDP 梯度双缓冲精度问题。OpenClaw v2026.4.2 恢复 Task Flow 基础设施并完成 Android 助理集成。
tags: [Inference, Attention, Multimodal, vLLM]
---

过去 24 小时，AI Infra 层有一个罕见的节奏现象：Google Gemma 4 在同一天进入了 vLLM 和 llama.cpp 的主干，两个框架各自的主支持 PR 当天均被一条 bugfix PR 紧随跟进。这种"型号同步上线"的模式意味着主要推理框架的型号追踪机制已经足够成熟，能够以近乎协调发布的方式处理新架构的首日支持。

与此同时，SGLang 将 TRT-LLM 的稀疏注意力内核从"可选接入"升格为 Blackwell 平台 DSA 预填充路径的默认选项，完成了一次从引用到依赖的关键跃升——对生产用户而言，不再需要任何配置，即可自动获得 Blackwell 优化路径。

## 一、推理侧：Gemma 4 协调落地，Blackwell 默认路径收紧

### Gemma 4 同日进入 vLLM 与 llama.cpp

**vLLM PR #38826** 是这一批变化中改动最大的单个 PR：26 个文件，覆盖 Gemma 4 的稀疏 MoE 路由（共享专家 + 路由专家并行）、SigLIP 变体视觉编码器、推理模式（思维链采样）以及工具调用（专用 Gemma4ToolParser）[[1]](https://github.com/vllm-project/vllm/pull/38826)。几小时后，PR #38847 修复了 ToolParser 初始化时缺少 `tools` 参数的问题[[1]](https://github.com/vllm-project/vllm/pull/38826)——当天追修，说明测试覆盖在新架构的工具调用路径上尚有盲区。

llama.cpp 通过 #21309 合入了 Gemma 4 的视觉 + MoE 支持（当前不含音频），随即以 #21326 跟进模板解析修复[[2]](https://github.com/ggml-org/llama.cpp/pull/21309)。当天 llama.cpp 连发五个构建（b8634 至 b8639），其中 b8639 附带了 ggml-webgpu 的矢量化 Flash Attention 内核，WebGPU 后端的注意力计算效率将有可见提升。

两个框架在同一日内完成型号主支持并进入 bugfix 节奏——这种协调节奏在此前的型号跟进中较为少见。

### SGLang：TRT-LLM 内核升格为默认，FlashMLA 最新版恢复

SGLang PR #21914 将 TRT-LLM 的稀疏注意力内核设为 Blackwell 平台 DSA（Dynamic Sparse Attention）预填充批次的默认内核[[3]](https://github.com/sgl-project/sglang/pull/21914)。这是一次从"支持"到"默认"的质变：在此之前，用户需要显式配置才能使用 TRT-LLM 的 Blackwell 优化路径；从今天起，凡是在 H200/B200 上运行 SGLang 的部署，无需修改任何配置即可自动走 TRT-LLM 稀疏注意力路径。

PR #21922 撤销了此前对 FlashMLA 的版本回滚，恢复了最新版 FlashMLA，表明触发降级的兼容性问题已经解决[[4]](https://github.com/sgl-project/sglang/pull/21922)。PR #21920 将 ngram 投机解码语料库从 torch cpp_extension 迁移到 TVM FFI JIT 内核，减少外部编译依赖，加快冷启动时间[[5]](https://github.com/sgl-project/sglang/pull/21920)。

### llama.cpp：SWA KV 精度修复，SYCL 大缓存死锁解除

PR #21277 是今天 llama.cpp 中另一个有实际影响的修复[[6]](https://github.com/ggml-org/llama.cpp/pull/21277)：滑动窗口注意力（SWA）的 KV 缓存此前会被统一量化，但 SWA 层对量化误差的容忍度明显低于全局注意力层，强制量化会导致 PPL 提升。本次明确将 SWA KV 缓存排除在量化路径之外，Gemma 系列和部分使用 SWA 的 Qwen 变体将在精度上有可见改善。

PR #21283 修复了 SYCL 后端在 KV 缓存超过约 5 GB 时发生死锁的问题[[7]](https://github.com/ggml-org/llama.cpp/pull/21283)——在消费级 GPU 上运行 32B 以上模型时此问题容易被触发，影响面较广。

### vLLM：Flash Attention 4 同步，DeepSeek V3.2 Indexer 融合提速

PR #38690 将 Flash Attention 4 更新至最新上游版本，延续对新一代注意力内核的持续跟踪[[8]](https://github.com/vllm-project/vllm/pull/38690)。PR #38684 为 DeepSeek V3.2 的 Indexer 层引入融合权重投影，通过一次矩阵乘替代此前两次独立投影，在 MoE prefill 阶段有明显吞吐提升[[9]](https://github.com/vllm-project/vllm/pull/38684)。CPU MoE 路径新增 GELU 激活支持（#38770），扩展了 vLLM CPU 推理对 Gemma/PaLM 等模型的兼容性。

## 二、训练侧：Megatron FSDP 梯度精度修复，TRL 扩展仿真场景

### Megatron-LM：FSDP 梯度 reduce 双缓冲不足

PR #4054 修复了一个沉默但影响实际训练精度的问题[[10]](https://github.com/NVIDIA/Megatron-LM/pull/4054)：在 Megatron-FSDP 的多流并行梯度 reduce 场景下，双缓冲的分配数量不足以覆盖所有并发流，两个 reduce 操作会共享同一缓冲区，导致一个操作的中间结果被另一操作的数据部分覆盖。这类错误不会产生显式报错，而是表现为训练损失的非确定性波动，在 FP8/BF16 混合精度场景下尤为难以诊断。修复方法是根据活跃并发流数量动态调整双缓冲分配数量。PR #4063 将 Megatron-LM 的 chat completions 推理端点格式与 vLLM 对齐，降低框架切换的适配成本。

### TRL v1.0 后首个多模态 RL 仿真示例

TRL PR #5437 新增了一个 CARLA 自动驾驶仿真器 + VLM 在线强化学习训练的完整示例[[11]](https://github.com/huggingface/trl/pull/5437)，展示如何以视觉语言模型作为策略网络并通过 GRPO 在仿真场景中完成训练。这是三天前 TRL v1.0.0 发布后，官方示例库中首个在真实仿真环境中演示多模态 RL 训练的条目。PR #5423 修复了 vllm-0.10.2 及以上版本在 OnlineDPO 和 OpenEnv 训练器中的 ImportError[[12]](https://github.com/huggingface/trl/pull/5423)。

## 三、生产部署侧：TRT-LLM 推理稳定性收紧，Mooncake 传输层优化

### TRT-LLM：AutoDeploy 分页缓存，分离式推理多项关键修复

PR #12642 为 TRT-LLM 的 AutoDeploy 路径接入 Triton paged attention[[13]](https://github.com/NVIDIA/TensorRT-LLM/pull/12642)，使快速部署模式也能享受分页 KV 缓存的内存管理能力（此前 AutoDeploy 只能使用非分页缓存，不适合长上下文生产场景）。

分离式推理方向有两项重要修复。PR #12667 修复了块驱逐后复用时 decode 节点永久挂起的问题[[14]](https://github.com/NVIDIA/TensorRT-LLM/pull/12667)——表现为 prefill 完成后 decode 侧请求永不继续，在高内存压力下持续驱逐的集群中容易触发。PR #12682 修复了 context chunking 与 KV 缓存复用共同使用时的 token 计数错误，该错误会使 KV 占用率统计失真，进而错误触发驱逐[[15]](https://github.com/NVIDIA/TensorRT-LLM/pull/12682)。

PR #12679 回滚了 Blackwell SageAttention 特性，表明该特性的稳定性验证尚未完成，Blackwell 优化的推进仍在分步落地。

### Mooncake：无锁路由缓存，HA 生命周期统一

PR #1793 为 Mooncake Store 引入无锁 P2P 路由缓存[[16]](https://github.com/kvcache-ai/Mooncake/pull/1793)：通过原子操作替代互斥锁实现路由表的并发读取，在大规模集群高并发 KV 传输场景下减少路由查找的锁争用。PR #1777 将高可用运行时拆分为独立子系统，统一 standby 节点的生命周期管理，使主备切换路径更加确定[[17]](https://github.com/kvcache-ai/Mooncake/pull/1777)。

SGLang 通过昨日合入的 PR #21844 集成了 Mooncake v0.3.10.post1，上述改进已在 SGLang 侧生效，P2P KV 传输优化对 SGLang 用户透明可用。

### Ray 2.55.0 准备，worker 崩溃可观测性增强

PR #62307 将 Ray 主干版本号推进至 2.55.0，下一个 minor release 准备窗口开启（上一个 Ray-2.54.1 发布于 3 月 25 日）[[18]](https://github.com/ray-project/ray/pull/62307)。PR #62297 新增了 worker 意外失败指标和对应的 dashboard 面板[[19]](https://github.com/ray-project/ray/pull/62297)——此前此类失败只能通过日志追查，现在可以在 Ray dashboard 中直接看到异常崩溃的统计趋势，便于生产环境快速定位 worker 稳定性问题。

## 四、应用侧：OpenClaw v2026.4.2 三层扩展

OpenClaw 发布 v2026.4.2，变化集中在三个方向[[20]](https://github.com/openclaw/openclaw/releases/tag/v2026.4.2)。

**Task Flow 基础设施完整回归。** 这一版本恢复了包含 managed/mirrored 同步模式、持久化流状态和版本追踪的完整 Task Flow 底层，`openclaw flows` 命令支持对后台任务编排进行检查和恢复[[21]](https://github.com/openclaw/openclaw/pull/59805)。子任务管理新增 sticky cancel intent：外部编排器可以立即停止调度，等待活跃子任务自然收尾后父 Task Flow 才会进入 `cancelled` 状态，避免强制终止。Plugin SDK 新增 `api.runtime.taskFlow` seam，插件层可以直接创建并驱动 managed Task Flow。

**Android 系统级助理集成。** PR #59721 新增助理角色入口和 Google Assistant App Actions 元数据，Android 用户可以通过系统语音助理触发词直接打开 OpenClaw 并将提示词传入聊天界面[[22]](https://github.com/openclaw/openclaw/pull/59721)。这是 OpenClaw 首次实现系统级助理集成，将 AI 代理的触发入口从 app 内部延伸到操作系统层。

**插件解耦完成两条 Breaking 迁移。** xAI 的 `x_search` 配置和 Firecrawl 的 `web_fetch` 配置均从 core 迁移到插件自有路径，现有配置可通过 `openclaw doctor --fix` 自动迁移。Exec 路径默认行为调整为 `security=full, ask=off`（YOLO 模式），简化了 gateway/node 主机侧的执行审批流程；新增 `before_agent_reply` 插件钩子，允许插件在 LLM 前短路合成回复。Feishu Drive 评论事件流同步上线，支持文档评论的线程化协作工作流。

---

今天没有单一的里程碑发布，但信号密度较高。Gemma 4 的协调上线说明型号追踪机制已进入协调发布阶段；SGLang 的 Blackwell 默认路径切换说明平台优化正在从"可选"走向"透明"；TRT-LLM 多个分离式推理 bugfix 的同日落地说明该功能正在进入稳定化阶段。Megatron-LM 的 FSDP 梯度双缓冲 bug 小而重要——它不会显式报错，只会让混合精度训练悄悄地"错"。OpenClaw 的 Task Flow 回归加上 Android 助理集成，是向下扩展到操作系统层的具体一步。

## 参考来源

[1] [vLLM #38826 Gemma 4 完整架构支持（MoE、多模态、推理、工具调用）](https://github.com/vllm-project/vllm/pull/38826)

[2] [llama.cpp #21309 Gemma 4 视觉 + MoE 支持](https://github.com/ggml-org/llama.cpp/pull/21309)

[3] [SGLang #21914 TRT-LLM 内核设为 Blackwell DSA 默认路径](https://github.com/sgl-project/sglang/pull/21914)

[4] [SGLang #21922 撤销 FlashMLA 旧版本回滚](https://github.com/sgl-project/sglang/pull/21922)

[5] [SGLang #21920 ngram 语料库迁移至 TVM FFI JIT](https://github.com/sgl-project/sglang/pull/21920)

[6] [llama.cpp #21277 SWA KV 缓存禁止量化](https://github.com/ggml-org/llama.cpp/pull/21277)

[7] [llama.cpp #21283 SYCL 后端 5 GB KV 缓存卡死修复](https://github.com/ggml-org/llama.cpp/pull/21283)

[8] [vLLM #38690 Flash Attention 4 上游同步](https://github.com/vllm-project/vllm/pull/38690)

[9] [vLLM #38684 DeepSeek V3.2 Indexer 融合权重投影](https://github.com/vllm-project/vllm/pull/38684)

[10] [Megatron-LM #4054 FSDP 梯度 reduce 双缓冲不足修复](https://github.com/NVIDIA/Megatron-LM/pull/4054)

[11] [TRL #5437 CARLA VLM 强化学习训练示例](https://github.com/huggingface/trl/pull/5437)

[12] [TRL #5423 vllm-0.10.2 兼容性修复](https://github.com/huggingface/trl/pull/5423)

[13] [TRT-LLM #12642 AutoDeploy 接入 Triton paged attention](https://github.com/NVIDIA/TensorRT-LLM/pull/12642)

[14] [TRT-LLM #12667 分离式推理块复用后卡死修复](https://github.com/NVIDIA/TensorRT-LLM/pull/12667)

[15] [TRT-LLM #12682 KV 缓存复用 + context chunking token 计数修复](https://github.com/NVIDIA/TensorRT-LLM/pull/12682)

[16] [Mooncake #1793 无锁 P2P 路由缓存](https://github.com/kvcache-ai/Mooncake/pull/1793)

[17] [Mooncake #1777 HA 运行时拆分与 standby 生命周期统一](https://github.com/kvcache-ai/Mooncake/pull/1777)

[18] [Ray #62307 2.55.0 版本号推进](https://github.com/ray-project/ray/pull/62307)

[19] [Ray #62297 worker 异常崩溃指标与 dashboard 面板](https://github.com/ray-project/ray/pull/62297)

[20] [OpenClaw v2026.4.2 release](https://github.com/openclaw/openclaw/releases/tag/v2026.4.2)

[21] [OpenClaw #59805 task domain runtime surfaces](https://github.com/openclaw/openclaw/pull/59805)

[22] [OpenClaw #59721 Android 助理自动发送提示词](https://github.com/openclaw/openclaw/pull/59721)
