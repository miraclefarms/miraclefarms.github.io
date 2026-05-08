---
title: AI Infra 早报｜MoE 弹性容错进入默认路径，推理栈可靠性演进加速
date: 2026-03-29 05:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: SGLang v0.5.10rc0 将弹性 Expert Parallelism 正式纳入 release 候选，MoE 部署首次获得无需全量重启的 GPU 故障容忍能力；vLLM 与 SGLang 同日完成 ROCm 7.2.1 全栈升级，AMD 路径加速追平 CUDA 构建节奏；llama.cpp 持续完善 reasoning 格式控制与 WebUI 交互体验；LMCache 新增 Blackwell SM120 轮包与 bench CLI；OpenClaw 以 20 个 PR 收口渠道稳定性与安全边界。
tags: [MoE, Inference, vLLM, SGLang]
---

今天的主线是**可靠性的演进**——不是某个单点的突破，而是分散在多个项目里同步推进的稳定化动作，在大规模生产部署的前夜形成合力。

最显眼的是 SGLang v0.5.10rc0，它把弹性 Expert Parallelism（Elastic EP）正式推进到 release candidate。这个特性的意义在于：过去 MoE 集群里一块 GPU 挂了，服务要停下来全量重启重新分配；现在系统可以自动把故障节点上的 expert weights 重新路由，服务不间断地继续响应。这不是小修补——它是 MoE 部署可靠性的一次实质性跃升。与此同时，Piecewise CUDA Graph Capture 也在这个版本转为默认开启，对具有复杂控制流的模型在内存和吞吐上都有改善。

AMD 方向的动作同样值得关注。vLLM 在昨天完成了 ROCm 7.2.1 + torch 2.10 + triton 3.6 的全栈升级，SGLang 也在同日补入了 AMD MI35x 平台的 GLM-4.7-FP8 精度 CI 覆盖。两个主流推理框架在同一天完成 AMD 路径的里程碑更新，说明 ROCm 生态的成熟度正在进入一个新阶段——不再是"能跑"，而是开始追求"稳定跑"，并建立质量保障体系。

## 一、SGLang v0.5.10rc0 的多线并进

除了弹性 EP 和 CUDA Graph 默认化，这个 RC 版本还带入了几条值得单独说的更新。

HiSparse 稀疏注意力后端的集成[[1]](https://github.com/sgl-project/sglang/releases/tag/v0.5.10rc0)是面向超长上下文的：对于大部分 token 之间注意力权重接近零的场景，稀疏感知计算可以大幅跳过无效计算量。这类方法在学术界已经验证多时，落地到 SGLang 主分支意味着离生产可用又近了一步。

FlashInfer MXFP8 内核（microscaling FP8）的引入则是精度-吞吐权衡的新选项。相比标准 FP8，microscaling 方案通过更细粒度的缩放因子来控制量化误差，特别是在 MoE 路由后的 expert GEMM 上，精度损失更可控[[1]](https://github.com/sgl-project/sglang/releases/tag/v0.5.10rc0)。

transformers 版本从 4.57.1 跳到 5.3.0 不是日常小升级——这里面有架构层面的变化，GLM-5 从此不再需要维护独立的定制镜像[[1]](https://github.com/sgl-project/sglang/releases/tag/v0.5.10rc0)。配套的 prefill KV cache fused Triton kernel 和可配置 MLA attention 稀疏阈值，是针对 DeepSeek V3.2 和 GLM-5 这类长上下文模型的专项优化。

扩散模型方向也在持续推进：overlay model materialization 功能合入[[4]](https://github.com/sgl-project/sglang/pull/21600)，加上前日的内核融合优化系列，SGLang-Diffusion 的深度和完整性正在快速追上语言模型路径。

## 二、llama.cpp：边缘侧的细节打磨

llama.cpp 昨天 5 个 build release（b8570～b8574），看起来平静，实际上有几处改动值得关注。

`reasoning_format=none` 选项[[2]](https://github.com/ggml-org/llama.cpp/pull/21094)让用户可以对 gpt-oss backend 完全绕过 reasoning wrapper 的包装，输出格式控制更灵活。结合前日已合入的 `reasoning_content` 多轮保留修复，llama.cpp 在推理链格式这一层的能力正在逐步闭环。

Vulkan 后端新增非连续 GLU 支持[[3]](https://github.com/ggml-org/llama.cpp/pull/21081)，对在 AMD GPU、Intel GPU 或其他非 CUDA 平台上跑现代 LLM 架构的用户有直接意义——GLU 是 Llama 系列、Mistral 系列等模型的标准组件，此前在 Vulkan 路径下有覆盖空白。

WebUI 层新增了对话分支功能[[14]](https://github.com/ggml-org/llama.cpp/pull/21021)：用户可以在对话历史的任意节点另起一条分支，而不必回到头重新开始。这在调试 prompt 工程和探索不同推理路径时很有用，llama.cpp 的本地 WebUI 体验在继续向专业工具方向演进。

## 三、LMCache：Blackwell 准备就绪，工具链补齐

LMCache 这个窗口有三件相对独立的事情同步落地。

SM120（NVIDIA Blackwell 架构）的 wheel 构建支持[[12]](https://github.com/LMCache/LMCache/pull/2873)是时效性很强的一步：Blackwell GPU 的部署窗口已经打开，KV cache 管理层对新架构的支持不应该成为瓶颈。

`lmcache bench engine` CLI[[13]](https://github.com/LMCache/LMCache/pull/2889)提供了标准化的 KV cache 性能测量工具。在此之前，不同团队可能各自用不同方式测量 KV cache 吞吐和延迟，数字难以横向比较；有了标准 CLI 之后，性能退化和优化效果都有了统一的衡量基准。

GPU-direct Storage（GDS）方向，cuFile 文件系统的并行 I/O 线程池[[https://github.com/LMCache/LMCache/pull/2802]](https://github.com/LMCache/LMCache/pull/2802)正式合入，KV 快照的落盘速度将有实质提升，对长前缀缓存和 checkpoint 恢复场景影响明显。

## 四、OpenClaw：20 个 PR 的稳定化周期

OpenClaw 昨天合了 20 个 PR，密度较高，基本都是 bugfix 和安全加固，没有新特性。这种"密集修复日"在开源项目里通常意味着一件事：在某个里程碑版本发布后，团队在集中处理积压的问题单。

安全方向最值得关注的是 HTTP OpenAI 兼容路由缺少 `operator.write` 权限校验的修复[[18]](https://github.com/openclaw/openclaw/pull/56618)——这是个权限绕过漏洞，影响通过 HTTP 路由访问 OpenClaw 的场景。web search key 的安全审计[[23]](https://github.com/openclaw/openclaw/pull/56540)也在同日完成，确保密钥访问控制覆盖所有内置 provider。

稳定性方向，ACP stale binding 的修复[[20]](https://github.com/openclaw/openclaw/pull/56476)解决了 ACP runtime 退出后子 agent 消息路由失效的问题；Gemini 3.1 模型解析[[21]](https://github.com/openclaw/openclaw/pull/56567)和 Telegram 路径的三处修复[[22]](https://github.com/openclaw/openclaw/pull/56595)都直接影响日常使用体验。NO_REPLY 的 JSON 包裹问题[[19]](https://github.com/openclaw/openclaw/pull/56612)被抑制前，使用 TTS 工具后偶尔会看到 `{"reply": "NO_REPLY"}` 这样的原始 JSON 出现在 channel 里，现在这个漏洞已经补上。

## 结语

今天没有发布会，没有新的 benchmark 世界纪录，但有 SGLang MoE 弹性容错走向生产候选，有两个主流框架同日完成 AMD 路径升级，有 LMCache 为 Blackwell 时代备好工具链。这种扎实推进的节奏，是 AI Infra 这个阶段最需要的——性能的天花板固然重要，但让已有能力在生产中真正可靠地运转，才是下一波大规模部署的前提。

## 参考来源

[1] [SGLang v0.5.10rc0 发布：弹性 EP + Piecewise CUDA Graph 默认化](https://github.com/sgl-project/sglang/releases/tag/v0.5.10rc0)

[2] [llama.cpp reasoning_format=none 支持](https://github.com/ggml-org/llama.cpp/pull/21094)

[3] [llama.cpp Vulkan noncontiguous GLU 支持](https://github.com/ggml-org/llama.cpp/pull/21081)

[4] [SGLang 扩散模型 overlay model materialization](https://github.com/sgl-project/sglang/pull/21600)

[5] [SGLang VLM ShmPointerMMData 多进程安全优化](https://github.com/sgl-project/sglang/pull/21465)

[6] [SGLang LoRA 自动检测目标模块](https://github.com/sgl-project/sglang/pull/21439)

[7] [DeepSpeed superoffload 多组更新 CPU buffer 修复](https://github.com/deepspeedai/DeepSpeed/pull/7906)

[8] [TRL GenerationConfig disable_config 迁移](https://github.com/huggingface/trl/pull/5384)

[9] [vLLM ROCm 7.2.1 + torch 2.10 + triton 3.6 构建升级](https://github.com/vllm-project/vllm/pull/38252)

[10] [vLLM ROCm 7.2.1 variant 正式切换](https://github.com/vllm-project/vllm/pull/38413)

[11] [SGLang AMD MI35x GLM-4.7-FP8 CI](https://github.com/sgl-project/sglang/pull/21534)

[12] [LMCache SM120 wheel 构建支持](https://github.com/LMCache/LMCache/pull/2873)

[13] [LMCache bench engine CLI](https://github.com/LMCache/LMCache/pull/2889)

[14] [llama.cpp WebUI 对话分支功能](https://github.com/ggml-org/llama.cpp/pull/21021)

[15] [llama.cpp CLI /glob 命令](https://github.com/ggml-org/llama.cpp/pull/21084)

[16] [Ray Serve 日志吞吐优化](https://github.com/ray-project/ray/pull/62146)

[17] [Ray Serve Gang 自动伸缩震荡修复](https://github.com/ray-project/ray/pull/62128)

[18] [OpenClaw HTTP API write scope 安全修复](https://github.com/openclaw/openclaw/pull/56618)

[19] [OpenClaw NO_REPLY JSON 包裹抑制修复](https://github.com/openclaw/openclaw/pull/56612)

[20] [OpenClaw ACP stale binding 修复](https://github.com/openclaw/openclaw/pull/56476)

[21] [OpenClaw Gemini 3.1 模型解析修复](https://github.com/openclaw/openclaw/pull/56567)

[22] [OpenClaw Telegram 消息长度分割与 reply 稳定性](https://github.com/openclaw/openclaw/pull/56595)

[23] [OpenClaw web search keys 安全审计](https://github.com/openclaw/openclaw/pull/56540)
