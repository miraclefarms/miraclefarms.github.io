---
title: AI Infra 早报｜vLLM v0.21.0 划定 C++20 硬门槛，DeepSeek V4 跨框架进入量产优化期
date: 2026-05-15 08:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: vLLM v0.21.0 正式发布，367 个 commit 划定 C++20 构建门槛并正式弃用 transformers v4；同日 DeepSeek V4 MegaMoE 在 SGLang、TRT-LLM、LMCache 三框架协同推进，MLA kernel 在 Blackwell 上展开新一轮性能竞速。
tags: [vLLM, MoE, Attention, Disaggregation, Inference]
---

vLLM v0.21.0 今天正式发布——367 个 commit、202 位贡献者，49 位是第一次出现在 release note 里。一个版本不足以让推理框架的世界发生质变，但 v0.21.0 做了两件事：把 C++20 写成硬门槛，同时正式告别 transformers v4。前者意味着下游部署环境的编译器矩阵需要整体升级；后者则是 CI/CD 流水线和自定义模型加载逻辑都需要适配的信号。同一天，DeepSeek V4 的 MegaMoE 路径在三个框架里各自迈进——如果说五月上旬是"DSV4 能跑了"，今天的故事是"DSV4 怎么跑得更快"。

## 一、vLLM v0.21.0：C++20 硬门槛，Model Runner V2 成为默认

v0.21.0 的版本升级不只是在 release page 放一行 changelog。C++20 的编译器要求意味着所有依赖 C++ 编译构建的插件和自定义算子都需要更新工具链[[1]](https://github.com/vllm-project/vllm/releases/tag/v0.21.0)；transforms v4 的正式弃用把迁移压力从 vLLM 核心团队转移到生态下游，所有依赖 v4 的模型加载逻辑、自定义 processor 和训练后量化工具都必须在 v0.22 之前完成 v5 适配。

工程层面最值得关注的是 Model Runner V2。Oracle 机制已上线：未设置环境变量时自动使用 V2 的 dense model 路径，Qwen3-0.6B 和 OPT-125M 作为验证基线[[2]](https://github.com/vllm-project/vllm/pull/39337)。配合 CudaGraph capture 修复——DeepSeek V4 Flash 在 full CUDA graph 模式下曾因 lazy attention state 初始化时机错误输出乱码[[3]](https://github.com/vllm-project/vllm/pull/42444)——V2 正在从实验分支走向默认路径。DP 负载均衡方面，#41626 把请求计数发布时机提前到每个 engine step 开头，**反馈延迟的缩短让数据并行调度器的决策粒度从 step 级优化到 step 内部**[[4]](https://github.com/vllm-project/vllm/pull/41626)。

v0.21.0 还带来一项架构级重构：用 `ModelRunnerOutput` 传输替代 shared memory + fcntl.flock 的 routed experts 跨进程搬运[[5]](https://github.com/vllm-project/vllm/pull/39568)。旧的方案把 worker 绑死在 `cpu().numpy()` 的每步拷贝上；新方案引入 HTTP 传输支持，为分布式环境里的容错和数据面分离铺了一层地基。

## 二、DeepSeek V4 MegaMoE 效率战：三框架同日推进，MLA 在 Blackwell 上开跑

SGLang 合入 DeepSeek V4 w4a4 MegaMoE，精度损失的补偿策略已经在 MegaMoE 框架层面处理完毕[[6]](https://github.com/sgl-project/sglang/pull/25052)。TRT-LLM 几乎同时启用 `MEGAMOE_DEEPGEMM` 后端并绑定 CI 测试[[7]](https://github.com/NVIDIA/TensorRT-LLM/pull/14129)，LMCache 也完成了 DSV4 的多进程连接器适配[[8]](https://github.com/LMCache/LMCache/pull/3171)。**三框架对 MegaMoE 的同日推进说明一个事实：DSV4 推理效率的核心杠杆已经不在"能不能支持"上，而在 MoE 路由与通信的 kernel 层优化。**

kernel 层的竞速同样密集。vLLM 接入 `tokenspeed_mla` CuTe DSL kernel 作为新 MLA backend，覆盖 DSR1 和 Kimi K25 的 prefill 与 decode，目标平台 Blackwell (SM100)[[9]](https://github.com/vllm-project/vllm/pull/41778)。SGLang 侧则用 TMA bulk-store 重写了 MLA paged KV 的 scatter-write 路径，在较小 batch 下最高 12 倍加速[[10]](https://github.com/sgl-project/sglang/pull/25311)——原本的 1D Triton kernel 在 `BLOCK=128` 和 `grid=(n_loc, ceil(total_dim/BLOCK))` 下会退化，TMA 的硬件批量写大幅缓解了这个问题。TRT-LLM 的 CuTe DSL FP4 paged MQA logits decode kernel 则是 NVFP4 路径上的 kernel 级补充[[11]](https://github.com/NVIDIA/TensorRT-LLM/pull/13929)。

## 三、Serving 层标准化：API 兼容、配置可扩展、KV 缓存盐值

推理框架的 serving 层今天在做三件事：统一 API 语义、让配置变成可插拔的、给 KV 缓存加身份标签。

llama.cpp 合入了 `continue_final_message` 标志，实现与 vLLM 和 transformers API 的行为兼容[[12]](https://github.com/ggml-org/llama.cpp/pull/23012)。SGLang 引入 `--model-config-parser` 注册表，**外部包可以通过注册机制插入自定义 config-loading 路径**，无需修改 SGLang 核心代码[[13]](https://github.com/sgl-project/sglang/pull/25050)；同时 `SGLANG_OPT_FP8_WO_A_GEMM` 从 `False` 翻为 `True` 的默认值[[14]](https://github.com/sgl-project/sglang/pull/25181)，意味着 FP8 weight-only 的激活量化加速现在默认开启。TRT-LLM 在 KV cache v2 管理器中补上了 `cache_salt_id` 支持，**同一模型实例可以根据请求携带的 salt 区分不同的 KV 缓存片段**——这对多租户共享 KV pool 的场景有直接工程意义[[15]](https://github.com/NVIDIA/TensorRT-LLM/pull/13793)。

多模态模型推理也在本周加速就位。TRT-LLM 正式合入 Kimi K2.5 多模态视觉支持，包括图像和视频预处理管线[[16]](https://github.com/NVIDIA/TensorRT-LLM/pull/12788)，GLM-5 的 MoE router GEMM 示例化补齐了 6144 hidden_dim 的编译期路径[[17]](https://github.com/NVIDIA/TensorRT-LLM/pull/13740)。vLLM 的 PD disaggregation 在 NIXL connector 层面补上了 Qwen3.5 的 GDN（Gated Delta Net）conv-state 布局支持[[18]](https://github.com/vllm-project/vllm/pull/41869)——**disaggregation 不再只服务 standard attention 模型，Mamba/hybrid 架构也开始进入 P/D 分离的视野。**

## 今天真正值得记住的判断

vLLM v0.21.0 的 C++20 和 transforms v5 不是"版本号升级需要跑一遍 CI"那么简单——它是推理框架基础设施层的一次代际升级，所有依赖 C++ 编译的生态组件都必须跟随。DSV4 MegaMoE 三框架同日推进则是另一维度：一周前 DSV4 推理的关键词是"能跑通"，今天的信号是"量产优化已经启动，MegaMoE 和 MLA kernel 是主要竞争面"。

---

## 参考来源

[1] [vLLM v0.21.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.21.0)

[2] [vLLM #39337: Model Runner V2 oracle for dense models](https://github.com/vllm-project/vllm/pull/39337)

[3] [vLLM #42444: Fix lazy attention state initialization during cudagraph capture](https://github.com/vllm-project/vllm/pull/42444)

[4] [vLLM #41626: Publish request counts at start of each engine step](https://github.com/vllm-project/vllm/pull/41626)

[5] [vLLM #39568: Replace shared-memory routed experts with ModelRunnerOutput transfer](https://github.com/vllm-project/vllm/pull/39568)

[6] [SGLang #25052: DeepSeek V4 w4a4 MegaMoE](https://github.com/sgl-project/sglang/pull/25052)

[7] [TRT-LLM #14129: Enable MEGAMOE_DEEPGEMM backend for DeepSeek V4](https://github.com/NVIDIA/TensorRT-LLM/pull/14129)

[8] [LMCache #3171: Support DeepSeek V4](https://github.com/LMCache/LMCache/pull/3171)

[9] [vLLM #41778: Add TOKENSPEED_MLA backend for Blackwell prefill+decode](https://github.com/vllm-project/vllm/pull/41778)

[10] [SGLang #25311: MLA TMA bulk-store up to 12x](https://github.com/sgl-project/sglang/pull/25311)

[11] [TRT-LLM #13929: CuTe DSL FP4 paged MQA logits decode kernel](https://github.com/NVIDIA/TensorRT-LLM/pull/13929)

[12] [llama.cpp #23012: Accept continue_final_message flag for vLLM API compat](https://github.com/ggml-org/llama.cpp/pull/23012)

[13] [SGLang #25050: Add model-config-parser registry](https://github.com/sgl-project/sglang/pull/25050)

[14] [SGLang #25181: Enable FP8 WO A GEMM by default](https://github.com/sgl-project/sglang/pull/25181)

[15] [TRT-LLM #13793: Support cache_salt_id in KV cache v2](https://github.com/NVIDIA/TensorRT-LLM/pull/13793)

[16] [TRT-LLM #12788: Kimi K2.5 multimodal vision support](https://github.com/NVIDIA/TensorRT-LLM/pull/12788)

[17] [TRT-LLM #13740: Add hidden_dim=6144 router GEMM for GLM-5](https://github.com/NVIDIA/TensorRT-LLM/pull/13740)

[18] [vLLM #41869: PD disagg with NIXL Connector GDN support](https://github.com/vllm-project/vllm/pull/41869)
