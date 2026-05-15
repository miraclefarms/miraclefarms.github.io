---
title: 今日焦点：vLLM v0.21.0 发布，DeepSeek V4 MegaMoE 量产化跨框架推进
wechat_variant: brief
intro: vLLM v0.21.0 划定 C++20 硬门槛、弃用 transformers v4；DeepSeek V4 MegaMoE 三框架协同推进，MLA kernel 在 Blackwell 上展开新一轮竞速。
---

**📅 2026-05-15**

![题图](assets/2026-05-15/ai-infra-daily-brief-cover.jpg)

> 新模型上线窗口正在压缩——DSV4 跑通不到两周，SGLang、TRT-LLM、LMCache 已同步把 MegaMoE 推到就绪状态。vLLM v0.21.0 的 C++20 和 transformers v5 不是版本号升级，是所有下游部署必须面对的基础设施换代。

---

## 推理侧

**vLLM v0.21.0 正式发布，367 个 commit 划定 C++20 硬门槛**[1] - transformers v4 被正式弃用，C++20 编译器成为构建前提，所有依赖 C++ 编译的插件和自定义算子都必须升级工具链。版本还对 Model Runner V2 启用了 Oracle 机制——未设置环境变量时自动走 V2 dense model 路径，Qwen3-0.6B 和 OPT-125M 作为验证基线[2]。配合 DeepSeek V4 Flash 在 full CUDA graph 下的 lazy attention state 初始化修复[3]，V2 正从实验分支走向默认。

**DP 负载均衡决策粒度细化**[4] - 请求计数发布从每步之后提前到 engine step 开头，数据并行调度器反馈延迟缩短，决策粒度从 step 级进到 step 内部。

**routed experts 跨进程传输从共享内存升级为 ModelRunnerOutput**[5] - 旧的 shared memory + fcntl.flock 方案把 worker 绑死在每步 `cpu().numpy()` 拷贝上；新方案引入 HTTP 传输，为分布式环境数据面分离铺了地基。

**DeepSeek V4 w4a4 MegaMoE 三框架同日推进**[6][7][8] - SGLang 合入 DSV4 w4a4 MegaMoE 并处理完精度补偿策略，TRT-LLM 启用 MEGAMOE_DEEPGEMM 后端并绑定 CI，LMCache 完成多进程连接器适配。DSV4 推理效率的核心杠杆已从"能不能跑"转向 MoE 路由与通信的 kernel 优化。

**MLA kernel 在 Blackwell 上展开新一轮竞速**[9][10][11] - vLLM 接入 tokenspeed_mla CuTe DSL kernel，覆盖 DSR1 和 Kimi K25 的 prefill 与 decode，目标平台 SM100。SGLang 用 TMA bulk-store 重写 MLA paged KV scatter-write，较小 batch 下最高 12 倍加速。TRT-LLM 则补齐了 CuTe DSL FP4 paged MQA logits decode kernel。

---

## 生产部署侧

**Serving 层 API 标准化**[12] - llama.cpp 合入 continue_final_message 标志，实现与 vLLM 和 transformers API 的行为兼容。SGLang 引入 model-config-parser 注册表，外部包可通过注册机制插入自定义 config-loading 路径，无需修改核心代码[13]。

**FP8 weight-only 激活量化默认开启**[14] - SGLang 将 SGLANG_OPT_FP8_WO_A_GEMM 默认值从 False 翻为 True，FP8 加速现在对大多数用户自动生效。

**KV 缓存盐值支持多租户隔离**[15] - TRT-LLM 在 KV cache v2 管理器补上 cache_salt_id，同一模型实例可根据请求携带的 salt 区分不同 KV 缓存片段，对共享 KV pool 的多租户场景有直接工程意义。

**多模态推理加速就位**[16][17][18] - TRT-LLM 合入 Kimi K2.5 多模态视觉支持（含图像和视频预处理管线），补齐 GLM-5 6144 hidden_dim 的 router GEMM 编译期路径。vLLM PD disaggregation 在 NIXL connector 层面支持 Qwen3.5 GDN conv-state 布局——disaggregation 不再只服务 standard attention，Mamba/hybrid 架构也进入 P/D 分离视野。

---

> 一句话结论：**vLLM v0.21.0 的 C++20 和 transformers v5 是推理框架基础设施的代际升级；DeepSeek V4 MegaMoE 三框架同日就位则标志着 DSV4 从"能跑通"正式进入"怎么跑更快"的量产竞争期。**

---

## 参考

[1] vLLM v0.21.0 release：https://github.com/vllm-project/vllm/releases/tag/v0.21.0

[2] Model Runner V2 oracle for dense models：https://github.com/vllm-project/vllm/pull/39337

[3] Fix lazy attention state init during cudagraph capture：https://github.com/vllm-project/vllm/pull/42444

[4] Publish request counts at start of each engine step：https://github.com/vllm-project/vllm/pull/41626

[5] Replace shared-memory routed experts with ModelRunnerOutput：https://github.com/vllm-project/vllm/pull/39568

[6] SGLang DeepSeek V4 w4a4 MegaMoE：https://github.com/sgl-project/sglang/pull/25052

[7] TRT-LLM MEGAMOE_DEEPGEMM for DSV4：https://github.com/NVIDIA/TensorRT-LLM/pull/14129

[8] LMCache Support DeepSeek V4：https://github.com/LMCache/LMCache/pull/3171

[9] vLLM TOKENSPEED_MLA backend for Blackwell：https://github.com/vllm-project/vllm/pull/41778

[10] SGLang MLA TMA bulk-store up to 12x：https://github.com/sgl-project/sglang/pull/25311

[11] TRT-LLM FP4 paged MQA logits decode kernel：https://github.com/NVIDIA/TensorRT-LLM/pull/13929

[12] llama.cpp continue_final_message flag for vLLM API compat：https://github.com/ggml-org/llama.cpp/pull/23012

[13] SGLang model-config-parser registry：https://github.com/sgl-project/sglang/pull/25050

[14] SGLang Enable FP8 WO A GEMM by default：https://github.com/sgl-project/sglang/pull/25181

[15] TRT-LLM cache_salt_id in KV cache v2：https://github.com/NVIDIA/TensorRT-LLM/pull/13793

[16] TRT-LLM Kimi K2.5 multimodal vision：https://github.com/NVIDIA/TensorRT-LLM/pull/12788

[17] TRT-LLM GLM-5 router GEMM instantiation：https://github.com/NVIDIA/TensorRT-LLM/pull/13740

[18] vLLM PD disagg with NIXL Connector GDN support：https://github.com/vllm-project/vllm/pull/41869
