---
title: AI Infra 早报｜安全补丁潮与推理链修复并进，基础设施正在为下一轮高速扩张筑底
date: 2026-03-28 05:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: 两个 CVSS 9.8 RCE 漏洞（SGLang ZMQ 未认证反序列化、Ray Data Parquet 扩展类型 cloudpickle）在同一天修复；llama.cpp 关闭多轮 CoT 截断缺口，reasoning_content 跨轮保留正式落地；Mooncake Store 引入 Redis snapshot catalog，etcd 不再是 HA 的唯一路径；OpenClaw 延续系统性安全收口，Discord 审批链与设备配对授权批量加固。
tags: [Inference, Agents, SGLang, llama.cpp]
---

今天的主线不是性能数字，而是两件本应更早发生的事情终于发生了：一个多月来悬在 SGLang 生产服务上的 RCE 漏洞被修补，Ray Data 处理不可信数据时的任意代码执行路径被封堵。与此同时，llama.cpp 关闭了多轮思维链对话中一个基础但影响显著的缺口——模型再也不会在每一轮对话开始时忘掉自己上一次是怎么思考的。

这是一个基础设施自我修补、为下一轮高速扩张筑底的典型窗口。

## 一、两个 RCE，同日关闭

SGLang 修复了三个 CVSS 9.8 的 CVE（CVE-2026-3059、CVE-2026-3060、CVE-2026-3989）[[8]](https://github.com/sgl-project/sglang/pull/21435)。根源是同一个模式：`pickle.loads()` 在未经认证的 ZMQ socket 上反序列化数据。攻击者只需要能向对应端口发送数据，就可以在运行 SGLang 的服务器上执行任意代码。受影响的组件包括多模态生成 ZMQ broker、Encoder Parallel Disaggregation 和 replay_request_dump 端点。这次修复是系列补丁的第一步，将 ZMQ socket 绑定到 localhost，后续系列将完善完整认证机制。如果你在生产环境中运行 SGLang 且向外部暴露了服务，需要优先评估。

同一天，Ray Data 修复了 Arrow 扩展类型从 Parquet 文件反序列化的 RCE 路径（GHSA-mw35-8rx3-xf9r）[[15]](https://github.com/ray-project/ray/pull/62056)。攻击向量是恶意构造的 Parquet 文件：Ray 2.49-2.54 在读取包含 Arrow tensor 扩展类型的 Parquet 时，会通过 `cloudpickle.loads()` 执行嵌入在文件里的 payload。修复将默认反序列化方式切换为 JSON-only，旧格式兼容通过环境变量 `RAY_DATA_AUTOLOAD_CLOUDPICKLE_TENSOR_METADATA=1` opt-in。如果你的数据处理管道读取来自外部的 Parquet 文件，这个修复版本需要尽快部署。

两个漏洞的共同特征是**信任边界缺失**——框架假设调用方是可信的，没有为不可信输入设防。这在 AI Infra 领域尤其值得警惕：训练数据、评测数据集、推理请求，都可能来自外部。

## 二、llama.cpp：多轮 CoT 终于能"记住"自己的推理

llama.cpp 修复了一个长期存在的行为问题：WebUI 在每次新轮发送消息时，会剥离所有 `<reasoning>` 标签内的内容，再发送给 `/v1/chat/completions`[[1]](https://github.com/ggml-org/llama.cpp/pull/21036)。

这意味着 GLM-4.7-Flash、DeepSeek-R1、QwQ 这类依赖多轮 chain-of-thought 的模型，在每一轮对话开始时都是从零出发——上一轮的推理过程对下一轮完全不可见。

修复的逻辑是把 reasoning 内容序列化到 `reasoning_content` 字段回传给模型，而不是丢弃。服务端的 `common_chat_msgs_parse_oaicompat` 和 Jinja 模板都已支持该字段作为一等公民输入。这不是一个小修复——它意味着多轮 CoT 对话从"每次从头推"走向"继承历史推理上下文"，这两种行为在复杂推理任务上的差距是显著的。

配合这个修复，另一个 PR 解决了 grammar sampler 与推理链的冲突[[2]](https://github.com/ggml-org/llama.cpp/pull/20970)：约束解码（grammar sampling）在推理 budget 激活期间若提前介入，会打断思维链的生成节奏。现在 grammar sampler 会严格等待推理阶段结束后再介入，两者联合使用时的正确性恢复。

在硬件量化覆盖上，llama.cpp 同日为 Qualcomm Hexagon HMX NPU 新增了 IQ4_NL 和 MXFP4 支持[[3]](https://github.com/ggml-org/llama.cpp/pull/21018)，并修复了 AMD CDNA3（MI300 系列）的 fp8 格式问题[[4]](https://github.com/ggml-org/llama.cpp/pull/21040)——CDNA3 需要 fnuz 格式的 fp8，而不是通用的 fp8 编码。Apple Metal 上的一个 matmul2d 描述符维度约束违规导致的崩溃也同步修复[[5]](https://github.com/ggml-org/llama.cpp/pull/21048)。llama.cpp 的硬件覆盖范围正在同时向 Qualcomm、AMD 和 Apple 三个方向延伸，单日五个版本（b8554-b8559）的发布频率也说明了这个项目当前的迭代密度。

## 三、vLLM 和 SGLang：投机解码与扩散模型的工程细节

vLLM 的 V2 Model Runner 修复了一个影响 EAGLE 投机解码实际性能的 bug[[6]](https://github.com/vllm-project/vllm/pull/38311)：在 EAGLE decode 的 cudagraph 执行中，position>0 时的 attention metadata 没有被重建，导致 draft token 使用了过期的 metadata，接受率显著偏低。重建之后，Llama3-8B 和 Qwen3-8B 的 draft token 接受率均有明显改善。投机解码的实际加速比在很大程度上取决于接受率，这个修复对使用 EAGLE 的部署有直接的吞吐影响。

SGLang 则在扩散模型侧做了两步内核优化[[11]](https://github.com/sgl-project/sglang/pull/21440)：qknorm+rope 融合 JIT 内核减少了两个操作之间的 kernel launch 间隔，而 `qknorm_across_heads` 内核改为每次同时处理多个 head，减少了内存访问的往返次数。这两个优化方向都指向同一个目标：在扩散模型推理中减少 GPU 的"等待时间"。

SGLang 新增的 `--gc-threshold` 参数[[10]](https://github.com/sgl-project/sglang/pull/21481) 是一个实用的运维工具。Python 默认 GC 策略在内存压力下每次触发可能耗时数百毫秒，这对 p99 latency 有严格要求的在线服务是无法接受的。通过调高 GC 触发阈值，可以在接受更高内存占用的前提下换取更稳定的尾延迟分布。

## 四、Mooncake：KV cache 持久层向 Redis 双路演进

Mooncake 在本窗口内完成了 KV cache 持久层的两步 Redis 建设。

第一步是 snapshot catalog 的抽象化[[18]](https://github.com/kvcache-ai/Mooncake/pull/1739)：此前 snapshot catalog 逻辑嵌入在 MasterService 里，强耦合导致引入新后端困难。重构将 catalog 逻辑提取为独立接口，Redis 实现与原有的 etcd 路径并列，etcd-free 的 Mooncake Store 部署成为可能。

第二步是 Redis ACL 用户认证[[19]](https://github.com/kvcache-ai/Mooncake/pull/1757)：通过 `MC_REDIS_USERNAME` 环境变量支持 ACL 身份认证，使 Redis 路径在安全性上与 etcd 路径对齐，同时重组了 HA 代码布局，让后续扩展有更清晰的落脚点。

同日，Mooncake 的 PyTorch backend（`mooncake-pg`）将 `barrier()` 从 CPU-only 路径改到 CUDA task 路径[[20]](https://github.com/kvcache-ai/Mooncake/pull/1751)，支持训练+推理混合部署中的 GPU 侧屏障同步。

## 五、生产部署：Ray Serve 监控升级与 LMCache 稳定性

Ray Serve 的 LLM 监控面板经历了一次完整的重构[[16]](https://github.com/ray-project/ray/pull/62069)：从 29 个 flat panel 重写为 7 行 44 个面板，3 列布局更易于并排比对。修复了 `deployment` 变量 "All" 解析为空字符串的历史 bug（这个 bug 会导致筛选"全部 deployment"时实际上什么都不显示）。对于 vLLM-only 指标，新版面板使用 WorkerId join pattern 在图例中区分 `deployment:replica`，多副本部署的异常定位更清晰。Ray Serve 也在本窗口落地了多轮对话 benchmark 框架的第一步[[17]](https://github.com/ray-project/ray/pull/62080)，后续系列将覆盖完整的多轮性能评测。

LMCache 在本窗口完成了两个稳定性修复。MP 模式控制器[[21]](https://github.com/LMCache/LMCache/pull/2883)的主循环加了 try-except 保护，单次迭代的意外异常不再终止整个后台控制线程；logger 切换到 `logger.exception`，异常时保留完整堆栈，调试可见性大幅提升。LRU cache policy[[22]](https://github.com/LMCache/LMCache/pull/2860) 补上了缺失的并发锁，多进程写入时的竞态条件消除。

TensorRT-LLM 引入了 FlexKV 支持[[12]](https://github.com/NVIDIA/TensorRT-LLM/pull/12512)，为外部 KV cache connector 提供标准化接口，包括初始化同步、metadata 处理、request 完成状态查询和 connector-matched token metrics。Megatron-LM 将 Transformer Engine 升至 2.14[[13]](https://github.com/NVIDIA/Megatron-LM/pull/4025)，底层 FP8 训练优化随之带入。

## 六、OpenClaw：安全收口的第二天

延续前日的节奏，OpenClaw 在本窗口继续批量落地安全补丁[[23]](https://github.com/openclaw/openclaw/pull/56015)。这次集中在 Discord 审批链（文字 `/approve` 命令对齐授权策略，组件交互施加 policy 门控）、设备配对收紧（节点配对审批防止未授权设备接入，撤销设备时主动断开其 session）、exec 绕过拦截（拒绝通过 wrapper 绕过 allow-always 限制）。另外修复了 WSL2 下 Ollama provider 的网络发现问题[[24]](https://github.com/openclaw/openclaw/pull/55435)，以及 `/status` 在 per-agent 模型覆盖场景下显示错误配置的问题[[25]](https://github.com/openclaw/openclaw/pull/55425)。

---

两天密集的安全补丁之后，今天的 AI Infra 生态像是在做一次有意识的系统体检——不是为了追赶新 benchmark，而是为了让现有的基础足够稳固，承载下一轮的高速扩张。

## 参考来源

[1] [llama.cpp reasoning_content 多轮保留](https://github.com/ggml-org/llama.cpp/pull/21036)

[2] [llama.cpp 禁止 lazy grammar sampler 干扰推理链](https://github.com/ggml-org/llama.cpp/pull/20970)

[3] [llama.cpp Hexagon IQ4_NL 和 MXFP4 支持](https://github.com/ggml-org/llama.cpp/pull/21018)

[4] [llama.cpp AMD CDNA3 fnuz fp8 修复](https://github.com/ggml-org/llama.cpp/pull/21040)

[5] [llama.cpp Apple Metal matmul2d 修复](https://github.com/ggml-org/llama.cpp/pull/21048)

[6] [vLLM EAGLE decode attention metadata 修复](https://github.com/vllm-project/vllm/pull/38311)

[7] [vLLM CPU W4A16 量化扩展](https://github.com/vllm-project/vllm/pull/38219)

[8] [SGLang ZMQ socket 未认证 RCE 修复（CVE-2026-3059/3060/3989）](https://github.com/sgl-project/sglang/pull/21435)

[9] [SGLang RL pause 后 tensor mismatch 修复](https://github.com/sgl-project/sglang/pull/21514)

[10] [SGLang GC threshold 参数](https://github.com/sgl-project/sglang/pull/21481)

[11] [SGLang 扩散模型内核优化](https://github.com/sgl-project/sglang/pull/21440)

[12] [TensorRT-LLM FlexKV 支持](https://github.com/NVIDIA/TensorRT-LLM/pull/12512)

[13] [Megatron-LM 升级 Transformer Engine 2.14](https://github.com/NVIDIA/Megatron-LM/pull/4025)

[14] [TRL datasets>=4.7.0 Json dtype 修复](https://github.com/huggingface/trl/pull/5376)

[15] [Ray Data Parquet RCE 修复（GHSA-mw35-8rx3-xf9r）](https://github.com/ray-project/ray/pull/62056)

[16] [Ray Serve LLM 监控面板重构](https://github.com/ray-project/ray/pull/62069)

[17] [Ray Serve 多轮 benchmark 框架（1/4）](https://github.com/ray-project/ray/pull/62080)

[18] [Mooncake Redis snapshot catalog 后端](https://github.com/kvcache-ai/Mooncake/pull/1739)

[19] [Mooncake Redis ACL 用户认证](https://github.com/kvcache-ai/Mooncake/pull/1757)

[20] [Mooncake PyTorch backend GPU barrier 支持](https://github.com/kvcache-ai/Mooncake/pull/1751)

[21] [LMCache MP 控制器稳定性加固](https://github.com/LMCache/LMCache/pull/2883)

[22] [LMCache LRU 并发锁修复](https://github.com/LMCache/LMCache/pull/2860)

[23] [OpenClaw Discord 审批链与设备配对安全加固](https://github.com/openclaw/openclaw/pull/56015)

[24] [OpenClaw WSL2 Ollama 网络修复](https://github.com/openclaw/openclaw/pull/55435)

[25] [OpenClaw /status 运行时模型配置修复](https://github.com/openclaw/openclaw/pull/55425)
