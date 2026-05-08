---
title: AI Infra 早报｜推理框架向 B200 适配冲刺，TRT-LLM 一次性打包 Qwen3 全家桶
date: 2026-03-25 05:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: TRT-LLM v1.3.0rc9 一次性带来了 Qwen3.5 文本、GLM 5 支持与 DeepSeek V3.2 NVFP4 路由优化，KVCacheManagerV2 新增 Python 调度器接口；SGLang 在 B200 上修复 DSA 性能下降并合并 CuTeDSL KDA 解码内核；Mooncake 完成 MACA 构建路径初始接入，KV cache 传输引擎开始向国产加速卡延伸；OpenClaw 修复 LanceDB embedding bootstrap 路径，记忆搜索功能恢复正常。
tags: [Inference, Quantization, KV Cache]
---

过去 24 小时，最密集的工程推进发生在 Blackwell 适配和 NVFP4 量化这两条轨道上。TRT-LLM 用 v1.3.0rc9 告诉市场：1.3.0 正式版快了；SGLang 在 B200 硬件细节上逐个攻坚。与此同时，Mooncake 悄悄迈出了一步——把 KV cache 传输引擎引向国产 GPU 架构，这个方向的商业逻辑值得单独讨论。

## 一、TRT-LLM v1.3.0rc9：Qwen3 全家桶与 KV Cache 调度器

TRT-LLM 今天发布了 **v1.3.0rc9**[[1]](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.3.0rc9)，是这轮 rc 系列中体量最大的一次更新。

模型支持侧一口气推进了三件事：**Qwen3.5 文本模型**现在可以在 PyTorch 后端以 BF16 或 FP8 运行；**GLM 5** 正式进入支持矩阵；**DeepSeek V3.2 的 NVFP4 路径**也得到了系统性优化——indexer GEMMs 和路由内核都被重新打磨，进一步逼近 NVFP4 量化下 MoE 路由的理论吞吐上限。

更值得关注的是 **KVCacheManagerV2 的 Python 调度器接口**。这个改动引入了基于约束的内存分区机制，并在 Python 层暴露了一个可自定义的 KV cache 调度器。此前调度逻辑封闭在 C++ 内部，现在研究者和工程师可以在不动 C++ 的情况下实验不同的调度策略。对做 KV cache 优化研究的团队来说，这是一个重要的接口扩展。

与此同时，SM100f（Blackwell 子架构）上运行大规模 MoE 模型时触发的 **Triton resmooth 内核崩溃**[[2]](https://github.com/NVIDIA/TensorRT-LLM/pull/12397)也在今天修复，Blackwell 上大 MoE 推理的稳定性又前进一步。

## 二、SGLang：B200 适配进入细节攻坚

SGLang 今天在 Blackwell 适配上推进了两件事，方向不同但都指向同一个目标：让 B200 生产可用。

**B200+DP 的 DSA 性能下降**[[3]](https://github.com/sgl-project/sglang/pull/21337) 是一个已知问题，根因还在排查中，但工程上不能等——今天合并了绕道方案，保证 B200 在数据并行模式下 DSA 不会因性能退化影响生产调度。这种"先绕道、再追因"的工程取舍在硬件迁移期很常见，但也意味着技术债务还在，后续要回来还。

另一件事是 **CuTeDSL KDA 解码内核**[[4]](https://github.com/sgl-project/sglang/pull/21203) 合入主线。KDA（Kernel Decode Attention）此前的 Triton 实现在硬件控制粒度上受到限制，CuTeDSL 实现可以更直接地操作 NVIDIA CuTe 的 tile 抽象，理论上针对特定 SM 架构有更大的调优空间。合入后 SGLang 的解码路径有了新的高性能备选。

值得一提的还有 **Flux.1 的 NVFP4 支持**[[5]](https://github.com/sgl-project/sglang/pull/20137)——扩散模型路径和文本推理路径在量化体系上的统一进程又向前迈了一步。

## 三、vLLM：NFS 预取与工具调用修复

vLLM 今天没有架构级的大动作，但有两个值得关注的实用改进。

**NFS 路径自动 prefetch**[[6]](https://github.com/vllm-project/vllm/pull/37673) 解决了一个常见的生产痛点：使用 NFS 挂载模型权重的集群，此前需要手动配置预取才能获得较好的加载性能。现在 vLLM 会自动检测 NFS 并启用预取，同时设置 RAM 用量上限，防止大模型预取时把内存打满。对使用 NFS 存储权重的多节点部署来说，这是零配置的性能改善。

**OpenAI 工具调用解析器 IndexError**[[7]](https://github.com/vllm-project/vllm/pull/37958) 影响的是多轮工具调用场景——`prev_tool_call_arr` 的越界访问在特定调用序列下会触发，修复后这条链路恢复稳定。

## 四、Megatron-LM：MimoOptimizer 与 EP CUDA Graph

Megatron-LM 今天合并了两个面向大规模训练优化的改动。

**MimoOptimizer**[[9]](https://github.com/NVIDIA/Megatron-LM/pull/4019) 是一种面向异构并行场景的优化器封装。异构集群（不同型号 GPU 混部，或混合 CPU/GPU）在参数更新时，不同设备的计算/通信开销差异巨大，MimoOptimizer 提供了在不同并行组之间灵活分配这些开销的机制。随着训练集群越来越多出现异构配置，这类优化器的实用价值会越来越高。

**EP Overlap 的全迭代 CUDA Graph 支持**[[10]](https://github.com/NVIDIA/Megatron-LM/pull/3820) 解决了一个长期存在的兼容性问题：专家并行的通信-计算 overlap 与全迭代 CUDA Graph 捕获此前不能同时开启，二选一。引入动态计算流后，这两项优化可以协同工作，MoE 模型训练吞吐有望在大规模配置下进一步释放。

## 五、Mooncake：MACA 迈出第一步

Mooncake 今天合并的两个 PR 方向截然不同，但各有意思。

**HA 控制平面线程拆分**[[11]](https://github.com/kvcache-ai/Mooncake/pull/1736) 是 HA 存储方向的持续推进——控制平面线程与数据线程解耦后，网络抖动产生的 zero-seg heap 误报被消除，HA 路径的可观测性更干净。这是上周 HA 抽象落地后的工程收尾。

**MACA 构建路径**[[12]](https://github.com/kvcache-ai/Mooncake/pull/1731) 则更有战略意义。MACA 是海光（Hygon）DCU 系列的计算框架，语义接近 CUDA 但并非完全兼容。Mooncake 为 Transfer Engine 新增了 MACA 构建路径和 CUDA-like 适配层，意味着 KV cache 传输能力开始向国产 GPU 延伸。

这个方向背后的逻辑不难理解：国内大量模型推理部署在 DCU 等国产加速卡上，但 KV cache 基础设施（Mooncake、LMCache 这类）的国产硬件支持一直是短板。MACA 适配迈出的是第一步，后续能走多远，要看生产需求的反馈。

## 六、LMCache 与 OpenClaw

**LMCache** 今天在多进程模式下新增了推理请求 ID 日志[[13]](https://github.com/LMCache/LMCache/pull/2812)。多进程部署时，某条 vLLM 请求卡在哪里很难排查——现在可以通过请求 ID 直接在 LMCache 日志里追踪完整的处理链路。这类可观测性改进不显眼，但在生产运维中往往比功能新特性更实用。

**OpenClaw** 今天发布了 **v2026.3.23**[[14]](https://github.com/openclaw/openclaw/releases/tag/v2026.3.23)，主要修复了全局安装时插件 bundled runtime 文件缺失的问题，以及 ClawHub 版本比对逻辑中的硬编码过期版本号。此外还合并了几个重要修复：插件 hook 终止语义强化[[15]](https://github.com/openclaw/openclaw/pull/54241)，LanceDB embedding bootstrap proxy 路径修复[[16]](https://github.com/openclaw/openclaw/pull/54119)，以及 gateway 频道启动失败的隔离处理[[17]](https://github.com/openclaw/openclaw/pull/54215)——后者在之前的版本中，WhatsApp 等单个频道启动失败会导致整个 gateway 进程崩溃。

## 参考来源

[1] [TRT-LLM v1.3.0rc9 release](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.3.0rc9)

[2] [TRT-LLM SM100f MoE resmooth 修复](https://github.com/NVIDIA/TensorRT-LLM/pull/12397)

[3] [SGLang B200+DP DSA 性能绕道](https://github.com/sgl-project/sglang/pull/21337)

[4] [SGLang CuTeDSL KDA 解码内核](https://github.com/sgl-project/sglang/pull/21203)

[5] [SGLang Flux.1 NVFP4 支持](https://github.com/sgl-project/sglang/pull/20137)

[6] [vLLM NFS prefetch 自动开启](https://github.com/vllm-project/vllm/pull/37673)

[7] [vLLM OpenAI 工具调用解析器修复](https://github.com/vllm-project/vllm/pull/37958)

[8] [llama.cpp token embedding norms b8508](https://github.com/ggml-org/llama.cpp/releases/tag/b8508)

[9] [Megatron-LM MimoOptimizer 异构并行](https://github.com/NVIDIA/Megatron-LM/pull/4019)

[10] [Megatron-LM EP Overlap 全迭代 CUDA Graph](https://github.com/NVIDIA/Megatron-LM/pull/3820)

[11] [Mooncake HA 控制平面线程拆分](https://github.com/kvcache-ai/Mooncake/pull/1736)

[12] [Mooncake Transfer Engine MACA 构建路径](https://github.com/kvcache-ai/Mooncake/pull/1731)

[13] [LMCache 多进程请求 ID 日志](https://github.com/LMCache/LMCache/pull/2812)

[14] [OpenClaw v2026.3.23 release](https://github.com/openclaw/openclaw/releases/tag/v2026.3.23)

[15] [OpenClaw 插件 hook 终止语义](https://github.com/openclaw/openclaw/pull/54241)

[16] [OpenClaw LanceDB embedding bootstrap 修复](https://github.com/openclaw/openclaw/pull/54119)

[17] [OpenClaw gateway 频道隔离](https://github.com/openclaw/openclaw/pull/54215)
