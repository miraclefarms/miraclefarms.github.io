---
title: AI Infra 早报｜推理框架三线并进量产准备——新模型、新硬件、核心架构同时收口
date: 2026-05-08 08:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: SGLang v0.5.11 把 CUDA 13 扶正为默认基线，vLLM 路由架构重写，TRT-LLM Blackwell FP4 indexer 落地——三大推理框架在新模型、新硬件、核心架构三条线上同步进入量产冲刺。speculative decoding 正在从实验特性变成新模型首发标配。
---

![题图](/assets/2026-05-08-ai-infra-daily-brief/cover.png)


今天的信号很集中：**三大推理框架几乎在同一时间窗口完成了各自的关键架构升级**——SGLang v0.5.11 把 CUDA 13 + PyTorch 2.11 扶正为默认基线，vLLM 用 device-cache 方案重写了 disaggregated serving 的路由层，TRT-LLM 在 Blackwell 上集成了 FP4 indexer 并引入 SWA prefill 内存复用。与此同时，speculative decoding 在三天内被三个框架分别适配到 Gemma 4 MTP、Qwen3.5 Mamba 混合和 Laguna DFlash 三个完全不同的模型家族上——这个节奏说明它已经不再是少数模型的实验选项，而是新模型推理路径的默认考量。DeepSeek-V4 的 MXFP4 量化路径在 SGLang 落地，Ray Serve 完成了直接流式转发架构，Mooncake 把触角伸向 AMD CDNA4。推理部署的硬件选择面和架构灵活性都在快速扩张。

## 一、SGLang v0.5.11：CUDA 13 成为默认基线，API 层开始承压

SGLang v0.5.11 把 CUDA 版本从 12 提升到 13.0、PyTorch 从 2.9 升到 2.11[[1]](https://github.com/sgl-project/sglang/releases/tag/v0.5.11)。这是整个推理框架生态里第一个把 CUDA 13 作为默认基线的 major release——意义不在于版本号本身，而在于 SGLang 选择在新硬件适配上持续领跑。

同一天合并的三个 PR 更值得注意：reasoning parser 自动检测（`--reasoning-parser auto`）[[2]](https://github.com/sgl-project/sglang/pull/23952)、two-phase reasoning grammar + `--enable-strict-thinking` [[3]](https://github.com/sgl-project/sglang/pull/23953)、以及 reasoning 配置字段兼容修复 [[4]](https://github.com/sgl-project/sglang/pull/23951)。这三个 PR 都不是 kernel 层的优化，而是在 API/调度层做的开发体验改进。**SGLang 在"首发支持新模型"上的竞争力正在从底层 kernel 上移到上层接口设计。**

## 二、DeepSeek-V4 MXFP4 推理路径落地

DeepSeek-V4 的 MXFP4（E8M0）量化推理在 SGLang 的 JIT kernel 路径上完成了移植——具体是把 Marlin MoE kernel 搬到了 JIT 编译框架里[[5]](https://github.com/sgl-project/sglang/pull/24490)。MXFP4 此前更多是实验性支持，这次移植让它走向可用。

配套的修复也在同步推进：attn kernel 的 early exit 加上了 cuda-graph 支持[[6]](https://github.com/sgl-project/sglang/pull/24584)，NSA prefill context parallel 修复了 CP 进程组引用错误导致的 crash[[7]](https://github.com/sgl-project/sglang/pull/24560)。TRT-LLM 侧也在做 DeepSeek-V4 的集成覆盖——dis-agg CI（GB200 only）[[8]](https://github.com/NVIDIA/TensorRT-LLM/pull/13803) 和 indexer compressed lengths 修复[[9]](https://github.com/NVIDIA/TensorRT-LLM/pull/13802)。**两个框架同时在 DeepSeek-V4 上做稳定性打磨，说明这个模型的推理部署已经进入工程收敛期。**

## 三、vLLM 路由架构重写——disaggregated serving 的地基重铺

vLLM #39917 用 device-cache 方案替换了原有的 routing replay 机制[[10]](https://github.com/vllm-project/vllm/pull/39917)。旧方案在 CUDA graphs、多节点 TP 和 data parallelism 场景下有正确性问题，新方案通过 device cache + async D2H pipeline 让路由逻辑正确适配这些场景。同一天，nixl transfer design 的重构 PR 也合并了[[11]](https://github.com/vllm-project/vllm/pull/40731)。

SGLang 侧也在做类似的 KV 传输异步化[[12]](https://github.com/sgl-project/sglang/pull/23967)。**两个框架在 disaggregated serving 路径上几乎同时在做架构级清理，这背后很可能是大规模部署中暴露出的性能和正确性瓶颈在倒逼重构。**

## 四、TRT-LLM v1.3.0rc14——Blackwell 首 token 延迟的系统性压缩

TRT-LLM v1.3.0rc14[[13]](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.3.0rc14) 里有几个针对性很强的优化：在 Blackwell（SM100+）上集成了 **FP4 indexer** 用于 DSA[[14]](https://github.com/NVIDIA/TensorRT-LLM/pull/13340)；SWA prefill 引入 scratch slots 做内存复用——窗口外的 KV block 在 prefill 结束后立即释放[[15]](https://github.com/NVIDIA/TensorRT-LLM/pull/13368)；KvCacheAwareRouter 把 tokenize + block-hash offload 到独立线程，从关键路径移除了约 50ms 的 CPU 开销[[16]](https://github.com/NVIDIA/TensorRT-LLM/pull/13377)。还有 Llama 8B decode 的三重优化——silu_mul backend 切换、quant+silu_mul fusion、QKV passthrough[[17]](https://github.com/NVIDIA/TensorRT-LLM/pull/12507)。

这些改动指向同一个目标：**压缩 Blackwell 上的首 token 延迟**。从量化到内存管理到计算内核，每一层都在挤出开销。

## 五、Speculative decoding 正在变成新模型首发标配

三天内三个框架分别给不同模型家族加了 speculative decoding 支持：SGLang 给 **Gemma 4** 加了 MTP（Multi-Token Prediction）[[18]](https://github.com/sgl-project/sglang/pull/24436)；vLLM 给 **Qwen3.5** 加了 Mamba 混合模型支持（Model Runner V2）[[19]](https://github.com/vllm-project/vllm/pull/35520)；vLLM 给 **Laguna** 加了 DFlash[[20]](https://github.com/vllm-project/vllm/pull/41880)。TRT-LLM 也在 v1.3.0rc14 里给 Mamba 混合模型（含 Qwen3.5）加了 prefix caching[[13]](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.3.0rc14)，SGLang 还做了 TRT-LLM draft extend 的 decode kernel 适配[[21]](https://github.com/sgl-project/sglang/pull/24566)。

MTP、Mamba 混合、DFlash——三种完全不同的加速策略，三个不同的模型家族，同步落地。**这不是巧合，而是一个趋势的确认：speculative decoding 已经从"少数模型的实验性特性"变成了"每个新模型首发时必须考虑的推理路径"。**

## 六、Ray Serve 直接流式转发 + Mooncake AMD CDNA4

Ray Serve 用 5 个 PR 完成了 LLM 应用的直接流式转发架构[[22]](https://github.com/ray-project/ray/pull/63167)——核心变化是 HAProxy ingress router 把请求 pin 到具体 replica，绕过了原来的 Serve proxy 转发层，减少了中间环节[[23]](https://github.com/ray-project/ray/pull/62669)。Mooncake 给 Tent 子系统加了 **AMD CDNA4（ROCm/HIP）** 支持[[24]](https://github.com/kvcache-ai/Mooncake/pull/2021)，同时做了 EP comprehensive test 去除对 DeepEP 的依赖[[25]](https://github.com/kvcache-ai/Mooncake/pull/1695)。

两个项目从不同角度在扩展推理部署的选择面：Ray Serve 在架构层面让流式推理更直接，Mooncake 在硬件层面把 KV cache 传输搬到 AMD 平台。

## 今天真正值得记住的判断

推理框架正在三条线上同步做量产准备——**新模型**（DeepSeek-V4 MXFP4、Gemma 4 MTP）、**新硬件**（Blackwell FP4、AMD CDNA4）、**核心架构**（vLLM routing 重写、Ray Serve 直接流式转发）。SGLang v0.5.11 把 CUDA 13 扶正是其中最具标志性的动作——它意味着新硬件适配的领先优势正在从"可选竞争力"变成"基本入场券"。而 speculative decoding 在三个模型家族上的同步落地，则说明推理加速已经不再是后置优化，而是新模型推理路径设计的第一天就要考虑的问题。

---

## 参考来源

[1] [SGLang v0.5.11 Release](https://github.com/sgl-project/sglang/releases/tag/v0.5.11)

[2] [SGLang Reasoning parser auto-detect](https://github.com/sgl-project/sglang/pull/23952)

[3] [SGLang Two-phase reasoning grammar](https://github.com/sgl-project/sglang/pull/23953)

[4] [SGLang reasoning.enabled mapping fix](https://github.com/sgl-project/sglang/pull/23951)

[5] [SGLang MXFP4 Marlin MoE JIT kernel](https://github.com/sgl-project/sglang/pull/24490)

[6] [SGLang DeepSeek-V4 attn kernel early exit with cuda-graph](https://github.com/sgl-project/sglang/pull/24584)

[7] [SGLang NSA prefill context parallel crash fix](https://github.com/sgl-project/sglang/pull/24560)

[8] [TRT-LLM DeepSeek-V4 dis-agg CI](https://github.com/NVIDIA/TensorRT-LLM/pull/13803)

[9] [TRT-LLM DeepSeek-V4 indexer compressed lengths fix](https://github.com/NVIDIA/TensorRT-LLM/pull/13802)

[10] [vLLM Replace routing replay with device cache](https://github.com/vllm-project/vllm/pull/39917)

[11] [vLLM Nixl refactor: new transfer design](https://github.com/vllm-project/vllm/pull/40731)

[12] [SGLang Nixl async transfer](https://github.com/sgl-project/sglang/pull/23967)

[13] [TRT-LLM v1.3.0rc14 Release](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.3.0rc14)

[14] [TRT-LLM FP4 indexer for DSA on Blackwell](https://github.com/NVIDIA/TensorRT-LLM/pull/13340)

[15] [TRT-LLM SWA prefill memory reuse](https://github.com/NVIDIA/TensorRT-LLM/pull/13368)

[16] [TRT-LLM KvCacheAwareRouter tokenize+block-hash offload](https://github.com/NVIDIA/TensorRT-LLM/pull/13377)

[17] [TRT-LLM Llama 8B decode triple optimization](https://github.com/NVIDIA/TensorRT-LLM/pull/12507)

[18] [SGLang Gemma 4 MTP speculative decoding](https://github.com/sgl-project/sglang/pull/24436)

[19] [vLLM Qwen3.5 Mamba hybrid for Model Runner V2](https://github.com/vllm-project/vllm/pull/35520)

[20] [vLLM Laguna DFlash speculative decoding](https://github.com/vllm-project/vllm/pull/41880)

[21] [SGLang TRT-LLM draft extend decode kernel](https://github.com/sgl-project/sglang/pull/24566)

[22] [Ray Serve direct streaming proxy (5/5)](https://github.com/ray-project/ray/pull/63167)

[23] [Ray Serve HAProxy ingress request router](https://github.com/ray-project/ray/pull/62669)

[24] [Mooncake AMD CDNA4 platform support](https://github.com/kvcache-ai/Mooncake/pull/2021)

[25] [Mooncake EP comprehensive test](https://github.com/kvcache-ai/Mooncake/pull/1695)