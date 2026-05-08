---
title: AI Infra 早报｜推理框架的竞争点从"能跑新模型"转向"通用路径上跑稳"
date: 2026-05-07 08:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: TRT-LLM 发 Helix Parallelism 博文并去掉模型专用补丁，SGLang 将 P2P 权重传输迁入主线，vLLM 修 PP 并发 token 丢失——推理框架的竞争重心正在从"首发支持"转向"通用路径上跑稳"。
tags: [Inference, Disaggregation]
---

![题图](/assets/2026-05-07-ai-infra-daily-brief/cover.jpg)


推理框架的竞争点正在发生一次不易察觉但意义深远的位移。今天 TRT-LLM 发布了新的并行策略 Helix Parallelism 的官方博文，但更值得关注的是 AutoDeploy 连续两个 PR 去掉了 Llama 4、Qwen3 Next、Qwen3 MoE 的模型专用补丁——这些此前需要逐个手写的 special case，现在走通用部署路径就能搞定。同一天，SGLang 把 P2P 实时权重传输基础设施从实验分支 sglang-miles 批量迁入主线，vLLM 修了 Pipeline Parallelism 模式下并发请求导致 token 丢失的生产级 bug。这三件事指向同一个判断：**推理框架的护城河正在从"能跑新模型"转向"在通用路径上跑稳"**。

## 一、TRT-LLM：Helix Parallelism 博文落地，AutoDeploy 去掉模型特殊补丁

TRT-LLM 今天有两个方向值得关注。一方面是并行策略的推进——Helix Parallelism 的官方博文作为 PR 合入主仓库[[1]](https://github.com/NVIDIA/TensorRT-LLM/pull/13547)，这标志着 NVIDIA 在多维度并行（TP/PP/EP/CP 之外）的布局有了新的官方叙事。同时 Sparse FMHA 加入了 multi-cta-kv 支持[[6]](https://github.com/NVIDIA/TensorRT-LLM/pull/13410)，MoE cubins 也做了更新[[5]](https://github.com/NVIDIA/TensorRT-LLM/pull/12440)，底层 kernel 的覆盖面在持续扩展。

但**更有信号意义的是 AutoDeploy 的通用化方向**。PR #13247 去掉了 Llama 4 / Qwen3 Next / Qwen3 MoE 的模型专用补丁[[2]](https://github.com/NVIDIA/TensorRT-LLM/pull/13247)，让这些模型成为自动部署的一等公民而非需要特殊照顾的 case。配合 decode 调度路径的 C++ 开销优化[[3]](https://github.com/NVIDIA/TensorRT-LLM/pull/13012)和 Qwen3.5 GDN 的 elementwise kernel fusion[[4]](https://github.com/NVIDIA/TensorRT-LLM/pull/12966)，TRT-LLM 的部署管线正在从"为每个模型写补丁"收敛到"通用路径自洽"。这对用户意味着：新模型上线的时间会越来越短，维护成本会越来越低。

## 二、SGLang：P2P 权重传输从实验分支迁入主线，PD 分离部署故障链路收紧

SGLang 在同一天连续 cherry-pick 了 6 个 weight_checker 相关 PR（#24532–#24538）从实验分支 sglang-miles 到 main[[7]](https://github.com/sgl-project/sglang/pull/24532)[[8]](https://github.com/sgl-project/sglang/pull/24533)[[9]](https://github.com/sgl-project/sglang/pull/24534)[[10]](https://github.com/sgl-project/sglang/pull/24536)[[11]](https://github.com/sgl-project/sglang/pull/24537)[[12]](https://github.com/sgl-project/sglang/pull/24538)，涵盖 FP8 dequant 修复、non-persistent buffer 跳过、buffer pattern 重构、checksum 支持和端到端测试。**P2P 实时权重更新能力正从实验阶段正式进入主线**——这对需要不停服更新模型的推理场景（比如热切换 checkpoint）是一个关键的成熟度信号。

与此同时，PD（Prefill-Decode）分离部署路径的故障恢复也在同天收紧：abort 状态传播修复覆盖了所有 KV backend[[13]](https://github.com/sgl-project/sglang/pull/24522)，prevent update_status from cleared entries 修复了状态被意外清除的问题[[14]](https://github.com/sgl-project/sglang/pull/24539)，KV transfer metrics 的修复让监控链路闭合[[15]](https://github.com/sgl-project/sglang/pull/24416)。此外 HiSparse FP8 KV cache 路由到 flashmla_kv backend[[16]](https://github.com/sgl-project/sglang/pull/23013)、SWA HiCache 加入 unified radix cache[[17]](https://github.com/sgl-project/sglang/pull/23391)、CP KV cache allgather symmetric memory 注册[[18]](https://github.com/sgl-project/sglang/pull/24040)等 PR 也在推进 KV cache 的多 backend 统一。

## 三、Megatron-LM：训练链路正确性集中修复，legacy GPT 代码整体删除

Megatron-LM 今天落了几个影响训练正确性的修复，其中**最严重的是 layerwise param all-gather overlap 导致的梯度损坏**[[19]](https://github.com/NVIDIA/Megatron-LM/pull/4609)。这个问题影响使用 Muon 等分布式优化器的用户，梯度在 overlap 过程中被破坏会导致训练结果静默出错——这种 bug 尤其危险，因为它不会 crash，只会让模型效果悄悄变差。

FlashInfer sampling[[20]](https://github.com/NVIDIA/Megatron-LM/pull/2456)将采样路径从自定义实现切换到 FlashInfer，可能对解码性能有正面影响。SHA-256 替换 polynomial rolling hash 做 prefix caching[[21]](https://github.com/NVIDIA/Megatron-LM/pull/4612)消除了哈希碰撞的风险——polynomial rolling hash 在 prefix 空间巨大时确实存在碰撞概率，换成 SHA-256 是一个正确性优先的选择。同时 legacy GPT 代码被整体删除[[22]](https://github.com/NVIDIA/Megatron-LM/pull/4322)，CSA/HCA hybrid attention prototype 进入 HybridModel[[23]](https://github.com/NVIDIA/Megatron-LM/pull/4569)，代码库在加速清理历史包袱。

## 四、DeepSpeed v0.19.0 发布

DeepSpeed 发布 v0.19.0[[24]](https://github.com/deepspeedai/DeepSpeed/releases/tag/v0.19.0)。版本号 bump PR 在 release 前后各有一笔[[25]](https://github.com/deepspeedai/DeepSpeed/pull/7995)[[26]](https://github.com/deepspeedai/DeepSpeed/pull/7996)，属于连续迭代中的新稳定版本。具体功能变更需要等官方 release notes 补充。

## 五、Ray：K8s 1.35 原地 Pod 扩缩容 + vLLM 升级到 0.20.0

Ray 在 Kubernetes 侧实现了基于 1.35 IPPR（In-Place Pod Resizing）的 Pod 原地扩缩容[[27]](https://github.com/ray-project/ray/pull/55961)。**这是推理和训练集群弹性调度的基础设施前提**——不需要杀 Pod 重调度就能调整 CPU/内存资源，对在线推理服务的弹性伸缩有直接价值。同时 Ray LLM 组件将 vLLM 升级到 0.20.0 并切换到 CUDA 13 + Python 3.12 镜像[[28]](https://github.com/ray-project/ray/pull/62970)，HAProxy ingress request router 进入第四阶段[[29]](https://github.com/ray-project/ray/pull/62669)。Ray Train + torchft 的 replica group restart hang 问题也被修复[[30]](https://github.com/ray-project/ray/pull/62651)。

## 六、vLLM：PP 并发 token 丢失 + HF tokenizer 线程安全

vLLM 修了两个影响生产正确性的 bug。**Pipeline Parallelism 模式下并发请求导致 token 丢失**[[31]](https://github.com/vllm-project/vllm/pull/41133)这个问题在 Qwen3-8B GSM8K 评测中暴露——精度从 0.8741 下降，说明 token 在 pipeline 阶段间传递时被丢弃。这是典型的只有生产并发流量才会触发的 bug，单请求测试根本发现不了。HuggingFace fast tokenizer 的 `RuntimeError: Already borrowed` 并发问题也被修复[[32]](https://github.com/vllm-project/vllm/pull/41181)，加了线程安全 wrapper。两个修复都指向同一个结论：**多用户并发是推理框架正确性的试金石**。

此外 Qwen3 streaming content routing 修复[[33]](https://github.com/vllm-project/vllm/pull/40820)、DeepSeekV32/v4 string attribute 和 argument unwrap 修复[[34]](https://github.com/vllm-project/vllm/pull/41801)、torchtitan rl 场景的 codegen for unqualified names 修复[[35]](https://github.com/vllm-project/vllm/pull/40726)也在同天合入。

## 七、今天真正值得记住的判断

推理框架的竞争正在进入新阶段。早期比的是"谁先支持 Llama 4、谁先跑通 Qwen3 MoE"，现在比的是"谁的通用部署路径能覆盖最多的模型而不需要写特殊补丁"。TRT-LLM 的 AutoDeploy 去补丁、SGLang 的 P2P 权重传输迁入主线、vLLM 修 PP 并发正确性——这三个项目在同一天的动作都指向同一个方向。训练框架这边，Megatron-LM 的梯度损坏修复提醒我们，**分布式训练的正确性仍然是比性能更优先的问题**，尤其是那些不会 crash 只会让模型静默变差的 bug。

---

## 参考来源

[1] [TRT-LLM Helix Parallelism blog post](https://github.com/NVIDIA/TensorRT-LLM/pull/13547)

[2] [TRT-LLM AutoDeploy remove model patches](https://github.com/NVIDIA/TensorRT-LLM/pull/13247)

[3] [TRT-LLM AutoDeploy decode scheduling C++ overhead optimization](https://github.com/NVIDIA/TensorRT-LLM/pull/13012)

[4] [TRT-LLM Qwen3.5 GDN elementwise kernel fusion](https://github.com/NVIDIA/TensorRT-LLM/pull/12966)

[5] [TRT-LLM MoE cubins update](https://github.com/NVIDIA/TensorRT-LLM/pull/12440)

[6] [TRT-LLM Sparse FMHA multi-cta-kv support](https://github.com/NVIDIA/TensorRT-LLM/pull/13410)

[7] [SGLang weight checker FP8 dequant fix cherry-pick](https://github.com/sgl-project/sglang/pull/24532)

[8] [SGLang weight checker non-persistent buffer pattern cherry-pick](https://github.com/sgl-project/sglang/pull/24533)

[9] [SGLang weight checker fp32 buffer skip cherry-pick](https://github.com/sgl-project/sglang/pull/24534)

[10] [SGLang weight checker unit test and e2e test](https://github.com/sgl-project/sglang/pull/24536)

[11] [SGLang weight checker checksum support](https://github.com/sgl-project/sglang/pull/24537)

[12] [SGLang weight checker buffer pattern refactor](https://github.com/sgl-project/sglang/pull/24538)

[13] [SGLang PD abort state propagation fix](https://github.com/sgl-project/sglang/pull/24522)

[14] [SGLang PD prevent update_status from cleared entries](https://github.com/sgl-project/sglang/pull/24539)

[15] [SGLang PD KV transfer metrics fix](https://github.com/sgl-project/sglang/pull/24416)

[16] [SGLang HiSparse FP8 KV cache](https://github.com/sgl-project/sglang/pull/23013)

[17] [SGLang SWA HiCache for unified radix cache](https://github.com/sgl-project/sglang/pull/23391)

[18] [SGLang CP KV cache allgather symmetric memory registration](https://github.com/sgl-project/sglang/pull/24040)

[19] [Megatron-LM fix layerwise param all-gather overlap gradient corruption](https://github.com/NVIDIA/Megatron-LM/pull/4609)

[20] [Megatron-LM FlashInfer sampling](https://github.com/NVIDIA/Megatron-LM/pull/2456)

[21] [Megatron-LM SHA-256 prefix caching](https://github.com/NVIDIA/Megatron-LM/pull/4612)

[22] [Megatron-LM delete legacy GPT code](https://github.com/NVIDIA/Megatron-LM/pull/4322)

[23] [Megatron-LM CSA/HCA hybrid attention prototype](https://github.com/NVIDIA/Megatron-LM/pull/4569)

[24] [DeepSpeed v0.19.0 release](https://github.com/deepspeedai/DeepSpeed/releases/tag/v0.19.0)

[25] [DeepSpeed version bump pre-release](https://github.com/deepspeedai/DeepSpeed/pull/7995)

[26] [DeepSpeed version bump post-release](https://github.com/deepspeedai/DeepSpeed/pull/7996)

[27] [Ray K8s 1.35 in-place Pod resizing](https://github.com/ray-project/ray/pull/55961)

[28] [Ray LLM upgrade vLLM to 0.20.0](https://github.com/ray-project/ray/pull/62970)

[29] [Ray HAProxy ingress request router dispatch path](https://github.com/ray-project/ray/pull/62669)

[30] [Ray Train + torchft fix hang on replica group restarts](https://github.com/ray-project/ray/pull/62651)

[31] [vLLM fix PP mode token loss](https://github.com/vllm-project/vllm/pull/41133)

[32] [vLLM HF fast tokenizer thread safety wrapper](https://github.com/vllm-project/vllm/pull/41181)

[33] [vLLM Qwen3 streaming content routing fix](https://github.com/vllm-project/vllm/pull/40820)

[34] [vLLM DeepSeekV32/v4 string attribute and argument unwrap fix](https://github.com/vllm-project/vllm/pull/41801)

[35] [vLLM codegen for unqualified names fix](https://github.com/vllm-project/vllm/pull/40726)