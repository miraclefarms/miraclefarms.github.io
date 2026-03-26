---
title: AI Infra 早报｜回滚即信号——CI 质量门拦截内核冒进，KV cache 底层同日三路扩张
date: 2026-03-27 05:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: vLLM 和 SGLang 同日各回滚一个激进内核合并（NVFP4 CuteDSL MoE 与 FlashMLA），CI 质量门在高速合并窗口有效拦截；LMCache 24 小时内完成 block-id MP 内核、HND 格式、AMD hipFile GPU-direct 三路扩张；DeepSpeed 将 Muon 优化器推进至 ZeRO Stage 3；OpenClaw 单日落地 14+ 渠道认证/授权安全补丁。
---

今天最值得先讲的，是两条回滚。vLLM 把昨天刚合并的 FlashInfer NVFP4 CuteDSL MoE 内核撤回来了，SGLang 把 FlashMLA 回滚到旧版本。两件事同日发生，不是巧合，也不是退步——这是高速合并节奏下 CI 质量门正常运转的标志。内核合进主干，发现 CI 失败，当天回滚，找到根因，下次再来。这个循环在两个项目里同时跑，说明工程纪律在。

另一个值得单独说的，是 LMCache 在同一个 24 小时窗口里，在三个完全不同的方向上同时落地了底层扩展：block-id 级的 MP 模式 KV 传输内核、HND KV 格式支持、以及 AMD ROCm 环境的 hipFile GPU-direct storage。方向各异，但背后的逻辑一致：KV cache 的传输和存储已经进入精细化的硬件适配阶段，不再只是"找个地方放"，而是在格式、传输粒度、GPU-direct 路径上逐一打通硬件极限。

## 一、回滚这件事

vLLM 的 NVFP4 CuteDSL MoE 内核集成（PR#38050）是前一天合入的，基于 FlashInfer 的 CuTe DSL 实现，目标是在 Blackwell 架构上释放更多 NVFP4 MoE 推理性能。但 nightly CI 随即出现一个新增失败，次日回滚 [[1]](https://github.com/vllm-project/vllm/pull/38169)。

SGLang 的情况类似：FlashMLA 是针对多头潜在注意力（MLA）的高性能 Flash 实现，回滚理由是稳定性考量 [[2]](https://github.com/sgl-project/sglang/pull/21430)。

这两件事说明一个现实：处于研发密集期的推理框架，每天都在把还不够稳定的内核推进主干边界。CI 的价值就在于此——它是允许冒进合并、又能快速撤回的安全网。NVFP4 CuteDSL MoE 内核和 FlashMLA 都会回来，只是不带 CI 失败。

同日，vLLM 还修复了 DeepGemm 在 Blackwell B200 上运行 Qwen3.5-35B-A3B-FP8 时的精度退化问题 [[3]](https://github.com/vllm-project/vllm/pull/38083)——DeepGemm 强制使用的 E8M0 power-of-2 ceiling 格式在 B200 上损失了精度，约 12pp 的差距；以及多节点 allreduce fusion 的修复 [[4]](https://github.com/vllm-project/vllm/pull/38136)，flashinfer trtllm allreduce backend 此前在多节点下不工作。

在功能侧，vLLM 合并了 `/v1/chat/completions/batch` 批量端点 [[5]](https://github.com/vllm-project/vllm/pull/38011)，单请求可携带多条独立对话，面向批量评估和离线处理场景；同时为 InternVL/InternViT 视觉编码器添加了 torch.compile 支持 [[6]](https://github.com/vllm-project/vllm/pull/38049)，以及 Cohere Transcribe 语音模型的正式启用 [[7]](https://github.com/vllm-project/vllm/pull/38120)。

SGLang 在修复侧也有几个值得注意的 PR：Kimi K2.5 在 DP Attention + 投机解码组合下的启动崩溃修复 [[8]](https://github.com/sgl-project/sglang/pull/21391)、MxInt4 MoE 的静默输出错误（写入了正确 buffer 但返回了错误变量）[[9]](https://github.com/sgl-project/sglang/pull/21348)，以及 Qwen3-next 量化权重加载时的 property setter 问题 [[10]](https://github.com/sgl-project/sglang/pull/21313)。这些都是量化模型路径在 MoE/MLA 复杂组合下暴露的细节问题。

AMD 侧，SGLang 为 AMD GPU 的 MoE softmax scoring 集成了 aiter 的融合 topk 内核 [[11]](https://github.com/sgl-project/sglang/pull/21421)，减少多次 kernel launch 的开销；同时扩散模型的 rotary embedding 内核改为每 token 处理多个 head [[12]](https://github.com/sgl-project/sglang/pull/21387)，降低 kernel launch 频率。对于国产 GPU，SGLang 新增了 Moore Threads MUSA 设备的约束解码支持 [[13]](https://github.com/sgl-project/sglang/pull/21296)。

## 二、LMCache 的三路底层扩张

LMCache 在这个窗口的三个合并方向值得放在一起看。

第一，block-id 级 MP 模式传输内核 [[14]](https://github.com/LMCache/LMCache/pull/2838)。此前 MP 模式下的 KV 传输是 flat token-level slot 映射操作，粒度偏细，存在冗余计算。新的 `multi_layer_block_kv_transfer` 内核以 block ID 为单位操作，与现代 attention 实现的 paged attention 粒度对齐，向 GPU 显存带宽极限靠近。

第二，HND KV 格式支持 [[15]](https://github.com/LMCache/LMCache/pull/2826)。KV cache 的张量排布通常有 NHD（Nodes-Head-Dim）和 HND（Head-Nodes-Dim）两种主流格式，不同的 attention kernel 可能采用不同排布。LMCache 增加 HND 支持，意味着它可以和更广泛的 attention 实现直接对接，不再需要在接口层做格式转换。

第三，AMD hipFile GPU-direct storage [[16]](https://github.com/LMCache/LMCache/pull/2799)。这是 ROCm 生态里等同于 NVIDIA cuFile 的能力——允许 GPU 直接从存储设备读写数据，绕过 CPU 内存。对于需要把 KV cache 持久化到 NVMe 的大规模 PD 分离部署，AMD GPU 用户此前缺乏这条快速路径，现在有了。

三个方向同日落地，指向 LMCache 在 KV cache 底层工程上的系统性推进，不再是零散补丁。

## 三、训练侧：Muon 向规模上限推进，Megatron 精度灵活性提升

DeepSpeed 今天把 Muon 优化器扩展到了 ZeRO Stage 3 [[17]](https://github.com/deepspeedai/DeepSpeed/pull/7919)。Muon 是基于 Newton-Schulz 迭代近似正交化的优化器，相比 Adam 在某些场景下收敛更好；此前它只支持 ZeRO Stage 1/2，Stage 3 是参数和优化器状态同时分片的最高分布式级别，支持后 Muon 可用于显存极度受限的超大规模训练。

Megatron-LM 侧，两个方向的进展值得关注。一是局部 FP32 梯度累积 [[18]](https://github.com/NVIDIA/Megatron-LM/pull/4028)：允许对 embedding 等数值敏感参数独立使用 FP32 梯度累积，其余参数维持 BF16，在不全面升精度的前提下为关键参数提供更高稳定性。这对超大词汇表的训练或容易出现梯度数值问题的模型结构有实际意义。

二是前缀缓存感知路由的负载均衡改善 [[19]](https://github.com/NVIDIA/Megatron-LM/pull/3930)：在最长前缀匹配相同的多个 rank 中，优先选择负载较低的 rank，避免热点副本过载。这是推理集群在缓存感知调度上的细节打磨，对大规模多副本部署的服务均匀性有实际改善。

同时，Muon 优化器新增了 `--muon-coefficient-type` 参数 [[20]](https://github.com/NVIDIA/Megatron-LM/pull/3927)，支持在 simple/quintic/polar_express/aol 等多种 Newton-Schulz 多项式系数集间切换，方便研究者在同一框架下探索不同配置。

## 四、Ray 和 llama.cpp 的稳定性修复

Ray 2.54.1 [[21]](https://github.com/ray-project/ray/releases/tag/ray-2.54.1) 是一个 patch 版本，主要修复是禁用了一个阻塞式的 hanging issue 检测器——该检测器调用 Ray State API 会阻塞调度循环，在高负载下严重降低调度性能。暂时禁用是临时方案，后续会以非阻塞方式重新实现。

Ray Serve 侧，controller histogram 指标的 replica 标签被移除 [[22]](https://github.com/ray-project/ray/pull/62088)。这个标签会使 histogram 序列数随副本数成倍增长，大规模部署下 /metrics 端点的 payload 会大到让 Prometheus 抓取超时。移除后，可观测性开销回到合理范围。

llama.cpp 统一了所有缓存路径到 `LLAMA_CACHE` 环境变量 [[23]](https://github.com/ggml-org/llama.cpp/pull/21009)，包括模型缓存、HF 缓存和 imatrix 缓存。对于在 systemd 服务中运行或使用自定义挂载路径的场景，此前需要分别设置不同的环境变量，现在统一到一个。高端 Adreno GPU 现在可以使用超出默认 `CL_DEVICE_MAX_MEM_ALLOC_SIZE` 上限的大 OpenCL buffer [[24]](https://github.com/ggml-org/llama.cpp/pull/20997)，移动端加载更大 GGUF 模型的路径打通。

## 五、OpenClaw 的安全收口

OpenClaw 今天合并了超过 14 个认证/授权相关补丁，几乎涵盖所有渠道和接入路径 [[25]](https://github.com/openclaw/openclaw/pull/55308)。Telegram、Feishu、BlueBubbles、Synology Chat 各自的 webhook 前置增加了限速，防止暴力猜测密钥；plugin HTTP runtime 恢复最小权限原则；子 agent 的 session 删除操作增加调用者 scope 验证；后端静默重连的 scope 升级被拦截；Matrix 的验证通知和 Telegram 的 callback query 都收到了 DM 访问门控。

这种集中密度——14+ 个安全 PR 在同一天合并——通常出现在一个发布周期完成功能扩展之后的系统性收口阶段。逐个修复安全点，而不是打一个大补丁，是可维护性更好的方式，也意味着每个漏洞路径都经过了独立的分析和测试。

功能侧，Telegram 轮询卡死后的自动 transport 重建 [[26]](https://github.com/openclaw/openclaw/pull/55014) 解决了长期运行中需要手动重启的痛点；Slack 新增了 upload-file 动作 [[27]](https://github.com/openclaw/openclaw/pull/54987)；MS Teams 插件新增全文消息搜索 [[28]](https://github.com/openclaw/openclaw/pull/54832)；BlueBubbles 群组参与者会用 macOS 通讯录中的真实姓名丰富显示 [[29]](https://github.com/openclaw/openclaw/pull/54984)。

TRL 侧，两个值得注意的合并：一是支持即将发布的 VLM 模型引入的 `pixel_position_ids` 视觉键 [[30]](https://github.com/huggingface/trl/pull/5374)，提前适配新格式；二是知识蒸馏训练路径集成 `trl.generation.VLLMGeneration` 统一接口 [[31]](https://github.com/huggingface/trl/pull/5351)，与 TRL 的 vLLM 集成路径对齐。

---

今天没有大版本发布，但每个方向都在踏实推进。两个回滚说明高速合并的 CI 在正常工作；LMCache 的三路扩张说明 KV cache 层的工程化正在进入硬件精细适配阶段；DeepSpeed 和 Megatron-LM 的训练侧进展说明大规模训练的优化器选择和数值稳定性正在得到更多关注。这些都是不会上头条、但三个月后会体现在生产系统稳定性上的那种进展。

## 参考来源

[1] [vLLM 回滚 FlashInfer NVFP4 CuteDSL MoE 内核](https://github.com/vllm-project/vllm/pull/38169)

[2] [SGLang FlashMLA 回滚](https://github.com/sgl-project/sglang/pull/21430)

[3] [vLLM DeepGemm E8M0 精度退化修复（Qwen3.5 FP8 B200）](https://github.com/vllm-project/vllm/pull/38083)

[4] [vLLM 多节点 allreduce fusion 修复](https://github.com/vllm-project/vllm/pull/38136)

[5] [vLLM 批量 chat completions 端点](https://github.com/vllm-project/vllm/pull/38011)

[6] [vLLM InternVL torch.compile 支持](https://github.com/vllm-project/vllm/pull/38049)

[7] [vLLM Cohere Transcribe 支持](https://github.com/vllm-project/vllm/pull/38120)

[8] [SGLang Kimi K2.5 DP Attention 崩溃修复](https://github.com/sgl-project/sglang/pull/21391)

[9] [SGLang MxInt4 MoE 静默输出错误修复](https://github.com/sgl-project/sglang/pull/21348)

[10] [SGLang Qwen3-next 量化权重加载修复](https://github.com/sgl-project/sglang/pull/21313)

[11] [SGLang AMD aiter 融合 topk 内核](https://github.com/sgl-project/sglang/pull/21421)

[12] [SGLang 扩散模型 rotary embedding 内核优化](https://github.com/sgl-project/sglang/pull/21387)

[13] [SGLang Moore Threads MUSA 约束解码支持](https://github.com/sgl-project/sglang/pull/21296)

[14] [LMCache block-id 级 MP 模式 KV 传输内核](https://github.com/LMCache/LMCache/pull/2838)

[15] [LMCache HND KV 格式支持](https://github.com/LMCache/LMCache/pull/2826)

[16] [LMCache AMD hipFile GPU-direct Storage](https://github.com/LMCache/LMCache/pull/2799)

[17] [DeepSpeed Muon 优化器 ZeRO Stage 3 扩展](https://github.com/deepspeedai/DeepSpeed/pull/7919)

[18] [Megatron-LM 局部 FP32 梯度累积](https://github.com/NVIDIA/Megatron-LM/pull/4028)

[19] [Megatron-LM 前缀缓存路由负载均衡改善](https://github.com/NVIDIA/Megatron-LM/pull/3930)

[20] [Megatron-LM Muon 优化器系数类型选项](https://github.com/NVIDIA/Megatron-LM/pull/3927)

[21] [Ray 2.54.1 调度循环性能修复](https://github.com/ray-project/ray/releases/tag/ray-2.54.1)

[22] [Ray Serve controller replica 标签移除](https://github.com/ray-project/ray/pull/62088)

[23] [llama.cpp 统一 LLAMA_CACHE 缓存路径](https://github.com/ggml-org/llama.cpp/pull/21009)

[24] [llama.cpp Adreno GPU 大 OpenCL buffer 支持](https://github.com/ggml-org/llama.cpp/pull/20997)

[25] [OpenClaw 系统性安全加固系列](https://github.com/openclaw/openclaw/pull/55308)

[26] [OpenClaw Telegram polling 自动 transport 重建](https://github.com/openclaw/openclaw/pull/55014)

[27] [OpenClaw Slack 文件上传动作](https://github.com/openclaw/openclaw/pull/54987)

[28] [OpenClaw MS Teams 消息全文搜索](https://github.com/openclaw/openclaw/pull/54832)

[29] [OpenClaw BlueBubbles 群组参与者名称丰富化](https://github.com/openclaw/openclaw/pull/54984)

[30] [TRL pixel_position_ids 视觉键支持](https://github.com/huggingface/trl/pull/5374)

[31] [TRL 知识蒸馏 VLLMGeneration 集成](https://github.com/huggingface/trl/pull/5351)
