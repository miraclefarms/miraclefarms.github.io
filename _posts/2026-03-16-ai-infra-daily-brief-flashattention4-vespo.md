---
title: AI Infra 早报｜Transformers FlashAttention4 正式启航，TRL VESPO 带来强化学习优化新范式
date: 2026-03-16 05:30:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: Transformers 首次支持 FlashAttention4（要求 torch >= 2.3.3），TRL 推出 VESPO 强化学习优化算法（比官方快 25%），vLLM 新增 GDN 内核选择配置，AI Infra 框架层进入"新一代注意力"与"强化学习优化"双重迭代窗口。
tags: [Attention, Training, Transformers]
---

AI Infra 框架层今天迎来两个重要里程碑——Transformers 正式引入 FlashAttention4 初始支持（要求 torch >= 2.3.3），TRL 推出 VESPO 强化学习优化算法。这两个更新分别从注意力机制和训练优化两个方向推动框架能力边界，标志着 AI Infra 在"功能军备竞赛"后进入"深度优化"与"新一代基础设施"并行的阶段。

## 一、FlashAttention4 启航：框架级注意力机制升级

**Transformers 首次支持 FlashAttention4[[1]](https://github.com/huggingface/transformers/pull/42435)** 是今天最重要的更新。

本次更新带来一个重大 breaking change：**Flash Attention 2 只支持 torch >= 2.3.3 的版本**。这意味着依赖旧版 PyTorch 的项目需要升级依赖。官方明确指出这一变化的原因：旧版本维护负担较重，且 FA 2/FA 3 已超过 2 年生命周期。

从技术角度看，FlashAttention4 在支持 Hopper 等新硬件时能获得更优的注意力计算性能。官方初步测试显示，在 Hopper 架构 GPU 上可以获得显著的性能提升。

这是自 3 月 13 日以来量化精度落地的进一步深化——从具体量化方法到注意力机制本身的代际升级。

## 二、TRL VESPO：强化学习优化新范式

**TRL 实现 VESPO (Variational Sequence-Level Soft Policy Optimization)[[2]](https://github.com/huggingface/trl/pull/5199)** 是另一个值得重点关注的框架级更新。

VESPO 基于论文 2602.10693，是一种新的强化学习优化方法。TRL 实现相较官方版本快约 25%，同时保持等价性。这是 3 月 13 日 DPO VLM bugfix 后，TRL 首次引入的新型优化算法。

此外，TRL 还有两项配置变更：
- **vLLM mode 默认改为 colocate**[[3]](https://github.com/huggingface/trl/pull/5255)：从独立部署改为合并部署，提供 v0→v1 迁移指南
- **支持 nullable logprobs**[[4]](https://github.com/huggingface/trl/pull/5203)：允许 vLLM 响应中的 logprobs 为空值

## 三、vLLM 功能补全：GDN 与 Azure Storage

**vLLM 新增 GDN 内核选择配置**[[5]](https://github.com/vllm-project/vllm/pull/36647)为 Generalized Differential Normalization 添加内核选择开关。GDN 内核虽然性能更优，但需要 C++17 进行 JIT 编译，配置切换可在兼容性和性能间取得平衡。使用方式为 `vllm serve --gdn-prefill-backend flashinfer`。

**vLLM 新增 Azure Blob Storage 支持**[[6]](https://github.com/vllm-project/vllm/pull/34614)为 RunAI Model Streamer 添加 Azure Blob Storage 后端，扩大云存储兼容范围。

## 四、其他更新

**SGLang** 修复量化文档错误[[7]](https://github.com/sgl-project/sglang/pull/20619)并增强 IPv6 双栈支持[[8]](https://github.com/sgl-project/sglang/pull/20491)。

**Ray** 节流应用状态 gauge 报告[[9]](https://github.com/ray-project/ray/pull/61603)，减少控制循环中的不必要开销。

**OpenClaw** 实现运行时懒加载优化[[10]](https://github.com/openclaw/openclaw/pull/47593)[[11]](https://github.com/openclaw/openclaw/pull/47536)，提升启动性能和内存效率。

---

**一句话结论：今天最值得关注的信号是 AI Infra 正在从"单点功能迭代"转向"基础设施代际升级"——Transformers FlashAttention4 要求 torch >= 2.3.3 意味着旧版技术债务正在被系统性清理，TRL VESPO 的 25% 加速则表明强化学习训练的效率优化进入新阶段。这两个方向的同步推进，将在未来 1-2 个月内逐步传导至推理部署侧的性能提升。**

## 参考

[1] [Transformers FlashAttention4 初始支持](https://github.com/huggingface/transformers/pull/42435)

[2] [TRL VESPO 实现](https://github.com/huggingface/trl/pull/5199)

[3] [TRL vLLM mode 默认改为 colocate](https://github.com/huggingface/trl/pull/5255)

[4] [TRL 支持 nullable logprobs](https://github.com/huggingface/trl/pull/5203)

[5] [vLLM 新增 GDN 内核选择配置](https://github.com/vllm-project/vllm/pull/36647)

[6] [vLLM 新增 Azure Blob Storage 支持](https://github.com/vllm-project/vllm/pull/34614)

[7] [SGLang 修复量化文档](https://github.com/sgl-project/sglang/pull/20619)

[8] [SGLang IPv6 双栈支持](https://github.com/sgl-project/sglang/pull/20491)

[9] [Ray 节流应用状态 gauge 报告](https://github.com/ray-project/ray/pull/61603)

[10] [OpenClaw 懒加载非交互式插件提供者运行时](https://github.com/openclaw/openclaw/pull/47593)

[11] [OpenClaw 懒加载模型选择器提供者运行时](https://github.com/openclaw/openclaw/pull/47536)
