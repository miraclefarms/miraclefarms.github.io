---
title: AI Infra 早报｜投机解码与弹性 MoE 同步推进
date: 2099-01-01 08:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: vLLM 在 MRV2 合并 probabilistic rejection sampling，SGLang 引入弹性 EP 通信路径，推理框架在投机解码和 MoE 调度两个方向同步迈向生产可靠性。
---

推理框架在投机解码与弹性 MoE 两条路线上同步推进，今天的更新说明生产可靠性正在成为核心竞争点。

## 一、vLLM 投机解码走向 Production-Ready

vLLM 在 MRV2 分支合并了 probabilistic rejection sampling 支持[[1]](https://github.com/vllm-project/vllm/pull/35461)，这是投机解码从 draft-verification 模型走向 production-ready 的关键一步。

## 二、SGLang 弹性 EP 通信路径落地

SGLang 合并了 Elastic NIXL-EP 通信路径[[2]](https://github.com/sgl-project/sglang/pull/19248)，MoE 推理在跨节点场景下的容错能力得到加强。

## 三、今天真正值得记住的判断

推理框架的竞争重心正在从"能跑"转向"生产可靠"。vLLM 和 SGLang 今天的更新都不是新功能，而是对已有机制的稳定性和容错性加固——这是框架走向 default path 的典型信号。

---

## 参考来源

[1] [vLLM Add probabilistic rejection sampling support in MRV2](https://github.com/vllm-project/vllm/pull/35461)

[2] [SGLang Add Elastic NIXL-EP communication path](https://github.com/sgl-project/sglang/pull/19248)
