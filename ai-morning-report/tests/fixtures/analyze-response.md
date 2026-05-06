# 素材包 — 2099-01-01

## 一、vLLM 推进 MRV2 投机解码

**主线判断：** vLLM 在 MRV2 分支合并了对 probabilistic rejection sampling 的支持，这是投机解码从 draft-verification 走向 production-ready 的关键一步。

**证据：**
- [#35461] Add probabilistic rejection sampling support in MRV2
  merged: 2099-01-01T06:12:00Z
  url: https://github.com/vllm-project/vllm/pull/35461

## 二、SGLang 引入弹性 EP 通信路径

**主线判断：** SGLang 合并了 Elastic NIXL-EP 通信路径，MoE 推理在跨节点场景下的容错能力得到加强。

**证据：**
- [#19248] Add Elastic NIXL-EP communication path
  merged: 2099-01-01T08:33:00Z
  url: https://github.com/sgl-project/sglang/pull/19248

## 今日主线判断

推理框架在投机解码和 MoE 弹性调度两个方向同步推进，说明生产可靠性正在成为下一阶段的核心竞争点。
