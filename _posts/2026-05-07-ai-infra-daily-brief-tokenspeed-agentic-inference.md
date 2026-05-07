---
title: AI Infra 早报｜agentic 推理开始按用户体感重写引擎
date: 2026-05-07 08:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: LightSeek 发布 TokenSpeed 性能预览，把 coding agent 的长上下文、多轮对话和 TPS/User 下限作为推理引擎第一约束；这会把竞争焦点从离线吞吐推向用户体感、调度正确性和 kernel 专用化。
---

TokenSpeed 这篇发布文章释放的信号，已经超出“又一个推理引擎开源”。更关键的是，agentic coding 已经把推理系统的评价坐标改写了一遍：服务端不能只追求总吞吐，还要在长上下文、多轮会话、并发用户和可感知生成速度之间找 Pareto 前沿。LightSeek 给出的 TokenSpeed 预览，把 TPS/User 下限和 TPM/GPU 同时放到台面上[[1]](https://lightseek.org/blog/lightseek-tokenspeed.html)，这比传统 benchmark 更接近 Claude Code、Codex、Cursor 这类工作负载的真实压力。

这也是今天最值得记住的变化：推理引擎竞争正在从“模型能不能跑起来”进入“用户能不能持续感觉它快”的阶段。TokenSpeed 还处在 preview，官方也明确说生产硬化仍在进行[[2]](https://github.com/lightseekorg/tokenspeed)，但它给出的系统取舍已经很清楚：为 agentic workload 单独设计建模层、调度层、kernel 层和 CPU 入口；把 coding agent 当成普通聊天服务的一个流量变种，已经解释不了这类系统设计。

## 一、评价指标从总吞吐转向 TPS/User floor

LightSeek 对 coding agent 的描述很直接：上下文经常超过 50K tokens，会话常常跨越几十轮，公开 benchmark 很难覆盖这种行为[[1]](https://lightseek.org/blog/lightseek-tokenspeed.html)。TokenSpeed repo 也单独保留了 agentic benchmark 目录，分别放置 TokenSpeed 和 TensorRT-LLM 的测试路径[[3]](https://github.com/lightseekorg/tokenspeed/tree/main/test/agentic_benchmark)。因此，文章没有停留在每张 GPU 能吐多少 token 这一层，进一步把 TPS/User 作为横轴、TPM/GPU 作为纵轴，沿着并发扫描出性能前沿。这个画法的含义是：系统必须先保证单个用户仍然“觉得快”，再谈整 fleet 的吞吐效率。

![TokenSpeed 与 TensorRT-LLM 在 Kimi K2.5 / B200 agentic workload 上的 Pareto 曲线](/assets/2026-05-07-ai-infra-daily-brief-tokenspeed-agentic-inference/fig-1-kimi-k25-pareto.png)

*图 1：TokenSpeed 把 TPS/User 作为横轴，直接把用户体感速度纳入吞吐比较；在 70 TPS/User 以上，LightSeek 称 Attention TP4 + MoE TP4 配置相对 TensorRT-LLM 覆盖了更高的 Pareto 前沿。来源：LightSeek TokenSpeed 博文。*

图里的关键数字有两个：在 coding agent 关注的 70 TPS/User 以上区间，LightSeek 认为 Attention TP4 + MoE TP4 是最佳配置；相对 TensorRT-LLM，它在 batch size 1 的低延迟场景约快 9%，在 100 TPS/User 附近约高 11% 吞吐[[1]](https://lightseek.org/blog/lightseek-tokenspeed.html)。这个结论有明确边界：测试聚焦 Kimi K2.5、NVIDIA B200 和单体部署，PD disaggregation 还在清理中。也就是说，它更像是一次方向预告，而非完整生产版选型报告。

## 二、TokenSpeed 的主线是把 agentic workload 写进系统结构

TokenSpeed 的 repo README 把目标说得更工程化：TensorRT-LLM 级性能、vLLM 级易用性，并面向生产 agentic workload 追求最高性能[[2]](https://github.com/lightseekorg/tokenspeed)。为了达到这个目标，它把几个容易互相拖累的层拆开处理，而 kernel 只是其中一环。

建模层采用 local-SPMD 设计，开发者在模块边界声明 I/O placement，轻量静态编译器生成 collective 通信；调度层把控制面放到 C++ 有限状态机里，用类型系统约束 KV cache 状态转移和资源复用；执行面仍保留 Python，以便研究和工程迭代更快[[1]](https://lightseek.org/blog/lightseek-tokenspeed.html)。这种结构的价值在于，它把“长上下文、多轮、多用户、KV 资源迁移”从运行时约定变成可检查的控制逻辑。

另一个容易被低估的点是 CPU 入口。TokenSpeed 提到集成 SMG，目标是降低 CPU 侧 request entrypoint 开销[[1]](https://lightseek.org/blog/lightseek-tokenspeed.html)。这和 PyTorch 官方 SMG 文章里的观察相呼应：长输入下 HTTP/JSON 序列化成本会随 prompt 长度线性增长，gRPC/protobuf 在 7800 输入 token 场景可以带来明显吞吐优势，极端配置下输出吞吐最高达到 3.5x[[5]](https://pytorch.org/blog/lightseek-smg/)。当 GPU kernel 越来越快，CPU 编排和序列化就会更早暴露为瓶颈。

## 三、MLA kernel 是这次预览里最硬的性能证据

TokenSpeed 最具体的性能抓手是 MLA kernel。LightSeek 说 decode kernel 会把 query-sequence 轴折进 head 轴，以便更充分利用 BMM1 的 `M` tile；binary prefill kernel 则对 softmax 做了细调[[1]](https://lightseek.org/blog/lightseek-tokenspeed.html)。这类优化已经进入通用“框架调度”触及不到的区域，要求开发者对模型结构、Blackwell Tensor Core 行为和实际 agentic batch 形态同时下手。

![TokenSpeed MLA prefill 与 decode 延迟对比](/assets/2026-05-07-ai-infra-daily-brief-tokenspeed-agentic-inference/fig-2-mla-prefill-decode.png)

*图 2：上半部分显示 TokenSpeed MLA 在五类长 prefix prefill workload 上的延迟对比，下半部分显示 speculative decoding 场景下 batch size 4/8/16 的 decode 延迟变化；这张图解释了为什么 TokenSpeed 把 MLA 单独作为核心优化。来源：LightSeek TokenSpeed 博文。*

这条线也已经开始外溢到主流推理框架。vLLM PR #41778 引入 `TOKENSPEED_MLA` 后端，覆盖 Blackwell 上的 MLA prefill 和 decode，并要求用户显式通过 engine config 启用[[4]](https://github.com/vllm-project/vllm/pull/41778)。PR 描述给出的推荐边界同样很清楚：小 batch 单用户 decode 仍可能优先 FlashInfer MLA；生产 decode、尤其 decode + speculative 场景，TokenSpeed MLA 才是它想证明优势的区间[[4]](https://github.com/vllm-project/vllm/pull/41778)。这类边界比“全面更快”更可信。

## 四、今天真正值得记住的判断

TokenSpeed 仍是预览版，README 也提醒还有 Qwen 3.6、DeepSeek V4、MiniMax M2.7、PD、EPLB、KV store、VLM、metrics、Hopper/MI350 优化等工作在清理中，不建议直接用于生产部署[[2]](https://github.com/lightseekorg/tokenspeed)。因此今天不该把它读成一个马上替换 vLLM 或 TensorRT-LLM 的结论。

更合理的读法是：agentic coding 正在逼推理系统重新定义“快”。过去的快经常是离线 benchmark 的 tokens/s，现在的快要同时满足长上下文输入、用户级 TPS 下限、KV 资源安全复用、CPU 入口低开销和模型特定 kernel 优化。TokenSpeed 仍远未到终局答案，但它把这组约束放在了同一张图和同一个系统设计里。接下来 vLLM、TensorRT-LLM、SGLang 之间真正有意思的竞争，也会越来越多发生在这些交叉点上。

---

## 参考来源

[1] [TokenSpeed: A Speed-of-Light LLM Inference Engine for Agentic Workloads](https://lightseek.org/blog/lightseek-tokenspeed.html)

[2] [lightseekorg/tokenspeed GitHub repository](https://github.com/lightseekorg/tokenspeed)

[3] [TokenSpeed agentic benchmark directory](https://github.com/lightseekorg/tokenspeed/tree/main/test/agentic_benchmark)

[4] [vLLM PR #41778: Add TOKENSPEED_MLA backend](https://github.com/vllm-project/vllm/pull/41778)

[5] [PyTorch blog: SMG, The Case for Disaggregating CPU from GPU in LLM Serving](https://pytorch.org/blog/lightseek-smg/)
