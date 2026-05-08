---
title: AI Infra 早报｜主流项目聚焦边界场景稳定性优化，高并发与冷启动问题成为关注重点
date: 2026-03-15 06:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: vLLM 修复高并发 sporadic stalls，Ray Serve 修补冷启动 autoscaling 放大问题，llama.cpp 继续深耕 CPU/GPU 底层性能优化，AI Infra 竞争正从功能迭代转向稳定性治理。
tags: [Inference, KV Cache, vLLM]
---

过去 24 小时内，AI Infra 主流项目从"功能迭代"转向"稳定性治理"——vLLM 修复高并发场景下的 sporadic stalls，Ray Serve 修补冷启动期间的 autoscaling 放大问题，llama.cpp 在 CPU/GPU 层面继续深耕性能优化。这些变化共同指向一个趋势：**生产级推理部署的边界场景正在被系统性收敛**。

## 一、vLLM 高并发 sporadic stalls 修复：生产稳定性的深度优化

今天最值得重点关注的更新是 **vLLM 修复 async_copy_to_gpu 中 pin_memory 导致的高并发 sporadic stalls[[1]](https://github.com/vllm-project/vllm/pull/37006)**。

这个 bug 的特别之处在于：它只在高并发场景下才会暴露。pin_memory() 在 32 并发以上时会导致 5-12ms 的 sporadic stalls——注意是"sporadic"，意味着不是每次都出现，而是在特定并发条件下偶发。这对于需要稳定延迟的在线推理服务来说非常棘手：测试阶段难以复现，上线后却会偶发抖动。

更关键的是 benchmark 数据揭示的问题深度：大 tensor（256、512 维度）场景下，pin_memory() 比 direct copy **慢约 300 倍**，P50 从 0.025ms 飙升到 7.5ms，P99 最高达到 12ms。这个差距意味着什么？如果你的 TTS 服务或其他生成式任务需要处理较大的中间 tensor，理论上的 8ms 推理步时间会偶发变成 70ms+ 的抖动。

修复方案很直接：移除 pin_memory() 调用，改用直接 copy。效果也很显著：生产环境 TTS 工作负载的可持续并发从 24 提升到 48，接近翻倍。

这个更新之所以重要，不仅在于它解决了一个具体的性能 bug，更在于它揭示了一个更深层的趋势：**推理框架正在从"单请求性能优化"转向"高并发稳定性治理"**。当benchmark 分数越来越接近理论上限时，真正的分水岭变成 了"谁能在生产环境的复杂负载下保持稳定"。

## 二、Ray Serve 冷启动放大：autoscaling 反馈回路的根因修复

**Ray Serve 修复冷启动期间的 autoscaling 反馈回路放大问题[[2]](https://github.com/ray-project/ray/pull/61731)** 是另一个值得重点关注的稳定性修复。

问题的根因很有意思：冷启动时 current_num_replicas=0，scaling factor 会把整个 target 作为 delta 来放大。用具体数字感受一下：

- Tick 0: target=2 → ceil(2.0 × 2) = 4
- Tick 1: target=4 → ceil(2.0 × 4) = 8  
- Tick 2: target=8 → ceil(2.0 × 8) = 16， clamped to max=10

3 个 tick 内，从 min_replicas=2 直接飙升到 max_replicas=10——**整个过程没有任何实际流量**。这不是bug是什么？是"在没有任何需求的情况下自我放大"。

根因追溯到 #60851：这个 PR 移除了 cold start fallback，原本是为了让自定义策略（如 AsyncInferenceAutoscalingPolicy）能够检测队列工作。但这个改动暴露了默认策略的一个设计缺陷——scaling factor 本应控制的是"从基准点的变化率"，而不是"从零开始的放大倍数"。

修复方案也很 elegant：当 current_num_replicas == 0 时，跳过 scaling factor 放大。这保留了自定义策略检测队列工作的能力（它们仍然会运行并返回期望值），同时让默认策略在冷启动时不再盲目扩容。

## 三、llama.cpp 底层优化：CPU 与 Metal 的持续深耕

**llama.cpp 增加原生 AVX512-FP16 支持[[3]](https://github.com/ggml-org/llama.cpp/pull/20529)** 是今天的第三个亮点。

这个优化的 benchmark 结果很有意思：减少 27 亿条指令执行。背后的原理是：AVX512-FP16 允许在单条指令中完成多个 FP16 操作，减少了指令调度和执行的开销。

更值得关注的是 benchmark 描述中的这句话："CPU 现在计算速度超过 RAM 数据传输速度"。这意味着 llama.cpp 在 CPU 推理上的瓶颈正在从"计算"转向"内存带宽"——这是一个标志性的转变。

**llama.cpp Metal 后端增加 FA 专门化优化[[4]](https://github.com/ggml-org/llama.cpp/pull/20549)** 则继续推进 Apple Silicon 端侧推理的优化。为特定 HSK=320、HSV=256 配置添加专门的 Flash Attention kernel，解决的是"通用 kernel 在特定配置下效率不达最优"的问题。

结合 3 月 14 日 SGLang 新增 MLX 后端来看，端侧推理正在成为 AI Infra 的重要战场——不只是"能用"，而是"好用"。

## 四、OpenClaw 持续迭代

**OpenClaw 在 Control UI 中显示 gateway restart 原因[[5]](https://github.com/openclaw/openclaw/pull/46580)** 解决了 dashboard UX 的一个痛点：配置更改触发的 gateway 重启会显示为"disconnected (1006): no reason"，排查时很难区分是"真崩溃"还是"正常重启"。

现在 dashboard 会捕获并显示 gateway 发出的 shutdown 事件原因——这个问题看似小，但实际影响排查效率，尤其是自动化运维场景下需要根据重启原因做不同处理。

**OpenClaw 恢复非原生 openai-completions providers 的 usage tracking[[6]](https://github.com/openclaw/openclaw/pull/46500)** 则补上了多 provider 环境下的成本可视化缺口。

## 五、趋势判断

今天的更新虽然分散在 vLLM、Ray、llama.cpp、OpenClaw 四个项目，但**它们回答的是同一个问题：如何在生产环境中把"理论性能"真正转化为"可持续的稳定吞吐"**。

这和前几天"功能军备竞赛"的氛围有明显区别。3 月 12 日大家在比拼 async 调度、量化格式；3 月 13 日大家在推 KV 剪枝、QLoRA 训练；3 月 14 日 TensorRT-LLM 发布大版本强调"全栈可配"。而今天，所有人的焦点都收敛到一个更务实的问题：**边界场景的稳定性**。

这个转变可能比任何一个单点 feature 都更重要。因为当功能趋同之后，真正的竞争壁垒不在于"我能做什么"，而在于"我什么条件下都能稳定地做"。

---

## 参考

- [1] vLLM 修复 async_copy_to_gpu 中 pin_memory 导致的高并发 sporadic stalls：https://github.com/vllm-project/vllm/pull/37006
- [2] Ray Serve 修复冷启动期间的 autoscaling 反馈回路放大问题：https://github.com/ray-project/ray/pull/61731
- [3] llama.cpp 增加原生 AVX512-FP16 支持：https://github.com/ggml-org/llama.cpp/pull/20529
- [4] llama.cpp Metal 后端增加 FA 专门化优化：https://github.com/ggml-org/llama.cpp/pull/20549
- [5] OpenClaw 在 Control UI 中显示 gateway restart 原因：https://github.com/openclaw/openclaw/pull/46580
- [6] OpenClaw 恢复非原生 openai-completions providers 的 usage tracking：https://github.com/openclaw/openclaw/pull/46500
