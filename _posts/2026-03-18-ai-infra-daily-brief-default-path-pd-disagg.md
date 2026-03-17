---
title: AI Infra 早报｜vLLM 将 Flashinfer 稀疏 MLA 立为默认后端，PD 解耦健壮性全线收紧
date: 2026-03-18 05:30:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: vLLM 将 Flashinfer sparse MLA 设为 FP8 KV Cache 默认后端，DeepSeek 系模型 FP8 推理无需手动配置即可受益；SGLang 在 PD 解耦侧完成多处健壮性修复（health check 语义精确化、hicache 配置前置校验、AllReduce 融合回滚）；Ray Serve 修复引入自 2025 年 8 月的 P99 延迟回归根因。AI Infra 框架层正在将"功能堆叠"遗留的欠债以稳步方式偿还。
---

过去 24 小时，AI Infra 框架层的重心不在"新功能"，而在"把已有功能做到最优"。vLLM 将 Flashinfer sparse MLA 立为 FP8 KV Cache 的默认后端，SGLang 集中修复 PD 解耦侧的多处边界问题，Ray Serve 追溯到一个潜伏近 7 个月的 P99 延迟回归根因。这些更新共同指向同一判断：**主流推理框架正在进入"默认行为对齐生产最优"的成熟阶段。**

## 一、vLLM：把最优性能设为默认

### Flashinfer sparse MLA 成为 FP8 KV Cache 默认后端

今天最值得关注的变化，是 **vLLM 将 Flashinfer sparse MLA 设为 FP8 KV Cache 的默认后端**[[1]](https://github.com/vllm-project/vllm/pull/37252)。

背景在于：DeepSeek-V3 等使用 MLA（Multi-head Latent Attention）架构的模型，在 FP8 量化部署时，此前需要手动切换才能获得 Flashinfer sparse 的最优性能。PR 附带的 E2E benchmark 涵盖 TP=1/4/8 配置，结果显示 Flashinfer 在多数场景的 pareto 最优上显著领先 flashMLA（仅 TP=2 略有落后）。从今天起，这成为开箱默认行为。

对生产用户而言，这意味着：**使用 vLLM + DeepSeek 系模型 + FP8 量化的部署，升级后无需任何配置变更即可获得性能提升。** 这是框架层"默认行为对齐生产最优"的典型范式——不需要用户懂内核选择，直接受益。

### OpenAI 渲染层重构完成

与此同时，**vLLM 完成了 OpenAI 渲染层的完整重构**[[2]](https://github.com/vllm-project/vllm/pull/37287)，将 ServingTokens、ServingPooling、ServingChat 等所有前处理逻辑委托给 `OpenAIServingRender`（GPU-less 渲染层），清理了基类中的冗余副本。

这是自 #36166 以来多 PR 系列的收尾。重构的意义不在于单次改动，而在于架构方向：**前端渲染逻辑（tokenization、template 处理）现在完全与 GPU 计算分离**，为 P/D 解耦架构下的前端独立部署铺平道路。

**vLLM 还支持通过 Plugin 覆盖 KV Connector 元数据构建逻辑**[[3]](https://github.com/vllm-project/vllm/pull/37336)，将 `build_connector_meta` 提取为独立方法，允许插件传递额外的 SchedulerOutput 信息而无需修改核心 struct。对自定义 KV 缓存连接器（外部 KV 存储、跨节点 KV 共享）的开发者来说，这是一个关键扩展点。

## 二、SGLang：PD 解耦从"可用"走向"可控"

SGLang 今天在 PD 解耦侧完成了多处修复，合在一起看是一个清晰的信号：**这套部署模式正在经历从"功能可用"到"运维可控"的关键转变。**

### 健康检查语义精确化

**SGLang 修复了 PD 解耦中 `is_fully_idle` 的 false-positive 问题**[[4]](https://github.com/sgl-project/sglang/pull/20756)。`bootstrap_queue`、`prealloc_queue`、`transfer_queue` 三个队列可能在没有 GPU 请求运行的情况下仍有内容（stuck handshake、KV cache 满、stalled transfer），此前这会导致 `process_output` 的健康检查结果被错误捆绑。修复后，健康检查 idle 条件仅由 `running_batch` + `waiting_queue` 决定，语义更为精确。

### 配置错误前置拦截

**SGLang 新增了 PD 解耦 Decode 侧启用 KV caching 时的 hicache-storage-backend 校验**[[5]](https://github.com/sgl-project/sglang/pull/20732)，防止用户在未配置 hicache 存储后端的情况下启用 KV caching——此前这会在运行时静默失败，排查成本极高。

### AllReduce 融合 API 回滚

**SGLang 回滚了 TRT-LLM MNNVL AllReduce 与 Flashinfer AllReduce fusion 的统一 API**[[6]](https://github.com/sgl-project/sglang/pull/20792)。这一回滚释放的信号值得注意：NVLink multi-node 场景下 AllReduce 融合的生产化比预期更复杂，目前尚不具备发布条件。

### Kimi-K2.5 支持修缮

**SGLang 修复了 Kimi-K2.5 模型的 piecewise CUDA graph 支持**[[7]](https://github.com/sgl-project/sglang/pull/20747)，在 `KimiK25ForConditionalGeneration` 中补充 `self.model` alias，对齐 piecewise CUDA graph 的接口约定。

## 三、Ray Serve：追溯 7 个月的 P99 回归根因

**Ray Serve 修复了一个 P99 延迟回归**[[8]](https://github.com/ray-project/ray/pull/61755)，根因定位颇具教育意义。

问题出在路由器的队列长度 cache：`on_send_request` 时递增，但请求完成后**从未递减**。这导致 cache 条目长期"卡死"在 `>= max_ongoing_requests` 的值，迫使每次路由决策都降级为阻塞 probe RPC，P99 延迟因此显著抬升。

该回归由 2025 年 8 月路由器重构引入，潜伏近 7 个月后被定位。受影响最明显的场景是 `max_ongoing_requests=1` 的部署。修复方式简洁：在请求完成时正确递减 cache 计数。

这类"引入容易、发现难"的回归，在推理服务框架的成熟化过程中几乎不可避免。值得关注的不只是修复本身，而是 Ray 团队愿意公开溯源的态度。

## 四、训练侧：TRL DPO VLM 持续修缮 [持续更新]

**TRL 修复了 DPO VLM "keep_end" 截断模式的训练数据损坏问题**[[9]](https://github.com/huggingface/trl/pull/5286)：当 `truncation_mode="keep_end"` 与 VLM 训练同时使用时，会导致训练数据静默损坏。修复方案选择"快速失败"策略——初始化时直接抛出 `ValueError`，拒绝该危险组合。

另外，**TRL 消除了 CausalLM 作为 reward model 加载时 `lm_head.weight` 的误报警告**[[10]](https://github.com/huggingface/trl/pull/5295)，减少了训练日志中的噪音。

DPO VLM 相关修复自 3 月 13 日起已连续五天推进，方向从 mm_token_type_ids 处理，到今天的截断模式语义修复，逐步收敛到更严格的 VLM 训练安全保障。

## 五、应用侧：OpenClaw 扩展生态与安全加固

**OpenClaw 新增 Chutes 内置扩展**[[11]](https://github.com/openclaw/openclaw/pull/49136)，将 Chutes（AI-native 推理 provider）作为 bundled extension 集成，支持 plugin-owned OAuth 和 API key 认证，同时泛化了内置 provider 默认启用与认证发现机制，为后续更多 provider 的集成打通了通用路径。

**OpenClaw 加固了主机执行沙箱**[[12]](https://github.com/openclaw/openclaw/pull/49025)，将 `JAVA_TOOL_OPTIONS`、`_JAVA_OPTIONS`、`JDK_JAVA_OPTIONS`、`PYTHONBREAKPOINT`、`DOTNET_STARTUP_HOOKS` 五个危险环境变量加入 blockedKeys。这几个变量的共同特点是：它们都允许在进程启动前注入任意代码，是 agent 执行沙箱中绕过访问控制的典型向量。

## 结论

今天最值得关注的判断是：**AI Infra 框架层正在从"添加功能"转向"把已有功能做到最优"。** vLLM 将 Flashinfer sparse MLA 立为 FP8 KV Cache 默认后端，是这一转变最清晰的例证——最优路径不再需要用户主动选择，而是成为开箱行为。SGLang PD 解耦侧的系统性修复，则表明这一部署范式正在走完"功能可用"到"运维可控"之间的最后一段路。

---

## 参考

[1] [vLLM 将 Flashinfer sparse MLA 设为 FP8 KV Cache 默认后端](https://github.com/vllm-project/vllm/pull/37252)

[2] [vLLM 完成 OpenAI 渲染层完整重构](https://github.com/vllm-project/vllm/pull/37287)

[3] [vLLM 支持通过 Plugin 覆盖 KV Connector 元数据构建](https://github.com/vllm-project/vllm/pull/37336)

[4] [SGLang 修复 PD 解耦 is_fully_idle false-positive](https://github.com/sgl-project/sglang/pull/20756)

[5] [SGLang PD 解耦 Decode 侧 hicache 配置前置校验](https://github.com/sgl-project/sglang/pull/20732)

[6] [SGLang 回滚 TRT-LLM MNNVL AllReduce 融合 API](https://github.com/sgl-project/sglang/pull/20792)

[7] [SGLang 修复 Kimi-K2.5 piecewise CUDA graph](https://github.com/sgl-project/sglang/pull/20747)

[8] [Ray Serve 修复 P99 延迟回归（队列长度 cache 未递减）](https://github.com/ray-project/ray/pull/61755)

[9] [TRL 修复 DPO VLM "keep_end" 截断模式数据损坏](https://github.com/huggingface/trl/pull/5286)

[10] [TRL 修复 CausalLM reward model 加载误报警告](https://github.com/huggingface/trl/pull/5295)

[11] [OpenClaw 新增 Chutes 内置扩展](https://github.com/openclaw/openclaw/pull/49136)

[12] [OpenClaw 加固沙箱阻断 JVM/Python/.NET 注入向量](https://github.com/openclaw/openclaw/pull/49025)
