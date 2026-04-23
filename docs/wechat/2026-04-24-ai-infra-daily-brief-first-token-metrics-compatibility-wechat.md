# 今日焦点：首包延迟、指标语义与兼容层正确性进入主路径

**📅 2026-04-24**

> 今天最值得记住的变化，不是谁又多卷出一个快 kernel，而是几个项目同时把第一包请求、观测语义和兼容层正确性收回了默认主路径。

---

## KV 与缓存

**Mooncake 重构 EFA SRD 共享端点[1]** - 旧模型里每增加一个 peer 都会继续消耗 QP，48 个 peer 左右就会碰到 768 QP 上限，首包握手也很重。现在它改成每个本地 NIC 共享一个 `fid_ep`，用 `fi_addr_t` 管全部 peer，cold submit 从 99ms 降到 26ms，warmup 从 17 秒降到 1.1 秒，稳态带宽还维持在约 365 GB/s。这条更新说明跨节点 KV transport 已经不能只看稳态吞吐，首包和扩展性正在变成正式指标。

**vLLM 为 KV offload connector 增加 multiple KV groups 的 load 支持[2]** - 这条改动没有 headline 数字，但信号很清楚。只要 KV cache 开始按组、按层级或按异构资源拆分，装载路径就不能继续假设只有一个 group。offload 正在从实验能力变成正式资源层，主框架必须先把语义补齐，属于 **[持续更新]**。

---

## 可观测性

**LMCache 用单调时钟修复 CUDA host callback trace 时间戳[3]** - 过去 `mp.store` 这类 span 会在 NTP 微调时出现结束时间早于开始时间，Jaeger 最后显示成极大的溢出时长。表面上 trace 没丢，实际上语义已经坏了。LMCache 这次修的不是展示层，而是最底层的时间语义。

**Ray Serve 一边过滤 `serve_long_poll_latency_ms` 里的 bootstrap 陈旧观测[4]，一边让 HTTP proxy 路径真正支持 per-request timeout 和 disconnect header[5]** - 前者解决“新副本刚起来，指标却像延迟了几十分钟”的假象；后者解决 direct ingress 和 proxy 两条入口对同一请求给出不同超时语义的问题。这两条放在一起看，Serve 平台现在更在意“这个指标和这个 header 到底是不是同一回事”。

**SGLang 为 diffusion 路径补上 OpenTelemetry tracing[6]** - 这意味着多模态 runtime 不再满足于“能出图”，而是开始进入现有生产观测体系。只要 LLM 路径可 trace、diffusion 路径还是黑盒，多模态系统就还没真正进入同一个运维面，属于 **[持续更新]**。

---

## 兼容层与正确性

**llama.cpp 修复 Anthropic API 路径下的 prefix caching[7]** - 问题出在 `x-anthropic-billing-header` 里的 `cch` 值会变，导致长上下文虽然只改了后面一小段，前缀匹配却总在很早位置断掉。修完之后，Anthropic 兼容层终于不再持续破坏 cache reuse。兼容层现在已经是 agentic coding 和 tool-use workload 的正式入口，不再只是 adapter。

**llama.cpp 同时修掉 `n_discard` 负值触发的 heap-buffer-overflow 漏洞[8]** - 这条更新提醒得很直接：只要 `llama-server` 被更多工具和客户端直接调用，请求解析边界就是公开攻击面，不能再当成内部参数处理。

**TensorRT-LLM 修复 AutoDeploy 多流 MoE 里的 TP deadlock[9]，DeepSpeed 修复 ZeRO-1/2 CPU offload 多次 backward 时只保留最后一段梯度的问题[10]** - 前者把 CPU 侧同步换成 `record_stream`，避免 NCCL collective 在多 rank 下互相卡死；后者则补上训练框架在复杂 accumulation 边界里的真实梯度语义。它们都在说明一件事：复杂执行路径已经进入真实用户路径，不能再靠默认前提兜底。

---

> 一句话结论：**AI Infra 下一阶段的差异，不只看谁更快，而看谁先把第一包、观测语义和复杂路径正确性一起写成默认能力。**

---

## 参考

[1] Mooncake 重构 EFA SRD 共享端点，消除 per-peer QP cliff 并降低首包时延：https://github.com/kvcache-ai/Mooncake/pull/1944

[2] vLLM 为 KV offload connector 增加 multiple KV groups 的 load 支持：https://github.com/vllm-project/vllm/pull/39402

[3] LMCache 使用单调时钟修复 CUDA host callback trace 时间戳：https://github.com/LMCache/LMCache/pull/3103

[4] Ray Serve 过滤 `serve_long_poll_latency_ms` 中的 bootstrap 陈旧观测：https://github.com/ray-project/ray/pull/62868

[5] Ray Serve 在 HTTP proxy 路径支持 per-request timeout 和 disconnect 语义：https://github.com/ray-project/ray/pull/62867

[6] SGLang 为 DiffGenerator 和 diffusion worker 增加 OpenTelemetry tracing：https://github.com/sgl-project/sglang/pull/21254

[7] llama.cpp 修复 Anthropic API 路径下的 prefix caching：https://github.com/ggml-org/llama.cpp/pull/21793

[8] llama.cpp 修复 `n_discard` 负值触发的 heap-buffer-overflow 漏洞：https://github.com/ggml-org/llama.cpp/pull/22267

[9] TensorRT-LLM 修复 AutoDeploy 多流 MoE 中的 TP deadlock：https://github.com/NVIDIA/TensorRT-LLM/pull/13220

[10] DeepSpeed 修复 ZeRO-1/2 CPU offload 多次 backward 时的梯度丢失：https://github.com/deepspeedai/DeepSpeed/pull/7981
