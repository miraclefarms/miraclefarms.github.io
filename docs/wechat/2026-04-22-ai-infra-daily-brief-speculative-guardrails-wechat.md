---
wechat_published: true
---
# 今日焦点：推理快路径进入生产护栏期

**📅 2026-04-22**

> 中文：清晨的数据中心控制台里，推理服务的 speculative decoding、KV Store 批处理、trace replay、Prometheus metrics 和 Serve 路由拓扑同时展开，工程师观察状态回退与健康检查面板，无文字，16:9
>
> English: A dawn AI datacenter control room showing speculative decoding, KV Store batching, trace replay, Prometheus metrics and Serve routing topology on operational dashboards, engineers monitoring fallback states and health checks, no text, 16:9

> 今天更重要的变化，是这些快路径开始补状态初始化、回退、批处理收尾、trace replay 和 metrics。

---

## 推理侧

**vLLM 修复 MRv2 stale token 导致的 speculative decoding 准确率回退[1]** - 在 PD disaggregation 下，decode 侧接收 fully prefilled 请求时，第一步就是 decode；如果 runner 复用旧 request slot，就可能读到上一个请求留下的 `last_sampled_tokens` 和 `draft_tokens`。这次修复把 GSM8K 分数从 0.82 拉回到 0.94，说明 speculative decoding 真进拆分式服务后，状态边界会直接影响模型正确性。

**vLLM 在 Mamba speculative decoding 下默认使用 `align` cache mode[2]**，**SGLang 为 adaptive speculative decoding 增加 unsupported config guard[3]** - 前者为了避开当前 `all` mode 在 SpecDec 组合下的不稳定性，先选择行为更一致的默认值；后者在遇到 DP attention、overlap scheduler、多层 EAGLE、two batch overlap、PDMux 等未支持组合时，回退到静态 speculative 参数。这组变化说明，默认快路径必须有清晰边界。

**llama.cpp 新增 `--spec-default`[4]**，把 speculative decoding 的默认配置显式暴露出来。它和前面几条放在一起看，方向很明确：speculative decoding 正在从高级用户手动拼参数，走向“可以默认开启，但必须知道什么时候退”的生产能力。

---

## 性能与模型路径

**vLLM 利用 batch invariant 优化 fused RMSNorm 调用[5]** - 这次给出约 2.1% 的端到端 latency 改善。数字不夸张，但思路很典型：当系统知道某个算子在 batch 维度上不变，就不该继续走多余 kernel 调用。

**SGLang 在 KDA 中融合 gate+cumsum，并复用 chunk index[6]** - PR 里的 microbench 显示多个配置下有 2 倍以上速度提升。随着 Kimi、Mamba、linear attention 这类非标准 attention 路径进入框架主线，优化目标也开始从传统 dense attention 外溢到更多真实模型路径。

**llama.cpp server 引入 speculative checkpointing[7]** - 对 recurrent modules 来说，部分 draft 被接受后可以回到 checkpoint 再执行更短 batch。PR 也承认这条路径不总是最快，但在重复性很强的场景里能带来明显收益。成熟的 speculative decoding 不会只讲加速，还要把回滚成本和状态保存讲清楚。

---

## 生产部署侧

**Mooncake Store 使用 `BatchGetWriteRoute` 优化 batch_put 路由查询[8]**，**并修复 transfer failure 时提前完成 batch 导致 use-after-free 的风险[9]** - 这两条都在补批处理语义。批量请求不能只是多个单请求打包，还要保证路由选择、失败传播和生命周期结束都按整批语义成立，属于 **[持续更新]**。

**Mooncake Store 为 Client 增加 HTTP metrics endpoint[10]**，**LMCache 新增 `lmcache trace` CLI[11]** - 前者让客户端暴露 Prometheus metrics、summary 和 health endpoint；后者让 storage-level trace 可以离线 info、replay 和统计。KV 数据面一旦进生产，真正麻烦的是出问题后无法复现。

**Ray Serve 暴露 backend HTTP endpoint metadata[12]**，**又接入 experimental `ray-haproxy` binary resolution[13]** - 这些改动还没有改变请求流，但它们为 replica_id 到 host:port 的解析、HAProxy 接入和后端 endpoint 管理铺好了控制面底座。

---

> 一句话结论：**快路径只有同时处理状态、失败、观测和默认值，才有资格成为生产默认路径。**

---

## 参考

[1] vLLM 修复 MRv2 stale token 导致的 speculative decoding 准确率回退：https://github.com/vllm-project/vllm/pull/39833

[2] vLLM 在 Mamba speculative decoding 下默认使用 align cache mode：https://github.com/vllm-project/vllm/pull/40454

[3] SGLang 为 adaptive speculative decoding 增加 unsupported config guard：https://github.com/sgl-project/sglang/pull/23289

[4] llama.cpp 新增 speculative decoding 默认配置开关：https://github.com/ggml-org/llama.cpp/pull/22223

[5] vLLM 利用 batch invariant 优化 fused RMSNorm 调用：https://github.com/vllm-project/vllm/pull/40413

[6] SGLang 在 KDA 中融合 gate+cumsum 并复用 chunk index：https://github.com/sgl-project/sglang/pull/23038

[7] llama.cpp server 引入 speculative checkpointing：https://github.com/ggml-org/llama.cpp/pull/19493

[8] Mooncake Store 使用 BatchGetWriteRoute 优化 batch_put 路由查询：https://github.com/kvcache-ai/Mooncake/pull/1947

[9] Mooncake Store 等待整批 transfer task 完成后再结束 batch：https://github.com/kvcache-ai/Mooncake/pull/1906

[10] Mooncake Store 为 Client 增加 HTTP metrics endpoint：https://github.com/kvcache-ai/Mooncake/pull/1934

[11] LMCache 新增 lmcache trace CLI 支持离线 trace 分析与 replay：https://github.com/LMCache/LMCache/pull/3075

[12] Ray Serve 暴露 backend HTTP endpoint metadata：https://github.com/ray-project/ray/pull/62667

[13] Ray Serve 增加 experimental ray-haproxy binary resolution：https://github.com/ray-project/ray/pull/62589
