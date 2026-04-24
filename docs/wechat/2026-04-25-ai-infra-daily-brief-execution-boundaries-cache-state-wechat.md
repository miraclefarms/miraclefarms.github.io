# 今日焦点：执行边界与缓存状态开始被写回主路径

**📅 2026-04-25**

> 今天最值得记住的变化，是几个主流项目同时把“复杂执行边界”收回了默认主路径，系统不再只为理想路径而设计。

---

## 推理侧

**SGLang 引入 breakable piecewise CUDA graph[1]** - 这条更新的真正价值，不是又多了一个 graph 开关，而是它开始拆掉“图必须整段连续捕获”的前提。运行时终于承认，真实线上环境里的图执行经常会被切开、被打断、被局部重组。只要这一步成立，CUDA graph 才更像生产能力，而不只是 benchmark 技巧。

**TensorRT-LLM 修掉 DSA、host KV cache offload 和 CUDA graph 叠加时的非法地址访问[2]** - 问题本质是 offload 交换后，block ID 和真实 memory pool index 不再一一对应，但 replay 还按旧假设去算偏移，最后把地址打出 buffer。它说明复杂推理组合路径已经进入真实用户路径，单独每条链路都能跑，并不代表叠起来仍然成立。

**DeepSpeed 让 dynamic offload 兼容 static optimizer offload[3]** - 训练侧看起来是小修，信号却很强。框架开始默认用户会把多种资源调度策略叠在一起用，而不是只跑一条最干净的配置路径。

---

## KV 与缓存

**Megatron-LM 为 prefix caching 增加 per-block MoE routing storage[4]** - 这条更新很关键，因为它让 prefix cache 保存的内容从 token 本身扩展到了生成 token 时走过的路由状态。只要对象是 MoE 模型，前缀复用想真正成立，路由信息就必须被一起保存。

**LMCache 在 MP 模式加入 S3 L2 adapter[5]** - 这意味着远端对象存储开始被正式纳入缓存分层体系。它不只是“多一个后端”，而是把容量统计、驱逐、失败熔断和对象命名一起写进了多进程缓存主链路。

**Mooncake 在 Get 路径实现 batch route query[6]** - 读路径终于开始正面处理“查路由本身就是成本”这件事。只要 KV 系统越来越分层、越来越分布式，批量路由查询就会从优化项变成基础能力。

---

## 生产部署侧

**Ray Serve 修复 controller 在 shutdown 中途崩溃时产生 orphaned actors 的问题[7]** - 旧路径会过早删除 checkpoint，导致重启后的 controller 根本不知道还有哪些 deployment 没清干净。现在它把 shutdown 进行中的状态先持久化，再把 checkpoint 删除放到最后，控制面开始把“恢复到一半的 shutdown”当成正式状态处理。

**LMCache 为 BlendEngineV2 增加 per-request root OTel span 和完整的 CB 事件订阅体系[8]** - 这条变化让 cache blending 这类异步路径终于能和主请求生命周期连成一棵完整的 span 树。观测只有在这种时候才是真的，不然看到的只是很多散碎事件，属于 **[持续更新]**。

**llama.cpp 让 WebGPU 在没有 subgroup matrix 的浏览器里也能启用 `FLASH_ATTN_EXT`[9]** - 它当然还没有最强路径那么快，但方向已经很明确：浏览器后端不再满足于“勉强能跑”，而是开始补真正能用的 attention 快路径。

---

> 一句话结论：**AI Infra 下一阶段的分水岭，在于谁先把被打断、被分层、被恢复后的复杂路径也写成默认能力。**

---

## 参考

[1] SGLang 引入实验性的 breakable piecewise CUDA graph：https://github.com/sgl-project/sglang/pull/22218

[2] TensorRT-LLM 修复 DSA 与 host KV cache offload 在 CUDA graph 下的非法地址访问：https://github.com/NVIDIA/TensorRT-LLM/pull/13124

[3] DeepSpeed 让 dynamic offload 兼容 static optimizer offload：https://github.com/deepspeedai/DeepSpeed/pull/7979

[4] Megatron-LM 为 prefix caching 增加 per-block MoE routing storage：https://github.com/NVIDIA/Megatron-LM/pull/4301

[5] LMCache 为 MP 模式增加 S3 L2 adapter：https://github.com/LMCache/LMCache/pull/3064

[6] Mooncake 在 Get 路径实现 batch route query：https://github.com/kvcache-ai/Mooncake/pull/1970

[7] Ray Serve 修复 shutdown 中途崩溃导致的 orphaned actors：https://github.com/ray-project/ray/pull/62823

[8] LMCache 为 BlendEngineV2 增加 per-request root OTel span 与 SpanRegistry：https://github.com/LMCache/LMCache/pull/3062

[9] llama.cpp 让 WebGPU 在无 subgroup matrix 的浏览器中启用 FLASH_ATTN_EXT：https://github.com/ggml-org/llama.cpp/pull/22199
