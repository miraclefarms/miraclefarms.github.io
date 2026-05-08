---
title: AI Infra 早报｜虚假优化路径开始被主动下线，租户与观测语义补进主干
date: 2026-04-17 05:30:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: 过去三天，TRL 直接弃用更慢且更耗显存的 paged 训练路径，vLLM 只在真正需要时才初始化 SSU dispatch；另一边，LMCache 把 cache_salt、用户级驱逐与请求级 tracing 补进主干，Mooncake 则让跨机 SSD offload 和 SSD 指标真正进入可部署状态。
tags: [Training, KV Cache, Inference]
---

过去三天，更值得写的变化不是谁又多了一个 headline feature，而是几个项目开始主动承认一件事：很多“理论上更高级”的路径，如果没有在真实环境里证明自己，就不该继续占据默认位置。TRL 直接给 `use_transformers_paged` 判了死缓，因为它既没有实际用户，又比默认 padded 路径更慢、还多吃了六倍显存；vLLM 也把 SSU dispatch 初始化改成按需触发，没有对应 layer 时就直接变成 no-op。框架团队开始主动清理伪能力，而不是继续把它们当作功能表里的加分项。

与此同时，缓存与 offload 底座也在补另一类长期欠账。LMCache 把 `cache_salt`、用户级驱逐扩展点和 request root span 一起并入主干，说明多租户隔离与请求级可观测性已经不再是附加功能；Mooncake 则连续修掉了跨机 SSD offload 的绑定地址问题、补上 SSD 指标，并处理了 eRDMA 环境里的配置和 NUMA 崩溃。这几条更新放在一起看，AI Infra 正在从“把复杂路径做出来”转向“只保留真实有效的路径，并把缺失的生产语义补齐”。

## 一、框架开始主动撤掉“看起来更高级”的伪路径

**TRL 弃用了 `use_transformers_paged`[[1]](https://github.com/huggingface/trl/pull/5544)**，理由非常直接：这条路径几乎没人用，而且实际 benchmark 结果并不占优。官方给出的 Qwen3-0.6B GRPO 对比里，paged 路径训练时间从 1188.9 秒增加到 1428.1 秒，峰值显存从 9.60 GiB 涨到 58.17 GiB，既更慢也更重。更关键的是，维护这条路径还会把 trainer 内部 generation 分支继续分裂成 `vLLM / paged / default` 三套逻辑。这里最重要的信号不是“删了一个配置项”，而是项目方开始把未经真实工作负载验证的优化路径主动降级。

**vLLM 则把 SSU dispatch 初始化改成严格按需[[2]](https://github.com/vllm-project/vllm/pull/40039)**。如果模型里根本没有 layer 会调用这条 dispatch 路径，初始化就直接跳过，不再为空转准备一整套状态。这类 PR 看起来比不上新 kernel 或新并行策略显眼，但它代表着另一种成熟度：系统开始默认假设“额外路径本身也有成本”，没有真实调用点，就不该继续预热。换言之，框架竞争已经不只是“谁支持得更多”，而是“谁更愿意把无效复杂度从默认路径里拿掉”。

把 TRL 和 vLLM 放在一起看，一个更清楚的趋势正在出现。过去大家喜欢把分叉路径当作潜在优势先挂在那里，哪怕它只是实验性能力；但现在主流项目更倾向于让路径先接受真实 workload 审判，再决定是否保留。这意味着默认路径的可信度，正在变成比功能面更重要的资产。

## 二、缓存系统把租户边界和请求级观测补进主链路

**LMCache 给 MP adapter 全面加上了 `cache_salt` 参数[[3]](https://github.com/LMCache/LMCache/pull/3029)**，随后又在 eviction policy 上新增 `is_user_level` 扩展点和同名 `cache_salt` 输入[[4]](https://github.com/LMCache/LMCache/pull/3032)。这两条改动单独看都还只是接口预埋，但连在一起就不再是小修：它们说明 LMCache 已经明确朝“按用户或租户作用域做缓存隔离和配额控制”的方向推进。缓存系统过去更像全局共享池，谁先命中谁先受益；一旦引入 salt 和用户级驱逐接口，L2 cache 就开始具备真正的多租户治理语义。

**同一时间，LMCache 又为 MP server tracing 补上了 per-request root OTel span 与 SpanRegistry[[5]](https://github.com/LMCache/LMCache/pull/3033)**。这不是简单加几个日志点，而是把 lookup_prefetch、retrieve、store 这些子操作正式挂到统一的 request span 下面，并处理 GPU callback 尚未完成时的 span 关闭时机。它还顺手给了一个 Collector、Tempo、Grafana 的完整 Compose 栈。也就是说，LMCache 现在不只想让你“知道系统在工作”，而是想让你看清一整个请求在缓存层到底卡在哪一段、跨了哪些子阶段。

这组变化真正重要的地方，在于它们把隔离语义和观测语义一起往前推。前者解决“谁该占用多少缓存”的边界问题，后者解决“这次请求究竟在缓存链路的哪一层变慢”的定位问题。缓存系统一旦开始同时补这两件事，就说明它正在从性能插件变成真正的共享基础设施。

## 三、跨机 offload 不再停留在单机演示

**Mooncake 修掉了 standalone client RPC 硬编码绑定 `127.0.0.1` 的问题[[6]](https://github.com/kvcache-ai/Mooncake/pull/1900)**，把地址改成 `FLAGS_host`，默认与 master 侧一致，允许真正跨机器访问 SSD offload 服务。这个修复的价值远大于那几行代码本身，因为它直接指出此前一条“看起来能远程部署”的路径其实在 TCP 层就会被拒绝。只有把 bind address 从 localhost 拿开，跨机 SSD offload 才不是文档里的部署想象，而是能被真实客户端打到的网络服务。

可部署只是第一步，**Mooncake 还给 SSD offload 正式补上了吞吐、IOPS 和 P50/P90/P99 延迟指标[[7]](https://github.com/kvcache-ai/Mooncake/pull/1879)**。读写路径分别有 bytes、ops、histogram 和 summary，说明团队已经不满足于在 VLOG 里看一次 batch 时间，而是开始把 SSD 层当成需要长期观测和聚合的生产部件。配合前面的跨机修复，这意味着 Mooncake 正在把 SSD offload 从“有这项能力”推进到“出了问题也能量化定位”。

**另一条同步合并的 TENT 修复则补上了 eRDMA 环境下最容易把系统直接打崩的边角条件[[8]](https://github.com/kvcache-ai/Mooncake/pull/1894)**。一方面，`tent_set_config` 通过字符串注入配置时会触发错误类型解析；另一方面，eRDMA 设备返回的 `numa_node = -1` 会导致 rail monitor 直接越界。修完之后，TCP 和 RDMA 模式都不再因为这些假设失效而崩溃。到这一步可以看得很清楚，Mooncake 的重点已经不是“再多接一条传输路径”，而是先把现有跨机与异构部署路径变成不会轻易翻车的主线能力。

## 四、今天真正值得记住的判断

今天最值得记住的，不是某个项目新增了什么能力，而是谁开始主动承认哪些能力其实不该继续留在默认路径里。TRL 把看起来更高级、实际却更差的 paged 路径往外推，vLLM 把不会被调用的初始化逻辑收回去；与此同时，LMCache 和 Mooncake 都在补那些过去最容易被忽略、却最决定生产质量的语义层: 多租户隔离、请求级 tracing、跨机网络可达性和 SSD 数据面指标。

如果这个趋势继续下去，下一阶段 AI Infra 的竞争就会更像一次“主路径清算”。真正留下来的，不会是 feature list 上最花哨的那一条，而是那些在真实 workload、真实租户边界和真实故障场景下仍然值得保留的默认能力。

---

## 参考来源

[1] [TRL 弃用 use_transformers_paged，因其更慢且更耗显存](https://github.com/huggingface/trl/pull/5544)

[2] [vLLM 仅在存在调用层时初始化 SSU dispatch](https://github.com/vllm-project/vllm/pull/40039)

[3] [LMCache 为 MP adapter 接口引入 cache_salt 参数](https://github.com/LMCache/LMCache/pull/3029)

[4] [LMCache 为 EvictionPolicy 增加用户级驱逐扩展点与 cache_salt](https://github.com/LMCache/LMCache/pull/3032)

[5] [LMCache 为 MP server tracing 增加 per-request root OTel span 与 SpanRegistry](https://github.com/LMCache/LMCache/pull/3033)

[6] [Mooncake 修复 standalone client RPC 的 127.0.0.1 绑定，允许跨机 SSD offload](https://github.com/kvcache-ai/Mooncake/pull/1900)

[7] [Mooncake 为 SSD offload 增加吞吐、IOPS 与延迟指标](https://github.com/kvcache-ai/Mooncake/pull/1879)

[8] [Mooncake 修复 eRDMA 场景下的配置类型推断与负 NUMA 节点崩溃](https://github.com/kvcache-ai/Mooncake/pull/1894)
