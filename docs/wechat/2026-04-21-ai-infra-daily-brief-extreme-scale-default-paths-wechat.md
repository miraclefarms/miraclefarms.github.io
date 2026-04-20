---
wechat_published: true
---
# 今日焦点：极端规模路径开始被收回默认主路径

**📅 2026-04-21**

> 中文：清晨的数据中心机房里，巨型 KV 缓存池、跨节点高速互联、长上下文推理面板与并行调度拓扑同时点亮，工程师在控制台观察超大内存注册、分布式通信与缓存回收状态，无文字，16:9
>
> English: A dawn AI datacenter with massive KV cache pools, cross-node high-speed fabric, long-context inference dashboards and parallel scheduling topology glowing together, engineers monitoring huge memory registration, distributed communication and cache lifecycle status, no text, 16:9

> 过去三天，更重要的变化不是又多了几个性能点，而是越来越多框架开始把长上下文、超大 KV 池、复杂并行拓扑和异构运行时这些“例外场景”收回默认实现。

---

## 推理侧

**SGLang 把 EAGLE bigram key 改成 `RadixKey` 上的 O(1) 视图[1]** - 过去一到百万上下文，热路径里光 key 物化和匹配就会吃掉明显 CPU 时间；现在它直接在原始 token 序列上暴露 bigram 语义，完整 `cache_unfinished_req` 类周期从 70ms 降到 23ms。长上下文不再被当成额外 case，而是在逼框架重写主路径。

**vLLM 在 WideEP 里移除了 naive all2all，收敛到 `allgather_reducescatter` 默认实现[2]**，**SGLang 又把 `moe_dp_size = 1` 与不同 `attention_cp_size` 的组合正式接进来[3]** - 前者是在删掉历史后门，后者是在让复杂拓扑真正可用。并行配置这件事，正在从“支持很多名字”变成“敢不敢只留下站得住的默认实现”。

**TensorRT-LLM 的 v1.2.1 release 修掉了 KV cache corruption[4]**，主干里又继续推进新的 sharding infrastructure[5]。这说明 NVIDIA 也在补同一件事：并行和部署不能继续拆成一堆模型特例，得往统一组织方式上收，属于 **[持续更新]**。

---

## 生产部署侧

**Mooncake 在 EFA transport 里加入 PTE-aware auto-split 大内存注册[6]** - 500GB 到 1500GB 的单块内存池不再因为 MR 上限和 PTE 预算直接掉出主路径。PR 给出的结果很直白：在 hugepages 和 full NIC coverage 下，吞吐能稳定到约 108 GB/s。超大 KV 池现在不只是“理论上能配”，而是要真正能作为默认部署形态成立。

**Mooncake 又把 BatchPut / BatchGet 改成两阶段并发模型[7]** - 过去每个 key 串行排队，本地 memcpy、TransferEngine 传输和远端 RPC 都会彼此阻塞；现在先把 handle 发出去，再统一等待结果，批处理终于开始像批处理。超大缓存池一旦进主路径，这类串行写法就会先暴露问题，属于 **[持续更新]**。

---

## 缓存与工具链

**SGLang 修掉了 CUDA 13.0 下 `cudaMemcpyBatchAsync` 签名变化导致的 segfault[8]** - 这不只是一个版本兼容修复，而是在说明新 CUDA 小版本也必须被当成正式运行环境，而不是让用户自己回退驱动解决。

**Mooncake 修复了 GPU VRAM 指针进入 disk-replica / SSD-offload 路径时的崩溃[9]**，**LMCache 则一边把 `use_cufile` 抽象成更通用的 `use_gds` / `gds_backend` 配置[10]，一边修掉了 MP connector 的 store bookkeeping 错误[11]** - 这些改动指向的是同一个问题：系统不能再假设所有指针、所有介质、所有命中记账都天然一致。接口必须承认真实的设备边界和后端差异，属于 **[持续更新]**。

---

> 一句话结论：**AI Infra 下一阶段的差异，不在于谁多挂了几个特性，而在于谁先把极端规模场景写成默认、稳定、可维护的主路径。**

---

## 参考

[1] SGLang 将 EAGLE bigram key 改为 `RadixKey` 的 O(1) 视图：https://github.com/sgl-project/sglang/pull/23106

[2] vLLM 在 WideEP 中移除 naive all2all，收敛到 allgather_reducescatter：https://github.com/vllm-project/vllm/pull/33728

[3] SGLang 支持 `moe_dp_size = 1` 与不同 `attention_cp_size` 的组合：https://github.com/sgl-project/sglang/pull/22003

[4] TensorRT-LLM v1.2.1 修复 KV cache corruption 并升级底层依赖：https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.2.1

[5] TensorRT-LLM 引入新的 sharding infrastructure：https://github.com/NVIDIA/TensorRT-LLM/pull/12419

[6] Mooncake 为 EFA transport 增加 PTE-aware 大内存自动切分注册：https://github.com/kvcache-ai/Mooncake/pull/1912

[7] Mooncake 将 BatchPut / BatchGet 改为两阶段并发读写模型：https://github.com/kvcache-ai/Mooncake/pull/1921

[8] SGLang 修复 CUDA 13.0 下 `cudaMemcpyBatchAsync` 的签名兼容问题：https://github.com/sgl-project/sglang/pull/23136

[9] Mooncake 修复 GPU VRAM 指针进入 disk-replica / SSD-offload 路径时的崩溃：https://github.com/kvcache-ai/Mooncake/pull/1892

[10] LMCache 将 `use_cufile` 重构为通用的 `use_gds` / `gds_backend` 配置：https://github.com/LMCache/LMCache/pull/2858

[11] LMCache 修复 MP connector store bookkeeping 导致的 KV block 静默丢失：https://github.com/LMCache/LMCache/pull/3012
