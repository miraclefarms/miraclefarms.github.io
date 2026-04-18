# 今日焦点：缓存生命周期开始长出正式数据面 contract

**📅 2026-04-19**

> 中文：清晨的数据中心机房里，GPU 服务器、NVMe 阵列与高速网卡组成分层推理集群，屏幕上显示缓存命中、数据回放与跨机传输路径，冷色工业风，无文字，16:9
>
> English: A dawn AI inference datacenter with GPU servers, NVMe storage arrays, and high-speed NICs forming a layered serving cluster, dashboards showing cache replay, isolation boundaries, and cross-node data paths, industrial cool tone, no text, 16:9

> 这几天更关键的变化，不是功能表又多了一项，而是缓存与拆分式 serving 的数据路径，开始被写成正式 contract。

---

## 推理侧

**LMCache 把持久化写进缓存接口[1]** - 新增 persistence interface 和 `nixl_store_dynamic` 适配器之后，缓存已经不只是进程活着时的加速层，而开始具备重启后恢复、磁盘二次查找和长期容量管理的语义。

**LMCache 补上 cache_salt 隔离[2]** 与 **StorageManager 二进制 trace[3]** - 前者让“同内容不同用户”不再天然共用一份 key，后者则把缓存 API 调用记录成可离线读取和回放的 trace 文件。换句话说，缓存系统开始同时补边界和观测，这属于 **[持续更新]**。

---

## 生产部署侧

**TensorRT-LLM 重写批量 addSequence 生命周期[4]** - 新的两阶段 claim/onboard 机制先认领可复用 block，再统一做 onboard 和分配，避免 host offload 把本来还能复用的块先挤掉。这不是小优化，而是在把 reuse 改写成调度器的前置条件。

**TensorRT-LLM 修正 reuse 与 chunking 的 token 记账[5]** - 旧逻辑把复用后的 chunk 计算量算小了，新的实现改成按真实窗口右移后的执行语义记账。调度器只有先把成本算对，后面的吞吐、容量和并发预期才不会漂。

---

## 工具链

**Mooncake 强化跨机数据面[6]** - `fi_read`、endpoint LRU eviction 和 multi-NIC striping 被放进同一条 EFA transport 主线后，跨机 KV 传输已经不只是“链路通了”，而是开始明确谁来拉数据、端点怎么复用、网卡怎么分摊大块传输。

**Mooncake 暴露实例级 SSD offload path[7]** 与 **vLLM 让拆分式 `/inference/v1/generate` 正式承接多模态特征[8]** - 前者解决多 TP worker 共用单一路径的老问题，后者则让 coordinator 可以直接把 `pixel_values` 这类预处理特征交给 worker。一个在补落盘介质的作用域，一个在补协议对象的边界，都是在把原来靠经验兜底的环节变成正式 contract。

---

> 一句话结论：**AI Infra 正在从“加功能”转向“把缓存和 serving 的隐式假设写成可恢复、可隔离、可观测的数据面 contract”。**

---

## 参考

[1] LMCache 增加 persistence interface 与 nixl 动态持久化适配器：https://github.com/LMCache/LMCache/pull/2938

[2] LMCache 将 cache_salt 写入 ObjectKey 与 IPC key，实现缓存隔离：https://github.com/LMCache/LMCache/pull/3042

[3] LMCache 为 StorageManager 增加二进制 trace 记录能力：https://github.com/LMCache/LMCache/pull/3063

[4] TensorRT-LLM 用两阶段 claim/onboard 重写批量 addSequence：https://github.com/NVIDIA/TensorRT-LLM/pull/13029

[5] TensorRT-LLM 修复 KV reuse 与 context chunking 下的 token accounting：https://github.com/NVIDIA/TensorRT-LLM/pull/12976

[6] Mooncake 为 EFA transport 增加 fi_read、LRU eviction 与 multi-NIC striping：https://github.com/kvcache-ai/Mooncake/pull/1821

[7] Mooncake 将 SSD offload path 暴露为 Python setup 参数：https://github.com/kvcache-ai/Mooncake/pull/1884

[8] vLLM 为拆分式 /inference/v1/generate 增加多模态特征支持：https://github.com/vllm-project/vllm/pull/38405
