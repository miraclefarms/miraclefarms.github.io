---
title: AI Infra 早报｜极端规模路径开始被收回默认主路径
date: 2026-04-21 05:30:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: 过去三天，更值得记住的不是谁又多卷出一个快 kernel，而是长上下文、超大 KV 池、异构 CUDA 版本和复杂并行拓扑这些过去常靠特殊分支兜底的路径，开始被项目方收回默认实现。
tags: [KV Cache, Long Context, Inference, Networking]
---

过去三天翻完几家主干仓库，我最强烈的感受不是“又多了几个 headline feature”，而是越来越多团队开始对一件事失去耐心: 那些只在极端规模、异构环境或复杂拓扑里才会触发的特殊路径，不能再继续作为主路径旁边的临时补丁存在。

这件事为什么重要？因为 AI Infra 走到今天，真正拖慢生产系统的，往往已经不是默认 case，而是那批“平时不常触发、但一旦上规模就一定会踩中”的角落。长上下文下的 radix key 构造、超大内存池在 EFA 上的注册方式、MoE 与 attention 的并行组合、CUDA 小版本升级后的 memcpy ABI、GPU 指针落到 SSD offload 路径时到底该怎么处理，这些问题过去都容易被当成例外。最近三天的更新说明，主干项目已经开始把这些例外收回默认实现。

## 一、长上下文与超大 KV 池开始被当成正常场景来写

**SGLang 把 EAGLE bigram key 改成 `RadixKey` 上的 O(1) 视图[[1]](https://github.com/sgl-project/sglang/pull/23106)**，这条更新很能说明方向。过去 `cache_unfinished_req` 的热路径会把 token 序列物化成 `List[Tuple[int, int]]`，一到百万上下文，光 key 构造和匹配就能吞掉大块 CPU 时间。现在它直接在原始 token 序列上暴露 bigram 语义，PR 给出的 microbench 结果是完整 `cache_unfinished_req` 类周期从 70ms 降到 23ms。这个数字的意义不只是“快了三倍”，而是项目方已经不再接受“长上下文场景下先多建一层临时数据结构再说”这种写法。

**Mooncake 随后在 EFA transport 里加入了 PTE-aware auto-split 的大内存注册路径[[2]](https://github.com/kvcache-ai/Mooncake/pull/1912)**，把 500GB 到 1500GB 的单块内存池也拉回了可正式支持的范围。它不只是在 buffer 超过 `max_mr_size` 时切块，而是连 backing page size、PTE 预算、NIC 覆盖范围和批量注册都一起纳入默认逻辑。PR 里给出的结果很硬: 在 hugepages 和 full NIC coverage 下，500GB 与 1500GB 池子的传输吞吐都能稳定到约 108 GB/s。换句话说，Mooncake 已经不想再把“超大 KV 池怎么注册、怎么不掉带宽”留给部署方自己绕过去。

**同一时间，Mooncake 又把 batch read/write 改成了两阶段并发路径[[3]](https://github.com/kvcache-ai/Mooncake/pull/1921)**。过去无论本地还是远端，`BatchPut` / `BatchGet` 基本都是按 key 串行推进；现在先把 handle 全部发出去，再统一等待结果。本地 memcpy、TransferEngine 传输和 RPC retry 都被纳入同一套 pending handle 模型里。这条改动看起来像工程重构，但它真正说明的是: 当 KV 数据面开始面向超大池和高并发批处理时，系统已经不能再假设“每个 key 依次排队也没关系”。

把这三条放在一起看，长上下文和超大缓存池正在失去“特殊场景”的身份。谁还把这些场景挂在主路径旁边做临时兼容，谁就会先在 CPU 开销、注册时延和批处理效率上掉队。

## 二、并行与通信路径开始压缩成更少、更硬的默认实现

**vLLM 直接移除了 WideEP 里的 naive all2all 实现，把 `allgather_reducescatter` 收成默认路径[[4]](https://github.com/vllm-project/vllm/pull/33728)**。这条 PR 表面上是在删代码，实际上是在释放非常明确的信号: 团队已经不想继续维护一个广播式 all2all 作为备用后门，因为更高效的通信路径已经足够成熟，可以承担默认职责。很多基础设施项目真正走向稳定，不是靠“支持更多后端名字”，而是靠“敢删掉那个没人愿意长期维护、但又一直挂在那里的旧分支”。

**SGLang 则把 `moe_dp_size = 1` 与不同 `attention_cp_size` 的组合正式接进来[[5]](https://github.com/sgl-project/sglang/pull/22003)**。这条更新之所以值得写，不只是因为它把配置矩阵从“只能 attention_cp_size == moe_dp_size”放宽了，而是它正面承认现实部署里 attention 并行和 MoE 并行未必会按同一把尺子走。PR 给出的数字也很说明问题: 测试里输出吞吐从约 941 token/s 提到约 1953 token/s。这意味着复杂拓扑不再只能靠用户手工避坑，框架开始主动把它写成可成立的默认组合。

**TensorRT-LLM 的 v1.2.1 release 一边修掉了 KV cache corruption，一边升级 xgrammar 和 flashinfer[[6]](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.2.1)**；更重要的是，主干里还合入了新的 sharding infrastructure[[7]](https://github.com/NVIDIA/TensorRT-LLM/pull/12419)，把 auto-deploy、模型 registry 和 sharding hint 往统一部署层面继续推。把 release 修复和这条 PR 放在一起看，更像是 NVIDIA 在补同一件事: 并行与部署这套底层组织方式，不能再拆成一堆每个模型各自拼接的特殊规则。

这一组变化合起来，指向的是同一个判断。AI Infra 的并行和通信竞争，正在从“我支持多少种配置”转向“我敢不敢把一小撮真正站得住的路径收成默认实现，再把其余历史包袱删掉”。

## 三、异构运行时和真实内存语义开始反过来约束接口设计

**SGLang 修掉了 CUDA 13.0 下 `cudaMemcpyBatchAsync` 因函数签名变化导致的 segfault[[8]](https://github.com/sgl-project/sglang/pull/23136)**，这类 PR 很容易被当成版本兼容补丁，但我反而觉得它特别说明阶段变化。过去很多项目默认假设“跑在主流 CUDA 版本就够了”，一旦升级到新驱动或新 toolkit，边角 ABI 断裂只能靠用户自己回退环境。现在 SGLang 直接在运行时按 driver version 切换签名，说明这些框架已经开始把“新 CUDA 小版本能否平滑跟上”视作正式主路径的一部分。

**Mooncake 另一条更典型的修复，是把 GPU VRAM 指针进入 disk-replica / SSD-offload 路径时的语义补全[[9]](https://github.com/kvcache-ai/Mooncake/pull/1892)**。在即将接入 vLLM connector 的路径上，GPU `data_ptr()` 传进 C++ 层后，如果继续被 CPU 线程当成普通 host pointer 去 `memcpy`，系统会直接崩掉。这个 PR 用同步 D2H staging 和 pinned buffer pool 把这层语义明确写进实现里。它重要的地方在于，项目终于不再假装“offload 路径拿到的指针都一样”，而是正面承认真实内存对象有设备边界。

**LMCache 这几天的几条更新也在补同样的接口诚实度。它一方面把 `use_cufile` 重构成更通用的 `use_gds` / `gds_backend` 配置[[10]](https://github.com/LMCache/LMCache/pull/2858)**，把后端选择从 NVIDIA 特定名词里抽出来；另一方面又修掉了 MP connector store bookkeeping 的两个错误，避免混合命中场景里 KV block 被静默丢弃[[11]](https://github.com/LMCache/LMCache/pull/3012)**。这两条放在一起看，不是在做“配置美化”和“小 bug 修复”，而是在说缓存系统终于开始承认两个事实: 底层介质不止一种，命中的记账也不能错一点点。

## 四、今天真正值得记住的判断

今天真正值得记住的，不是哪个项目又多支持了一种组合，而是越来越多项目开始主动关闭“特殊分支永远只是特殊分支”这条退路。

长上下文、超大缓存池、复杂并行拓扑、异构 CUDA 版本、GPU 指针落盘，这些问题过去总容易被留到部署阶段再处理。但最近三天的更新说明，主干项目已经越来越不愿意把它们继续留在默认路径之外。谁能先把这些极端规模场景收回主实现、写成可观察、可部署、可维护的默认 contract，谁才更有机会拿到下一阶段的基础设施优势。

---

## 参考来源

[1] [SGLang 将 EAGLE bigram key 改为 `RadixKey` 的 O(1) 视图](https://github.com/sgl-project/sglang/pull/23106)

[2] [Mooncake 为 EFA transport 增加 PTE-aware 大内存自动切分注册](https://github.com/kvcache-ai/Mooncake/pull/1912)

[3] [Mooncake 将 BatchPut / BatchGet 改为两阶段并发读写模型](https://github.com/kvcache-ai/Mooncake/pull/1921)

[4] [vLLM 在 WideEP 中移除 naive all2all，收敛到 allgather_reducescatter](https://github.com/vllm-project/vllm/pull/33728)

[5] [SGLang 支持 `moe_dp_size = 1` 与不同 `attention_cp_size` 的组合](https://github.com/sgl-project/sglang/pull/22003)

[6] [TensorRT-LLM v1.2.1 修复 KV cache corruption 并升级底层依赖](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.2.1)

[7] [TensorRT-LLM 引入新的 sharding infrastructure](https://github.com/NVIDIA/TensorRT-LLM/pull/12419)

[8] [SGLang 修复 CUDA 13.0 下 `cudaMemcpyBatchAsync` 的签名兼容问题](https://github.com/sgl-project/sglang/pull/23136)

[9] [Mooncake 修复 GPU VRAM 指针进入 disk-replica / SSD-offload 路径时的崩溃](https://github.com/kvcache-ai/Mooncake/pull/1892)

[10] [LMCache 将 `use_cufile` 重构为通用的 `use_gds` / `gds_backend` 配置](https://github.com/LMCache/LMCache/pull/2858)

[11] [LMCache 修复 MP connector store bookkeeping 导致的 KV block 静默丢失](https://github.com/LMCache/LMCache/pull/3012)
