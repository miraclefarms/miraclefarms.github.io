---
title: AI Infra 早报｜从边缘到数据中心，推理硬件路径全线提速
date: 2026-03-30 05:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: llama.cpp b8579 重写 MoE GEMV 核，边缘侧多 token 推理吞吐系统性提升；vLLM 消除 pooling 路径冗余设备拷贝，E2E 吞吐意外提升 48.9%。Mooncake 让弹性 EP rank 恢复变为异步非阻塞，LMCache 引入 Intel DSA 作为 KV cache 第三条传输路径。OpenClaw 2026.3.28 发布，plugin 工具通过 MCP server 打通 ACP session，FTS5 CJK 检索补齐。这是一个把已知瓶颈逐一打穿的迭代日。
tags: [Inference, KV Cache, MoE, vLLM]
---

今天的主线不是某个重大发布，而是多个项目在各自关键路径上同时"打穿"了一个已知瓶颈。llama.cpp 边缘侧 MoE 核重写、vLLM 推理服务 pooling 路径去冗余、Mooncake 弹性容错异步化、LMCache 异构传输扩展——每一件单独看都是合理的工程优化，合在一起看则是一张从端侧到数据中心的推理加速图在同步向前推进。

## 一、llama.cpp b8579：MoE GEMV 核重写

llama.cpp 本轮最重要的变更是 [[1]](https://github.com/ggml-org/llama.cpp/releases/tag/b8579) 对 MoE 多 token（Batch Size > 1）GEMV 核的架构级重写。旧版核以三维 grid `(nrows_x, nchannels_dst, ncols_dst)` 组织，每个 block 有效工作量极少；新版 `mul_mat_vec_q_moe` 核改为二维 grid `(ceil(nrows_x/rpb), nchannels_dst)`，每个 warp 独立处理两行，只做 warp-level reduction，彻底去掉了 shared memory sync 的开销。同时，MMVQ 核最大 batch size 从默认值提升到 8。

对使用 MoE 模型做 Chain-of-Thought 或投机解码等多步推理的端侧场景，这次改写带来的是系统性吞吐提升，而不是边际优化。同日合并的 b8580 [[2]](https://github.com/ggml-org/llama.cpp/releases/tag/b8580) 则补齐了 MiniCPM 模型缺少的 `ROPE_FACTORS_LONG/SHORT` 参数，修复使用长短 RoPE scaling 的 MiniCPM 变体推理准确性问题。

Hexagon DSP 后端也收到一次 DMA 路径修复 [[3]](https://github.com/ggml-org/llama.cpp/pull/21137)，主要解决近期引入的性能回归，影响在 Snapdragon 设备上跑量化模型的高通端侧部署场景。

## 二、vLLM：小改动，大收益

vLLM 今天最值得关注的 PR [[4]](https://github.com/vllm-project/vllm/pull/38139) 来自一个看似不起眼的问题：pooling 路径的 token IDs 原本经历了 CPU→GPU→CPU 的冗余往返，而这些 token IDs 自始至终都只在 CPU 上被使用。去掉这两次拷贝之后，BGE-M3 等 embedding/pooling 模型的 E2E 吞吐提升了 48.9%。改动的代码量很少，但揭示了一类在推理栈里容易被忽视的隐性瓶颈——设备间拷贝的成本会随请求量线性放大，在高并发场景里尤其明显。

同日还落地了 QeRL（Quantized Runtime Loading）的修复 [[5]](https://github.com/vllm-project/vllm/pull/38442)：捕获正确的 load device context，修复 fp8 MoE scales 实例化在错误设备的 bug，并将量化重载 CI 测试从 skip 切换为正式运行。量化模型的热重载路径从"存在但未硬化"推进到"有 CI 覆盖的生产路径"。

## 三、分布式 KV cache 传输的多路扩展

**Mooncake PR #1744** [[6]](https://github.com/kvcache-ai/Mooncake/pull/1744) 解决了弹性 EP 场景中一个"小概率高代价"的问题：此前 GPU 故障后的 rank 恢复需要与所有健康 rank 进入同一通信阶段，在被恢复 rank 执行昂贵的 CUDA graph capture 期间，整个健康推理进程都被阻塞。新版引入异步路径：recovered rank 先孤立完成本地初始化，再通过显式 `joinGroup()` 接入活跃进程组。这延续了昨日 SGLang Elastic NIXL-EP 的思路——降低弹性容错的系统代价，让 MoE 故障恢复真正"静默化"。

**LMCache PR #2897** [[7]](https://github.com/LMCache/LMCache/pull/2897) 在 Mooncake connector 中新增 Intel DSA（Data Streaming Accelerator）支持，为 KV cache 跨节点传输提供 CPU 侧的硬件数据移动加速。在 RDMA 和 GPU-Direct 之外，这是第三条传输硬件路径，对无 InfiniBand 网络的标准 x86 集群具有实际价值。

## 四、SGLang：向 Blackwell 迈步

SGLang 本轮没有大型功能 PR 合并，但新增 GB300（NVIDIA Grace Blackwell Superchip）nightly benchmark CI 测试套件 [[8]](https://github.com/sgl-project/sglang/pull/21487) 值得记录：这意味着 SGLang 开始在下一代旗舰 GPU 平台上建立持续性能基准，为 v0.5.10 正式版的 Blackwell 支持做前置质量保障。同日还修复了无 layers 模型的 piecewise CUDA graph 禁用逻辑，防止无效 graph 捕获导致的异常。

## 五、OpenClaw 2026.3.28：plugin 工具进入 ACP session

OpenClaw v2026.3.28 [[9]](https://github.com/openclaw/openclaw/releases/tag/v2026.3.28) 昨夜发布，今日合并了几个配套 PR，其中最重要的两笔是：

**plugin tools MCP server** [[10]](https://github.com/openclaw/openclaw/pull/56867)：新建独立 MCP server，将所有 plugin 注册工具（memory_recall/store/forget 等）暴露给在 ACP session 中运行的 Claude Code / Codex，通过 acpx MCP proxy 机制实现。这打通了"ACP coding agent + 持久化 plugin 工具"的完整工作流——在此之前，ACP session 中的编程 agent 无法调用 OpenClaw 的 memory、calendar 等扩展能力。

**FTS5 CJK tokenizer 支持** [[11]](https://github.com/openclaw/openclaw/pull/56707)：新增 `memorySearch.store.fts.tokenizer` 配置项（unicode61 / trigram），FTS5 index 在 tokenizer 变更时自动重建，1-2 字节 CJK 短查询通过 substring fallback 处理。中日韩文本的内存语义检索从"部分可用"变为"系统性可用"。

此外，release notes 中还提到 plugin before_tool_call hook 新增 async requireApproval [[9]](https://github.com/openclaw/openclaw/releases/tag/v2026.3.28)（plugin 可暂停工具执行等待用户审批），sandbox fail-closed 安全加固 [[12]](https://github.com/openclaw/openclaw/pull/56800)，以及 xAI provider 迁移至 Responses API 并新增 x_search 工具 [[9]](https://github.com/openclaw/openclaw/releases/tag/v2026.3.28)。

---

今天没有单一的架构革命，但推理栈从边缘到云端的多个关键路径都收到了有实质意义的改进。llama.cpp 和 vLLM 各自挖出了一块"本该早点修"的性能空间；Mooncake 和 LMCache 在分布式传输层向异步化和硬件多样化延伸；OpenClaw 则在 agent 工具链层面补齐了 plugin 与 ACP session 的互通缺口。短期看，SGLang v0.5.10 正式版和 TensorRT-LLM v1.3.0 的 rc 退出进度是接下来值得持续跟踪的两条线。

## 参考来源

[1] [llama.cpp b8579 MoE GEMV 核架构重写](https://github.com/ggml-org/llama.cpp/releases/tag/b8579)

[2] [llama.cpp b8580 MiniCPM ROPE_FACTORS 修复](https://github.com/ggml-org/llama.cpp/releases/tag/b8580)

[3] [llama.cpp Hexagon DMA 优化修复回归](https://github.com/ggml-org/llama.cpp/pull/21137)

[4] [vLLM pooling 路径去冗余拷贝 +48.9% 吞吐](https://github.com/vllm-project/vllm/pull/38139)

[5] [vLLM QeRL 在线量化重载修复](https://github.com/vllm-project/vllm/pull/38442)

[6] [Mooncake 异步 recovered-rank 初始化 + deferred join](https://github.com/kvcache-ai/Mooncake/pull/1744)

[7] [LMCache Mooncake connector Intel DSA 支持](https://github.com/LMCache/LMCache/pull/2897)

[8] [SGLang GB300 nightly benchmark CI](https://github.com/sgl-project/sglang/pull/21487)

[9] [OpenClaw v2026.3.28 发布说明](https://github.com/openclaw/openclaw/releases/tag/v2026.3.28)

[10] [OpenClaw plugin tools MCP server for ACP sessions](https://github.com/openclaw/openclaw/pull/56867)

[11] [OpenClaw FTS5 CJK tokenizer 支持](https://github.com/openclaw/openclaw/pull/56707)

[12] [OpenClaw sandbox fail-closed 加固](https://github.com/openclaw/openclaw/pull/56800)
