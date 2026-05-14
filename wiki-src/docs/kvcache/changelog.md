# KVCache Wiki 更新日志

记录 wiki 每次内容更新的范围、涉及章节和变更摘要。

---

## 2026-05-14 — Phase 1: 首轮内容下沉

**说明：** 扫描全部 20 篇主站 essay/reading（7 essay + 13 reading），将实质性技术判断和生产数据下沉到对应 wiki 章节。同时扫描 10 篇 brief 提取框架 PR 更新。

**涉及章节（按变更量排序）：**

| 章节 | 版本 | 主要变更 |
|------|------|----------|
| prefix-cache.md | v0.1→v0.2 | +三框架 Prefix Cache 实现对比（Radix Tree/链式哈希/两阶段Claim）、+分布式 Prefix Cache（Mooncake Store 实测/LMCache 多后端）、+HiCache 生产数据 |
| frameworks.md | v0.1→v0.2 | +各框架实质性技术细节（vLLM Mooncake Store/多级 offload/TurboQuant、SGLang HiCache/HiSparse/ShadowRadix/PD Staging、TRT-LLM 三层存储/两阶段 Claim/Disaggregated serving、LMCache 多后端生态、Mooncake 分布式 Store 详情）、更新对比表增加分布式 KV 列 |
| offload.md | v0.1→v0.2 | +CXL 作为 KV Offload 通道（Beluga 实测数据 7.35× QPS、五层分类、限制） |
| pd-disaggregation.md | v0.1→v0.2 | +PPD 分离（多轮 append-prefill 本地化）、+Prefill-as-a-Service（跨数据中心 hybrid attention）、+ZeRO-Prefill（MoE KV 放置） |
| compression-quantization.md | v0.1→v0.2 | +TurboQuant（3.5-bit 近无损、三框架集成路线）、+HACK（同态 INT2 量化、JCT 38.6–61.6% 降低） |
| evaluation.md | v0.1→v0.2 | +SCBench KV 生命周期四阶段评估框架及核心发现、+Agent + KV Cache 评估空白分析 |
| lifecycle.md | v0.1→v0.2 | +SCBench 生命周期视角（四阶段、Sub-O(n) 多轮退化、O(n) 稀疏注意力稳健性） |
| workloads.md | v0.1→v0.2 | +Agent 章节：Claude Code 缓存工程实证（90%失效来自服务端路由、10× cache 价格差）、Mooncake Store agentic trace 实测 |
| attention-variants.md | v0.1→v0.2 | +§5 Attention Sink（形成机制四步放大链路、可迁移性、对 KVCache 设计的启示） |
| storage-hierarchy.md | v0.1→v0.2 | +§6 CXL 存储层前瞻（Beluga 实测 7.35×、CXL-RPC 2.11μs vs RDMA-RC 8.39μs） |

**未修改章节：** first-principles.md, basics.md, sparsity.md, runtime-architecture.md, parallelism.md, routing.md, elasticity.md, crossings.md, future.md, glossary.md, references.md

**已扫描文章：** 见 `scanned_references.md`

---

## 2026-05-14 — Phase 0: 框架搭建

**说明：** 完成全部 23 个章节文件的创建与框架级内容填充。各章节建立了基本概念、核心结构和关键引用，但内容尚处于"骨架"阶段，需要后续从主站文章下沉实质性技术判断和数据。

**涉及章节（全部）：**

| 章节 | 初始版本 | 状态 |
|------|----------|------|
| index.md | v0.1 | 综述框架完成 |
| first-principles.md | v0.1 | 核心心智模型 |
| basics.md | v0.1 | 基础概念 |
| attention-variants.md | v0.1 | GQA/MQA/MLA/SWA 概述 |
| sparsity.md | v0.1 | 稀疏化方法分类 |
| compression-quantization.md | v0.1 | 压缩与量化方法分类 |
| runtime-architecture.md | v0.1 | 引擎架构框架 |
| storage-hierarchy.md | v0.1 | 存储层级框架 |
| paged-kv.md | v0.1 | PagedAttention 基本原理 |
| prefix-cache.md | v0.1 | 前缀缓存框架 |
| offload.md | v0.1 | Offload 框架 |
| lifecycle.md | v0.1 | 生命周期框架 |
| pd-disaggregation.md | v0.1 | PD 分离中的 KV |
| parallelism.md | v0.1 | 并行切分框架 |
| routing.md | v0.1 | 路由与亲和性 |
| elasticity.md | v0.1 | 弹性与故障 |
| workloads.md | v0.1 | 工作负载维度 |
| crossings.md | v0.1 | 维度交叉 |
| evaluation.md | v0.1 | 评估方法框架 |
| future.md | v0.1 | 未来方向 |
| frameworks.md | v0.1 | 框架对比初稿 |
| glossary.md | v0.1 | 术语表初稿 |
| references.md | v0.1 | 参考资料初稿 |

**mkdocs.yml 导航:** 已配置完整导航结构。
