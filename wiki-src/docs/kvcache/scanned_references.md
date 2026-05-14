# 已扫描参考文献清单

记录每个主站文章是否已被纳入 wiki。每次更新只扫描状态为"⏳ 待扫描"的文章。

---

## 扫描状态说明

| 状态 | 含义 |
|------|------|
| ⏳ 待扫描 | 尚未扫描，等待纳入 |
| 🔍 扫描中 | 当前正在扫描 |
| ✅ 已纳入 | 内容已提取并更新到对应 wiki 章节 |
| ⏭️ 跳过 | 经审查无 KV Cache 相关内容 |

---

## Phase 1 扫描结果 (2026-05-14)

### Essay（深度分析）

| 文章路径 | 日期 | 扫描状态 | 扫描日期 | 纳入章节 |
|----------|------|----------|----------|----------|
| `_posts/2026-03-12-vllm-kvcache-runtime-architecture.md` | 2026-03-12 | ✅ 已纳入 | 2026-05-14 | prefix-cache, frameworks |
| `_posts/2026-03-14-sglang-kvcache-runtime-architecture.md` | 2026-03-14 | ✅ 已纳入 | 2026-05-14 | prefix-cache, frameworks |
| `_posts/2026-04-08-kvcache-agent-long-context-benchmark.md` | 2026-04-08 | ✅ 已纳入 | 2026-05-14 | evaluation, workloads |
| `_posts/2026-05-08-claude-code-context-kvcache-engineering.md` | 2026-05-08 | ✅ 已纳入 | 2026-05-14 | workloads, prefix-cache |
| `_posts/2026-05-09-trtllm-kvcache-runtime-architecture.md` | 2026-05-09 | ✅ 已纳入 | 2026-05-14 | frameworks, runtime-architecture |
| `_posts/2026-05-13-cxl-kvcache-survey.md` | 2026-05-13 | ✅ 已纳入 | 2026-05-14 | offload, storage-hierarchy |
| `_posts/2026-05-13-kvcache-prefix-matching-design.md` | 2026-05-13 | ✅ 已纳入 | 2026-05-14 | prefix-cache |

### Reading（论文解读）

| 文章路径 | 日期 | 扫描状态 | 扫描日期 | 纳入章节 |
|----------|------|----------|----------|----------|
| `_posts/2026-03-27-turboquant-kvcache-3bit.md` | 2026-03-27 | ✅ 已纳入 | 2026-05-14 | compression-quantization |
| `_posts/2026-04-03-hierarchicalkv-gpu-cache-semantic-hash-table.md` | 2026-04-03 | ⏭️ 跳过 | 2026-05-14 | GPU embedding 哈希表，非 LLM Attention KV Cache |
| `_posts/2026-04-19-prefill-as-a-service-cross-datacenter-kvcache.md` | 2026-04-19 | ✅ 已纳入 | 2026-05-14 | pd-disaggregation |
| `_posts/2026-04-20-mainstream-attention-algorithms-overview.md` | 2026-04-20 | ✅ 已纳入 | 2026-05-14 | attention-variants |
| `_posts/2026-04-21-scbench-kv-cache-lifecycle-analysis.md` | 2026-04-21 | ✅ 已纳入 | 2026-05-14 | lifecycle, evaluation |
| `_posts/2026-04-21-turboquant-vllm-sglang-trtllm-integration.md` | 2026-04-21 | ✅ 已纳入 | 2026-05-14 | compression-quantization, frameworks |
| `_posts/2026-05-07-vllm-mooncake-store-distributed-kv-cache.md` | 2026-05-07 | ✅ 已纳入 | 2026-05-14 | prefix-cache, workloads, frameworks |
| `_posts/2026-05-08-hack-homomorphic-kv-cache-disaggregated-inference.md` | 2026-05-08 | ✅ 已纳入 | 2026-05-14 | compression-quantization |
| `_posts/2026-05-08-ppd-disaggregation-multiturn-llm-serving.md` | 2026-05-08 | ✅ 已纳入 | 2026-05-14 | pd-disaggregation |
| `_posts/2026-05-11-attention-sink-variance-super-neurons.md` | 2026-05-11 | ✅ 已纳入 | 2026-05-14 | attention-variants |
| `_posts/2026-05-13-beluga-cxl-kvcache-memory-pool.md` | 2026-05-13 | ✅ 已纳入 | 2026-05-14 | offload, storage-hierarchy |
| `_posts/2026-05-14-forcing-kv-hybrid-cache-compression-video-diffusion.md` | 2026-05-14 | ⏭️ 跳过 | 2026-05-14 | 视频扩散 KV，非 LLM 推理 KVCache；attention head 分化可备参考 |
| `_posts/2026-05-14-zeRO-prefill-async-ep-moe-prefill-serving.md` | 2026-05-14 | ✅ 已纳入 | 2026-05-14 | pd-disaggregation |

### Brief（日报 — 已在 Phase 1 扫描）

| 文章路径 | 日期 | 扫描状态 | 扫描日期 | 纳入章节 |
|----------|------|----------|----------|----------|
| `_posts/2026-03-19-ai-infra-daily-brief-moe-tuning-kvcache-convergence.md` | 2026-03-19 | ✅ 已纳入 | 2026-05-14 | frameworks |
| `_posts/2026-03-25-ai-infra-daily-brief-nvfp4-diffusion-moe-lora-kvcache-connectors.md` | 2026-03-25 | ✅ 已纳入 | 2026-05-14 | frameworks, prefix-cache |
| `_posts/2026-03-27-ai-infra-daily-brief-ci-rollback-kvcache-triexpansion.md` | 2026-03-27 | ✅ 已纳入 | 2026-05-14 | frameworks |
| `_posts/2026-04-07-ai-infra-daily-brief-inference-multimodal-cache.md` | 2026-04-07 | ✅ 已纳入 | 2026-05-14 | frameworks |
| `_posts/2026-04-15-ai-infra-daily-brief-moe-kv-real-workloads.md` | 2026-04-15 | ✅ 已纳入 | 2026-05-14 | frameworks, offload |
| `_posts/2026-04-18-ai-infra-daily-brief-default-paths-prefix-reuse.md` | 2026-04-18 | ✅ 已纳入 | 2026-05-14 | frameworks, prefix-cache |
| `_posts/2026-04-19-ai-infra-daily-brief-cache-lifecycle-data-contracts.md` | 2026-04-19 | ✅ 已纳入 | 2026-05-14 | frameworks, prefix-cache |
| `_posts/2026-04-25-ai-infra-daily-brief-execution-boundaries-cache-state.md` | 2026-04-25 | ✅ 已纳入 | 2026-05-14 | frameworks, offload |
| `_posts/2026-05-07-ai-infra-daily-brief-tokenspeed-agentic-inference.md` | 2026-05-07 | ✅ 已纳入 | 2026-05-14 | frameworks |
| `_posts/2026-05-14-ai-infra-daily-brief.md` | 2026-05-14 | ✅ 已纳入 | 2026-05-14 | frameworks, offload |

### Field Note

| 文章路径 | 日期 | 扫描状态 | 扫描日期 | 纳入章节 |
|----------|------|----------|----------|----------|
| `_posts/2026-05-14-field-notes-opening.md` | 2026-05-14 | ⏭️ 跳过 | 2026-05-14 | 站点元文章，无 KV Cache 内容 |

---

## 待后续 Phase 扫描的 Brief

以下 brief 文件在初始 grep 中匹配到 KV Cache 关键词，但 Phase 1 暂未扫描（后续阶段处理）：

| 日期 | 文件 |
|------|------|
| 2026-03-13 ~ 2026-05-13 | 约 40 篇涉及 KV Cache 的日报，详见 `_posts/2026-0[3-5]-*-*.md` |

这些 brief 中 KV Cache 的相关内容通常为 PR 动态、小修复、发布说明，适合在后续批量更新 frameworks.md 时统一处理。
