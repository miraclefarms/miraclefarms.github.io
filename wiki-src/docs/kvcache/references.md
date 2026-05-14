# 参考资料

本页汇总 KVCache 相关的重要论文、框架文档、博客和 GitHub 资源。标注 ⭐ 的为强烈推荐的入门材料。

---

## 核心论文

### KVCache 基础与分页管理

- ⭐ **Efficient Memory Management for Large Language Model Serving with PagedAttention**
  Kwon et al., 2023. vLLM 和 PagedAttention 的原始论文。
  [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)

- **Attention is All You Need**
  Vaswani et al., 2017. Transformer 和 Multi-Head Attention 的原始论文。
  [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

### Attention 变体与架构

- **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints**
  Ainslie et al., 2023. GQA（Grouped-Query Attention）的提出，显著减小 KVCache 体积。
  [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)

- **Fast Transformer Decoding: One Write-Head is All You Need**
  Shazeer, 2019. MQA（Multi-Query Attention）的提出。
  [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)

- **DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model**
  DeepSeek, 2024. 包含 MLA（Multi-head Latent Attention）的详细描述。
  [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)

- **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**
  Gu & Dao, 2023. 状态空间模型路线，恒定大小状态替代 KVCache。
  [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)

- **RWKV: Reinventing RNNs for the Transformer Era**
  Peng et al., 2023.
  [arXiv:2305.13048](https://arxiv.org/abs/2305.13048)

### KV 稀疏化

- **Efficient Streaming Language Models with Attention Sinks（StreamingLLM）**
  Xiao et al., 2023. 发现并利用 attention sink，构造滑动窗口 + sink 的无限长上下文方案。
  [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)

- **H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs**
  Zhang et al., 2023. token 级重要性评分驱动的 KV 剪枝。
  [arXiv:2306.14048](https://arxiv.org/abs/2306.14048)

- **SnapKV: LLM Knows What You are Looking for Before Generation**
  Li et al., 2024. Prefill 末端预测重要 token，一次性丢弃次要 K/V。
  [arXiv:2404.14469](https://arxiv.org/abs/2404.14469)

- **Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference**
  Tang et al., 2024. 基于 query 的动态稀疏 attention。
  [arXiv:2406.10774](https://arxiv.org/abs/2406.10774)

- **YOCO: You Only Cache Once**
  Sun et al., 2024. 跨层共享 KV，KVCache 体积大幅下降。
  [arXiv:2405.05254](https://arxiv.org/abs/2405.05254)

- **Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation**
  Feng et al., 2024. head 级自适应 KV 剪枝。
  [arXiv:2407.11550](https://arxiv.org/abs/2407.11550)

### KV 量化

- **KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache**
  Liu et al., 2024. K per-channel + V per-token 的极致 2-bit 量化。
  [arXiv:2402.02750](https://arxiv.org/abs/2402.02750)

- **KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization**
  Hooper et al., 2024. 4-bit 量化 + RoPE 对齐。
  [arXiv:2401.18079](https://arxiv.org/abs/2401.18079)

- **Atom: Low-bit Quantization for Efficient and Accurate LLM Serving**
  Zhao et al., 2023. 推理全栈低比特化。
  [arXiv:2310.19102](https://arxiv.org/abs/2310.19102)

- **QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving**
  Lin et al., 2024.
  [arXiv:2405.04532](https://arxiv.org/abs/2405.04532)

- **TurboQuant: Extremely Low-Bit KV Cache Quantization for Negligible Accuracy Loss**
  Mao et al., 2025. 3.5-bit 近似无损，WHT + QJL 残差纠偏。
  [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)

- **HACK: Homomorphic Acceleration via Compression of the Key-Value Cache for Disaggregated LLM Inference**
  Luo et al., 2025. 同态 INT2 量化，KV 不还原直接在 attention 内计算。
  [arXiv:2502.03589](https://arxiv.org/abs/2502.03589)

### Prefix Cache 与 KV 复用

- ⭐ **SGLang: Efficient Execution of Structured Language Model Programs**
  Zheng et al., 2024. 包含 RadixAttention 的描述。
  [arXiv:2312.07104](https://arxiv.org/abs/2312.07104)

- **Prompt Cache: Modular Attention Reuse for Low-Latency Inference**
  Gim et al., 2023. 位置无关 prefix cache 的早期探索。
  [arXiv:2311.04934](https://arxiv.org/abs/2311.04934)

- **CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion**
  Yao et al., 2024. RAG 场景下文档块级 KV 复用。
  [arXiv:2405.16444](https://arxiv.org/abs/2405.16444)

### 长上下文与序列并行

- **Ring Attention with Blockwise Transformers for Near-Infinite Context**
  Liu et al., 2023. 长上下文 SP 的代表实现。
  [arXiv:2310.01889](https://arxiv.org/abs/2310.01889)

### PD 分离

- ⭐ **Splitwise: Efficient Generative LLM Inference Using Phase Splitting**
  Patel et al., 2023. PD 分离早期的重要论文。
  [arXiv:2311.18677](https://arxiv.org/abs/2311.18677)

- **Mooncake: Kimi's KVCache-centric Architecture for LLM Serving**
  Qin et al., 2024. 以 KVCache 为核心的 PD 分离调度系统。
  [arXiv:2407.00079](https://arxiv.org/abs/2407.00079)

- **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving**
  Zhong et al., 2024.
  [arXiv:2401.09670](https://arxiv.org/abs/2401.09670)

### KV Offload 与多级存储

- **InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management**
  Lee et al., 2024. KV offload + 动态管理。
  [arXiv:2406.19707](https://arxiv.org/abs/2406.19707)

- **CacheGen: KV Cache Compression and Streaming for Fast LLM Serving**
  Liu et al., 2024.
  [arXiv:2310.07240](https://arxiv.org/abs/2310.07240)

### CXL 与 KVCache

- **Beluga: CXL Memory Pooling for Disaggregated LLM Inference**
  Li et al., 2025. CXL 2.0 内存池化 KV Cache，TTFT 1.36s vs RDMA 13.00s。
  [arXiv:2511.20172](https://arxiv.org/abs/2511.20172)

- **TraCT: CXL-based KV Cache Transfer for Disaggregated LLM Inference**
  CXL 作为 PD KV 传输通道，TTFT 最高 9.8× 提升。
  [arXiv:2512.18194](https://arxiv.org/abs/2512.18194)

- **CXL-SpecKV: Speculative KV Cache Lookup with CXL Memory**
  CXL 池作为独立 KVCache 服务。
  [arXiv:2512.11920](https://arxiv.org/abs/2512.11920)

### Agent 与长上下文

- **SCBench: A KV Cache-Centric Analysis of Long-Context Methods**
  Tan et al., 2024. KV 生命周期四阶段拆解评估框架。
  [arXiv:2412.10319](https://arxiv.org/abs/2412.10319)

- **Don't Break the Cache: Caching-Aware Prompt Optimization for LLM Agents**
  Determinant of KV cache hit rate in agent systems.
  [arXiv:2601.06007](https://arxiv.org/abs/2601.06007)

### PD 分离与多轮

- **PPD: Prefill-Prefill-Decode Disaggregation for Efficient Multi-Turn LLM Serving**
  Full prefill vs append-prefill 的分野。
  [arXiv:2603.13358](https://arxiv.org/abs/2603.13358)

- **Prefill-as-a-Service: Cross-Datacenter LLM Inference with Hybrid Attention**
  Hybrid attention 降低跨数据中心 KV 传输带宽至 3–8 Gbps。
  [arXiv:2604.15039](https://arxiv.org/abs/2604.15039)

- **ZeRO-Prefill: Asynchronous Expert Parallelism for MoE Prefill Serving**
  MoE 推理中 KV 的"原地位"策略。
  [arXiv:2605.02960](https://arxiv.org/abs/2605.02960)

### Attention 机制

- **Why Does the Attention Sink Occur?** (ICML 2026)
  Attention sink 的统计-结构形成机制，HeadNorm 缓解。
  [arXiv:2605.06611](https://arxiv.org/abs/2605.06611)

### FlashAttention

- ⭐ **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**
  Dao et al., 2022.
  [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

- **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**
  Dao, 2023.
  [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)

---

## 框架文档

- [vLLM 官方文档](https://docs.vllm.ai/) — 包含 PagedAttention、APC、Chunked Prefill 等特性说明
- [SGLang 官方文档](https://docs.sglang.ai/) — 包含 RadixAttention、KVCache 管理的说明
- [TensorRT-LLM 文档](https://nvidia.github.io/TensorRT-LLM/) — 包含 KVCache 配置和量化选项
- [LMCache 文档](https://lmcache.readthedocs.io/) — 多级 KVCache 存储系统

---

## 博客文章

- ⭐ **vLLM Blog: PagedAttention** (2023)
  vLLM 团队对 PagedAttention 的介绍，含图示说明。
  [blog.vllm.ai](https://blog.vllm.ai/2023/06/20/vllm.html)

- **Towards 100x Speedup: Full Stack Transformer Inference Optimization** (待补充 URL)

---

## 相关 GitHub 资源

- [vLLM](https://github.com/vllm-project/vllm) — PagedAttention 原始实现
- [SGLang](https://github.com/sgl-project/sglang) — RadixAttention、高性能 KV 管理
- [LMCache](https://github.com/LMCache/LMCache) — 多级 KVCache 分层存储
- [Mooncake](https://github.com/kvcache-ai/Mooncake) — KVCache-centric 调度系统
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) — 分页 KVCache 的高效 Attention kernel 库
- [AIBrix](https://github.com/vllm-project/aibrix) — vLLM 生态下的 K8s 原生推理基础设施，含 cache-aware autoscaler
- [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) — NVIDIA 推出的分布式推理基础设施
- [TokenSpeed](https://github.com/ai-dynamo/tokenspeed) — NVIDIA Dynamo 生态的高性能推理调度引擎
- [XConn XC50256](https://www.xconn-tech.com/) — CXL 2.0 switch，Beluga 等 CXL KV 方案的关键硬件

---

## MiracleFarms 站内相关文章

**Essay（深度分析）：**

- [vLLM KVCache Runtime 架构解析](/notes/2026/03/12/vllm-kvcache-runtime-architecture/) — PagedAttention、BlockPool、调度器集成
- [SGLang KVCache Runtime 架构解析](/notes/2026/03/14/sglang-kvcache-runtime-architecture/) — RadixAttention、HiCache、HiSparse、ShadowRadix
- [KV Cache Agent 长上下文 Benchmark](/notes/2026/04/08/kvcache-agent-long-context-benchmark/) — 研究现状与评估空白
- [Claude Code 上下文工程与 KV Cache](/notes/2026/05/08/claude-code-context-kvcache-engineering/) — Anthropic prompt caching 接口分析
- [TRT-LLM KVCache Runtime 架构解析](/notes/2026/05/09/trtllm-kvcache-runtime-architecture/) — 三层存储、事件驱动路由、NIXL 传输
- [CXL + KVCache 现状调研报告](/notes/2026/05/13/cxl-kvcache-survey/) — CXL 硬件/软件生态与五层分类
- [KV Cache 前缀匹配的设计分野](/notes/2026/05/13/kvcache-prefix-matching-design/) — Radix Tree / 链式哈希 / 两阶段 Claim

**Reading（论文解读）：**

- [TurboQuant 详解](/notes/2026/03/27/turboquant-kvcache-3bit/) — 3.5-bit 近似无损 KV 量化
- [Prefill-as-a-Service](/notes/2026/04/19/prefill-as-a-service-cross-datacenter-kvcache/) — Hybrid attention 跨数据中心
- [主流 Attention 算法全景](/notes/2026/04/20/mainstream-attention-algorithms-overview/)
- [SCBench KV 生命周期分析](/notes/2026/04/21/scbench-kv-cache-lifecycle-analysis/)
- [TurboQuant 框架集成路线图](/notes/2026/04/21/turboquant-vllm-sglang-trtllm-integration/)
- [vLLM × Mooncake Store](/notes/2026/05/07/vllm-mooncake-store-distributed-kv-cache/) — Agentic KV 分布式池
- [HACK 同态 KV 压缩](/notes/2026/05/08/hack-homomorphic-kv-cache-disaggregated-inference/)
- [PPD Disaggregation](/notes/2026/05/08/ppd-disaggregation-multiturn-llm-serving/) — 多轮 Prefill 分化
- [Attention Sink 的结构起点](/notes/2026/05/11/attention-sink-variance-super-neurons/)
- [Beluga：CXL 内存池化 KV Cache](/notes/2026/05/13/beluga-cxl-kvcache-memory-pool/)
- [ZeRO-Prefill：MoE Prefill 的 KV 放置](/notes/2026/05/14/zeRO-prefill-async-ep-moe-prefill-serving/)

---

*如有遗漏的重要论文或资源，欢迎通过 [GitHub Issues](https://github.com/miraclefarms/miraclefarms.github.io/issues) 提交补充建议。*

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.2 | 2026-05-14 | 新增 TurboQuant、HACK、Beluga、TraCT、CXL-SpecKV、SCBench、PPD、PrfaaS、ZeRO-Prefill、Attention Sink 等论文；补充 CXL 论文分类；补充主站全部 18 篇 KV Cache 相关文章链接；新增 TokenSpeed/XConn 相关资源 |
