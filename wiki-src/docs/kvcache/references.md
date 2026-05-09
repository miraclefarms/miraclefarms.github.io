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

### Attention 变体与 KVCache 压缩

- **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints**
  Ainslie et al., 2023. GQA（Grouped-Query Attention）的提出，显著减小 KVCache 体积。
  [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)

- **Fast Transformer Decoding: One Write-Head is All You Need**
  Shazeer, 2019. MQA（Multi-Query Attention）的提出。
  [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)

- **DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model**
  DeepSeek, 2024. 包含 MLA（Multi-head Latent Attention）的详细描述。
  [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)

### Prefix Cache 与 KV 复用

- ⭐ **SGLang: Efficient Execution of Structured Language Model Programs**
  Zheng et al., 2024. 包含 RadixAttention 的描述。
  [arXiv:2312.07104](https://arxiv.org/abs/2312.07104)

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
  TODO：待补充。

- **CacheGen: KV Cache Compression and Streaming for Fast LLM Serving**
  Liu et al., 2024.
  [arXiv:2310.07240](https://arxiv.org/abs/2310.07240)

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

---

## MiracleFarms 站内相关文章

以下是 Miracle Farms 博客中与 KVCache 相关的深度文章：

- [TRT-LLM KVCache Runtime 架构解析](/notes/2025/04/28/trtllm-kvcache-runtime-architecture/) — TensorRT-LLM KVCache 运行时内部架构的详细分析

---

*如有遗漏的重要论文或资源，欢迎通过 [GitHub Issues](https://github.com/miraclefarms/miraclefarms.github.io/issues) 提交补充建议。*
