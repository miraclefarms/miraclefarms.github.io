# 内容标签规范

每篇文章的 front matter 里用 `tags` 字段标注 2–5 个标签，帮助读者按主题筛选文章。

---

## 当前标签体系

| 标签 | 适用场景 |
|------|---------|
| `Agents` | AI agent 框架、多 agent 协作、agent loop、agent 评测、AI 编程助手（如 Claude Code、Cursor）的工程分析 |
| `Attention` | Attention 机制、FlashAttention 系列、稀疏 attention、attention kernel 优化 |
| `Disaggregation` | Prefill-Decode 分离（P/D disaggregation）、计算与存储解耦 |
| `Evaluation` | LLM benchmark、评测框架、能力评估、对比测试 |
| `Inference` | LLM 推理服务（通用）、推理框架对比、吞吐量/延迟优化（无法归入更细分类时使用） |
| `KV Cache` | KV cache 管理策略、prefix caching、paged attention、缓存命中率优化、KV cache 量化 |
| `Long Context` | 长上下文处理、超长 context window、长文档推理 |
| `MoE` | Mixture of Experts 架构、专家路由、expert parallelism、MoE 推理优化 |
| `Mooncake` | Mooncake（月光石）分布式 prefill 系统的专项分析 |
| `Multimodal` | 多模态模型、视觉语言模型、图像/音频 token 处理 |
| `Networking` | AI 集群网络、RDMA、InfiniBand、拓扑优化、网络带宽瓶颈 |
| `Quantization` | 模型量化（INT4/FP8/FP4）、量化感知训练、量化推理精度权衡 |
| `SGLang` | SGLang 框架的功能更新或源码分析 |
| `Speculative Decoding` | 推测解码、draft model、speculative prefill |
| `TRT-LLM` | TensorRT-LLM 框架的功能更新或源码分析 |
| `Training` | 模型训练（预训练、RLHF、GRPO）、分布式训练、训练基础设施 |
| `Transformers` | Transformer 核心架构改动、位置编码、归一化层等架构层面分析 |
| `llama.cpp` | llama.cpp 的功能更新或源码分析 |
| `vLLM` | vLLM 框架的功能更新或源码分析 |

---

## 选标签规则

### 数量

- **2–4 个**是常态；5 个是上限，不要为了凑数强行加到 5。
- 精准覆盖文章的核心主题即可，不需要把所有涉及的词都列上。

### 框架专属标签（vLLM / SGLang / TRT-LLM / llama.cpp / Mooncake）

只在该框架是**文章的主要分析对象**时使用。

- ✅ "深度解析 vLLM PagedAttention 实现" → `vLLM`
- ✅ 当天日报三条里两条都是 SGLang PR → `SGLang`
- ❌ 日报里某条 SGLang PR 只是顺带一提 → 不加 `SGLang`
- ❌ 某论文用 vLLM 跑实验 → 不加 `vLLM`（除非文章重点分析 vLLM 的工程细节）

### `Inference` 的使用边界

`Inference` 是兜底标签，**只在无法归入更细分类时使用**。如果文章主要讲 KV Cache，就用 `KV Cache` 而非 `Inference`；如果两者都是重点，可以同时标注。

### Brief vs Essay 的标签粒度

- **Brief**：反映当天覆盖的技术领域组合，同一天如果同时有 KV Cache 和 MoE 更新，两个都标。
- **Essay**：反映文章的技术深度与主题焦点，通常 2–3 个，聚焦核心主题。

### 关于新增标签

当现有标签体系无法准确覆盖文章主题时，**直接新增标签**，无需等待用户确认：

1. 确认现有标签确实无法近似覆盖（别把"有点相关"当"能覆盖"）。
2. 按以下原则决定新标签名称：
   - **技术领域类**：使用该领域的标准英文术语，首字母大写，例如 `CUDA Graph`、`Scheduling`、`Compilation`。
   - **框架/系统类**：使用项目官方名称，例如 `Triton`、`Megatron`、`DeepSpeed`。
   - 避免过于宽泛（如 `Performance`）或过于细碎（如 `H100 Tensor Core`）的标签。
3. 将新标签加入本文件的**当前标签体系**表格，补充适用场景说明。
4. 在文章 front matter 里正常使用新标签。

---

## 操作步骤

写完文章正文、完成自检清单后，在输出文件之前执行：

1. 阅读文章正文，识别主要技术主题（通常 2–4 个）。
2. 在当前标签体系中找到最匹配的标签，确认数量在 2–5 范围内。
3. 将标签写入 front matter：`tags: [Tag1, Tag2, Tag3]`（YAML 行内列表格式，大小写与表格一致）。

### 示例

**文章主题**：分析 vLLM 中 KV cache 的分层压缩策略
→ `tags: [KV Cache, Inference, vLLM]`

**文章主题**：解析 FlashAttention-4 的 Blackwell 优化
→ `tags: [Attention, Inference]`

**文章主题**：Claude Code 的提示词缓存与上下文管理工程
→ `tags: [KV Cache, Agents, Inference]`

**文章主题**：当天日报，覆盖 MoE 路由、SGLang 更新、KV Cache 量化三个主题
→ `tags: [MoE, KV Cache, Quantization, SGLang]`
