# 四维联动范式

> 把所有维度拼起来，才能看到工业界的真实部署形态。每种范式都是算法 × 系统 × 部署 × 工作负载四维空间里的一个具体点。

## Mooncake 范式

**核心思路：以 KV 存储为中心重构推理架构**

Kimi 团队的核心判断：KVCache 是一等公民资源，调度决策应该围绕 KV 的存储和复用来组织。

| 维度 | 选择 |
|------|------|
| 算法 | MLA / GQA + FP8 KV 量化 |
| 系统 | 四级存储（HBM→DRAM→SSD→跨节点 RDMA 池） |
| 部署 | PD 分离 + cache-aware 全局路由（以 KV 命中为路由核心依据） |
| 工作负载 | 通用（多轮 + RAG + Coding + Agent 都覆盖） |

**关键设计决策：**

- 分布式 RDMA KV 池（Mooncake Store）作为独立产品，而非引擎内部组件
- block-hash 全局寻址：任意节点可以查询任意 prefix 是否在全局池中
- 路由器感知 KV 命中：请求路由到已有最长匹配 prefix 的实例，无论该实例是 P 节点还是 D 节点

**实测数据（agentic trace）：** cache hit rate 1.7% → 92.2%，吞吐 3.8×，P50 TTFT −46×。

---

## SGLang 范式

**核心思路：Radix Tree 精确匹配 + 结构化生成**

| 维度 | 选择 |
|------|------|
| 算法 | 标准 GQA（主要靠系统优化，不依赖算法压缩） |
| 系统 | RadixAttention（任意 token 边界的 prefix cache）+ HiCache 三层 offload |
| 部署 | Cache-aware router（选择 radix tree 匹配最深的副本） |
| 工作负载 | 结构化输出（JSON / 约束生成）+ 多轮对话 + API 服务 |

**关键设计决策：**

- RadixAttention 突破了 block_size 对齐的限制，可以在任意 token 边界命中——特别有利于结构化 prompt 的细粒度复用
- SLRU 淘汰策略：相比纯 LRU，热点 prefix block 更难被驱逐
- ShadowRadix：HiCache 的扩展，支持超长上下文（4K → 900K）几乎无吞吐损失

---

## vLLM 生态范式

**核心思路：开放生态 + 可插拔扩展**

| 维度 | 选择 |
|------|------|
| 算法 | GQA + 可选 FP8/INT8 量化（TurboQuant / 原生 FP8） |
| 系统 | PagedAttention（原创）+ LMCache 多级存储（可插拔） |
| 部署 | vLLM Production Stack（路由 + 自动扩缩容）+ Mooncake Store（分布式 KV） |
| 工作负载 | 通用 API 服务，生态覆盖最广 |

**关键设计决策：**

- 开放插件接口：KV 后端（LMCache / Mooncake）、attention backend（FlashAttention / FlashInfer / Triton）、路由层（Production Stack）均可替换
- 代价：组件间接口版本兼容性是持续的维护负担
- MultiConnector 统一 PD + Store 拓扑管理，是向 Mooncake 范式收敛的方向

---

## TensorRT-LLM 范式

**核心思路：NVIDIA 全栈垂直整合**

| 维度 | 选择 |
|------|------|
| 算法 | FP8 / NVFP4 原生量化（NVIDIA 自研格式） |
| 系统 | KVCacheManager（两阶段 Claim + BlockKey 多维编码）+ GDS NVMe |
| 部署 | Disaggregated serving（NIXL 传输）+ NVLink-C2C 高带宽互联 |
| 工作负载 | 高性能 API 服务，NVIDIA 硬件（H100/H200/GB200/GH200）专属优化 |

**关键设计决策：**

- BlockKey 多维编码（LoRA ID + 多模态哈希 + cache_salt）：在 block 级别同时支持多 LoRA 服务和多租户隔离
- Priority-based LRU：高优先级 block（如 system prompt）不参与普通驱逐，命中率 +20% vs 纯 LRU
- GH200 NVLink-C2C 900 GB/s：PD 分离的 KV 传输带宽是 PCIe 的 7×，基本消除传输瓶颈

来源：主站 essay [TRT-LLM KVCache Runtime 架构](/notes/2026/05/09/trtllm-kvcache-runtime-architecture/)

---

## 新兴范式（待完善）

### AIBrix 范式

- K8s 原生的弹性伸缩 + cache-aware 调度
- 以 Kubernetes CRD 管理 KV pool 的生命周期
- 目标：让 KVCache 管理与云原生基础设施无缝集成

### Dynamo 范式

- NVIDIA 推出的分布式推理基础设施
- 与 TRT-LLM 深度集成，统一管理 prefill / decode / KV 传输的调度
- 适合大规模多节点部署场景

---

## 范式选型参考

| 需求 | 推荐范式 |
|------|---------|
| 通用 API 服务，需要开放生态 | vLLM 生态范式 |
| 追求最高 prefix cache 命中率，Agent / 多轮为主 | Mooncake 范式 |
| 结构化输出，精细 prefix 复用 | SGLang 范式 |
| NVIDIA 硬件，极致性能，多 LoRA 服务 | TRT-LLM 范式 |
| 云原生 K8s 部署，弹性伸缩 | AIBrix 范式（成熟度待观察） |

---

## 关联章节

- 各框架的详细实现：[框架对比](frameworks.md)
- 评估各范式的方法学：[评估方法](evaluation.md)
- 未来可能出现的新范式：[未来方向](future.md)

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 从维度交叉总览拆分，补充各范式的四维分解表和选型参考 |
