# TensorRT-LLM

[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) 是 NVIDIA 官方推理框架，以极致的 GPU 性能优化和与 NVIDIA 硬件的深度整合为目标。

!!! warning "内容时效性"
    TRT-LLM 更新极快。请以项目的最新文档和 Release Notes 为准。

---

## KVCache 架构

### KVCacheManager

TRT-LLM 的分页 KV 管理核心组件：

- **两阶段 Claim**：`addSequenceBatch()` 先锁定可复用的 Block，再批量 onboard，解决 C++ 环境下的 TOCTOU 竞态问题
- Block size 可配置
- 支持 Chunked Context：长 Prompt 切分成 chunks 处理，与 KVCache 管理协同

### BlockKey 多维编码

TRT-LLM 的 Block 哈希包含多个维度，实现多维缓存隔离：

- LoRA ID：不同 LoRA adapter 的 block 不互相污染
- 多模态哈希：图像/视频 token 的 block 单独标识
- `cache_salt`：租户/用户级别的隔离

### 三层存储

GPU HBM → Host DRAM → NVMe（通过 GDS）：

- 事件驱动路由：自动迁移 Block 到合适的 tier
- GDS（GPUDirect Storage）：NVMe 直接与 GPU 交换数据，绕过 CPU

### Priority-based LRU

高优先级 Block 获得保留槽位，不参与普通 LRU 驱逐：

- 实测：命中率 +20% vs 纯 LRU（在优先级分化明显的 workload 下）
- 应用场景：系统 prompt block 设为高优先级，用户历史 block 普通优先级

### Prefix Reuse 优化

PR #13139 将每个 pending request 的 radix tree 遍历从 5 次收敛为单次 `analyzePrefixReuse()`，减少调度开销。

---

## 主要特性

### PD 分离（Disaggregated Serving）

TRT-LLM 的 disaggregated serving 是当前最成熟的生产级实现之一：

**实测数据：**

| 模型 | 硬件 | 吞吐提升 | 备注 |
|------|------|---------|------|
| DeepSeek R1 | GB200 | 1.4–1.8× | 基础 PD 分离 |
| Qwen 3 | — | 最高 6.11× | 含 MTP 优化 |
| + MTP | — | 额外 +1.6–2.5× | 多 token 预测 |

**KV 传输：**

- NIXL 传输后端
- GH200 NVLink-C2C：900 GB/s = 7× PCIe Gen5 的带宽
- 跨节点 KV 传输在 NVLink-C2C 场景下几乎无瓶颈

### KV 量化

- FP8 KV Cache：原生支持，完整的 kernel 优化
- NVFP4：NVIDIA 自研格式，适配 Blackwell 架构
- MLA FP4 模型有 BF16 fallback pool

### 多模态 KV

BlockKey 的多模态哈希支持图像/视频 token 的 block 独立管理，为多模态 prefix cache 奠定基础。

---

## 与开源框架的关键差异

TRT-LLM 与 vLLM/SGLang 的主要区别：

| 维度 | TRT-LLM | vLLM / SGLang |
|------|---------|--------------|
| 目标 | NVIDIA 硬件极致性能 | 通用性 + 生态 |
| 开发语言 | 主要 C++，Python 接口 | 主要 Python |
| 自定义灵活性 | 较低（依赖 NVIDIA 工具链） | 较高 |
| PD 分离成熟度 | 生产级（NIXL + NVLink-C2C） | 进行中 / 较成熟 |
| BlockKey 多维度 | ✅（LoRA ID + 多模态 + salt） | 部分支持 |
| 与 NVIDIA 生态集成 | 深度（Dynamo、GDS、NVLink） | 有限 |

---

## 关联章节

- 两阶段 Claim 的 KV 分配机制：[Paged KV](paged-kv.md)
- PD 分离与 NIXL 传输：[PD 分离](pd-disaggregation.md)
- 框架总览与对比表：[框架对比](frameworks.md)

来源：主站 essay [TRT-LLM KVCache Runtime 架构](/notes/2026/05/09/trtllm-kvcache-runtime-architecture/)

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 从框架对比总览拆分，整理 KVCacheManager / BlockKey / PD 分离 / NIXL 等核心特性 |
