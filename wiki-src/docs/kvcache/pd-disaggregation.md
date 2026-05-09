# PD 分离中的 KVCache

## Prefill 与 Decode 的不同需求

在传统的单实例部署中，Prefill 和 Decode 交替发生在同一批 GPU 上。但两者的计算特征截然不同：

| 维度 | Prefill | Decode |
|------|---------|--------|
| 计算模式 | 并行处理整个 Prompt | 每步处理 1 个 token |
| 计算强度 | Compute-bound（矩阵乘为主） | Memory-bound（读取 KVCache 为主） |
| GPU 利用率 | 高 | 低（受 HBM 带宽限制） |
| 批次大小影响 | 大 batch 提升效率 | 大 batch 提升效率（但受 KVCache 限制） |
| 延迟贡献 | TTFT（首 token 延迟） | TPOT（每 token 延迟） |

混跑时，两者相互干扰：Prefill 占用 GPU 计算资源时，正在 Decode 的请求不得不等待，产生"卡顿"（stall），增加 TPOT；大量 Decode 请求占用 HBM 时，新来的 Prefill 请求又必须排队等待显存释放。

PD 分离（Prefill-Decode Disaggregation）将 Prefill 和 Decode 分配到不同的 GPU/节点上，各自优化，解除耦合。

## KVCache 在 PD 分离中的核心位置

PD 分离的关键工程挑战是：**Prefill 节点产生的 KVCache，必须传输给 Decode 节点才能继续生成**。

```
请求流向：

Client → Router → Prefill Node（生成 KV） → KV Transfer → Decode Node（消费 KV）
                                               ↑
                                         核心工程挑战
```

这一 KV Transfer 的效率直接影响：
- Prefill 完成到第一个 Decode token 之间的间隔（即 KV 传输延迟）
- 系统端到端的 TTFT（TTFT = Prefill 时间 + KV 传输时间）
- 网络带宽是否成为系统瓶颈

## KV Transfer 的方式

### 基于 RDMA / InfiniBand

最高性能的方案。Prefill 节点通过 RDMA 将 KV Block 直接写入 Decode 节点的 GPU HBM（需要 GPUDirect RDMA 支持）或 CPU DRAM。

- 带宽：InfiniBand HDR 约 200 Gbps（~25 GB/s），NDR 约 400 Gbps（~50 GB/s）
- 延迟：微秒级
- 适合：同一数据中心内的高性能集群

### 基于 NVLink / NVSwitch

在同一机器内的多 GPU 环境中，NVLink 提供更高的带宽（NVLink 4.0 双向 900 GB/s），但跨节点不适用。

### 基于以太网（TCP/IP）

带宽约 25-100 Gbps，延迟更高，适合普通云环境或成本敏感场景。延迟增加意味着 TTFT 劣化。

## 对调度与放置策略的影响

KV Transfer 引入了 Prefill 节点和 Decode 节点之间的**依赖关系**，调度系统必须处理：

### KV 放置决策

- Decode 节点需要提前预留 KVCache 空间给即将到来的 KV
- 如果 Decode 节点 HBM 不足，必须延迟 Prefill 完成后的 KV 传输，或提前驱逐其他 KV
- 不合理的放置会导致 Decode 节点成为传输和显存双重瓶颈

### 负载均衡

- Prefill 负载由请求数和 Prompt 长度决定（高度变化）
- Decode 负载由在线 Sequence 的数量和生成长度决定（相对稳定但也有波动）
- Router 需要将请求路由到合适的 Prefill-Decode 对，避免单点饱和

### 亲和性（Affinity）

对于多轮对话，用户每轮的新消息需要先 Prefill（增量部分），再 Decode。为了复用历史轮次的 KVCache（Prefix Cache），理想情况下历史 KV 应该在同一个 Decode 节点上。这对路由策略提出了亲和性要求。

## TP Size 不一致的问题

一个复杂的工程场景：Prefill 节点和 Decode 节点的**张量并行（TP）配置可能不同**。

- Prefill 需要高并发计算，可能使用 TP=8（8 卡并行处理 Attention）
- Decode 对计算要求低，可以用 TP=4 或 TP=2 节省资源

TP 配置不同时，同一 Sequence 的 KVCache 在两端的存储布局不同（K/V 按 head 分片的方式不一致），直接传输无法使用，必须做格式转换（KV reshape），增加传输和计算开销。这是 PD 分离工程中的一个已知难题，各框架的解法不同。

## 异构节点场景

不同 GPU 型号（如 H100 + A100，或 H100 + H800）在 HBM 带宽、显存大小和 RDMA 连接能力上存在差异，进一步复杂化 KV 放置决策。

## 未来方向（待补充）

- KV 压缩传输：传输前对 KV 进行量化（如 FP8 或更低精度），减少传输带宽
- 流水线传输：Prefill 计算与 KV 传输重叠，隐藏传输延迟
- 跨节点 Prefix Cache：Decode 节点通过 RDMA 直接读取其他节点的 KV，无需 Prefill
- 弹性 Prefill 池：根据请求流量动态增减 Prefill 节点数量
