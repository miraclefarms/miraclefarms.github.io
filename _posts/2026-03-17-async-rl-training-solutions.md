---
title: "强化学习异步训练方案深度解析：三派路线与技术选型"
date: 2026-03-17 20:00:00 +0800
author: Ethan
kind: essay
category: Essay
intro: 系统梳理 AReaL、verl、slime、StreamRL、AsyncFlow、LlamaRL、TRL AsyncGRPO 等七大异步 RL 训练方案，以激进派、保守派、工程派三条路线为框架，分析各方案的架构设计、staleness 处理策略与选型边界。
---

# 强化学习异步训练方案深度解析：三派路线与技术选型

2025 年是异步强化学习（Async RL）爆发的一年。主流训练框架密集发布异步方案，从学术原型到生产验证，经历了从“能否工作”到“如何选型”的范式转变。本文基于 AReaL、verl、slime、StreamRL、AsyncFlow、LlamaRL 等核心工作，系统梳理当前异步 RL 的技术路线分化，为 AI Infra 架构师和工程团队提供选型参考。

---

## 一、问题背景：为什么需要异步训练

同步 RL 训练的核心矛盾在于 **Rollout（生成）与 Training（训练）的时间不对称**。以 GRPO/PPO 为代表的 RL 算法需要大量 rollout 样本来计算 advantage 和更新策略，但 LLM 生成是显存密集且延迟不确定的任务——一个几千 token 的 response 生成可能耗时数十秒，而梯度更新通常在秒级完成。

这种不对称导致同步系统中 GPU 利用率低下。训练进程等待生成完成，生成进程又依赖训练后的模型权重，形成明显的流水线气泡（pipeline bubble）。对于长推理链任务（如数学推理、代码生成），这种气泡尤其严重——生成时间可能比训练时间高出一个数量级。

### 同步系统的扩展性困境

同步 RL 系统在大规模（千卡级）训练下面临额外的扩展性挑战。随着 GPU 数量增加，每个 GPU 分到的 batch size 减小，推理解码从计算密集型转向内存 IO 密集型，增加设备反而无法提升吞吐[1]。这与 RL 训练需要大量并行样本的需求形成根本矛盾。

### 异步训练的核心思路

异步训练通过 **解耦生成与训练**，让两者并行推进。但这引入了一个根本性挑战：**数据时效性（staleness）**。当训练 worker 使用的是由旧版本模型生成的样本时，策略梯度估计会偏离真实方向，影响训练稳定性甚至导致训练发散。

各框架的核心分歧，本质上就在于如何处理这个 tradeoff——是用算法改进抵消 off-policy 代价，还是通过工程手段减少异步程度，抑或两者兼顾。

---

## 二、核心框架深度解析

### 2.1 AReaL：激进算法路线的代表

**定位**：完全异步，学术系 + 工程系双重严谨

AReaL（Asynchronous Reinforcement Learning）来自清华大学 IIIS 实验室与蚂蚁集团（现置于 inclusionAI 组织下）[1]，是当前最彻底的异步 RL 方案。其核心理念是 **用算法改进抵消 off-policy 代价**，而不是通过工程手段减少异步程度。

![AReaL 架构图](/assets/async-rl-training-solutions-comparison/areal-arch.png)

**图 1：AReaL 系统架构**[1]

#### 核心设计：Staleness-aware PPO

传统 PPO 在 off-policy 样本上会导致策略退化，AReaL 通过两个机制解决这一问题：

1. **动态负载均衡**：平衡 rollout worker 和 training worker 的工作负载，将数据 staleness 控制在合理范围内
2. **Decoupled PPO 目标函数**：引入 proximal policy（近端策略）的概念，将行为策略（behavior policy）和近端策略解耦，使得来自不同模型版本的样本能够被正确处理[1]

这种设计的核心洞察是：与其严格限制 stale 程度，不如从算法层面增强对 stale 数据的容忍度。

#### 流式生成

AReaL 不等待一个完整 batch 的 rollout 完成，而是以流式方式持续生成新输出，每个 rollout worker 独立运行，模型更新后立即同步权重到所有 worker。这种设计对于长 response 场景尤为友好——长序列生成时间远大于训练时间，异步可以将生成完全隐藏在关键路径之外[1]。

#### 可中断 Rollout

AReaL 实现了可中断的 rollout worker。当新权重到达时，正在进行的生成请求会被安全中断，KV cache 被丢弃，用新权重重新计算。这种设计虽然引入了额外计算，但避免了等待最长序列完成的瓶颈[1]。

#### 性能数据

AReaL 展示了近似线性的扩展趋势，在 512 GPU 上达到 2.57x 训练加速。更重要的是，在数学和代码推理任务上不仅没有性能损失，反而有所提升[1]。

#### 最新进展（2026 年 2 月）

- **AReaL-SEA**：结合自进化数据合成引擎，使 235B MoE 模型在 τ²-bench 上达到与 Gemini 3.0 Pro 相当的表现
- **AReaL-lite**：以 20% 代码量保留 90% 性能，为研究者提供快速原型支持[1]

### 2.2 verl：生态最丰富的可选方案

**定位**：生态最丰富，异步作为可选 recipe

verl（Volcano Engine Reinforcement Learning）来自字节跳动火山引擎[2]，其主干设计是同步的（colocate 模式），异步作为 `verl.experimental` 下的独立 recipe 提供。这种设计的好处是：**用户可以根据任务特性选择合适的模式**，而不需要切换框架。

#### 异步方案三档位

verl 的异步方案分为三个档位，渐进式地引入异步程度：

| 模式 | 描述 | 风险等级 |
|------|------|----------|
| one_step_off_policy | rollout 和 train 之间异步一步，最保守 | 低 |
| fully_async_policy (streaming mode) | 真正的流水线重叠，支持 partial rollout | 中 |
| async stream pipeline + partial rollout (mode d) | 最激进，staleness 样本直接参与训练 | 高 |

#### old_log_prob 的处理

在完全异步策略中，一个关键的工程细节是 **old_log_prob 的处理**。verl 默认由 rollout 侧计算 old_log_prob，而非 trainer 侧。这是因为 importance sampling 需要知道生成样本时的策略概率，如果由 trainer 重新计算会引入误差[2]。

verl 同时提供了 Rollout Importance Sampling 选项用于进一步修正。

#### 性能数据

128 GPU 训练 Qwen2.5-7B 时实现 2.35x–2.67x 的性能提升，对最终效果没有显著影响[2]。

verl 的优势在于其灵活性——用户可以先用同步模式验证算法有效性，再逐步切换到异步模式追求吞吐提升。

### 2.3 slime：SGLang-native 的生产方案

**定位**：SGLang-native，同步/异步统一框架，轻量高度可定制

slime 来自清华大学 THUDM / GLM 团队[3]，是 GLM-4.5、4.6、4.7 背后的训练框架，已在生产环境中验证。与其他框架不同，slime 从设计之初就将 SGLang 作为一等公民，推理引擎与训练框架原生集成。

![slime 架构图](/assets/async-rl-training-solutions-comparison/slime-arch.png)

**图 2：slime 异步训练架构**[3]

#### 核心设计：--colocate 开关

slime 的核心设计亮点是通过 **单个 --colocate 开关** 切换两种部署模式：
- **colocated**：同一 GPU 上运行生成和训练
- **decoupled**：独立 GPU 分别运行生成和训练

借助 Ray 的 `.remote()` 异步执行能力，切换同步/异步行为只需移动 `ray.get` 的位置——这种极简的 API 设计大大降低了使用门槛[3]。

#### 三大核心组件

slime 异步架构的三个核心组件[3]：

1. **Training (Megatron)**：负责主训练循环，从 Data Buffer 读数据，训练后同步参数给 rollout
2. **Rollout (SGLang + Router)**：生成新数据（含 reward/verifier 输出），写入 Data Buffer
3. **Data Buffer**：桥接模块，管理 prompt 初始化、自定义数据和生成策略

#### 可中断 Rollout 与权重预empt

在完全异步解耦设计中，GPU 分区为 rollout 引擎和训练引擎两部分。新权重广播给 rollout server 后，正在飞行的请求会被安全 preempt，然后以新模型继续生成[3]。这个细节体现了工程上的成熟度。

#### APRIL 优化

**APRIL**（Active Partial Rollout）是 slime 应对长尾生成瓶颈的独特方案[3]：过采样启动更多 rollout 请求（如目标 32 个，则启动 64 个），达到目标批大小后立即中止剩余请求，未完成的轨迹缓存后在下一轮从中断处继续。

#### Router 外露

slime 的另一个差异化优势是通过 Router 对外暴露接口，外部 agent 可以直接调用内部 SGLang server，实现纯异步 agentic 训练。这对于需要与外部评估器、工具链交互的多轮任务尤其有价值[3]。

### 2.4 TRL AsyncGRPO：轻量实验方案

TRL（Transformer Reinforcement Learning）的 AsyncGRPO 目前仍是实验性功能（PR #5293，2026-03-16 提交）[4]，定位是轻量级异步方案。

其设计哲学是 **最小化异步偏移**——只允许一步或少量步骤的异步偏移，最大程度保留 on-policy 特性。

AsyncGRPO 使用 vLLM 作为独立推理进程，通过 NCCL 进行权重同步，HTTP trigger 触发更新。这种设计的优势是轻量（2 GPU 即可运行），但局限性也很明显：不支持 partial rollout，对多轮 agentic 训练的支持有限[4]。

---

## 三、补充工作：流式 Pipeline 路线

### 3.1 StreamRL（ByteDance Seed）

![StreamRL 架构图](/assets/async-rl-training-solutions-comparison/streamrl-arch.png)

**图 3：StreamRL Disaggregated 架构**[5]

StreamRL 的核心创新是 **Stream Generation Service（SGS）**——不等所有样本生成完毕就将完成的样本流式返回给 Trainer，使 Trainer 得以在样本就绪时立刻开始处理[5]。

这种设计实现了 **dynamic-batch pipelining**：计算重叠。对于异步 RL 算法，流式传输的额外价值在于 **使权重传输脱离关键路径**，生成和训练真正并行推进[5]。

StreamRL 不修改算法本身，属于系统优化而非算法松弛。

### 3.2 AsyncFlow（Huawei / MindSpeed）

AsyncFlow 引入了 **TransferQueue**——一个带分布式存储能力的中央化数据管理模块，作为异步流式 dataloader[6]。

其核心创新是将依赖粒度从 batch 级降低到 mini-batch 级：下游任务可以在一个 mini-batch 的 rollout 完成后立即推进 reward 和 train 流程。

这种设计的哲学是 **不改变算法**，而是通过更细粒度的流水线调度来消除气泡。本质上是在同步框架内实现异步效率[6]。

### 3.3 LlamaRL（Meta）

LlamaRL 在 Llama 3 后训练中实现了 **10.7x 加速**（405B 参数模型），其技术组合包括[7]：

- **Colocated 模型卸载**：在推理和训练间动态分配 GPU 资源
- **异步 off-policy 训练**：放宽对数据纯度的要求
- **NVLink 分布式直接内存访问**：加速权重同步

LlamaRL 的价值在于验证了异步方案在超大规模模型（400B+ 参数）上的可行性，以及 NVLink 在权重同步中的关键作用[7]。

---

## 四、横向对比

| | TRL AsyncGRPO | AReaL | verl fully_async | slime async |
|---|---|---|---|---|
| **推理引擎** | vLLM（独立进程） | SGLang / vLLM | vLLM / SGLang | SGLang（native）|
| **权重同步** | NCCL（HTTP trigger）| 内置同步 | NCCL / gloo | Ray + 自定义 |
| **Staleness 处理** | max_staleness 丢弃 | staleness-enhanced PPO | staleness_threshold | 可配置 |
| **Partial Rollout** | ✗（暂无） | ✓ | ✓ | ✓（APRIL） |
| **多轮 Agentic** | 有限支持 | ✓（ASearcher） | ✓（retool） | ✓（Router 外露） |
| **分布式规模** | 轻量（2 GPU）| 千卡级 | 百卡～千卡 | 百卡级 |
| **成熟度** | 实验性（PR #5293）[4]| 生产验证 | experimental recipe | 生产验证（GLM 系列）[3]|

---

## 五、技术分歧的本质

这些框架在一个核心问题上存在 **哲学分歧**：异步带来的 off-policy 程度如何处理？

### 激进派：AReaL、slime 完全异步模式

- **核心理念**：直接接受 staleness，用改进的重要性采样或丢弃策略补偿，追求最大吞吐
- **算法改动**：较大，需要修改 PPO 目标函数或引入新机制
- **适用场景**：长序列生成、多轮交互任务、千卡级大规模训练

### 保守派：verl one_step_off_policy、TRL AsyncGRPO

- **核心理念**：只允许一步或少量步骤的异步偏移，最小化算法风险
- **算法改动**：最小，基本保持原 PPO/GRPO 行为
- **适用场景**：算法验证、快速原型、训练稳定性优先的场景

### 工程派：StreamRL、AsyncFlow

- **核心理念**：不修改算法，通过流式 pipeline 在 on-policy 前提下消除 bubble
- **算法改动**：无，保持原算法的纯度
- **适用场景**：希望在保持算法纯度的同时提升效率的团队

### 选型的本质

同步和异步之间存在 **训练稳定性与硬件利用率的根本性权衡**。同步方法因其对 rollout 纯度的可控性，常被用于完成时间相对均匀的任务（如数学推理）；而异步系统在复杂交互式环境（如多步 agentic 任务）中优势更明显。

---

## 六、选型建议

### 选同步的情况

- 任务完成时间相对均匀（如数学推理、代码生成）
- 训练稳定性优先于硬件利用率
- 处于算法验证阶段
- 规模较小（百卡以下）

### 选异步（激进派）的情况

- 长 response 场景（生成时间 >> 训练时间）
- 多轮 agentic 任务
- 千卡级大规模训练
- 已完成算法验证，追求生产效率

### 选异步（保守派）的情况

- 想要异步的吞吐优势，但不希望引入太大算法复杂度
- 愿意在稳定性和效率之间做渐进式权衡
- 团队对 off-policy 算法的经验有限

### 选流式 pipeline 的情况

- 不希望修改算法本身
- 希望通过工程手段提升效率
- 团队对现有算法框架有强依赖

---

## 七、结论与展望

异步 RL 训练已经过了“从 0 到 1”的阶段，进入“从多样方案中选择最适合自己的”的成熟期。技术路线分化的背后是 **训练稳定性与硬件利用率** 的根本性权衡——没有免费的午餐，每种方案都有其适用边界。

对于 AI Infra 架构师而言，关键判断是：

1. **先明确任务特性**：长序列还是短序列、agentic 还是单轮、规模是大还是小
2. **再选择技术路线**：激进算法路线追求吞吐，保守工程路线追求稳定，流式 pipeline 路线追求兼容性
3. **最后选具体框架**：根据团队技术栈（PyTorch/Megatron-Ray）、规模预期、成熟度要求做最终决策

### 未来值得关注的变量

- 各框架在更大规模（万卡级）上的扩展性表现
- Staleness 处理机制的进一步优化
- 与新推理引擎（如 SGLang v2）的深度集成
- Agentic 训练场景的最佳实践积累

---

## 参考来源

[1] AReaL 官方仓库：https://github.com/inclusionAI/AReaL；论文：https://arxiv.org/abs/2505.24298

[2] verl 官方仓库：https://github.com/verl-project/verl；文档：https://verl.readthedocs.io

[3] slime 官方仓库：https://github.com/THUDM/slime

[4] TRL AsyncGRPO PR #5293: https://github.com/huggingface/trl/pull/5293

[5] StreamRL 论文：https://arxiv.org/abs/2504.15930

[6] AsyncFlow 论文：https://arxiv.org/abs/2507.01663

[7] LlamaRL 论文：https://arxiv.org/abs/2505.24034
