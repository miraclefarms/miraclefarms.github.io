---
title: 上下文窗口是第一约束：OpenDev 终端代理的工程逻辑
date: 2026-04-15 12:00:00 +0800
author: Ethan
kind: essay
category: Essay
intro: OpenDev 技术报告提出了一个不寻常的主张：在终端 AI 编程代理的设计中，上下文窗口管理不是附加功能，而是第一设计约束，由此衍生出复合 AI 路由、五层安全防线和扩展版 ReAct 循环等一系列结构性决策。
tags: [Agents]
---

构建一个能在终端中自主完成编程任务的代理，最难的问题是什么？不是工具调用，不是代码生成，也不是用户界面——而是**模型的工作记忆是有限的，但任务的时间跨度是无限的**。

OpenDev 的技术报告<a href="https://arxiv.org/html/2603.05344v1">[1]</a>试图回答这个问题。它不提供新算法，也不报告 benchmark 排名，而是系统地记录了一套终端原生 AI 代理的工程架构：为什么做这些决定，取舍是什么，在实践中踩过什么坑。这类"工程经验文档"在学术发表中相对罕见，但它往往比论文更接近系统构建者真正需要的东西。

## 一、上下文是稀缺资源，而非无限黑板

大多数代理教程把上下文窗口当成写入记录的地方：每次工具调用追加输入输出，积累到足够长的会话后截断。OpenDev 采取了相反的态度：上下文是预算，从设计第一天起就要主动管理。

这个判断带来的影响贯穿整个架构。系统提示被拆分为两层：可缓存的身份声明和静态策略放在前段，由 provider-level 缓存复用；动态上下文（当前任务状态、活跃记忆片段）放在后段，每轮重新构建。这样一来，多轮对话的 token 开销不会线性增长，因为前段命中缓存。

工具输出的处理也受同一原则约束。每类工具有独立的摘要策略：文件读取输出超过阈值时写入临时文件并只返回路径，Shell 命令输出做截断并附提示告知模型如何继续获取剩余内容。这些不是优化项，而是让代理能持续运行数小时的前提条件。

**自适应上下文压缩（Adaptive Context Compaction）**是这套策略的核心机制：当会话接近 token 预算上限时，系统对较旧的观察结果做渐进式摘要，释放 token 给后续推理。压缩是分阶段的——越旧的轮次压缩得越激进——而不是在达到阈值时一次性截断。这个设计的出发点是保留近期上下文的完整性，因为模型对近期信息的注意力权重更高。

## 二、五个角色，五个模型

![OpenDev 系统层级概览：session → agent → workflow → LLM 的四级结构](/assets/opendev-context-engineering-agent-design/fig-1-system-overview.png)
*图 1：OpenDev 把一次代理会话组织为 session → agent → workflow → LLM 的四级层级，每个 workflow 类型可以独立绑定不同的 LLM，实现按认知负载的模型路由。来源：论文 Figure 1。*

单一模型承担所有工作是最简单的架构，但不是最经济的。OpenDev 把代理的认知工作拆成五个角色，每个角色可以配置不同的模型：

- **Normal**：日常推理和工具调用，性能与成本的主要权衡点
- **Thinking**：需要长链路推理时启用，接受更高延迟
- **Critique**：对 Normal 的输出做自我批评和验证
- **VLM**：处理截图和视觉输入
- **Fast**：简单分类、简短查询，用最便宜的模型

这个设计的关键不是"用什么模型"，而是**把什么时候用哪种认知模式变成了显式的架构决策**，而不是在 system prompt 里写"think step by step"然后希望模型自己判断。Thinking 和 Critique 角色只在对应的 ReAct 阶段被调用，不会污染其他轮次的 token 预算。

每个角色支持 fallback 链：主模型不可用或响应失败时，自动切换到备用配置。这是在 provider 层面做的容错，用户不需要在 prompt 层面处理。

## 三、五层防线：结构替代信任

![OpenDev 代理执行循环（Harness）架构：六阶段 ReAct 循环与七个支撑子系统](/assets/opendev-context-engineering-agent-design/fig-2-agent-harness.png)
*图 2：代理 Harness 把六阶段 ReAct 循环（pre-check、thinking、critique、action、tool execution、post-processing）嵌入七个支撑子系统，上下文工程和安全系统是其中最核心的两层。来源：论文 Figure 4。*

"给模型写一段安全提示"是代理安全的最常见处理方式，也是最脆弱的方式——因为模型可能忘记，可能被后续指令覆盖，也可能在长会话中出现指令衰减（instruction fade-out）。

OpenDev 的回答是**防深度纵深**（defense-in-depth）：五个独立层次，每层在不同抽象级别拦截危险操作，任何一层失效不会危及其他层次。

1. **Prompt-level guardrails**：系统提示中的安全策略和行为约束，第一道防线
2. **Schema-level tool restrictions**：Plan Mode 下的工具白名单，subagent 的 `allowed_tools` 过滤，MCP 工具的显式授权门控
3. **Runtime approval system**：Manual / Semi-Auto / Auto 三级审批，权限可持久化（避免用户对高频操作的审批疲劳）
4. **Tool-level validation**：DANGEROUS_PATTERNS 黑名单、stale-read 检测（写入前必须先读）、输出截断防止 token 溢出
5. **Lifecycle hooks**：pre-tool 阻断、参数变换、JSON stdin 协议——用户可以在这一层注入自定义安全逻辑

每层的设计原则是：**在模型做出错误判断之前，而不是之后，做拦截**。Prompt-level 层的设计是 Model as reasoner；Schema-level 和 Tool-level 层不信任模型的判断，直接在基础设施层面限制能力范围。

值得关注的是 Plan Mode 的设计：这是一个只读工具子集的受限模式，代理在这个模式下只能探索和分析，不能写入文件或执行 Shell 命令。用户审批计划后才切换到 Normal Mode 执行。这个分离把"决定做什么"和"实际去做"拆开，让用户在最低成本的节点介入监督。

## 四、扩展版 ReAct：把思考和批评分开

标准 ReAct 循环（Reason-Act-Execute-Observe）是大多数代理框架的基础。OpenDev 在此之上加了两个显式阶段：**Thinking 阶段**（用 thinking 模型做链式推理，结果作为 Normal 模型的上下文输入）和 **Critique 阶段**（对 Action 阶段的输出做自我批评，检查是否存在遗漏或错误）。

加这两个阶段的代价是延迟和 token 消耗，收益是减少"第一个反应就执行"的错误。在代码修改这类高风险操作中，多花 10 秒思考通常比回滚代价低得多。

循环的另一个细节是**doom-loop 检测**：系统追踪连续的工具调用模式，如果检测到代理陷入重复执行相同操作的循环（典型场景：错误处理不当、工具输出解析失败），会主动中断并请求用户介入，而不是让 token 预算耗尽。

事件驱动的 System Reminders 解决了长会话中的指令衰减问题：系统不依赖模型"记住"初始系统提示里的内容，而是在检测到特定事件时（连续多轮工具调用失败、迭代次数过高、出现安全违规）主动向模型注入提示。这类似于在代码中用断言而不是假设——相信"指令一直有效"是假设，注入提醒是断言。

## 五、结论

OpenDev 的工程判断最终落在一句话上：**上下文窗口管理不是可以后期优化的功能，而是代理系统的骨架，需要在架构设计阶段就确定。** 复合 AI 模型路由、五层安全防线、扩展版 ReAct 循环、懒加载工具发现——这些设计的共同起点都是"token 是稀缺资源，模型的工作记忆不可靠"。

这个判断有一个明确的适用边界：它在长时间跨度、多工具调用、有真实破坏风险的场景下成立；对于单轮问答或短任务代理，这套架构的复杂度远超收益。

报告本身有一个显著的缺失：**没有量化评估**。对于一个强调上下文工程和安全架构的系统，我们不知道自适应压缩对任务完成率的实际影响，不知道五层防线在多大比例上捕获了 prompt-only 方法遗漏的危险操作。这类系统论文往往用工程严密性换取实证数据——这是理解该报告价值时需要认识到的前提。

---

## 参考资料

[1] Nghi D. Q. Bui. Building AI Coding Agents for the Terminal: Scaffolding, Harness, Context Engineering, and Lessons Learned. https://arxiv.org/html/2603.05344v1
