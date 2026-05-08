---
title: "LangChain 的 Harness 工程实践：从手工调优到可迭代的反馈系统"
date: 2026-04-07 10:00:00 +0800
author: Ethan
kind: reading
category: Reading
intro: LangChain 通过 harness 工程将其 coding agent 从 Top 30 提升至 Top 5，仅通过改动 harness（而非模型）将 Terminal Bench 2.0 得分从 52.8 提升至 66.5。本文深入分析其核心方法论，并与 Meta-Harness 的自动化思路形成对话。
tags: [Agents, Evaluation]
---

在前一篇关于 Meta-Harness 的文章中，我们讨论了**让 LLM 自动搜索并优化整个 Harness** 的思路——用 Harness Optimizer 替代人工调优，在搜索效率上取得了显著提升（4 次评估匹配其他方法 40 次的性能）。

但一个很自然的问题是：**这种自动化方法，是否是 harness 工程唯一的可行路径？**

LangChain 最近公布的实践给出了一个不同的答案。它们没有使用自动化搜索，而是在 Terminal Bench 2.0 上将编码 agent 从 52.8% 提升到 66.5%，同样仅修改了 harness，模型保持不变（GPT-5.2-Codex）[[1]](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/)。两种思路形成了有趣的对照：**一个是全自动搜索，一个是可迭代的反馈系统**。本文尝试深入分析 LangChain 的方法论，并与 Meta-Harness 的思路展开对话。

## 一、方法论差异：自动化搜索 vs 人工迭代

### 两种优化范式的核心区别

LangChain 的 harness 工程方法，本质上是一个**人工驱动的迭代优化流程**。核心步骤是：

1. **收集 Trace**：通过 LangSmith 记录 agent 的每次执行轨迹[[2]](https://docs.langchain.com/langsmith/observability-quickstart)
2. **错误分析**：并行启动多个分析 agent 诊断失败模式
3. **针对性修改**：基于分析结果修改 harness 的组件
4. **循环迭代**：回到步骤 1

这一流程与 Meta-Harness 的自动化搜索形成了鲜明对比。Meta-Harness 使用一个 LLM（Harness Optimizer）来自动完成"分析 → 生成新配置 → 测试"的循环；而 LangChain 则把这个过程**显式化为人机协作的工作流**，将"诊断错误"本身也作为一个 Agent Skill（Trace Analyzer Skill）。

| 维度 | Meta-Harness | LangChain |
|------|-----------|----------|
| 优化主体 | Harness Optimizer (LLM) | Human + Trace Analyzer Agent |
| 搜索方式 | 自动生成新配置 | 人工决策 + agent 辅助分析 |
| 反馈信号 | 评估分数 | 执行 Trace |
| 迭代粒度 | 每次自动评估 | 每次人工审核 |

### 两种方法的适用场景

这种差异并非偶然，它们指向不同的实际约束：

**Meta-Harness 的自动化搜索**适合的场景是：评估成本较低、任务定义清晰、需要快速收敛。它的核心优势在于"不用人介入"——这对于评估基准固定的任务（如 Terminal Bench）尤其有价值。

**LangChain 的人工迭代**适合的场景是：评估成本高昂、需要人类领域知识、或者优化目标本身在演化。LangChain 明确指出，人工在"第 3 步"（审核 proposed changes）时可以"很有帮助"——因为人类可以识别哪些改动"overfit 到某个任务"，从而避免在其他任务上产生回归[[1]](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/)。

**从架构视角看，两种方法的本质差异在于"反馈回路的闭合方式"**。Meta-Harness 把整个回路都交给 LLM；LangChain 则把"判断"这个环节保留给人，让 LLM 只做"分析"和"建议"。这可能是当前阶段更务实的选择——毕竟，判断一个 harness 改动是否"真的改进"还是"只是 overfit"，仍然需要超出纯文本能力的领域直觉。

## 二、核心改进：三个可调组件

### 2.1 优化空间的选择

LangChain 并没有对 harness 的所有组件同时优化，而是**刻意压缩优化空间**，聚焦于三个核心"knobs"[[1]](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/)：

- **System Prompt**：系统提示词
- **Tools**：工具集
- **Middleware**：在模型调用和工具调用周围的 hooks，LangChain 称之为"中间件"

![](/assets/langchain-harness-engineering/fig-1.png)

*图 1. LangChain 的 harness 三大可调组件：System Prompt、Tools、Middleware*

这与 Meta-Harness 的做法形成了有趣的呼应。Meta-Harness 同样不追求修改模型参数，而是在"整个 Harness"这个空间内做搜索——但它选择让 Optimizer 自己决定改什么；LangChain 则主动限制了搜索空间的维度，聚焦在三个组件上。

这种"聚焦"背后的考量是：**减少人工搜索时的认知负担**。当优化空间太大时，人工逐个尝试的效率很低；而明确的组件边界让迭代更有方向感。

### 2.2 构建-自验证循环

在所有改进中，LangChain 认为最有价值的是**为 agent 引入自验证（self-verification）能力**[[1]](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/)。

**问题的发现**：最常见的失败模式是 agent 写完代码后"重读自己的代码，确认看起来没问题"就直接结束——这本质上是在与自己对话，而不是与任务要求对话。模型倾向于接受自己第一个"看起来合理"的方案，而不是验证它是否真的满足任务要求。

**LangChain 的解决方案**：在 System Prompt 中加入明确的"四步指南"——

1. **Planning & Discovery**：阅读任务、扫描代码库、制定计划
2. **Build**：带着验证的目的实现，构建测试（如果任务没有提供）
3. **Verify**：运行测试、完整阅读输出、与任务要求对比
4. **Fix**：分析错误、回到原始 spec、修复问题

更关键的是，LangChain 还实现了一个 **`PreCompletionChecklistMiddleware`**，在 agent 退出前强制触发一次验证流程——这类似 [Ralph Wiggum Loop](https://ghuntley.com/loop/) 的思路：一个 hook 强制 agent 在"退出"时继续执行，而不是直接返回。

![](/assets/langchain-harness-engineering/fig-4.png)

*图 4. 自验证循环：agent 在退出前强制执行验证流程*

**与 Meta-Harness 的对话**：Meta-Harness 的核心贡献是"自动化搜索"；而 LangChain 的核心贡献是"在 harness 中嵌入自验证结构"。这两者并不冲突——实际上，一个很自然的方向是把 LangChain 的"验证循环"也纳入 Meta-Harness 的搜索空间，让 Optimizer 自己发现"加入 self-verification 比不加更好"。

### 2.3 环境上下文注入

第二个有价值的改进是**为 agent 注入环境上下文**[[1]](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/)。

Terminal Bench 任务带有目录结构、内置工具和严格的超时限制——但 agent 一开始对这些一无所知。LangChain 的做法是：

- **`LocalContextMiddleware`**：在 agent 启动时映射当前工作目录和父子目录结构，运行 `bash` 命令查找可用的 Python 环境等工具
- **指导可测试代码**：告诉 agent "你的工作将通过程序化测试来评估"，从而引导 agent 生成符合测试接口的代码
- **时间预算**：注入超时提醒，引导 agent 在时间耗尽前切换到验证模式

**核心洞察**：Harness 的本质不是"给模型更多知识"，而是**为模型准备和传递上下文**，使其能够自主完成工作[[1]](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/)。"prepare and deliver context so agents can autonomously complete work"——这是 LangChain 对 harness 工程师角色的重新定义。

### 2.4 循环检测与恢复

第三个有价值的改进是**检测和恢复"doom loops"**[[1]](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/)。

��� agent 决定了一个方案后，它可能会"执着"地小幅修改同一文件 10+ 次——陷入微小的迭代陷阱。LangChain 实现了 **`LoopDetectionMiddleware`**，追踪同一文件的编辑次数，在达到 N 次后注入"consider reconsidering your approach"（考虑重新考虑你的方法）的提示。

这是一个非常有意思的观察：**模型当前的缺陷，需要用 harness 来"兜底"**。LangChain 明确指出，这是"设计一个 heuristic 来绕过当前模型的感知缺陷"，随着模型改进，这些 guardrails"可能变得不必要"——但今天它们仍然有价值[[1]](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/)。

### 2.5 推理预算的"三明治"

最后一个值得关注的改进是**推理预算的分配策略**[[1]](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/)。

GPT-5.2-Codex 有 4 种推理模式：low, medium, high, xhigh。LangChain 发现：

- **全程 xhigh**：得分仅 53.9%，因为推理消耗了太多时间导致超时
- **全程 high**：得分 63.6%
- **"推理三明治"**（xhigh → high → xhigh）：得分 66.5%

![](/assets/langchain-harness-engineering/fig-5.png)

*图 5. 推理预算"三明治"策略*

"三明治"的逻辑是：规划阶段需要深度推理理解问题，验证阶段也需要深度推理 catch 错误，而中间的执行阶段可以用较低推理预算节省时间[[1]](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/)。

**这指向的方向是"自适应推理"（Adaptive Reasoning）**——类似 Claude 和 Gemini 模型自身支持的功能[[3]](https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking)。在多模型 harness 中，一个很自然的扩展是：用大模型做规划，小模型做执行。

## 三、Trace 作为一等公民

### 3.1 为什么是 Trace？

通读 LangChain 的方法论，最核心的发现可能是：**Trace（执行轨迹）是反馈回路的一等公民**[[1]](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/)。

模型在今天很大程度上仍然是黑箱——它们的内部机制难以解释。但我们可以看到它们的输入和输出（即文本空间），而 Trace 正是这个文本空间的结构化记录[[2]](https://docs.langchain.com/langsmith/observability-quickstart)。

**Meta-Harness 的反馈信号是"评估分数"**——只有对/错、得分高/低的二元判断。**LangChain 的反馈信号是"执行轨迹"**——可以看到模型在哪一步出了错、为什么出错、甚至它"以为"自己在做什么。

这两种反馈信号的差异，本质上是"优化目标"的设计差异：

| 反馈信号 | 信息量 | 适用方法 |
|---------|--------|---------|
| 评估分数 | 低（只有结果） | 自动化搜索（Meta-Harness） |
| 执行轨迹 | 高（包含过程） | 人工迭代（LangChain） |

### 3.2 Trace Analyzer Skill

LangChain 将"分析 trace"本身也做成了一个 **Agent Skill**——这让它可以重复使用。

![](/assets/langchain-harness-engineering/fig-3.png)

*图 3. Trace Analyzer Skill 工作流程*

工作流程是：

1. 从 LangSmith 获取实验 traces
2. 并行启动多个错误分析 agent，主 agent 综合发现和建议
3. 聚合反馈，制定针对性的 harness 改动

这与机器学习中 **boosting** 的思路非常相似——关注之前轮次的错误[[4]](https://en.wikipedia.org/wiki/Boosting_(machine_learning))，但不是调整模型参数，而是调整 harness 配置。

**这里露出了一个很深的洞见**：当"调整模型参数"不可行时（因为我们不能改模型权重），我们可以用 trace 作为"伪梯度"——通过分析错误模式来指导搜索方向。

## 四、与 Meta-Harness 的对话：两种优化哲学

### 4.1 互补而非替代

回顾两篇文章，LangChain 的"人工迭代"与 Meta-Harness 的"自动搜索"并非对立，而是互补���

| 维度 | Meta-Harness | LangChain |
|------|-----------|----------|
| 优化主体 | LLM (Optimizer) | Human + Trace Analyzer |
| 反馈信号 | 评估分数 | 执行 Trace |
| 迭代方式 | 自动生成新配置 | 人工审核改动 |
| 适用场景 | 评估成本低 | 评估成本高 |

**最务实的做法可能是两者的结合**：用 LangChain 的方法论定义"什么样的改动是有效的"（比如 self-verification、context injection），用 Meta-Harness 的自动化来搜索这些组件的最优配置。

### 4.2 收敛方向

两篇文章都指向同一个结论：**在模型固定的情况下，优化 harness 可以带来巨大的性能提升**。

- Meta-Harness：37.6% vs 27.5%（人工基线），相对提升 37%
- LangChain：66.5% vs 52.8%，相对提升 26%

这与 Meta-Harness 论文中"在模型固定的情况下，仅优化 Harness 就能带来巨大的性能提升"的判断完全一致。对实际应用的意义是：**换模型之前，先优化 harness**。

### 4.3 开放问题

两篇文章也都暴露了类似的开放问题：

1. **Harness 是否可泛化？** 一个为 Terminal Bench 优化的 harness，在其他任务上是否仍然有效？LangChain 明确提到需要"避免 overfit 到某个任务"[[1]](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/)
2. **多模型 harness？** LangChain 尝试了不同模型（Codex vs Claude），发现需要"为模型定制 harness"——这对多模型系统意味着什么？
3. **Harness 的"保质期"？** 随着模型更新，今天有效的 harness 是否需要重新调优？

LangChain 正在探索 RLMs（Reasoning Language Models）来更高效地从 trace 中挖掘信息[[5]](https://alexzhang13.github.io/blog/2025/rlm/)——这可能是自动化的下一个方向。

## 五、总结

LangChain 的 harness 工程实践，核心贡献在于：

1. **将 Trace 作为反馈回路的核心**，而非仅依赖评估分数
2. **明确定义了三个人工可调的组件**：System Prompt、Tools、Middleware
3. **实现了几个有效的 harness 模式**：自验证循环、环境上下文注入、循环检测、推理三明治

与 Meta-Harness 的自动化思路相比，LangChain 的方法更依赖人工判断，但保留了更大的解释性和可控性——这在今天仍然是务实的选择。

**两篇文章的对话指向一个更本质的问题**：当我们不能修改模型参数时，harness 就是我们唯一的"优化自由度"。无论这个自由度是通过自动化搜索（Meta-Harness）还是人工迭代（LangChain）来操作，唯一的要求是：**有一个有效的反馈回路**。

Meta-Harness 的反馈是分数，LangChain 的反馈是 trace——只要反馈存在，优化就能收敛。

---

## 参考资料

[1] [Improving Deep Agents with Harness Engineering](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/)

[2] [LangSmith Observability Quickstart](https://docs.langchain.com/langsmith/observability-quickstart)

[3] [Claude Adaptive Thinking](https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking)

[4] [Boosting (Machine Learning)](https://en.wikipedia.org/wiki/Boosting_(machine_learning))

[5] [RLMs: Reasoning Language Models](https://alexzhang13.github.io/blog/2025/rlm/)