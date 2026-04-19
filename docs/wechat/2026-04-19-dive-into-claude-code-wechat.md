# Claude Code 的设计哲学：当 98.4% 的代码是纯文本

**📅 2026-04-19**

![题图](assets/2026-04-19/dive-into-claude-code-cover.png)

> VILA Lab 对 Claude Code 的系统级逆向分析揭示了一个出人意料的结论：这套 agentic 编程系统的 JS 运行时基础设施只占 1.6%，其余 98.4% 是 prompt、markdown 和配置文本——而这个比例本身，就是其核心哲学"minimal footprint + trust in Claude's judgment"的最直接体现。

---

## 一行数字揭示的哲学

如果你打开 Claude Code 的源码，会发现它的核心是一个 while 循环：调用模型、执行工具、拿回结果、循环往复。VILA Lab 最近发表的论文[1]对这套架构做了系统的源码级逆向分析，核心发现出人意料：Claude Code 的 JS 运行时基础设施只占整个代码库的 **1.6%**，剩余 **98.4%** 是纯文本——prompt、markdown 文档、配置文件。

这不是编程风格的选择，而是一个哲学立场：把判断权交给模型，基础设施只负责保证判断被安全地执行。

---

## 五个价值观，十三条原则

论文从实现中提炼出五个驱动架构决策的核心价值观：人类决策权威、安全与安保、可靠执行、能力放大、上下文适应性。这五个价值观被追溯到 13 条具体设计原则，再落实到可观察的实现选择。

以"minimal footprint"为例——会话存储是追加式 JSONL 文件而不是数据库，subagent 在独立 worktree 中执行，工具调用结果直接喂回循环。简洁不是偷懒，而是对"能做的边界"的刻意克制。

---

## 54 个工具，9 步流水线

Claude Code 内置最多 **54 个工具**，覆盖文件操作、搜索、Shell 执行、网络访问和代码智能五大类。工具池动态组装：基础枚举 → 模式过滤 → 拒绝列表预筛 → MCP 整合 → 去重，不是所有工具都默认可用。

每轮对话的执行走一条 9 步流水线：设置解析 → 状态初始化 → 上下文组装 → 5 个 pre-model shaper → 模型调用 → 工具分发 → 权限门控 → 工具执行 → 停止条件。关键设计在于：**模型决定要做什么，权限系统决定是否允许做**。两者是独立代码路径，越狱的模型无法绕过基础设施层面的安全检查。

**27 个 hook 事件**分布在流水线关键节点，支持 shell 命令、LLM 评估、webhook、subagent 验证器四种执行类型，让用户在不修改 Claude Code 本身的前提下注入自定义逻辑。

---

## 上下文是最稀缺的资源

Claude Code 设置了 5 层上下文压缩机制：当上下文接近窗口上限，系统清除旧工具输出，触发完整 LLM 摘要，生成包含 9 个结构化章节的会话摘要，再加上最近文件内容（上限 50K token）重建上下文。

这套机制的关键结果是 **92% 的前缀复用率**——静态内容跨轮次缓存复用，多轮对话的 token 消耗不会线性增长。CLAUDE.md 中的 `Compact Instructions` 章节是用户唯一可以直接干预压缩结果的接口，把"会话记忆持久化"问题从代码层推给了用户配置层。

---

## 与 OpenClaw 的比较

论文把 Claude Code 和 OpenClaw 做了六个维度的对比，揭示的核心判断是：没有普适的 agent 架构，只有在特定约束下最合理的架构。

Claude Code 面向单一用户终端，权限模型是逐动作分类：每次工具调用经过 ML 分类器判断风险等级，7 种权限模式覆盖从完全手动到自动执行的光谱。OpenClaw 面向多租户网关，转向外围访问控制：agent 进入系统时做资质验证，而不是每次动作时介入。同样的问题，不同的部署上下文，产生了完全不同的技术答案。

---

> 一句话结论：**Claude Code 的 1.6%/98.4% 是一个哲学声明——它相信语言能比代码更有效地表达"应该怎么做"，而基础设施的职责是让这个判断被安全执行，而不是替代它。**

---

## 参考

[1] Dive into Claude Code: The Design Space of Today's and Future AI Agent Systems：https://arxiv.org/abs/2604.14228

[2] VILA-Lab/Dive-into-Claude-Code：https://github.com/VILA-Lab/Dive-into-Claude-Code

[3] How Claude Code Works：https://code.claude.com/docs/en/how-claude-code-works
