---
title: Claude Code 的 Token 经济学：提示词分层、上下文压缩与 KV Cache 工程
date: 2026-05-08 12:00:00 +0800
author: Ethan
kind: essay
category: Essay
tags: [KV Cache, Agents, Inference]
intro: 从 Claude Code 源码出发，解析其提示词分段缓存架构、自动压缩机制与 KV Cache 工程的设计逻辑——三套机制的真正主线是同一个约束：在不破坏 cache prefix 稳定性的前提下，用最少的 token 传递最关键的状态。
---

Claude Code 是目前公开信息最丰富的生产级 AI agent 之一，但它真正值得拆解的地方，不是那些显眼的 agent 能力，而是藏在底层的 token 工程。一个运行在 200K context window 内的编程助手，需要同时解决三件事：系统提示词如何在数万 token 规模下保持 cache 命中；对话上下文膨胀到极限时如何压缩而不丢失关键状态；服务端 KV cache 如何才能稳定命中而不因客户端的一个小改动全盘失效。

这三件事表面看是独立系统，实际上围绕同一个约束点联动——**cache prefix 的稳定性**。提示词分段的每个设计决策都在为 cache 命中率服务；压缩机制的触发时机和状态恢复在刻意回避"压缩顺带破坏 cache"；七种 cache 优化模式是对前两者漏网之鱼的专项修补。理解这个耦合关系，是读懂 Claude Code token 工程的真正入口。

## 一、提示词的分段缓存架构

Claude Code 的系统提示词不是一个硬编码字符串，而是一组独立段落的组合体。这个设计背后有三重工程压力：数万 token 的提示词需要通过缓存分层来控制成本；内容有静态（身份规范、工具定义）也有动态（当前环境、MCP 工具列表），必须采用不同的更新策略；还要兼容自定义模式、Agent 模式、协调器模式等多种运行上下文。

架构上的核心机制是 `SYSTEM_PROMPT_DYNAMIC_BOUNDARY`——一条将提示词内容劈成两半的分界线。边界之前的内容必须完全静态，可以跨所有用户会话全局缓存；边界之后的动态区每个会话单独计算。这条约束不只是软性规范，而是被写进代码审查摩擦里的硬约束：任何段落被标记为 `DANGEROUS_uncachedSystemPromptSection` 时，开发者必须在代码中提供书面理由，否则 review 通不过。这种摩擦设计本身就是工程文化的一部分——让"打破 cache"这件事在代码层面有足够高的成本感知。

每个段落按照可缓存性分为两类。`systemPromptSection` 走记忆化路径，相同输入只计算一次；`DANGEROUS_uncachedSystemPromptSection` 则每轮重新生成。运行时，`splitSysPromptPrefix` 函数根据当前条件选择三路策略之一：有 MCP 工具时降级到组织级缓存（因为 MCP 工具列表是会话特定的）；无工具时全局缓存加边界分离；第三方 provider 时使用组织级兜底。

工具的 `description` 字段同样是 cache prefix 的组成部分，因此也被当作"micro-harness"来设计，而不只是文档字符串。BashTool 的描述里明确写着哪些操作应该转给专用工具，FileEditTool 要求必须先读后编辑——这些行为约束同时出现在描述和运行时代码里，形成双重保护。而 SkillTool 把技能列表限定在 context window 的 1%（约 8000 字符）以内，按"完整描述 → 截断描述 → 仅名称"三级降格，直接控制了这部分提示词对 cache prefix 的影响范围。

## 二、自动压缩：断路器与三层保护

当对话积累到 context window 极限时，Claude Code 会触发自动压缩（autoCompact）。触发阈值由 `autoCompactThreshold` 计算得出：有效 context window 减去 13,000 token 缓冲。对 Claude Sonnet 4 的 200K 窗口，这个值约为 167,000 token——即 83.5% 利用率。

阈值之所以不设到 90% 甚至更高，是因为压缩本身需要消耗 token。系统在压缩时预留三层缓冲：20,000 token 留给压缩输出，13,000 token 作为安全缓冲，3,000 token 是硬性底线。这三层叠加确保压缩请求本身不会反过来触发另一次压缩——一个听起来简单但设计不当就会引发级联崩溃的场景。

压缩提示词采用九段模板，有一个细节能说明这套设计的工程严肃性："Do NOT call any tools"这条指令在模板的开头和结尾各出现一次。原因是压缩发生在单轮 API 调用里，如果模型在此时执行工具调用，整个压缩预算就白费了。两次重复不是笔误，而是故意的强调——在高重要性低容错的单次操作中，冗余比简洁更安全。模板还要求模型按时间顺序分析再总结，分析过程包裹在隐藏 tag 内，不会消耗压缩后的 context。

当压缩请求本身超出输入限制时（比如对话太长导致即使只发压缩提示词也超限），系统走 PTL（Prompt Too Long）重试循环：最多重试 3 次，每次截断最早的消息组，如果无法精确解析错误响应中的 token 差值，就以每次约 20% 的估算量递减。

断路器机制是另一个值得注意的设计。连续 3 次压缩失败后，系统停止尝试自动压缩。这个限制来自一次真实的生产事故：1,279 个会话出现了 50 次以上的连续失败，每天产生约 25 万次徒劳的 API 调用。断路器正是这个数字推动的工程决策——不是理论上的最佳实践，而是真实成本倒逼的结果。

## 三、压缩后的状态恢复

压缩最危险的副作用不是 token 消耗，而是状态丢失。Claude Code 用"快照-清空"模式来应对：压缩前保存 `readFileState` 缓存的完整快照，然后清空内存缓存；压缩完成后按预算选择性地恢复状态。关键设计在于"选择性"——不是恢复所有之前的状态，而是按照固定的五常数预算框架裁剪：

- 最多恢复 **5 个文件**（按最近访问时间排序）
- 每个文件上限 **5,000 token**
- 文件恢复总预算 **50,000 token**
- 每个 skill 上限 **5,000 token**
- skill 恢复总预算 **25,000 token**

文件恢复还需要通过多层过滤：plan 文件走独立恢复通道，CLAUDE.md 内存文件通过 system prompt 注入（不重复传），已经出现在保留消息尾部的文件跳过（避免重复），剩余文件按时间戳排序取前 5。

有一个刻意的"不恢复"决策更能说明设计思路：`sentSkillNames`（技能列表缓存）在压缩后**不会重置**。重新注入完整技能列表会消耗约 4,000 token，而实际价值极低——已经被调用的 skill 会在恢复时单独重建，skill 工具本身仍然保留在模型 schema 里。作者选择接受"技能发现能力略有下降"换取 4K token 的节省，这是 token 经济学在恢复策略上的直接体现。

## 四、微压缩：三条轻量裁剪路径

自动压缩是大手术，但大多数时候需要的是小手术。Claude Code 实现了三种"微压缩"机制，核心哲学是"最便宜的 token 是你从来不发的那个"。

**时间触发型微压缩**在 60 分钟以上无活动后触发。理由是服务端 KV cache 的默认 TTL 是 5 分钟，长时间停顿后 cache 已经过期，重新发送旧工具结果毫无意义。触发时，系统只保留最近 5 个工具结果，其余替换为占位文本。这个机制的逻辑是：当 cache 已经冷却，重建 cache 的成本已经既成事实，此时截断旧内容的代价最低。

**cache_edits 精准删除**利用 API 的 `cache_edits` 特性，在不修改本地消息记录的情况下，删除服务端 cache 中的特定内容。这个机制的价值在于它保护了 cache prefix 的连续性——本地消息数组不变，cache key 的计算基础也不变，只是服务端存储的内容被剪裁了。对活跃会话里的精准清理，这是比任何重发策略都更优雅的方案。

**API 上下文管理**是声明式策略：客户端告诉服务端"当输入超过 X token 时，清除 Y 类型内容，保留最近 Z 条"，由服务端自动执行。这把裁剪决策的执行从客户端移到了服务端，减少了往返次数。

三种机制之间有明确的优先级：时间触发型最先检查，命中则短路其他两种，因为冷 cache 和热 cache 需要完全不同的处理策略。

## 五、KV Cache 工程：前缀稳定性即架构

Claude Code 的 KV cache 建立在 Anthropic API 的 prefix 精确匹配模型之上：如果请求的 prefix 与之前的请求字节级完全一致，服务端就可以直接复用已缓存的 KV 状态。"字节级"是关键词——任何字符的变动都会产生不同的 hash，让整条 prefix 的 cache 失效。

缓存作用域分三级：**全局**（跨所有 Claude Code 用户共享，只能包含完全静态的内容）、**组织**（同一组织内共享，适合包含 MCP 工具等会话相关内容）、**无**（高动态内容不缓存）。TTL 方面，默认 5 分钟适合频繁交互的活跃会话；1 小时的扩展 TTL 适合长文档处理等场景，但写入成本是 2 倍（5 分钟版本是 1.25 倍），需要权衡使用频率。

![Anthropic Prompt Caching 的混合 TTL 计费模型](/assets/claude-code-context-kvcache/fig-1-cache-mixed-ttl.svg)

*图 1：Anthropic 的 prompt caching 支持在同一请求中混用 5 分钟和 1 小时 TTL。系统按三段边界（A/B/C）计算计费：A 之前是 cache read token，A 到 B 是 1 小时 cache write，B 到 C 是 5 分钟 cache write。来源：Anthropic 官方文档。*

Cache 失效的层级很重要：`tools` → `system` → `messages`，上游的任何变化都会让下游全部失效。这意味着修改一个工具定义会让系统提示词和所有消息的 cache 一并失效。Claude Code 用 **Latching 机制**来应对这个脆弱性：会话开始时将 Fast Mode、Auto Mode、Beta Header 等配置状态冻结，整个会话期间不再变化，只有 `/clear` 或 `/compact` 命令才能解冻重置。Beta Header 的"粘性开启"策略尤为典型——一旦某个 header 在会话中被启用，即使后续功能被关闭，该 header 在本次会话内也保持激活，以避免 50–70K token 的已缓存系统提示词因此失效。这是用"略微过时的配置"换取稳定 cache 命中的刻意权衡。

## 六、Cache Break 检测与七种优化模式

Cache 失效难以完全避免，但可以被检测和归因。Claude Code 的 cache break 检测分两个阶段：请求前，`recordPromptState()` 对系统提示词、工具 schema、模型设置等拍摄快照，与上一次快照对比记录差异；响应后，`checkResponseForCacheBreak()` 分析 API 返回的 usage 字段，用双阈值过滤噪声——cache token 必须同时下降超过 5% **且**超过 2,000 才触发告警，单一条件不足以确认真实失效。

`PreviousState` 对象维护 15 个以上的字段，包括系统提示词和工具 schema 的内容 hash、各功能模式的开关状态、每个工具的单独 hash 记录，以及 cache control 标记。当检测到 break 时，系统会尝试归因：客户端变化、TTL 到期（最后一条消息距今超过 5 分钟），或服务端因素（无客户端变化但有 break）。

这套检测系统带来了一个关键发现：**约 90% 的 cache break 来自服务端路由或驱逐，而非客户端改动**。这个数字根本性地改变了优化方向——与其穷尽所有客户端可变量，不如把精力集中在让可控变化保持稳定上。

七种缓存优化模式正是这个策略的具体落地：

**日期记忆化**：用 `memoize(getLocalISODate)` 防止每次跨天（午夜）导致系统提示词失效。代价是在极少数情况下模型看到一个过时一天的日期，但这比每天零点全体用户的 cache 全部重建要合算得多。

**月度粒度**：工具提示词中的时间标记使用"Month YYYY"格式而非"YYYY-MM-DD"，把变化频率从每日降至每月，直接减轻工具 schema 的 cache 压力。

**Agent 列表附件化**：动态的 Agent 列表从工具描述移到消息附件，而不是嵌入工具 schema。这一改动消除了 10.2% 的 `cache_creation_tokens`，是七种模式中单项效果最显著的。原因直接：工具 schema 是 cache 层级的顶端，任何变化都向下传播；消息附件在层级末端，失效影响最小。

**技能列表预算**：技能列表限定在 context window 的 1%（约 8,000 字符），通过三级降格保证列表规模可控，避免技能数量增长悄悄侵蚀 cache 稳定性。

**占位符替换**：把用户特定路径（如 `/tmp/user-abc123/`）统一替换为环境变量占位符（`$TMPDIR`），消除用户维度的差异，把组织级缓存提升为全局缓存。

**条件段落省略**：对于因功能开关状态决定是否插入的提示词段落，采用稳定的插入策略，避免 prefix 内容因功能抖动而变化，保证 prefix 的单调稳定性。

**工具 Schema 会话缓存**：用模块级 Map 在会话内锁定 schema，隔离 GrowthBook 远程配置系统翻转的影响，确保同一会话内工具 schema 的 hash 不变。

## 七、结论

三套机制的耦合不是设计之初就规划好的，更像是在真实成本反馈下逐步收敛到的一套一致架构。25 万次日废弃 API 调用推动了断路器；cache_creation_tokens 的可观测数据推动了 Agent 列表附件化；90% 服务端 break 的发现重新分配了优化资源。

对构建 AI agent 的工程师来说，这套经验的可迁移部分有三点：把提示词当架构组件而非字符串，建立段落级的 cache 层级意识；把"打破 cache"的操作设计成高摩擦的（需要显式理由或触发特定命令）；通过可观测性数据驱动优化，而不是靠直觉猜测哪里是热点。

还有一个开放问题值得关注：随着 context window 持续扩大（200K → 1M），压缩触发的时机和策略需要重新校准，而更长的 prefix 也意味着 cache break 的代价更高。Claude Code 目前的阈值和预算常数都是针对 200K 窗口调出来的——如果模型的 context window 翻倍，这些数字未必还是最优解。

---

## 参考资料

[1] [Mastering Engineering: From Claude Code Source to AI Coding Best Practices — Part 2: Prompt Engineering](https://zhanghandong.github.io/harness-engineering-from-cc-to-ai-coding/part2/ch05.html)

[2] [Mastering Engineering: From Claude Code Source to AI Coding Best Practices — Part 3: Context Management](https://zhanghandong.github.io/harness-engineering-from-cc-to-ai-coding/part3/ch09.html)

[3] [Mastering Engineering: From Claude Code Source to AI Coding Best Practices — Part 4: Prompt Cache Engineering](https://zhanghandong.github.io/harness-engineering-from-cc-to-ai-coding/part4/ch13.html)

[4] [Anthropic Prompt Caching Documentation](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
