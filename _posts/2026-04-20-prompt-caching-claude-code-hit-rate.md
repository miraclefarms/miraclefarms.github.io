---
title: 理解 Prompt Caching：从 Claude Code 92% 缓存命中率案例说起
date: 2026-04-20 12:00:00 +0800
author: Ethan
kind: essay
category: Essay
intro: 基于 Avi Chawla 原文整理一篇中文学习版摘要，解释 prompt caching 的工作机制、工程约束与 Claude Code 案例。
---

> 版权说明：本文基于 Avi Chawla 发表于 Daily Dose of Data Science 的文章《Prompt Caching in LLMs!》整理为中文学习版摘要。原文版权及配图版权归作者所有，本文仅用于技术学习与研究，不构成全文翻译或商业转载；原文与原图请以作者页面为准<a href="https://blog.dailydoseofds.com/p/prompt-caching-in-llms">[1]</a>。

> 说明：下文按原文的论证顺序重述核心内容，但不逐句对应原文表述。

> 配图说明：本文插图均为基于原文观点重绘的原创示意图，用于辅助理解，不是原文原图。

Prompt caching 常被理解成一个“打开就能省钱”的模型能力，但这篇文章真正想讲清的是：**它首先是一条系统设计纪律，其次才是一项 API 特性。** Claude Code 能把缓存命中率稳定在 92%，并不是因为它用了某个神秘优化，而是因为整个会话结构从一开始就围绕“让可复用前缀保持不变”来组织<a href="https://blog.dailydoseofds.com/p/prompt-caching-in-llms">[1]</a>。

换句话说，原文的主线不是介绍一个新名词，而是在回答一个更实用的问题：为什么长会话 agent 会越来越贵，以及怎样通过 prompt 结构和工程纪律，把这部分重复成本压下去<a href="https://blog.dailydoseofds.com/p/prompt-caching-in-llms">[1]</a>。

## 一、问题先不在模型，而在重复计算

文章开头先指出了一个很多 agent 系统都会踩到的成本陷阱：每当 agent 往前走一步，它通常都会把整段历史上下文重新发给模型，其中不仅包括新的用户消息，也包括系统指令、工具定义、项目上下文，以及几轮之前就已经读过的内容<a href="https://blog.dailydoseofds.com/p/prompt-caching-in-llms">[1]</a>。

如果这些内容每一轮都重新预填充一次，那么账单增长就不是来自“新信息”，而是来自“反复处理旧信息”。原文用一个很直观的例子来说明：一个 20,000 token 的 system prompt，跑 50 轮对话，就会产生大约 100 万 token 的重复计算，而且这些 token 每次都按全价计费<a href="https://blog.dailydoseofds.com/p/prompt-caching-in-llms">[1]</a>。这也是 prompt caching 要解决的根本问题。

## 二、为什么缓存能成立：上下文天然分成静态前缀和动态后缀

原文接着把一个 agent 请求拆成两部分。前半部分是跨轮次不变的静态前缀，例如 system instructions、tool definitions、project context 和行为规则；后半部分则是会不断增长的动态后缀，例如用户消息、assistant 回复、工具输出和终端观察<a href="https://blog.dailydoseofds.com/p/prompt-caching-in-llms">[1]</a>。

一旦这样划分，缓存逻辑就很自然了：推理基础设施可以把静态前缀对应的内部状态存下来，后续只要请求前缀完全一致，就不需要再把这一段从头算一遍，而是直接复用已缓存的结果<a href="https://blog.dailydoseofds.com/p/prompt-caching-in-llms">[1]</a>。文章强调，工程上所有重要决策其实都围绕这一点展开，因为只要你弄清楚“什么必须稳定、什么可以增长”，后面的最佳实践几乎都会顺理成章。

![Prompt caching 中的静态前缀与动态后缀](/assets/prompt-caching-claude-code-hit-rate/fig-1-static-prefix-dynamic-suffix.svg)

*图 1：一个 agent 请求里，真正适合长期缓存的是最上层的静态前缀；用户消息、工具输出和终端观察则应作为持续增长的动态后缀处理。基于原文观点重绘。*

## 三、底层机制并不神秘，本质是复用 KV cache

为了说明“为什么省得下来”，原文专门回到 Transformer 推理过程本身。一次推理大致分成 prefill 和 decode 两个阶段：prefill 负责处理整段输入上下文，要对所有 token 做密集计算，因此昂贵且偏算力受限；decode 则是一 token 一 token 往后生成，更多是在读取历史状态，因此偏内存受限<a href="https://blog.dailydoseofds.com/p/prompt-caching-in-llms">[1]</a>。

在 prefill 阶段，模型会为每个 token 计算 Query、Key 和 Value。文章解释说，其中 Key 和 Value 一旦为某段前缀算出来，只要前缀本身不变，它们也不会变化；问题只在于，如果系统不缓存，这些张量会在请求结束后被丢掉，下一轮又要重算一次<a href="https://blog.dailydoseofds.com/p/prompt-caching-in-llms">[1]</a>。

因此 prompt caching 的本质并不是“把整段文本存起来”，而是把那段静态前缀已经算好的 KV 状态保留在推理服务器上，并通过前缀 token 序列的哈希值来索引。只要新请求的前缀和旧请求完全相同，系统就能直接命中缓存，跳过那一大段 prefill 计算<a href="https://blog.dailydoseofds.com/p/prompt-caching-in-llms">[1]</a>。原文用这个机制说明，省下来的不是一点点 I/O，而是最贵的那部分重复注意力计算。

![Prompt caching 的 KV cache 复用路径](/assets/prompt-caching-claude-code-hit-rate/fig-2-kv-cache-reuse.svg)

*图 2：首轮请求先为静态前缀完成 prefill 并写入 KV cache，后续只要前缀哈希命中，就能直接读取缓存状态，跳过整段最贵的注意力计算。基于原文观点重绘。*

## 四、经济账为什么成立：贵的是首次写入，便宜的是后续读取

文章随后把这件事拉回到成本模型。它给出的价格逻辑很清楚：cache write 比普通输入贵，因为系统要额外把 KV 状态存起来；cache read 则便宜得多，因为后续只是在读取已存在的缓存<a href="https://blog.dailydoseofds.com/p/prompt-caching-in-llms">[1]</a>。原文给出的 Anthropic 价格表中，5 分钟 cache write 是基础输入价格的 1.25 倍，而 cache read 只要基础输入价格的 0.1 倍，也就是九折以上的降幅<a href="https://blog.dailydoseofds.com/p/prompt-caching-in-llms">[1]</a>。

但作者也明确提醒，这笔账只有在缓存命中率够高时才成立。如果前缀频繁变化，系统就会不断重建缓存，最后不仅省不到钱，还可能多付出一部分写入成本。所以原文才会把 Claude Code 作为例子，因为它展示的不是“缓存理论上可以省钱”，而是“在生产环境里怎样把命中率维持在高位”<a href="https://blog.dailydoseofds.com/p/prompt-caching-in-llms">[1]</a>。

## 五、Claude Code 的例子说明，真正重要的是会话结构

原文给了一段 30 分钟编码会话的账单式拆解。会话开始时，Claude Code 先加载 system prompt、工具定义和项目里的 `CLAUDE.md` 文件，这一大段内容超过 20,000 token，所以第一轮最贵；但只要这段基础前缀不变，后面每一轮都可以按缓存读价复用它<a href="https://blog.dailydoseofds.com/p/prompt-caching-in-llms">[1]</a>。

随着用户继续发指令、子代理探索代码库、工具不断返回输出，新增内容会被附加到动态后缀中，而不是回写进静态前缀。原文还特别提到，计划阶段不会直接把原始探索结果整段塞进去，而是通过摘要控制动态后缀的体积，因为这部分虽然不能像静态前缀那样长期复用，但同样会影响上下文长度和费用<a href="https://blog.dailydoseofds.com/p/prompt-caching-in-llms">[1]</a>。

文章给出的结果是：如果没有缓存，这次会话按 Sonnet 4.5 费率大约要花 6 美元；而在 92% 缓存效率下，绝大多数 token 都变成 cache read，最终成本降到约 1.15 美元，单任务成本下降约 81%<a href="https://blog.dailydoseofds.com/p/prompt-caching-in-llms">[1]</a>。这里最关键的启发并不是这个具体数字，而是背后的设计原则：把昂贵但稳定的部分放在上面，把变化但必须新增的部分放在下面。

![Claude Code 会话中的成本差距](/assets/prompt-caching-claude-code-hit-rate/fig-3-claude-code-cost-gap.svg)

*图 3：在长会话里，首轮写缓存会增加一次性成本，但只要后续绝大多数轮次都转成 cache read，整段任务的账单结构就会显著下移。基于文中示例数字重绘。*

## 六、缓存最脆弱的地方，在于它依赖“完全一致”的前缀

文章最有工程价值的一段，是它反复强调哈希式缓存的脆弱性。缓存并不是“语义相似就能命中”，而是要求从开头到断点之前的 token 序列完全一致；顺序变了、字段排序变了、时间戳多了，哪怕只是很小的改动，也会让整段前缀重新计算<a href="https://blog.dailydoseofds.com/p/prompt-caching-in-llms">[1]</a>。

原文列了几个在生产中真实发生过的失效场景：有人把时间戳注入 system prompt，导致每个请求都生成新的哈希；有人在不同请求里用不同顺序序列化工具 schema，结果同一个工具定义也无法命中；还有一种情况是会话中途更新了 AgentTool 参数，直接把整段 20,000 token 的前缀缓存作废<a href="https://blog.dailydoseofds.com/p/prompt-caching-in-llms">[1]</a>。

围绕这些失败案例，文章归纳出三条很硬的纪律：不要在会话中途增删工具；不要在会话中途切换模型；不要通过修改 system prompt 去更新状态，而要把新的提醒或状态附加到后续 user message 中<a href="https://blog.dailydoseofds.com/p/prompt-caching-in-llms">[1]</a>。这三条看起来像经验法则，其实都来自同一个约束：前缀一旦变了，缓存就白建了。

## 七、把这套方法用于自建 Agents，关键是先把顺序设计对

文章在最后把 Claude Code 的经验抽象成一套更普遍的 agent 组织方式。推荐的 prompt 顺序是：最上面放系统指令和行为规则，其次放所有工具定义，再往下放检索到的上下文和参考文档，最后才是对话历史与工具输出<a href="https://blog.dailydoseofds.com/p/prompt-caching-in-llms">[1]</a>。这个顺序的逻辑并不复杂，但它决定了哪些内容有资格成为“长期可复用前缀”。

![适合缓存的 agent prompt 排布](/assets/prompt-caching-claude-code-hit-rate/fig-4-agent-prompt-layout.svg)

*图 4：原文给出的经验可以简化成一条 prompt 排布纪律：越靠上的层越应该稳定，越靠下的层越适合承接变化与增量状态。基于原文观点重绘。*

原文还提到两个实操点。第一，如果在 Anthropic API 上启用了自动缓存，随着对话继续增长，缓存断点可以自动往后推进；第二，当上下文快满时，更稳妥的做法不是去改已有前缀，而是发起一次“缓存安全”的压缩分叉，让系统在保留既有前缀的前提下生成摘要，再用摘要重建后续上下文<a href="https://blog.dailydoseofds.com/p/prompt-caching-in-llms">[1]</a>。

最后，作者建议持续监控三个字段：`cache_creation_input_tokens`、`cache_read_input_tokens` 和 `input_tokens`。因为 prompt caching 是否真正生效，不该靠体感判断，而要像监控 uptime 一样监控缓存效率<a href="https://blog.dailydoseofds.com/p/prompt-caching-in-llms">[1]</a>。

## 八、结论

这篇文章最值得记住的判断是：prompt caching 不是一个后补优化项，而是 agent 基础设施的建模方式。你得先把 prompt 设计成“稳定的前缀在上、增长的后缀在下”，缓存才会自然成为成本优势；反过来，如果系统状态、工具定义和上下文装配顺序本身不稳定，再好的缓存能力也很难救场<a href="https://blog.dailydoseofds.com/p/prompt-caching-in-llms">[1]</a>。

因此，Claude Code 的 92% 命中率并不只是一个漂亮指标，它更像一个工程信号：当你的 agent 结构足够克制、上下文边界足够清楚、状态更新路径足够稳定时，缓存才能真正从“文档里的功能”变成“账单上的差异”<a href="https://blog.dailydoseofds.com/p/prompt-caching-in-llms">[1]</a>。

---

## 参考资料

[1] Avi Chawla, Prompt Caching in LLMs!, Daily Dose of Data Science. https://blog.dailydoseofds.com/p/prompt-caching-in-llms

[2] Avi Chawla, X post for the article. https://x.com/_avichawla/status/2044670188998803855?s=20
