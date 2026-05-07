# Essay 格式规范

## Frontmatter

```yaml
---
title: {描述性中文标题，直接点明技术主题和核心判断，无"早报"前缀}
date: 2026-MM-DD 12:00:00 +0800
author: Ethan
kind: essay
category: Essay
intro: {一句话摘要，说明文章的技术覆盖范围和核心判断，40-80字}
---
```

**规则：**

- `title`：无前缀，直接用描述性标题。例：`vLLM PagedAttention 的设计原理与工程权衡`
- `author: Ethan`（固定）
- `kind: essay`（小写）和 `category: Essay`（首字母大写）必须配对
- `intro`：一句话概括文章核心，面向阅读列表展示，不含引用编号

---

## 可选：版本声明

如果文章基于特定 commit 或版本分析源码，在 frontmatter 之后加：

```markdown
> **版本声明**：本文分析基于 {repo} commit `{hash}`（{date}）；除非特别说明，以下描述均基于此版本。
```

---

## 正文结构

```
{开篇段落：1-2 段，无 H2 标题}
{直接提出核心问题或核心判断；建立分析框架；让读者知道这篇文章要回答什么}

## 一、{第一节：子论点或背景铺垫}

{叙述内容，行内引用用 HTML anchor}

### 1.1 {子节（可选）}

## 二、{第二节}

...

## N、结论

{回应开篇的核心判断；说明适用边界和仍开放的问题}

---

## 参考资料

[1] [Title](https://url)

[2] [Title](https://url)
```

---

## 引用格式（与 Brief 不同，不要混用）

### 正文行内：HTML anchor

```markdown
SCBench 把长上下文评测推进到 KV cache lifecycle 视角<a href="https://arxiv.org/pdf/2412.10319">[1]</a>，明确拆成四个阶段。

LoCoBench-Agent 将静态任务改造为交互式 agent 环境<a href="https://arxiv.org/pdf/2511.13998">[2]</a>，覆盖 10K 到 1M token 上下文。
```

格式：`<a href="url">[N]</a>`（HTML anchor，数字在方括号内）

### 文末参考资料：隐藏 URL 的 Markdown 链接

```markdown
---

## 参考资料

[1] [SCBench: A KV Cache-Centric Analysis of Long-Context Methods](https://arxiv.org/pdf/2412.10319)

[2] [LoCoBench-Agent: An Interactive Benchmark for LLM Agents in Long-Context Software Engineering](https://arxiv.org/pdf/2511.13998)

[3] [vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
```

格式：`[N] [Title](URL)`（Markdown 链接；页面只显示标题，不直接展示裸 URL）

注意：**章节名是 `## 参考资料`**（资料），不是 `## 参考来源`（来源是 brief 用的）

---

## 写作风格细节

### 开篇：提出真正的问题，而不是背景铺垫

❌ 避免：
> vLLM 是一个广泛使用的推理框架。本文将介绍其 PagedAttention 机制。

✅ 应该：
> 为什么一个全新的内存管理机制会在 LLM 推理领域迅速成为事实标准？PagedAttention 的核心不是性能技巧，而是对内存碎片化这个根本问题提出了一种 OS 借鉴思路。

### 结论：回应开篇，设定边界

结论不是"总而言之，本文介绍了……"，而是：
- 用 1-2 段回到开篇提出的核心问题
- 说明在什么条件下本文的判断成立
- 留下值得继续追的开放问题

### 处理技术深度

- 专业概念第一次出现时给一句简明解释（"X 是指……"），之后直接用
- 代码片段用 ` ``` ` 块，但除非必要不大段引用代码
- 架构图或数据图引用自原始来源（论文 figure / 官方博客图），不要自己描述"有一个方形框代表……"

---

## 完整文章模板

```markdown
---
title: {描述性中文标题}
date: 2026-MM-DD 12:00:00 +0800
author: Ethan
kind: essay
category: Essay
intro: {一句话摘要}
---

> **版本声明**：本文分析基于 {repo} commit `{hash}`（{date}）。（可选）

{开篇 1-2 段，无 H2 标题。直接抛出核心问题或判断。}

## 一、{第一节}

{叙述内容。引用格式：<a href="url">[1]</a>}

## 二、{第二节}

{叙述内容。可以有多个段落。}

### 2.1 {子节（可选）}

{更细的分析}

## 三、{第三节（可选）}

## 四、结论

{回应开篇判断。说明适用条件和开放问题。}

---

## 参考资料

[1] [Title of Paper or Article](https://arxiv.org/...)

[2] [vLLM PR / Blog / Doc Title](https://github.com/...)

[3] [Another Reference](https://...)
```

---

## 常见错误速查

| 错误 | 正确做法 |
|------|---------|
| Essay 用了 `[[N]](url)` 引用格式 | Essay 只用 `<a href="url">[N]</a>` |
| 章节名写成 `## 参考来源` | Essay 用 `## 参考资料`（来源是 brief 专用的） |
| 参考格式用 `[N] Title. https://url` | Essay 文末用 `[N] [Title](https://url)`，不要直接展示裸 URL |
| 开篇段落先介绍背景 | 开篇直接提出核心问题 / 判断 |
| `author: 荔枝不耐思` | Essay 的作者是 `Ethan` |
| `kind: Essay`（大写） | 应为 `kind: essay`（小写） |
| 结论段写成"综上所述" | 结论直接回应开篇、给出边界 |
