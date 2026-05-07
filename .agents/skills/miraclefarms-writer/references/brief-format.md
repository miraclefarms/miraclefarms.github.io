# Brief 格式规范

## Frontmatter

```yaml
---
title: AI Infra 早报｜{主题描述，15-30 字，趋势判断式，非项目更新列表}
date: 2026-MM-DD 05:30:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: {一句话摘要，80-120 字，概括今日最重要的 2-3 条进展，无引用编号 [n]}
---
```

**必须牢记的规则：**

- `title` 中的竖线必须是**全角竖线** `｜`（U+FF5C），绝对不能用半角 `|`
- `date` 必须含 `+0800`；发布时间 ≤ 08:00，避免 future post 问题
- `kind: brief` 和 `category: Brief` 必须同时存在且配对
- `series: ai-infra-daily-brief` 对日报系列是必填的
- `intro` 不含 `[n]` 引用编号，是纯文字摘要

**标题风格**：主题描述部分要传达趋势判断，而不是"vLLM 更新了 XX 功能"。好的标题例子：
- `AI Infra 早报｜推理框架进入默认路径质量竞争期`
- `AI Infra 早报｜speculative decoding 全面走向生产部署，框架支持深度分化`

---

## 正文结构

```
{开篇段落：1-2 段，无 H2 标题}
{设定今日整体格局和核心判断；帮读者定位"今天最重要的事情是什么"}

## 一、{主题一：用趋势判断命名，不是项目名}

{叙述段落，可多段}
{引用格式：**项目名 功能简述[[1]](https://github.com/...)** 分析内容...}

## 二、{主题二}

{同上}

## 三、{主题三（可选）}

{同上}

## N、今天真正值得记住的判断   ← 可选，用于收尾的全局洞察

{1-2 段全局判断，不是前面内容的重复}

---

## 参考来源

[1] [描述文字](https://github.com/...)

[2] [描述文字](https://github.com/...)
```

---

## 引用格式（最常出错的地方）

### 正文行内

```markdown
vLLM 合并了 FP8 DeepGemm zero initializer 融合[[1]](https://github.com/vllm-project/vllm/pull/39547)，...
```

- 格式：`[[N]](url)` — 数字在双方括号内，URL 直接跟着
- 引用编号在**文章中首次出现时**标注，同一来源多次引用沿用同一编号

### 文末参考来源（每篇 brief 必须有）

```markdown
---

## 参考来源

[1] [SGLang 实现 H20 风格 KV 缓存剪枝初始版本](https://github.com/sgl-project/sglang/pull/20450)

[2] [vLLM FP8 DeepGemm zero initializer fusion](https://github.com/vllm-project/vllm/pull/39547)

[3] [另一条引用](https://...)
```

规则：
- `---` 分隔线放在 `## 参考来源` **之前**
- 每条引用单独一行，各条之间空一行
- 格式：`[N] [描述文字](url)`，描述文字应概括链接内容

---

## 完整文章模板

```markdown
---
title: AI Infra 早报｜{主题描述}
date: 2026-MM-DD 05:30:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: {80-120字摘要}
---

{开篇段落，1-2段，无H2标题。今日整体格局和核心判断。}

## 一、{主题一}

{叙述内容。**项目名 更新描述[[1]](url)**，分析文字。续行续段。}

{可以是多个段落，把相关的几条更新组织在一起讲，而不是每条更新单独一段。}

## 二、{主题二}

{同上}

## 三、今天真正值得记住的判断

{全局洞察。1-2段，不是对前面内容的简单重复，而是提炼出跨项目的趋势。}

---

## 参考来源

[1] [描述文字](https://...)

[2] [描述文字](https://...)

[3] [描述文字](https://...)
```

---

## 常见错误速查

| 错误 | 正确做法 |
|------|---------|
| 标题用半角 `\|` | 改为全角 `｜`（U+FF5C） |
| 日期写成 `-0400` 或其他时区 | 统一 `+0800` |
| Brief 用了 HTML anchor 引用 | Brief 只用 `[[N]](url)` |
| 没有 `## 参考来源` 章节 | 每篇 brief 结尾必须有 |
| `intro` 里含 `[1]` 引用编号 | intro 是纯文字摘要 |
| H2 标题直接写项目名 | 改成趋势判断式标题 |
| `kind: Brief`（大写） | 应为 `kind: brief`（小写） |
