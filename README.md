# MiracleFarms

MiracleFarms 是一个以 AI Infrastructure 为核心的公开研究站点，使用 Jekyll 构建，部署于 GitHub Pages。

站点地址：`https://miraclefarms.github.io`

---

## 本地开发

```bash
bundle install
bundle exec jekyll serve
```

推送到 `main` 分支后，GitHub Actions 自动构建并部署。

---

## Wiki

站点在 `/wiki/` 子路径下提供一个独立的技术知识库，使用 MkDocs Material 构建。

- **Wiki 入口**：`https://miraclefarms.github.io/wiki/`
- **源码目录**：`wiki-src/`（MkDocs 源码，不被 Jekyll 处理）
- **构建产物**：`wiki/`（已提交到仓库，被 Jekyll 原样服务）

### 当前主题

| 主题 | 路径 | 说明 |
|------|------|------|
| KVCache | `wiki-src/docs/kvcache/` | 10 个页面：基础概念、运行时架构、Prefix Cache、Paged KV、Offload、PD 分离、评估、框架对比、术语表、参考资料 |

### Wiki 更新流程

编辑 `wiki-src/docs/` 下的 Markdown 文件，push 到 `main`，GitHub Actions（`build-wiki.yml`）自动构建并将产物写回 `wiki/`，随后 `pages.yml` 重新部署整站。

### Wiki 本地预览

```bash
cd wiki-src
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdocs serve          # http://127.0.0.1:8000/
```

### 新增 Wiki 主题

1. 在 `wiki-src/docs/` 下新建主题目录（如 `pd-disaggregation/`）
2. 在 `wiki-src/mkdocs.yml` 的 `nav:` 中追加该主题的条目
3. Push 到 `main`，Actions 自动构建

---

## 发文规范与格式守则

本节是发文的权威参考，适用于 agent 和作者。以往出现的问题主要集中在**时区错误、引用格式混用、标题格式不一致**三类，本节逐一给出规则。

### 1. 文件命名

```
_posts/YYYY-MM-DD-slug.md
```

- slug 使用小写英文和连字符，不含中文
- 文件名日期必须与 front matter `date` 字段的日期完全一致

常见命名模式：

| 类型 | 命名示例 |
|------|----------|
| AI Infra 日报 | `ai-infra-daily-brief-quantization-usable-stage` |
| 技术 essay | `vllm-kvcache-runtime-architecture` |
| 特别报道 | `gtc-2026-briefing` |
| 站点说明 | `why-i-created-miraclefarms` |

---

### 2. Front Matter

```yaml
---
title: 文章标题
date: 2026-03-17 08:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief   # 可选
intro: 一句话摘要，显示在列表页，不超过 100 字。
---
```

**必填字段：** `title` / `date` / `author` / `kind` / `category` / `intro`

**可选字段：** `series`（系列日报使用，例如 `ai-infra-daily-brief`）

`kind` 与 `category` 必须对应：

| `kind` | `category` | 适用场景 |
|--------|------------|----------|
| `brief` | `Brief` | 日报、快讯、事件简报 |
| `essay` | `Essay` | 技术深度分析、源码解读 |
| `field-note` | `Field Note` | 调研笔记、现场观察 |
| `founding-note` | `Founding Note` | 站点理念、创刊说明 |

---

### 3. 时间与时区（最常见错误）

**所有文章统一使用 Asia/Shanghai（UTC+8）**：

```yaml
# 正确 ✓
date: 2026-03-17 08:00:00 +0800

# 错误 ✗ — 不要使用 -0400（美国东部时间）
date: 2026-03-17 12:40:00 -0400
```

各类文章的惯用发布时间：

| 文章类型 | 建议时间 |
|----------|----------|
| AI Infra 日报（brief） | `08:00:00 +0800` 或 `05:30:00 +0800` |
| 技术 essay | `12:00:00 +0800` |
| 特别报道 | 按实际时间，须含 `+0800` |

---

### 4. 标题格式

**Brief（日报）**：

```
AI Infra 早报｜{主题描述，15-30 字}
```

- 使用**全角竖线** `｜`（U+FF5C），不是半角 `|`
- 示例：`AI Infra 早报｜量化精度落地进入"可用"阶段，推理效率优化持续加速`

**Essay（技术分析）**：直接使用描述性中文标题，不加前缀。

---

### 5. 正文结构

#### Brief

```markdown
{无标题的开篇段落，1-2 段，设定今日主题和全局判断}

## 一、{第一个主题}

{分析内容，行内引用使用 [[N]](url) 格式}

## 二、{第二个主题}

...

## N、今天真正值得记住的判断   ← 可选收尾段

---

## 参考来源

[1] [链接描述](https://github.com/...)

[2] [链接描述](https://github.com/...)
```

#### Essay

```markdown
> **版本声明**：本文分析基于 {仓库} commit `{hash}`（{日期}）；...   ← 可选

{无标题的开篇段落，提出核心问题}

## 一、{第一节}

{分析内容，行内引用使用 HTML 格式 <a href="url">[N]</a>}

## 二、{第二节}

...
```

---

### 6. 引用格式（Brief 与 Essay 不同，不要混用）

#### Brief 引用

正文行内：

```markdown
SGLang 实现了 H20 风格 KV 缓存剪枝[[1]](https://github.com/sgl-project/sglang/pull/20450)，...
```

文末参考来源（每篇 brief 必须有）：

```markdown
---

## 参考来源

[1] [SGLang 实现 H20 风格 KV 缓存剪枝初始版本](https://github.com/sgl-project/sglang/pull/20450)

[2] [SGLang HiMamba Tree offloading 支持](https://github.com/sgl-project/sglang/pull/20457)
```

- 每条引用独占一行，条目间空一行
- `---` 分隔线放在 `## 参考来源` 之前

#### Essay 引用

正文行内使用 HTML anchor，不需要文末参考来源章节：

```markdown
SGLang 在早期文章里描述了共享前缀 workload<a href="https://lmsys.org/blog/...">[1]</a>。
```

---

### 7. H2/H3 标题编号

- **H2** 使用中文数字 + 顿号：`一、` `二、` `三、`
- **H3** 可使用阿拉伯数字：`1.1` `1.2`
- 正文不使用 H1（`#`），`title` 字段自动渲染页面 H1

---

### 8. 图片

图片资源放在 `/assets/{post-slug}/` 目录下：

```markdown
![图片描述](/assets/sglang-kvcache-runtime-architecture/image.jpg)
*图 N：说明文字，可含引用<a href="url">[N]</a>。*
```

---

### 9. 作者署名

| 作者 | 适用场景 |
|------|----------|
| `荔枝不耐思` | AI Infra 日报系列（ai-infra-daily-brief） |
| `Ethan` | 技术深度 essay |
| `MiracleFarms` | 站点说明、founding note |

---

### 10. 常见问题速查

| 问题 | 正确做法 |
|------|----------|
| 时区写成 `-0400` | 统一改为 `+0800` |
| 文件名日期与 front matter 日期不符 | 两者必须完全一致（年月日） |
| Brief 用了 HTML anchor 引用 | Brief 用 `[[N]](url)` + 文末 `## 参考来源` |
| Essay 用了 `[[N]](url)` 行内格式 | Essay 用 `<a href="url">[N]</a>` |
| 标题用了半角竖线 `\|` | 改为全角 `｜`（U+FF5C） |
| `kind` 和 `category` 不对应 | 见第 2 节枚举对照表 |
| H2 缺少中文数字编号 | 使用 `一、` `二、` `三、` |
| Brief 缺少 `## 参考来源` 章节 | Brief 结尾必须有 `---` + `## 参考来源` |
| `intro` 为空或过长 | 必填，建议 30-100 字的一句话摘要 |

---

### 11. Brief 完整模板

```markdown
---
title: AI Infra 早报｜{主题描述}
date: 2026-03-17 08:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: {一句话摘要，概括今日最重要的 2-3 个进展。}
---

{开篇段落：过去 24 小时的整体格局，不加 H2 标题}

## 一、{主题一}

**项目名 功能描述[[1]](https://github.com/...)**，{分析内容}

## 二、{主题二}

...

## N、今天真正值得记住的判断

{全局洞察与判断}

---

## 参考来源

[1] [描述文字](https://github.com/...)

[2] [描述文字](https://github.com/...)
```

---

### 12. Essay 完整模板

```markdown
---
title: {技术深度标题}
date: 2026-03-14 12:00:00 +0800
author: Ethan
kind: essay
category: Essay
intro: {一句话摘要，说明文章覆盖的技术范围和核心判断。}
---

> **版本声明**：本文分析基于 {仓库} commit `{hash}`（{日期}）；除非特别说明，...

{开篇段落：提出核心问题，建立分析框架}

## 一、{第一节}

{内容}<a href="url">[1]</a>

## 二、{第二节}

...
```
