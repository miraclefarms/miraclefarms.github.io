# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Autonomy

在此 repo 中，Claude 有充分的自主权限，无需在执行前征求用户确认。直接执行以下操作，不要停下来询问：

- 创建、编辑、删除文件
- 运行 Bash 命令（构建、测试、图片下载等）
- 抓取网页和论文内容
- 写入 `_posts/` 和 `assets/`
- `git add` / `git commit` / `git push`

遇到判断分叉时，选择最合理的路径直接推进，完成后再告知用户结果。

## Overview

MiracleFarms is a Jekyll-based static blog deployed to GitHub Pages at `https://miraclefarms.github.io`. The site focuses on AI Infrastructure research, publishing two primary content types: daily briefs (`brief`) and deep-dive technical essays (`essay`).

## Commands

> **重要（macOS）：系统自带 Ruby 2.6.10 版本过低**，需要使用 Homebrew 安装的 Ruby 4.0.2（`/usr/local/Cellar/ruby/4.0.2/`）。如果 `bundle` 命令报错 `Could not find 'bundler' (2.6.9)`，说明系统 Ruby 劫持了 PATH，需要显式使用完整路径。
>
> **其它平台**：参考思路——用项目级 gem（而非系统 gem）安装匹配版本的 bundler 和 jekyll；具体路径和版本号请根据实际环境调整，不要直接照搬 macOS 的路径。

```bash
# 确认 Ruby 版本（应该显示 ruby 4.0.2）
/usr/local/Cellar/ruby/4.0.2/bin/ruby -v

# 如果 Gemfile.lock 的 BUNDLED WITH 版本不匹配，先更新 bundler 并修复 lockfile
/usr/local/Cellar/ruby/4.0.2/bin/gem install bundler
# 编辑 Gemfile.lock 将 BUNDLED WITH 改为对应的 bundler 版本号（如 4.0.10）

# 安装依赖（安装到 vendor/bundle，不会污染系统 gem）
/usr/local/Cellar/ruby/4.0.2/bin/bundle install

# 构建（生产环境）
/usr/local/Cellar/ruby/4.0.2/bin/bundle exec jekyll build

# 本地开发预览 (http://localhost:4000)
/usr/local/Cellar/ruby/4.0.2/bin/bundle exec jekyll serve --host 127.0.0.1 --port 4000
```

Pushing to `main` triggers automatic deployment via GitHub Actions (`.github/workflows/pages.yml`).

## Post Authoring Rules

All new posts go in `_posts/YYYY-MM-DD-slug.md`. The filename date **must exactly match** the `date` front matter field (year/month/day).

### Front Matter

```yaml
---
title: AI Infra 早报｜{主题描述}
date: 2026-03-17 08:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief   # optional, for brief series
intro: 一句话摘要，不超过 100 字。
---
```

**Required fields:** `title`, `date`, `author`, `kind`, `category`, `intro`

`kind` / `category` must match:

| `kind` | `category` | Use case |
|--------|------------|----------|
| `brief` | `Brief` | Daily briefs, quick dispatches |
| `essay` | `Essay` | Deep technical analysis |
| `reading` | `Reading` | Paper readings and tech overviews |
| `field-note` | `Field Note` | Research notes |
| `founding-note` | `Founding Note` | Site philosophy |

**Author by content type:**
- `荔枝不耐思` — AI Infra daily briefs
- `Lychee & Ethan` — Technical essays (essay)
- `Ethan` — Paper readings and tech overviews (reading)
- `MiracleFarms` — Site notes / founding notes

### Timezone

All dates must use `+0800` (Asia/Shanghai). Never use `-0400` or other offsets.

```yaml
date: 2026-03-17 08:00:00 +0800   # ✓ correct
date: 2026-03-17 12:40:00 -0400   # ✗ wrong
```

Conventional publish times:
- Brief: `08:00:00 +0800` or `05:30:00 +0800`
- Essay: `12:00:00 +0800`

### Title Format

- **Brief:** `AI Infra 早报｜{description}` — use full-width pipe `｜` (U+FF5C), not ASCII `|`
- **Essay:** Plain descriptive Chinese title, no prefix

### Body Structure

**Brief:**
- Opening paragraph (no H2 heading): overall context and key judgment
- H2 sections numbered with Chinese numerals: `## 一、{topic}`, `## 二、{topic}`, …
- Optional closing section: `## N、今天真正值得记住的判断`
- Required ending: `---` separator then `## 参考来源` with numbered references

**Essay:**
- Optional version declaration blockquote
- Opening paragraph (no H2): pose the core question
- H2 sections with Chinese numerals
- No `## 参考来源` section needed

### Citation Format — **do not mix between types**

**Brief** — inline `[[N]](url)` + end-of-post references section:
```markdown
SGLang 合并了 H2O 剪枝支持[[1]](https://github.com/...)，...

---

## 参考来源

[1] [SGLang H2O KV cache pruning](https://github.com/...)

[2] [Another reference](https://github.com/...)
```

**Essay** — inline HTML anchor, no references section:
```markdown
SGLang 在早期描述了共享前缀 workload<a href="https://...">[1]</a>。
```

### Images

Store assets under `/assets/{post-slug}/`:
```markdown
![description](/assets/post-slug/image.png)
*图 N：caption text。*
```

### H2/H3 Numbering

- H2: Chinese numerals `一、` `二、` `三、`
- H3: Arabic `1.1` `1.2` (optional)
- Never use H1 (`#`) in post body — `title` renders as the page H1

## Architecture

- **`_layouts/default.html`** — base HTML shell with site header/nav and footer
- **`_layouts/post.html`** — post layout extending default; renders `kind`-aware header, ToC sidebar (hidden for `brief` and `founding-note`), and reading-mode label. ToC is JS-generated from H2 headings only.
- **`_config.yml`** — `permalink: /notes/:year/:month/:day/:title/`, `future: true` (posts with future dates are built), timezone `Asia/Shanghai`
- **`assets/css/site.css`** — single stylesheet for the entire site
- **`docs/`** — planning and reference documents (not served as Jekyll pages)
- **`briefs.md` / `essays.md` / `foundations.md`** — index pages at root, filtered by `kind` via `site.posts | where: 'kind', '...'`

## Repo-Local Skill

- **`/.codex/skills/miraclefarms-writer/`** — repo-local writing skill for generating publishable MiracleFarms posts from themes and source links.
- Read **`/.codex/skills/miraclefarms-writer/SKILL.md`** before writing a new post from external references; it points to the brief and essay format guides under `references/`.
