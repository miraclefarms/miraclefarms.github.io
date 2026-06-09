# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ 仓库架构（必读，先看这段再动手）

**本仓库 `miraclefarms.github.io` 只是渲染层（Jekyll 模板、样式、布局、部署配置）。文章本身不在这里。**

- 本仓库的 `_posts/` **故意是空的**（只有 `.gitkeep`）。真正的文章、Wiki 源码、文章配图都在**私有仓库 `miraclefarms/miraclefarms-content`**，由 `.github/workflows/pages.yml` 在 CI 构建时通过 `actions/checkout` + `cp -r _private/_posts/. _posts/` 现场注入（见 commit 14baeab）。
- **因此本地直接 `jekyll build` / `serve` 看到的板块页是 0 篇文章——这是正常的，不是 bug。** 要带内容预览，先从 content 仓库把 `_posts/` 和 `assets/` 复制过来（见 `README.md` 的「本地开发」）。
- 板块页/首页是独立的 React 单页（`assets/shared.jsx` + 各 `*/index.html`），文章列表由 Jekyll 在构建时用 Liquid 把 `site.posts` 注入成 `window.ALL_BRIEFS` 等全局数组。所以板块页的内容同样依赖私有 `_posts` 注入。
- `_private/assets/` 只含每篇文章的图片目录，**不含** `shared.jsx`/`notion.css`；workflow 的 `rsync ... --exclude=css --exclude=icons` 不会覆盖渲染层这两个文件，可放心在本仓库直接改。

### 部署陷阱（曾导致整站文章消失）

GitHub Pages 源**必须保持「GitHub Actions」（`build_type: workflow`），不能是「Deploy from a branch」（legacy）。**

- 若是 branch 源：每次 `git push` 本仓库都会额外触发 GitHub **内置的 `pages build and deployment`**，它直接用本仓库源码构建（`_posts` 为空）并部署 → **把带文章的部署覆盖成空站**（线上 `/notes/...` 全 404、板块页列表全空）。push **content 仓库**不受影响（它走 `repository_dispatch: content-updated`，只触发自定义 workflow）。
- **诊断**：`gh api repos/miraclefarms/miraclefarms.github.io/pages --jq .build_type` 应为 `workflow`。
- **恢复 / 改回**：`gh api -X PUT repos/.../pages -f build_type=workflow`，再 `gh workflow run pages.yml --ref main` 手动重部署注入内容。
- **快速自检线上是否有内容**：`curl -s https://miraclefarms.github.io/briefs/ | grep -c '"date":"'`（>0 即正常）。

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

> **重要（macOS）：系统自带 Ruby 2.6.10 版本过低**，需要使用 Homebrew 安装的 Ruby 4.0.3（`/opt/homebrew/Cellar/ruby/4.0.3/`）。如果 `bundle` 命令报错 `Could not find 'bundler'`，说明系统 Ruby 劫持了 PATH，需要显式使用完整路径。
>
> **其它平台**：参考思路——用项目级 gem（而非系统 gem）安装匹配版本的 bundler 和 jekyll；具体路径和版本号请根据实际环境调整，不要直接照搬 macOS 的路径。

```bash
# 确认 Ruby 版本（应该显示 ruby 4.0.3）
/opt/homebrew/Cellar/ruby/4.0.3/bin/ruby -v

# 如果 Gemfile.lock 的 BUNDLED WITH 版本不匹配，先更新 bundler 并修复 lockfile
/opt/homebrew/Cellar/ruby/4.0.3/bin/gem install bundler
# 编辑 Gemfile.lock 将 BUNDLED WITH 改为对应的 bundler 版本号（如 4.0.10）

# 安装依赖（安装到 vendor/bundle，不会污染系统 gem）
/opt/homebrew/Cellar/ruby/4.0.3/bin/bundle install

# 构建（生产环境）
/opt/homebrew/Cellar/ruby/4.0.3/bin/bundle exec jekyll build

# 本地开发预览 (http://localhost:4000)
/opt/homebrew/Cellar/ruby/4.0.3/bin/bundle exec jekyll serve --host 127.0.0.1 --port 4000
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

**Optional field — `locked` (all kinds):**

```yaml
locked: true                        # encrypt post body; requires password to read
password_hint: "联系荔枝获取密码"    # optional hint shown in the password gate UI
```

- Set `locked: true` to encrypt the post body content at CI build time (AES-256-GCM). **Never use `encrypted: true`** — that field is unrecognized and the body will render in plain text.
- The post title, intro, author and date remain visible in the list and on the post page; only the body is encrypted.
- A lock icon 🔒 appears after the title on all index pages.
- Password is stored in `_data/site_secrets.json` (private content repo, `lock_password` key).
- Omitting `password_hint` shows the default: "本文已加密，请输入密码继续阅读。"
- Do **not** combine with `updated` — locked drafts are not public revisions.

**Optional field — `updated` (essay / reading only):**

```yaml
updated: 2026-05-09   # date only (YYYY-MM-DD), no time/timezone needed
```

- Only set on `essay` and `reading` posts that have been substantively revised after first publication.
- Do **not** add to `brief` posts — briefs are dated snapshots and are never updated.
- When set, the Essays index page sorts this post by `updated` instead of `date`, and the post header displays "Updated YYYY.MM.DD" alongside the original publish date.
- The filename and permalink always reflect the original `date`; `updated` only affects display and sort order.

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
- Optional version history blockquote (use version table for essays that have been revised; see Essay 版本管理 section in the writer skill)
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
