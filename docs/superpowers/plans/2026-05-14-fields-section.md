# Fields Section Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone "Fields" index page and nav entry for `kind: field-note` posts, separate them from Essays, and publish the opening meta post.

**Architecture:** Five self-contained file changes. No new CSS or JS patterns needed — `fields.md` is a structural copy of `readings.md` with `field-`-prefixed JS selectors. The `_layouts/` changes are small targeted edits. Essays cleanup is a deletion-only change.

**Tech Stack:** Jekyll 4.x, Liquid templates, vanilla JS, GitHub Pages CI.

Build command (macOS):
```bash
/usr/local/Cellar/ruby/4.0.2/bin/bundle exec jekyll build 2>&1 | tail -20
```

Expected successful build output ends with:
```
                    done in X.XXX seconds.
 Auto-regeneration: disabled. Use --watch to enable.
```

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `fields.md` | Fields index page — filter, list/table view, year groups |
| Modify | `_layouts/default.html` | Add Fields nav link + active-state JS entry |
| Modify | `_layouts/post.html` | Route field-note breadcrumb to /fields/, add word count |
| Modify | `essays.md` | Remove field-note merge; essays-only content |
| Create | `_posts/2026-05-14-field-notes-opening.md` | Opening meta post |

---

## Task 1: Create `fields.md` index page

**Files:**
- Create: `fields.md`

- [ ] **Step 1: Create the file**

Create `/Users/anne/mycode/miraclefarms.github.io/fields.md` with the following content:

```markdown
---
layout: default
title: Fields
description: MiracleFarms Fields — AI Infra 实验与论文复现的动手研究日志。
permalink: /fields/
---

{% assign field_notes = site.posts | where: 'kind', 'field-note' | sort: 'date' | reverse %}

{% assign all_tag_str = "" %}
{% for post in field_notes %}
  {% for tag in post.tags %}
    {% assign all_tag_str = all_tag_str | append: "," | append: tag %}
  {% endfor %}
{% endfor %}
{% assign tag_list = all_tag_str | split: "," | uniq | sort %}

<main class="page">
  <div class="pg-meta">
    <a href="{{ '/' | relative_url }}">MiracleFarms</a>
    <span class="dot">›</span>
    <span>Fields</span>
  </div>

  <div class="pg-icon" aria-hidden="true">🔬</div>
  <h1 class="pg-title">Fields</h1>
  <p class="pg-sub">实验与复现：动手做的 AI Infra 研究。</p>
  <p class="pg-sub-small">
    Readings 是读别人写的东西；Fields 是自己动手做的东西——论文复现、工具基准测试、调试记录。一手数据，不是综述。
  </p>

  <aside class="callout">
    <span class="ico" aria-hidden="true">🔬</span>
    <div>
      <p>
        共 <strong>{{ field_notes.size }}</strong> 篇 · 覆盖 <strong>Inference / KV Cache / Agents / Evaluation</strong>。每篇聚焦一次实际动手的研究记录，可复现、可引用。
      </p>
    </div>
  </aside>

  <div class="filterbar" role="toolbar" aria-label="筛选与视图" id="fields-filterbar">
    <span class="lbl">Filter:</span>
    <button class="fchip on" data-tag="" type="button">
      全部 <span class="count">{{ field_notes.size }}</span>
    </button>
    {% for tag in tag_list %}
    {% if tag != "" %}
    {% assign t_count = field_notes | where_exp: "p", "p.tags contains tag" | size %}
    <button class="fchip" data-tag="{{ tag }}" type="button">
      {{ tag }} <span class="count">{{ t_count }}</span>
    </button>
    {% endif %}
    {% endfor %}
    <span style="flex:1"></span>
    <span class="viewtabs" role="tablist">
      <button class="on" data-view="list" type="button">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M8 6h13M8 12h13M8 18h13"/><circle cx="4" cy="6" r="1"/><circle cx="4" cy="12" r="1"/><circle cx="4" cy="18" r="1"/></svg>
        List
      </button>
      <button data-view="table" type="button">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="16" rx="1.5"/><path d="M3 10h18M3 15h18M9 4v16"/></svg>
        Table
      </button>
    </span>
  </div>

  <div id="fields-list-view">
    {% assign prev_year = "" %}
    {% for post in field_notes %}
      {% assign yr = post.date | date: "%Y" %}
      {% if yr != prev_year %}
        {% unless forloop.first %}</div></section>{% endunless %}
        {% assign yr_count = 0 %}
        {% for p in field_notes %}
          {% assign pyr = p.date | date: "%Y" %}
          {% if pyr == yr %}{% assign yr_count = yr_count | plus: 1 %}{% endif %}
        {% endfor %}
        <section class="field-year-group" data-year="{{ yr }}">
        <div class="year-row">
          <span class="yr">{{ yr }}</span>
          <span class="ln"></span>
          <span class="yr-count" data-total="{{ yr_count }}">{{ yr_count }} 篇</span>
        </div>
        <div class="pagelist" role="list">
        {% assign prev_year = yr %}
      {% endif %}
      <a class="pagelink field-item" href="{{ post.url | relative_url }}" role="listitem"
         data-tags="{{ post.tags | join: ',' }}">
        <span class="pl-handle" aria-hidden="true">⋮⋮</span>
        <span class="pl-ico" aria-hidden="true">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 014 4v14a3 3 0 00-3-3H2z"/><path d="M22 3h-6a4 4 0 00-4 4v14a3 3 0 013-3h7z"/></svg>
        </span>
        <div class="pl-body">
          <div class="pl-title">{{ post.title }}</div>
          {% if post.intro %}
          <div class="pl-excerpt">{{ post.intro | truncate: 140 }}</div>
          {% endif %}
          <div class="pl-meta">
            <span class="tag-kind">{{ post.category | default: post.kind }}</span>
            {% for tag in post.tags %}
            <span class="tag">{{ tag }}</span>
            {% endfor %}
          </div>
        </div>
        <div class="pl-date">{{ post.date | date: "%Y.%m.%d" }}</div>
      </a>
    {% endfor %}
    </div></section>
  </div>

  <div id="fields-table-view" style="display:none">
    <table class="ntable" role="table">
      <thead>
        <tr>
          <th class="col-date" style="width:120px">
            <span class="col-ico">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="5" width="18" height="16" rx="1.5"/><path d="M3 10h18M8 3v4M16 3v4"/></svg>
            </span>Date
          </th>
          <th class="col-title">
            <span class="col-ico">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 014 4v14a3 3 0 00-3-3H2z"/><path d="M22 3h-6a4 4 0 00-4 4v14a3 3 0 013-3h7z"/></svg>
            </span>Title
          </th>
          <th class="col-tags" style="width:200px">
            <span class="col-ico">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M20 12L12 20l-8-8V4h8z"/><circle cx="8" cy="8" r="1.5"/></svg>
            </span>Tags
          </th>
        </tr>
      </thead>
      <tbody>
        {% for post in field_notes %}
        <tr class="field-table-row" data-tags="{{ post.tags | join: ',' }}">
          <td class="col-date">{{ post.date | date: "%Y.%m.%d" }}</td>
          <td class="col-title"><a href="{{ post.url | relative_url }}">{{ post.title }}</a></td>
          <td class="col-tags">
            {% for tag in post.tags %}
            <span class="tag">{{ tag }}</span>
            {% endfor %}
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <div id="fields-empty" class="empty" style="display:none">没有匹配的 field note。</div>

  <hr class="divider">

  <p class="h2-sub">Field note</p>
  <h2 class="h2">关于 Fields 的定位</h2>
  <p class="muted">
    Fields 是 MiracleFarms 里最靠近原始数据的地方。Brief 捕捉"今天发生了什么"，Reading 消化"别人怎么写的"，Essay 给出"这意味着什么"——而 Fields 回答"我自己跑出来的结果是什么"。实验记录、复现笔记、基准测试、工具调试：这里写的东西都经过了亲手验证。
  </p>

  <footer class="pg-footer">
    <span>Less hype, more systems. · 公开生长，而不是一次性完成。</span>
    <span>© {{ 'now' | date: "%Y" }} MiracleFarms</span>
  </footer>
</main>

<script>
(function() {
  var chips      = document.querySelectorAll('#fields-filterbar .fchip');
  var viewBtns   = document.querySelectorAll('#fields-filterbar .viewtabs button');
  var listView   = document.getElementById('fields-list-view');
  var tableView  = document.getElementById('fields-table-view');
  var emptyMsg   = document.getElementById('fields-empty');
  var activeTag  = '';
  var activeView = 'list';

  function matchesTag(el) {
    if (!activeTag) return true;
    var tags = el.dataset.tags ? el.dataset.tags.split(',') : [];
    return tags.indexOf(activeTag) !== -1;
  }

  function applyFilter() {
    var listItems  = listView.querySelectorAll('.field-item');
    var tableRows  = tableView.querySelectorAll('.field-table-row');
    var yearGroups = listView.querySelectorAll('.field-year-group');
    var visible = 0;

    listItems.forEach(function(el) {
      var show = matchesTag(el);
      el.style.display = show ? '' : 'none';
      if (show) visible++;
    });
    tableRows.forEach(function(el) {
      el.style.display = matchesTag(el) ? '' : 'none';
    });
    yearGroups.forEach(function(group) {
      var groupItems = Array.from(group.querySelectorAll('.field-item'));
      var groupVisible = groupItems.filter(function(el) { return el.style.display !== 'none'; });
      group.style.display = groupVisible.length > 0 ? '' : 'none';
      var countEl = group.querySelector('.yr-count');
      if (countEl) {
        var total = countEl.dataset.total;
        countEl.textContent = activeTag
          ? groupVisible.length + ' / ' + total + ' 篇'
          : total + ' 篇';
      }
    });
    emptyMsg.style.display = visible === 0 ? '' : 'none';
  }

  chips.forEach(function(chip) {
    chip.addEventListener('click', function() {
      activeTag = this.dataset.tag;
      chips.forEach(function(c) { c.classList.toggle('on', c.dataset.tag === activeTag); });
      applyFilter();
    });
  });

  viewBtns.forEach(function(btn) {
    btn.addEventListener('click', function() {
      activeView = this.dataset.view;
      viewBtns.forEach(function(b) { b.classList.toggle('on', b.dataset.view === activeView); });
      listView.style.display  = activeView === 'list'  ? '' : 'none';
      tableView.style.display = activeView === 'table' ? '' : 'none';
    });
  });
})();
</script>
```

- [ ] **Step 2: Build and verify**

```bash
/usr/local/Cellar/ruby/4.0.2/bin/bundle exec jekyll build 2>&1 | tail -5
```

Expected: build succeeds, no Liquid errors.

Also confirm the file was generated:
```bash
ls _site/fields/index.html
```

Expected: file exists.

- [ ] **Step 3: Commit**

```bash
git add fields.md
git commit -m "Add Fields index page at /fields/"
```

---

## Task 2: Update nav in `_layouts/default.html`

**Files:**
- Modify: `_layouts/default.html`

- [ ] **Step 1: Add Fields nav link between Readings and Essays**

In `_layouts/default.html`, replace this block:

```html
          <a href="{{ '/readings/' | relative_url }}" id="nav-readings">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 014 4v14a3 3 0 00-3-3H2z"/><path d="M22 3h-6a4 4 0 00-4 4v14a3 3 0 013-3h7z"/></svg>
            <span>Readings</span>
          </a>
          <a href="{{ '/essays/' | relative_url }}" id="nav-essays">
```

with:

```html
          <a href="{{ '/readings/' | relative_url }}" id="nav-readings">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 014 4v14a3 3 0 00-3-3H2z"/><path d="M22 3h-6a4 4 0 00-4 4v14a3 3 0 013-3h7z"/></svg>
            <span>Readings</span>
          </a>
          <a href="{{ '/fields/' | relative_url }}" id="nav-fields">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M9 3H5a2 2 0 00-2 2v4m6-6h10a2 2 0 012 2v4M9 3v18m0 0h10a2 2 0 002-2v-4M9 21H5a2 2 0 01-2-2v-4m0 0h18"/></svg>
            <span>Fields</span>
          </a>
          <a href="{{ '/essays/' | relative_url }}" id="nav-essays">
```

- [ ] **Step 2: Add `/fields` to the active-state JS map**

In `_layouts/default.html`, replace:

```javascript
      var map = {
        '/': 'nav-home',
        '/briefs': 'nav-briefs',
        '/essays': 'nav-essays',
        '/readings': 'nav-readings'
      };
```

with:

```javascript
      var map = {
        '/': 'nav-home',
        '/briefs': 'nav-briefs',
        '/readings': 'nav-readings',
        '/fields': 'nav-fields',
        '/essays': 'nav-essays'
      };
```

- [ ] **Step 3: Build and verify**

```bash
/usr/local/Cellar/ruby/4.0.2/bin/bundle exec jekyll build 2>&1 | tail -5
```

Expected: build succeeds.

Confirm Fields link appears in built HTML:
```bash
grep -c 'nav-fields' _site/index.html
```

Expected: `1`

- [ ] **Step 4: Commit**

```bash
git add _layouts/default.html
git commit -m "Add Fields nav link between Readings and Essays"
```

---

## Task 3: Update breadcrumb and word count in `_layouts/post.html`

**Files:**
- Modify: `_layouts/post.html`

- [ ] **Step 1: Fix the breadcrumb for field-note**

In `_layouts/post.html`, replace:

```liquid
      {% if page.kind == 'brief' %}
        <a href="{{ '/briefs/' | relative_url }}">Briefs</a>
        {% elsif page.kind == 'essay' or page.kind == 'field-note' %}
        <a href="{{ '/essays/' | relative_url }}">Essays</a>
        {% else %}
        <a href="{{ '/foundations/' | relative_url }}">Foundations</a>
        {% endif %}
```

with:

```liquid
      {% if page.kind == 'brief' %}
        <a href="{{ '/briefs/' | relative_url }}">Briefs</a>
        {% elsif page.kind == 'reading' %}
        <a href="{{ '/readings/' | relative_url }}">Readings</a>
        {% elsif page.kind == 'field-note' %}
        <a href="{{ '/fields/' | relative_url }}">Fields</a>
        {% elsif page.kind == 'essay' %}
        <a href="{{ '/essays/' | relative_url }}">Essays</a>
        {% else %}
        <a href="{{ '/foundations/' | relative_url }}">Foundations</a>
        {% endif %}
```

- [ ] **Step 2: Add field-note to the word-count condition**

In `_layouts/post.html`, replace:

```liquid
        {% if page.kind == 'essay' or page.kind == 'reading' %}
        <span id="post-word-count" class="post-word-count-chip"></span>
        {% endif %}
```

with:

```liquid
        {% if page.kind == 'essay' or page.kind == 'reading' or page.kind == 'field-note' %}
        <span id="post-word-count" class="post-word-count-chip"></span>
        {% endif %}
```

- [ ] **Step 3: Build and verify**

```bash
/usr/local/Cellar/ruby/4.0.2/bin/bundle exec jekyll build 2>&1 | tail -5
```

Expected: build succeeds, no Liquid errors.

- [ ] **Step 4: Commit**

```bash
git add _layouts/post.html
git commit -m "Route field-note breadcrumb to /fields/, add word count"
```

---

## Task 4: Remove field-note merge from `essays.md`

**Files:**
- Modify: `essays.md`

- [ ] **Step 1: Remove the field-note fetch and merge**

In `essays.md`, replace the front-matter-following Liquid block:

```liquid
{% assign essays = site.posts | where: 'kind', 'essay' | sort: 'date' | reverse %}
{% assign field_notes = site.posts | where: 'kind', 'field-note' | sort: 'date' | reverse %}
{% assign all_posts = essays | concat: field_notes | sort: 'date' | reverse %}
```

with:

```liquid
{% assign essays = site.posts | where: 'kind', 'essay' | sort: 'date' | reverse %}
{% assign all_posts = essays %}
```

- [ ] **Step 2: Fix the tag loop to use essays (not all_posts) — verify it's already `all_posts`**

The tag loop at line 12 already iterates `all_posts`, so after the above change it automatically becomes essays-only. No further edit needed.

- [ ] **Step 3: Build and verify**

```bash
/usr/local/Cellar/ruby/4.0.2/bin/bundle exec jekyll build 2>&1 | tail -5
```

Expected: build succeeds.

Confirm field-notes no longer appear in Essays HTML (there are none yet, so just confirm the `field_notes` variable is gone from the built output):
```bash
grep -c 'field_notes' _site/essays/index.html 2>/dev/null || echo "0"
```

Expected: `0`

- [ ] **Step 4: Commit**

```bash
git add essays.md
git commit -m "Remove field-note merge from Essays index"
```

---

## Task 5: Write the opening Field Note post

**Files:**
- Create: `_posts/2026-05-14-field-notes-opening.md`

- [ ] **Step 1: Create the opening post**

Create `_posts/2026-05-14-field-notes-opening.md`:

```markdown
---
title: "Field Notes 开篇：这里放什么"
date: 2026-05-14 12:00:00 +0800
author: MiracleFarms
kind: field-note
category: Field Note
intro: Field Notes 是 MiracleFarms 的动手研究日志——论文复现、实验记录、工具调试。这里写的是做出来的东西，不是读到的东西。
---

MiracleFarms 一直有三层内容：Brief 捕捉当天发生的事，Reading 消化别人的论文和系统，Essay 把多条线索拼回到一个工程问题上。但还差一层——自己动手做的记录。这就是 Fields 的来由。

Fields 的核心约束只有一条：写进来的东西必须是亲手验证过的。不是"据论文称"，不是"理论上应该"，而是"我跑出来的结果是这样的"。

## 一、Fields 在 MiracleFarms 里的位置

MiracleFarms 的四个板块各自回答一个不同的问题：

- **Briefs**——今天 AI Infra 生态里发生了什么值得记录的事？
- **Readings**——这篇论文或这个系统是怎么工作的？
- **Essays**——这件事从更长的视角看意味着什么？
- **Fields**——我自己跑出来的结果是什么？

Readings 和 Fields 看起来最近，但方向相反。Reading 的输入是别人的文字，输出是自己的理解；Field Note 的输入是一个问题或一篇论文，输出是自己的数据。Readings 是消化，Fields 是生产。

## 二、这里会放什么

预计以下几类内容会在这里出现：

**论文复现。** 把论文里的关键实验跑一遍，看结果是否能对齐，记录偏差和原因。不追求完整复现，只追求"核心主张是否在我的环境下成立"。

**工具基准测试。** 对 vLLM、SGLang、llama.cpp 等推理框架做局部对比，固定变量，记录数字。结论不求推广，只求在当次实验条件下可信。

**调试与拆解记录。** 遇到某个行为不符合预期时，把诊断过程写下来。这类笔记对别人的参考价值往往比结论本身更高。

**实验性想法验证。** 有时候想法来了，跑一个小实验验证直觉，把过程和结果记下来，即使结论是"想法是错的"。

## 三、格式约定

Field Notes 的格式目前遵循 Readings 的写作规范——H2 节标题用汉字数字编号，正文以中文为主。

随着内容积累，格式会随具体需求演化。实验类文章可能需要代码块和数据表格；调试类文章可能更接近流水账。格式服从内容，不倒过来。
```

- [ ] **Step 2: Build and verify the post is generated**

```bash
/usr/local/Cellar/ruby/4.0.2/bin/bundle exec jekyll build 2>&1 | tail -5
```

Expected: build succeeds.

Confirm post appears in Fields index and was generated:
```bash
grep -l 'field-notes-opening' _site/notes/2026/05/14/ 2>/dev/null || find _site -name '*field-notes*' -type f
```

Expected: a path like `_site/notes/2026/05/14/field-notes-opening/index.html`

Also confirm it appears in the Fields index:
```bash
grep -c 'field-notes-opening' _site/fields/index.html
```

Expected: `1`

- [ ] **Step 3: Commit**

```bash
git add _posts/2026-05-14-field-notes-opening.md
git commit -m "Add Field Notes opening post"
```

---

## Task 6: Push to main

- [ ] **Step 1: Verify all commits are present**

```bash
git log --oneline -6
```

Expected: 5 new commits (fields.md, nav, post.html, essays.md, opening post) on top of the prior HEAD.

- [ ] **Step 2: Push**

```bash
git push origin main
```

Expected: push accepted, GitHub Actions deploys automatically.

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered by |
|---|---|
| New `fields.md` index page | Task 1 |
| Nav: Briefs → Readings → Fields → Essays | Task 2 Step 1 |
| Active-state JS for /fields | Task 2 Step 2 |
| Post breadcrumb: field-note → /fields/ | Task 3 Step 1 |
| Word count chip for field-note | Task 3 Step 2 |
| Remove field-note merge from essays.md | Task 4 |
| Opening meta post | Task 5 |
| Push to main | Task 6 |

**Placeholder scan:** No TBDs, TODOs, or incomplete steps found.

**Type/name consistency:** `field-item`, `field-table-row`, `field-year-group`, `fields-filterbar`, `fields-list-view`, `fields-table-view`, `fields-empty`, `nav-fields` — all used consistently across Tasks 1 and 2.
