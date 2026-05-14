# Fields Section Design

**Date:** 2026-05-14  
**Status:** Approved

## Summary

Add a standalone "Fields" section to miraclefarms.github.io for hands-on experiments and paper reproductions. Content type `kind: field-note` already exists in the codebase but is currently merged into the Essays index. This work separates it into its own nav entry and index page.

---

## Naming Decision

Nav label: **Fields**  
URL: `/fields/`  
Jekyll kind: `field-note` (unchanged)  
Category front matter: `Field Note` (unchanged)

Rationale: "Fields" is a single-word noun consistent with Briefs / Readings / Essays. It carries a MiracleFarms double meaning (agricultural fields = the place where things grow through hands-on work). "Lab" would be more literal but breaks the farm-metaphor vocabulary. Two-word "Field Notes" is accurate but over-specifies and limits future scope.

---

## Navigation Order

New order: **Home | Briefs | Readings | Fields | Essays | Wiki | GitHub**

Fields sits between Readings and Essays — positionally reflects the content gradient from "reading others' work" → "doing your own work" → "synthesizing".

---

## Files to Change

### 1. New: `fields.md`

- Permalink: `/fields/`
- Layout: `default`
- Content type: `kind: field-note`
- Structure: mirrors `readings.md` exactly — filter chips, list/table view toggle, year-grouped list view
- Page emoji: `🔬`
- H1: `Fields`
- `pg-sub`: `实验与复现：动手做的 AI Infra 研究。`
- `pg-sub-small`: explains how Fields differs from Readings (Readings = reading others' work; Fields = doing your own)
- Callout: shows post count and topic coverage
- Filter bar: same filter + view-toggle as Readings
- List view: year-grouped, same `.pagelink` structure, CSS class prefix `field-`
- Table view: Date + Title + Tags columns (same as Readings, no Updated column)
- Empty state: `没有匹配的 field note。`
- Section footer description (below divider): explains Fields positioning
- Page footer: same standard footer

### 2. `_layouts/default.html`

- Add nav link `<a href="/fields/" id="nav-fields">` between Readings and Essays
- SVG icon: flask/beaker shape (signals experiment)
- Label: `<span>Fields</span>`
- Active-state JS: add `'/fields': 'nav-fields'` to the path→id map

### 3. `_layouts/post.html`

Two changes:
1. Breadcrumb: add `{% elsif page.kind == 'field-note' %}` branch pointing to `/fields/` (remove field-note from the essay branch)
2. Word count chip: extend condition to include `field-note` (currently only essay + reading get word count)

### 4. `essays.md`

- Remove lines 9–10 that fetch and merge `field_notes`
- `all_posts` reverts to essays only: `{% assign all_posts = essays %}`
- Update callout and filter to reflect essays-only count
- Remove field-note references from empty state message

### 5. New: `_posts/2026-05-14-field-notes-opening.md`

**Front matter:**
```yaml
---
title: Field Notes 开篇：这里放什么
date: 2026-05-14 12:00:00 +0800
author: MiracleFarms
kind: field-note
category: Field Note
intro: Field Notes 是 MiracleFarms 的动手研究日志——论文复现、实验记录、工具调试。这里写的是做出来的东西，不是读到的东西。
---
```

**Structure** (Readings-style, H2 sections):
- Opening paragraph: what Fields is for
- `## 一、Fields 在 MiracleFarms 里的位置` — contrast with Briefs (observation) / Readings (others' papers) / Essays (synthesis)
- `## 二、这里会放什么` — experiments, reproductions, tool benchmarks, setup notes
- `## 三、格式约定` — brief note on format conventions (to be refined later)
- No `## 参考来源` section needed (meta post, no external citations)

---

## What Does NOT Change

- `kind: field-note` and `category: Field Note` front matter values — no rename
- `post.html` ToC: field-notes already show ToC (not in the force-hide list), keep as-is
- `foundations.md` — unaffected
- CSS (`assets/css/site.css`) — no new classes needed; existing `.pagelink`, `.pagelist`, `.filterbar`, `.fchip`, `.ntable`, `.reading-year-group` / `.reading-item` patterns are reused with `field-` prefix class names; no new CSS rules needed since the JS behavior is identical

---

## Spec Self-Review

- No placeholders or TBDs remaining
- Architecture consistent: fields.md mirrors readings.md; essays.md cleanup is isolated
- Scope: 5 discrete file changes, all focused on this feature
- No ambiguity: each file change has a specific, actionable description
