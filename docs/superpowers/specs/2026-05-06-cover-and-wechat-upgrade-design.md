# Cover Image & WeChat Style Upgrade Design

## Summary

Optimize the AI morning report pipeline's Stage 04 (cover image) and Stage 06 (WeChat format) with four improvements: AI-generated image prompts, template-driven image generation, cover image insertion into articles, and Doocs classic style for WeChat.

## Changes

### 1. Stage 04 Rewrite — AI-Generated Cover Prompts + Template System

**Current state:** `04-cover.js` uses a hardcoded English prompt, only reads title/intro from front matter.

**New flow (two steps within Stage 04):**

- **Step A — AI generates prompt** (same `runAI()` pattern as Stage 03):
  - Reads the full article from `_posts/{date}-ai-infra-daily-brief.md`
  - Calls AI via `cli-adapter.js` to produce a cover image prompt (≤500 chars Chinese)
  - AI is instructed to highlight the day's most important PR or overall trend
  - Saves intermediate file `{assets-dir}/cover-prompt.txt`
- **Step B — OpenRouter image generation**:
  - Reads `cover-prompt.txt`
  - Combines with image template from `wechat-cover-prompt-templates.json`
  - Calls OpenRouter (Gemini image model) to generate 9:16 vertical image
  - Saves to `{assets-dir}/cover.{ext}`
- **Step C — Insert cover into article**:
  - Inserts `![题图](/assets/{date}-ai-infra-daily-brief/cover.png)` after front matter
  - Idempotent: skips if cover already present

**Data flow:**
```
Stage 03 output (_posts/*.md)
  → Step A: AI call → cover-prompt.txt
  → Step B: cover-prompt.txt + template → OpenRouter → cover.png
  → Step C: insert into _posts article
```

### 2. Image Template

**File:** `scripts/config/wechat-cover-prompt-templates.json`

Reuse existing template system from `scripts/lib/wechat-cover-prompts.js`. The template provides:
- Style description (base prompt)
- Aspect ratio config (9:16)
- Image model config

Pipeline reads the template and combines it with the AI-generated prompt before calling OpenRouter.

### 3. Cover Image Insertion into Article

After image generation, `04-cover.js` inserts the cover into the GitHub.io post:
- Position: immediately after front matter, before the opening paragraph
- Format: `![题图](/assets/{date}-ai-infra-daily-brief/cover.{ext})`
- Also copies the image to the project's `assets/` directory for GitHub.io serving

### 4. WeChat Style — Doocs Classic

**File:** `ai-morning-report/wechat-themes/brief-emerald.css`

Rewrite to match Doocs "Classic" (default) theme structure with emerald green palette:
- **H2**: centered, solid emerald background bar, white text
- **H3**: left border accent, no background
- **blockquote**: left border + light background (kept)
- **Conclusion blockquote**: gradient background (kept)
- **strong, code, tables**: emerald color system

**File:** `ai-morning-report/wechat-themes/base.css`
- Minor adjustments for H2 padding/centering to support the solid background bar style

## Files Changed

| File | Change |
|------|--------|
| `ai-morning-report/src/stages/04-cover.js` | Full rewrite: AI prompt gen + template + image gen + article insertion |
| `ai-morning-report/wechat-themes/brief-emerald.css` | Rewrite to Doocs classic style |
| `ai-morning-report/wechat-themes/base.css` | Minor adjustments for H2 centering |

## Non-Goals

- Not merging the two publisher systems (pipeline vs scripts/)
- Not changing Stage 05/07 behavior
- Not adding new image models beyond OpenRouter
