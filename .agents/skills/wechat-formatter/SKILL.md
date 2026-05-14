---
name: wechat-formatter
description: >
  将已发布的 GitHub.io brief/essay 改写为微信公众号格式的 Markdown。
  触发条件：自动化流程传入 GitHub.io 版 markdown 路径，要求生成公众号适配稿。
  只负责语义改写（内容结构、链接处理、引用格式）；CSS 主题渲染由 wechat-renderer.js 完成。
---

# WeChat Formatter

将 GitHub.io 版 brief/essay 改写为符合微信公众号编辑器规范的 Markdown。

## 工作边界

- 负责：正文结构改写、链接去除、参考章节格式化、`wechat_variant` 判断
- 不负责：front matter 生成、CSS/HTML 渲染、图片上传（由脚本层完成）

## 输入

通过命令行参数接收 GitHub.io 版 `.md` 文件路径，读取其 front matter 和正文。

## 改写流程

### 1. 判断文章类型

读取 front matter 中的 `kind` 字段：
- `kind: brief` → 走 Brief 路径，输出 `wechat_variant: brief`
- `kind: essay` → 走 Essay 长文路径，输出 `wechat_variant: essay-longform`

### 2. Brief 改写规则

**结构：**
```
---
title: 今日焦点：{核心主题描述}
wechat_variant: brief
intro: {40-80字摘要，用于公众号摘要栏}
---

**📅 YYYY-MM-DD**

> {引导语：结合当天热点点明为什么现在值得读，不超过 90 字}

---

## {分类名}

**{条目标题}[N]** - {当前问题 → 这次变化做了什么 → 预期效果}，属于 **[持续更新]**（如适用）。

---

> 一句话结论：**{全文最核心的判断，不超过 60 字}**

---

## 参考

[1] {条目标题}：{完整 URL}

[2] {条目标题}：{完整 URL}
```

**正文 body 禁止重复标题**：Brief 和 Essay 一样，文章标题写在 front matter 的 `title` 字段里；正文 body 不再写 `# 今日焦点...`，推送脚本会把 front matter title 用作草稿箱标题。

**题图下方导语要有热点张力**：不要复述标题。必须抓住当下热点模型、框架或硬件变化（例如 DeepSeek、GPT-OSS、Blackwell、新模型首发、新硬件量产、跨框架同日就位），用一句话回答“为什么读者现在需要关心”。导语要把技术变化和现实压力连起来，比如“新模型上线窗口正在压缩，今天这些 PR 决定 NVFP4 能不能从实验格式进入默认服务”。

**H2 分类名约定**（按需选用）：推理侧 / 训练侧 / 生产部署侧 / 应用侧 / 工具链

**不使用中文数字编号**（`一、二、三、` 是 GitHub.io 版规范，公众号版不沿用）。

### 3. Essay 长文改写规则

维持论证顺序，不拆成日报式条目。保留与 GitHub.io 版相同的正文配图（路径写成相对 `docs/wechat/` 的相对路径）。使用经典蓝主题（在 front matter 写 `wechat_variant: essay-longform`）。

**正文 body 禁止重复标题**：`# 标题` 本身是文章的 H1；正文 body 的第一句话直接进入引导语或开篇正文，不要以任何形式重复文章标题（例如 "本文分析……"、"这篇文章将……" 等）。

### 4. 链接处理（必须严格执行）

- 正文主体中的 `[[N]](url)` → 改写为纯文本 `[N]`
- 正文主体中的 `<a href="url">[N]</a>` → 改写为纯文本 `[N]`
- 正文主体中的 `[文字](url)` Markdown 链接 → 改写为纯文字，URL 丢入参考章节
- **URL 只允许出现在 `## 参考` 章节**

### 5. 参考章节格式

章节名固定为 `## 参考`（不用 `## 参考来源` 或 `## 参考资料`）。

格式：`[N] {条目标题}：{完整 URL}`（冒号后直接接 URL，不加括号）。

### 6. 持续更新标签

如果某个条目在近期日报中已出现过、今天是继续跟进，结尾打：`属于 **[持续更新]**。`

## 输出要求

- 输出纯 Markdown，以 front matter 开头
- front matter 包含：`title`、`wechat_variant`、`intro`（40-80字摘要，用于公众号摘要栏）
- 不输出任何说明文字，直接给 markdown 全文
- 末尾必须有 `> 一句话结论：**...**` blockquote

## 自检

- 正文主体没有任何 Markdown 链接、HTML 链接
- URL 只出现在 `## 参考` 章节，格式为 `[N] 标题：URL`
- Brief 的 H2 直接写分类名，没有中文数字编号
- Brief 正文 body 没有 `# 今日焦点...` 或其他 H1，大标题只在 front matter `title` 中出现
- Brief 题图下方导语结合了当天热点，并说明为什么读者现在需要关心
- 结尾有 `> 一句话结论：**...**`
- 保留了原文的核心判断，不只是新闻摘要
- Essay 长文：已写 `wechat_variant: essay-longform`，保留正文配图
- Essay 长文：正文 body 没有以任何形式重复文章标题（没有 "本文分析……" 等开头）
