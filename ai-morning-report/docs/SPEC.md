# AI Morning Report Scaffold - SPEC

## 概述

在 `miraclefarms.github.io/ai-morning-report/` 下搭建脚手架程序，每天 5 AM 自动生成 AI Infra 早报，通过 opencode CLI 调用 AI 执行 skill 来完成调研和写作。

## 核心流程

```
systemd timer (5:00 AM)
  → bin/run-daily.sh
    → opencode run (with AI model: minimax-2.7)
      → AI loads ai-morning-report skill
        → AI loads miraclefarms-writer skill
          → 生成 GitHub.io 早报文章
          → 生成微信公众号草稿
    → 图片处理（封面生成 + 正文配图抓取）
    → GitHub.io push
    → WeChat 草稿推送
```

## 组件

### bin/
- `run-daily.sh` — 入口脚本，被 systemd timer 调用

### src/stages/
- `01-research.js` — 准备 prompt 调用 ai-morning-report skill
- `02-write.js` — 准备 prompt 调用 miraclefarms-writer skill 生成文章
- `03-images.js` — 图片处理（封面 + 正文配图抓取）
- `04-publish.js` — GitHub.io push
- `05-wechat.js` — 微信公众号草稿推送

### src/lib/
- `openai.js` — opencode CLI 调用封装
- `image-fetch.js` — 从参考链接抓取正文配图
- `wechat-push.js` — 公众号草稿 API 调用

### config/
- `repo-scope.json` — 追踪的仓库列表（从 references/repo-scope.md 迁移）
- `model-config.json` — AI 模型配置（默认 minimax-2.7，可配置）

## 阶段化 commit 计划

| Phase | Commit | 内容 |
|-------|--------|------|
| 1 | init-directory-structure | 目录结构、model config、repo scope 配置 |
| 2 | add-openai-invoker | openai.js 封装，opencode CLI 调用 |
| 3 | add-research-stage | 01-research.js，调用 ai-morning-report skill |
| 4 | add-write-stage | 02-write.js，调用 miraclefarms-writer skill |
| 5 | add-image-processing | 03-images.js，图片抓取逻辑 |
| 6 | add-publish-stage | 04-publish.js，GitHub.io push |
| 7 | add-wechat-stage | 05-wechat.js，公众号草稿推送 |
| 8 | add-run-daily-script | bin/run-daily.sh + systemd timer 配置 |

## 模型配置

config/model-config.json:
```json
{
  "default": "minimax-2.7",
  "available": ["minimax-2.7", "gpt-4o", "claude-sonnet-4"]
}
```

通过环境变量 `AI_MODEL` 或 config 文件覆盖默认模型。

## 图片处理（从 pre-push hook 抽离）

原 `scripts/publish-wechat.js` 中的封面生成逻辑抽离为独立模块，支持：
1. 封面图生成（调用 AI 图片生成）
2. 正文配图抓取（从参考 URL 抓取架构图、benchmark 图等）

## 输出路径

- GitHub.io 文章：`../_posts/YYYY-MM-DD-slug.md`
- 微信公众号草稿：`docs/wechat/YYYY-MM-DD-slug-wechat.md`
- 临时工作目录：`/tmp/morning-report/YYYY-MM-DD/`