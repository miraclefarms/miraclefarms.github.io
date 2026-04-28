# AI Morning Report Scaffold - SPEC

> Superseded by [docs/superpowers/specs/2026-04-27-ai-morning-report-llm-pipeline-design.md](/Users/lychee/mycode/miraclefarms.github.io/docs/superpowers/specs/2026-04-27-ai-morning-report-llm-pipeline-design.md).
> This scaffold document describes the retired `01-research -> 05-wechat` pipeline and is kept only for historical reference.

## 概述

在 `miraclefarms.github.io/ai-morning-report/` 下搭建脚手架程序，每天 5 AM 自动生成 AI Infra 早报，通过 opencode CLI 调用 AI 执行 skill 来完成调研和写作。

## 核心流程（修改后）

```
systemd timer (5:00 AM Asia/Shanghai)
  → bin/run-daily.sh
    → Stage 1: 调研 (调用 ai-morning-report skill)
    → Stage 2: 写作 (调用 miraclefarms-writer skill)
    → Stage 3: 图片处理（封面生成 + 正文配图抓取）← 原来在 pre-push hook 中
    → Stage 4: GitHub.io push
    → Stage 5: WeChat 草稿推送
```

**关键变更：** 微信公众号题图生成从 `pre-push hook` 移至 stage 3，在文章生成后、git push 之前完成。

## 组件

### bin/
- `run-daily.sh` — 入口脚本，被 systemd timer 调用

### src/stages/
- `01-research.js` — 调用 ai-morning-report skill 进行调研
- `02-write.js` — 调用 miraclefarms-writer skill 生成 GitHub.io + 微信文章
- `03-images.js` — 封面图生成（调用 OpenRouter）+ 正文配图抓取
- `04-publish.js` — GitHub.io push
- `05-wechat.js` — 微信公众号草稿推送

### src/lib/
- `openai.js` — opencode CLI 调用封装
- `image-fetch.js` — 从参考链接抓取正文配图（预留）
- `wechat-push.js` — 公众号草稿 API 调用（预留）

### config/
- `repo-scope.json` — 追踪的仓库列表
- `model-config.json` — AI 模型配置（默认 minimax-2.7，可通过 AI_MODEL 环境变量覆盖）

### docs/
- `SPEC.md` — 本设计文档
- `ai-morning-report.timer` — systemd timer 配置
- `ai-morning-report.service` — systemd service 配置

## 阶段化 commit 记录

| Phase | Commit | 内容 |
|-------|--------|------|
| 1 | 585be3b | 目录结构、model config、repo scope 配置 |
| 2 | f24eb55 | openai.js 封装 opencode CLI 调用 |
| 3-4 | b56eea5 | research + write stages，调用 ai-morning-report 和 miraclefarms-writer skills |
| 5-8 | 5308661 | image、publish、wechat stages、run-daily 脚本、systemd timer 配置 |
| - | (pending) | pre-push hook 改为 NOOP（图片生成已移至 stage 3） |

## 模型配置

config/model-config.json:
```json
{
  "default": "minimax-2.7",
  "available": ["minimax-2.7", "gpt-4o", "claude-sonnet-4"]
}
```

通过环境变量 `AI_MODEL` 覆盖默认模型。

## 图片处理流程（修改后的关键变更）

原流程：pre-push hook → 调用 publish-wechat.js → 生成题图 → push

**新流程：**
```
Stage 3 (03-images.js):
  1. 读取 docs/wechat/{date}-ai-infra-daily-brief-wechat.md
  2. 调用 OpenRouter API 生成封面图
  3. 将封面图写入 /tmp/morning-report/{date}/assets/cover.png
  4. 更新 wechat 文章中的题图路径（insertGeneratedTitleImageMarkdown）
  5. 抓取正文中的外部图片到本地 assets 目录
Stage 4: git add + commit + push（包含更新后的 wechat 文章和本地图片）
Stage 5: WeChat draft API（使用已生成的本地封面图作为 thumb_media_id）
```

## 输出路径

- GitHub.io 文章：`../_posts/YYYY-MM-DD-ai-infra-daily-brief.md`
- 微信公众号草稿：`docs/wechat/YYYY-MM-DD-ai-infra-daily-brief-wechat.md`
- 临时工作目录：`/tmp/morning-report/YYYY-MM-DD/`
- 封面图：`/tmp/morning-report/YYYY-MM-DD/assets/cover.png`

## systemd timer 安装（Linux）

```bash
cp ai-morning-report/docs/ai-morning-report.timer /etc/systemd/system/
cp ai-morning-report/docs/ai-morning-report.service /etc/systemd/system/
systemctl enable ai-morning-report.timer
systemctl start ai-morning-report.timer
```

## macOS launchd 安装

```bash
cp ai-morning-report/docs/ai-morning-report.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/ai-morning-report.plist
```

**注意：** `ai-morning-report.timer` 和 `ai-morning-report.service` 是 systemd 格式（Linux），macOS 使用 `ai-morning-report.plist`（launchd 格式）。

## 手动测试

```bash
cd /Users/lychee/mycode/miraclefarms.github.io
./ai-morning-report/bin/run-daily.sh
```
