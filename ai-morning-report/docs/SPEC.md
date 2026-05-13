# AI Morning Report — SPEC v2

每天 05:30 Asia/Shanghai 自动运行，生成 AI Infra 日报并发布到 GitHub.io + 微信公众号草稿箱，可选同步发布 X thread。

## 核心设计原则

- **AI 做 AI 的事，脚本做脚手架**：数据抓取用 `gh` CLI，AI 分析+写作调用 CLI，格式渲染用 Node 工具
- **Skills 是核心知识层**：提示词逻辑沉淀在 `.codex/skills/`，脚本不内嵌长提示词
- **Multi-CLI 可互换**：通过 `cli-adapter.js` 支持 claude / opencode / codex，`AI_CLI` 环境变量选择
- **配图不阻塞发布**：Stage 4 失败只告警，GitHub.io 和微信草稿箱均可无图发布

## 流程

```
05:30 launchd
  → bin/run-daily.sh
      ├── [1] 01-fetch.sh       纯 gh CLI，并行抓取 merged PR + release + commits
      ├── [2] 02-analyze.js     AI call 1：筛选/聚类 → material.md
      ├── [3] 03-write.js       AI call 2：写作 → _posts/YYYY-MM-DD-ai-infra-daily-brief.md
      ├── [4] 04-cover.js       封面图生成（非阻塞，失败继续）
      ├── [5] 05-publish.sh     git push → GitHub.io
      ├── [6] 06-wechat-format.js  AI 改写（wechat-formatter skill）+ CSS 渲染 → 临时 .md + .html，并归档 Markdown + 题图到 docs/wechat
      ├── [7] 07-wechat-push.js    WeChat draft API → 草稿箱，成功后提交并推送 docs/wechat 归档稿与题图资产
      └── [8] 08-x-push.js         AI 生成英文 thread → X API 发帖（默认关闭）
```

## 文件结构

```
ai-morning-report/
  bin/
    run-daily.sh              入口，launchd 调用
  src/
    lib/
      cli-adapter.js          Multi-CLI 封装（claude/opencode/codex）
      wechat-renderer.js      Markdown → Styled HTML（不依赖浏览器）
    stages/
      01-fetch.sh             并行 gh CLI 数据抓取
      02-analyze.js           AI：筛选 + 聚类
      03-write.js             AI：撰写 GitHub.io 日报
      04-cover.js             封面图生成（OpenRouter）
      05-publish.sh           git add/commit/push
      06-wechat-format.js     AI 改写 + CSS 渲染 + Markdown/题图归档
      07-wechat-push.js       WeChat API
      08-x-push.js            X thread 生成与发布
  wechat-themes/
    base.css                  公共基础样式
    brief-emerald.css         Brief 翡翠绿主题
    essay-classic-blue.css    Essay 经典蓝主题
    themes.json               主题注册表（wechat_variant → css 文件）
  config/
    repo-scope.json           追踪的仓库列表 + 时间窗口
    model-config.json         默认模型配置
  docs/
    SPEC.md                   本文档
    ai-morning-report.plist   macOS launchd 配置
```

## Skills（核心知识层）

| Skill | 路径 | 用途 |
|-------|------|------|
| `ai-morning-report` | `.codex/skills/ai-morning-report/` | 调研范围、筛选标准、聚类规则 |
| `miraclefarms-writer` | `.codex/skills/miraclefarms-writer/` | GitHub.io brief/essay 写作规范 |
| `wechat-formatter` | `.codex/skills/wechat-formatter/` | 公众号语义改写规则 |

## 环境变量

| 变量 | 必填 | 说明 |
|------|------|------|
| `AI_CLI` | 否 | `claude`/`opencode`/`codex`，未设则默认 `opencode` |
| `OPENROUTER_API_KEY` | 否 | 封面图生成（无则跳过）|
| `COVER_IMAGE_MODEL` | 否 | 默认 `google/gemini-2.0-flash-exp:free` |
| `WECHAT_APPID` | 是* | 微信公众号 AppID |
| `WECHAT_APPSECRET` | 是* | 微信公众号 AppSecret |
| `WECHAT_THUMB_MEDIA_ID` | 否 | 无封面图时的备用 media_id |
| `WECHAT_CONTENT_SOURCE_BASE_URL` | 否 | 原文链接域名，默认 `https://miraclefarms.github.io` |
| `SKIP_WECHAT=1` | 否 | 跳过 Stage 6-7（只发 GitHub.io）|
| `ENABLE_X_PUSH=1` | 否 | 开启 Stage 8；默认关闭，不做 X 改写也不推送 |
| `X_USER_ACCESS_TOKEN` | 是** | X OAuth 2.0 user access token，需包含 `tweet.write` |
| `X_USERNAME` | 否 | 用于输出最终 X URL |
| `X_DRY_RUN=1` | 否 | 只生成并打印 thread，不实际发帖 |
| `X_POST_MODE` | 否 | `thread` 或 `single`，默认为 `thread` |
| `X_AI_TIMEOUT_MS` | 否 | X thread AI 生成超时，默认 `90000` |
| `X_FORCE_POST=1` | 否 | 忽略本地记录，强制重发 |

*`SKIP_WECHAT=1` 时可不填。
**`ENABLE_X_PUSH` 未设为 `1` 或 `X_DRY_RUN=1` 时可不填。

## 门控

| Stage | 门控条件 | 失败行为 |
|-------|---------|---------|
| 02-analyze | 素材包主题 < 2 | 停止整个流程 |
| 03-write | front matter 缺少 title/date/intro | 停止 |
| 04-cover | 图片生成失败 | 告警，继续 |
| 05-publish | git push 失败 | 停止（不推微信）|
| 08-x-push | X 生成或发布失败 | 告警，保留 GitHub.io/微信结果 |

## WeChat CSS 主题管理

`wechat-themes/themes.json` 定义 `wechat_variant → [css files]` 映射。
`wechat-renderer.js` 合并 CSS、解析 `var()` 为静态值后注入 `<style>` 块。
切换主题：修改 `themes.json` 映射，或新增 CSS 文件。

## macOS launchd 安装

```bash
cp ai-morning-report/docs/ai-morning-report.plist ~/Library/LaunchAgents/
# 编辑 plist 中的路径和环境变量
launchctl load ~/Library/LaunchAgents/ai-morning-report.plist
```

## 手动运行

```bash
# 全流程
./ai-morning-report/bin/run-daily.sh

# 只发 GitHub.io
SKIP_WECHAT=1 ./ai-morning-report/bin/run-daily.sh

# GitHub.io + 微信 + X dry-run（需要显式开启 X Stage）
ENABLE_X_PUSH=1 X_DRY_RUN=1 ./ai-morning-report/bin/run-daily.sh

# 手动预览某篇文章的 X thread
npm run publish:x -- _posts/2026-05-08-ai-infra-daily-brief.md --dry-run

# 手动预览单条摘要模式
npm run publish:x -- _posts/2026-05-08-ai-infra-daily-brief.md --dry-run --single

# 指定 CLI
AI_CLI=opencode ./ai-morning-report/bin/run-daily.sh

# 单独测试某个 stage
node ai-morning-report/src/stages/02-analyze.js 2026-05-06 \
  /tmp/morning-report/2026-05-06/raw-data.md \
  /tmp/morning-report/2026-05-06/material.md \
  .
```
