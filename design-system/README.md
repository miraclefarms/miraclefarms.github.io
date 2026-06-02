# MiracleFarms Design System Reference

这里是设计参考文档，**不参与 Jekyll 构建**（`_config.yml` 里 `docs/` 已 exclude），不会发布到线上。

## 文件说明

- **`design-tokens.json`** — 五板块配色的规范数据源（canonical source of truth）。新增或调整颜色时以这个文件为准，再同步到其他地方。
- **`design-tokens.css`** — 同一套配色的 CSS 自定义属性写法，附行内注释，便于快速查阅。

## 各端使用位置

| 端 | 文件 | 更新方式 |
|----|------|----------|
| GitHub.io 运行时 CSS | `assets/notion.css`（`:root` + `html[data-section="…"]` 段落） | 直接编辑；颜色值与 `design-tokens.json` 保持一致 |
| WeChat 渲染主题 | `miraclefarms-content/scripts/lib/wechat-render.js` | 通过 `wechat-theme` 字段名对应，见下表 |
| 首页板块导航卡片 | `assets/notion.css`（`.secnav-card[data-sec="…"]`） | 硬编码值，来源于各板块的 `accent` + `tint` |

### WeChat 主题名对照

| 板块 | `wechat-theme` | accent |
|------|---------------|--------|
| briefs | `emerald-green` | `#3a5a40` |
| readings | `slate-reading` | `#3d5168` |
| fields | `warm-ochre` | `#b08d3a` |
| essays | `ink-violet` | `#322a50` |
| foundations | `earth-clay` | `#9c5a3c` |

## 字段含义

| 字段 | 作用 |
|------|------|
| `accent` | 主色——stamp 背景、title rule、active chip、callout 边框 |
| `accent-soft` | 柔化变体——小图标、kicker 文字、h2-sub 标签 |
| `tint` | 背景底色——tag 背景、expressive 强度档的 headband |
| `quote-text` | tint 底色上的可读文字色 |
