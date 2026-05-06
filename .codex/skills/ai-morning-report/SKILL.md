---
name: ai-morning-report
description: >
  在固定的 AI Infra 仓库范围内搜索最近三天的重要提交、已合并 PR、release 和官方博客更新，并将高价值变化整理成每日早报素材。
  当用户要求生成“今日早报 / daily brief / repo 更新日报”、要求扫描指定仓库最近三天动态、或要求输出 GitHub.io / 微信公众号版本的 AI Infra 日报时触发。
  这个 skill 只负责限定调研范围、筛选重要变化和组织素材；具体写作格式、配图规则、GitHub.io 与微信公众号成稿规范统一交给 miraclefarms-writer。
---

# AI Morning Report

把“最近三天、固定 repo 范围内的重要变化”整理成一份可直接交给 `miraclefarms-writer` 的日报素材包。

## 工作边界

- 这个 skill 负责：确定日期范围、扫描指定 repo、筛选重要更新、做主题聚类、去重。
- 这个 skill 不负责：定义 front matter、引用格式、标题风格、配图规则、GitHub.io 正文格式、微信公众号正文格式。
- 需要成稿时，直接调用同仓库下 `.codex/skills/miraclefarms-writer/SKILL.md`。

## 快速流程

### 1. 确定范围

- 默认时间窗口：以 Asia/Shanghai 为准，回看最近三天的活动。
- 默认输出：GitHub.io 版日报。
- 如果用户明确要求微信公众号版，在 GitHub.io 版确定后，再让 `miraclefarms-writer` 生成公众号适配稿。
- 如果用户提供 repo 列表，优先使用用户列表；否则读取 `references/repo-scope.md` 中的默认范围。

### 2. 采集候选更新

对每个 repo 优先关注：

- release / release notes
- 已合并 PR
- 最近三天内高影响 commit
- 官方博客或设计文档更新

提取时回答四个问题：

1. 做了什么
2. 解决了什么问题
3. 影响了哪条主链路
4. 为什么今天值得写

### 3. 过滤低价值噪音

默认降权以下内容，除非它们揭示了更大的方向变化：

- 纯文档修订
- 机械重构、rename、lint、注释调整
- 单测补丁或 CI 修复
- 没有进入主链路的实验性改动

默认升权以下内容：

- 推理性能、稳定性、吞吐、延迟、内存、KV cache、MoE、EP、调度、量化、通信路径
- 新模型支持且明显影响生产可用性
- 故障恢复、容错、跨节点传输、部署架构
- 能代表项目路线变化的 design / API / runtime 改动

### 4. 聚类成“今天的几件事”

- 不要把所有 commit 平铺成流水账。
- 先整理成 3-6 个主题，再决定每个主题下引用哪些 repo 更新。
- 如果同一主题有多条更新，优先选最能支撑判断的 1-3 条，其余只作为补充证据。

### 5. 做最近日报去重

- 检查最近 2-3 天的相关 brief，避免把同一件事连续多天重复写成主线。
- 如果某个项目只是持续小修，而没有新的方向变化，可以降为一句补充，或直接跳过。

### 6. 交给 miraclefarms-writer 成稿

交付给 `miraclefarms-writer` 的最小输入应包含：

- 报道日期
- 输出渠道：GitHub.io，或 GitHub.io + 微信公众号
- 主题判断：今天真正值得写的 1-2 个主线
- 证据列表：标题、URL、为何重要、属于哪个主题
  - 如果要生成微信公众号版本，URL 不能在素材阶段丢失；`miraclefarms-writer` 需要把它们写进参考资料括号中
- 候选图片：如果某条来源里有高价值图，顺带标记

## 输出要求

- 如果用户只要 GitHub.io 版：产出一篇 repo-local brief 素材并交给 `miraclefarms-writer` 落到 `_posts/`
- 如果用户还要微信公众号版：在 GitHub.io 版确定后，再让 `miraclefarms-writer` 生成公众号适配稿；公众号正文主体不放超链接，但参考资料要在括号中保留完整 URL
- 不要在这里重复维护写作模板；写作规则只保留在 `miraclefarms-writer`
