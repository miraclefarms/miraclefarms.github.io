# AI Morning Report LLM Pipeline Design

## 背景

MiracleFarms 需要一条真正可持续的“AI Infra 每日早报”自动化流水线，而不是简单的 GitHub 数据抓取和字符串拼接。目标产物是一篇每天定时生成、具备可读性和权威感的 AI Infra 早报，并同步发布到两个渠道：

- GitHub Pages：自动直推 `main`
- 微信公众号：推送到草稿箱，保留人工最终审核空间

现有 `ai-morning-report` 脚手架已经把流程拆成了 research/write/image/publish/wechat 几个阶段，但阶段边界仍然以“脚本执行顺序”为中心，而不是以“内容生产对象”为中心。结果是：

- 调研对象偏向 `release` 和 `commit`，没有把 PR 作为主对象
- 内容生成与渠道渲染耦合，GitHub 稿和微信稿在生成阶段就分叉
- 微信发布链路重复实现，没有复用仓库里更完整的发布器
- 临时图片、最终图片、渠道缩略图之间没有形成统一的资产闭环
- 失败时容易出现“上游失败、下游仍然成功”的假成功流水线

本设计的目标是把流程重构为“模板 + 大模型驱动”的内容生产系统，以结构化研究结果为基础，由 LLM 负责筛选、判断和成文，再由稳定的发布层投递到各个渠道。

## 目标

### 业务目标

- 每天定时自动调研一组固定 AI Infra 仓库的近期更新
- 主窗口以 `Asia/Shanghai` 的“当天”为核心，允许扩展到前 `1-3` 天
- 以 PR 为主对象筛选当天最值得关注的 `5-8` 条更新
- 由大模型总结每条更新的功能亮点、性能亮点和工程意义
- 按 MiracleFarms 的微信公众号日报模板生成正文、摘要和题图提示词
- 生成正式题图资产，并把图文并茂的文章投递到 GitHub Pages 与微信公众号草稿箱

### 质量目标

- 文章开篇必须先给判断，再展开事实和论据
- 每条重点更新都需要可追溯到官方证据
- 正文必须兼顾可读性、节奏感和工程可信度，不能退化为流水账
- 当某些信号不足时，系统必须显式保守，而不是自动补全未经证实的判断

### 运行目标

- GitHub 发布成功后，网站可自动上线
- 微信发布失败时，不回滚 GitHub 成果，但必须记录待补发状态
- 研究失败、筛选失败、成文失败时必须整条停止，不能产出模板化假日报

## 非目标

- 不直接自动群发微信公众号正式文章
- 不覆盖 essay 技术长文的生产流程
- 不构建通用 CMS 或跨站点发布平台
- 不在第一阶段引入数据库、消息队列或外部工作流编排系统

## 用户确认的产品决策

- 失败策略：研究和成文是硬门槛；题图或微信失败时允许降级为只发 GitHub，并记录待补发
- GitHub 发布策略：全自动直推 `main`
- 每日文章规模：精选 `5-8` 条最重要 PR，重点展开
- 如果当天高价值更新不足：允许从前 `2-3` 天窗口补充仍值得写的更新，避免凑数

## 设计原则

### 单一内容源

同一篇日报只能有一个事实源。GitHub 稿、微信稿、题图提示词、摘要、作者、分类都必须从同一个中间对象派生，而不是分别生成、再人工保持一致。

### PR 是一等公民

日报的主对象是“值得写的 PR”，而不是“仓库最近发生了什么”。`release` 和 `commit` 只作为证据补充，不作为默认选题主线。

### LLM 负责判断，不负责捏造事实

大模型应当用于：

- 对结构化候选项进行筛选和排序
- 总结每条候选项的价值
- 把结构化事实写成高可读性的文章
- 生成贴合主线判断的题图提示词

大模型不应当直接面对未经整理的原始 API 文本自由发挥，更不能在证据不足时“补完”事实。

### 内容生产与发布分离

研究、筛选、成文、渲染、发布必须是职责清晰的独立阶段。发布层只消费已经成型的文章对象，不承担内容判断职责。

### 失败显式化

所有关键阶段都要输出结构化状态和错误原因。系统允许降级，但不允许沉默失败。

## 目标架构

### 总体流程

```text
scheduler
  -> collect
  -> rank
  -> compose
  -> image
  -> render
  -> publish-github
  -> publish-wechat
```

### 阶段说明

#### 1. collect

负责从目标仓库收集候选素材，并输出结构化研究结果。

- 输入：
  - 仓库列表
  - 目标日期
  - 时间窗口配置（默认当天，必要时扩展到前 3 天）
- 输出：
  - `research.json`
  - `research.md`，供人工排查时阅读
- 主要行为：
  - 以 PR 为主抓取对象
  - 补充关联 release、关键 commit、必要标签和 diff 摘要
  - 标出性能相关、功能相关、稳定性相关、默认路径相关信号
  - 收集引用所需的官方链接

#### 2. rank

负责把候选素材筛选成“今天真正值得写的 `5-8` 条”。

- 输入：
  - `research.json`
- 输出：
  - `selection.json`
- 主要行为：
  - 优先选择当天更新
  - 当当天高价值项不足时，从前 `2-3` 天补充
  - LLM 对每条候选给出入选理由、主题分类、证据强度和重要性排序
  - 降低 docs-only、CI-only、测试-only 的优先级

#### 3. compose

负责生成“标准文章对象”，这是整条流水线的单一事实源。

- 输入：
  - `selection.json`
- 输出：
  - `article-source.json`
- 主要行为：
  - 生成标题、导语、段落结构、关键判断、引用列表、摘要
  - 生成与主线判断一致的题图提示词
  - 固定输出 MiracleFarms brief 模板所需的文章字段
  - 显式区分“事实”和“基于事实的判断”

#### 4. image

负责从 `article-source.json` 的题图提示词生成正式图片资产，并回填引用信息。

- 输入：
  - `article-source.json`
- 输出：
  - 仓库正式资产图片
  - 更新后的 `article-source.json`
- 主要行为：
  - 使用图像生成模型接口生成题图
  - 将题图存入正式资产目录，而不是临时目录
  - 把题图路径、mime type、生成时间回填到文章对象
  - 失败时允许降级，使用默认封面或无题图策略

#### 5. render

负责从同一文章对象渲染两个渠道版本。

- 输入：
  - 更新后的 `article-source.json`
- 输出：
  - GitHub 稿 `_posts/YYYY-MM-DD-slug.md`
  - 微信稿 `docs/wechat/YYYY-MM-DD-slug-wechat.md`
- 主要行为：
  - GitHub 稿遵守 MiracleFarms brief 格式规范
  - 微信稿保留相同内容主线，但调整为适合公众号阅读和后续 HTML 渲染的格式
  - 作者、摘要、分类、题图、引用、时间戳都由文章对象统一驱动

#### 6. publish-github

负责校验并发布 GitHub 稿及相关正式资产。

- 输入：
  - 渲染产物
- 输出：
  - Git 提交
  - push 到 `main`
  - `publish-record.json` 中的 GitHub 状态更新
- 主要行为：
  - 校验 front matter、引用、图片路径和文件完整性
  - stage 所有新建或已修改的日报相关文件
  - 成功后推送到 `main`

#### 7. publish-wechat

负责调用现有微信发布链路，把同一篇文章送入草稿箱。

- 输入：
  - 微信稿
  - 题图资产
  - 渠道元数据
- 输出：
  - 微信草稿
  - `publish-record.json` 中的 WeChat 状态更新
- 主要行为：
  - 复用现有 `scripts/publish-wechat.js`
  - 上传题图和正文图片
  - 渲染 HTML
  - 创建草稿并写入草稿 ID、发布时间和失败原因

## 中间产物设计

### `research.json`

用于表达原始研究结果，建议包含下列字段：

```json
{
  "target_date": "2026-04-27",
  "timezone": "Asia/Shanghai",
  "window": {
    "start": "2026-04-25T00:00:00+08:00",
    "end": "2026-04-27T23:59:59+08:00"
  },
  "repos": [
    {
      "name": "sgl-project/sglang",
      "items": [
        {
          "type": "pull_request",
          "number": 12345,
          "title": "Add X",
          "url": "https://github.com/...",
          "merged_at": "2026-04-27T10:15:00Z",
          "author": "alice",
          "labels": ["performance"],
          "summary": "Structured summary from source data",
          "signals": {
            "performance": true,
            "feature": true,
            "stability": false,
            "default_path": true
          },
          "evidence": [
            {
              "kind": "pr",
              "url": "https://github.com/..."
            }
          ]
        }
      ]
    }
  ]
}
```

### `selection.json`

用于表达 LLM 的筛选和排序结果。

```json
{
  "target_date": "2026-04-27",
  "selected": [
    {
      "rank": 1,
      "repo": "sgl-project/sglang",
      "pr_number": 12345,
      "why_selected": "Touches default runtime path and changes graph execution assumptions",
      "category": "runtime",
      "importance_score": 9.4,
      "evidence_strength": "high",
      "primary_angle": "工程路径变化"
    }
  ]
}
```

### `article-source.json`

这是渲染和发布唯一允许消费的文章对象。

```json
{
  "target_date": "2026-04-27",
  "kind": "brief",
  "title": "AI Infra 早报｜示例标题",
  "author": "荔枝不耐思",
  "category": "Brief",
  "series": "ai-infra-daily-brief",
  "intro": "一句话摘要",
  "thesis": "今天最值得记住的总判断",
  "cover_prompt": "Image generation prompt",
  "cover_asset": {
    "path": "/assets/...",
    "status": "generated"
  },
  "sections": [
    {
      "heading": "一、示例章节",
      "items": [
        {
          "title": "某个 PR",
          "what_changed": "事实",
          "why_it_matters": "判断",
          "references": [
            {
              "index": 1,
              "title": "Official PR",
              "url": "https://github.com/..."
            }
          ]
        }
      ]
    }
  ],
  "wechat": {
    "title": "AI Infra 早报｜示例标题",
    "author": "荔枝不耐思",
    "digest": "一句话摘要",
    "thumb_strategy": "generated-cover"
  }
}
```

### `publish-record.json`

用于支持幂等、重跑、补发和排障。

```json
{
  "target_date": "2026-04-27",
  "github": {
    "status": "published",
    "commit": "abc1234",
    "published_at": "2026-04-27T05:10:00+08:00"
  },
  "wechat": {
    "status": "pending_retry",
    "draft_media_id": null,
    "last_error": "uploadimg timeout",
    "last_attempt_at": "2026-04-27T05:13:00+08:00"
  }
}
```

## 时间窗口规则

- 所有时间计算严格使用 `Asia/Shanghai`
- 主判断对象是“目标日期当天发生的更新”
- 默认调研窗口为目标日期当天向前最多 3 天
- 排序时优先当天更新
- 只有在当天高价值更新不足时，才允许从前 `2-3` 天补充候选

这意味着窗口是“调研补足机制”，不是“任意三天摘要模式”。

## 文章模板策略

### 生成逻辑

文章模板以 WeChat brief 的阅读体验为主，再派生 GitHub 稿。这样做的原因是：

- 公众号是对节奏、段落和摘要最敏感的渠道
- GitHub 稿可以容纳更完整的引用呈现
- 如果先写成 GitHub 再机械压缩为微信，容易出现读感变差的问题

### 内容结构

一篇标准日报至少包含：

- 开篇判断段
- `2-4` 个主题分组章节
- 每个章节下若干重点 PR 分析
- 结尾判断段
- 完整参考来源

### 写作约束

- 开篇必须直接给总判断，不能先铺背景
- 每条重点更新必须同时回答“做了什么”和“为什么重要”
- 判断要明确边界，不能把推断写成确定事实
- 引用必须一一对应，不能复用错误编号
- 当证据不足以支撑强判断时，应使用保守措辞

## 题图与资产策略

### 正式资产路径

题图必须生成到仓库正式路径下，并作为可发布资产存在。不得把 `/tmp` 中的临时文件写入最终 Markdown。

建议资产路径：

- GitHub / Jekyll 可访问资产：`/assets/ai-infra-daily-brief/<date>-<slug>/cover.png`
- 微信渠道专用中间产物可以继续保留本地缓存，但不能作为最终引用路径

### 题图提示词来源

题图提示词必须从 `article-source.json` 的主线判断生成，而不是从关键词列表拼接。题图要服务于“今天的总判断”，而不是装饰页面。

### 失败降级

- 题图生成失败时：允许使用默认缩略图或无题图发布 GitHub
- 微信发布时：若缺少生成题图，可回退到默认 thumb 图
- 题图失败本身不阻塞 GitHub 正文发布

## 失败与降级策略

### 硬失败

以下阶段失败时，整条停止，不产出最终发布：

- `collect`
- `rank`
- `compose`
- `render`
- `publish-github`

### 软失败

以下阶段失败时，允许降级：

- `image`
  - GitHub 可继续发正文
  - WeChat 可使用默认 thumb 或延后补发
- `publish-wechat`
  - GitHub 结果保留
  - 记录为 `pending_retry`

### 明确禁止的行为

- 研究失败后继续产出模板早报
- 写作失败后发布旧稿或半成品
- GitHub 未成功推送时仍继续发微信草稿

## 代码组织建议

### 建议保留

- `scripts/publish-wechat.js`
- `scripts/lib/wechat-render.js`
- `scripts/lib/wechat-config.js`
- 现有微信题图 prompt 配置与相关测试

这些组件已经形成了比较完整的微信发布能力，应当作为稳定发布层保留。

### 建议重写

- `ai-morning-report/bin/run-daily.sh`
- `ai-morning-report/src/stages/01-research.js`
- `ai-morning-report/src/stages/02-write.js`
- `ai-morning-report/src/stages/03-images.js`
- `ai-morning-report/src/stages/04-publish.js`
- `ai-morning-report/src/stages/05-wechat.js`

原因是这些文件当前的职责划分建立在旧方案之上，继续修补会把错误的边界固化下来。

### 建议新增

- `ai-morning-report/src/stages/01-collect.js`
- `ai-morning-report/src/stages/02-rank.js`
- `ai-morning-report/src/stages/03-compose.js`
- `ai-morning-report/src/stages/04-image.js`
- `ai-morning-report/src/stages/05-render.js`
- `ai-morning-report/src/stages/06-publish-github.js`
- `ai-morning-report/src/stages/07-publish-wechat.js`
- `ai-morning-report/src/lib/date-window.js`
- `ai-morning-report/src/lib/github-research.js`
- `ai-morning-report/src/lib/article-schema.js`
- `ai-morning-report/src/lib/publish-record.js`

## 配置建议

### 仓库范围配置

继续保留 `repo-scope.json`，但其语义应当从“简单 repo 列表”扩展为：

- 仓库名单
- 关键词优先级
- 低优先级标签
- 默认窗口配置
- 每日目标条数范围

### 模型配置

模型配置需要按职责拆分，而不是一个默认模型通吃：

- `research_model`：用于摘要候选 PR
- `ranking_model`：用于筛选和排序
- `writing_model`：用于生成文章对象
- `image_model`：用于生成题图

不同阶段允许使用不同模型，以控制成本和输出质量。

## 观测与可运维性

### 运行日志

每次运行至少记录：

- target date
- time window
- 抓取候选数
- 入选条数
- 文章输出路径
- GitHub push 结果
- WeChat 发布结果
- 失败原因

### 幂等性

对同一 `target_date` 重跑时，系统应能识别：

- 已生成的 `research.json`
- 已发布的 GitHub 稿
- 已成功推送的微信草稿
- 需要补发的微信稿

### 补发能力

如果某天 GitHub 已成功、WeChat 失败，后续应允许仅执行 `publish-wechat` 重试，而不是整条重跑。

## 测试策略

### 单元测试

至少覆盖：

- 上海时区窗口计算
- 候选项筛选与优先级降权逻辑
- `selection.json` 和 `article-source.json` 的 schema 校验
- 渲染层的引用编号和 front matter 输出
- 发布记录的状态迁移

### 集成测试

至少覆盖：

- 从 `research.json` 到 GitHub / WeChat 渲染产物的整链 smoke test
- 题图生成失败后的降级路径
- WeChat 发布失败后 `pending_retry` 的状态记录
- 同一天重跑时对已跟踪文件的正确发布行为

### 回归测试

针对当前已暴露的问题，应新增回归测试保证：

- Stage 3/图像阶段必须真实执行
- 微信发布必须消费 HTML，而不是原始 Markdown
- 正式图片引用路径必须指向仓库内可发布资产
- GitHub 发布必须包含已跟踪文件的修改
- 研究失败不得继续进入成文阶段

## 迁移计划

建议分两步迁移：

### 第一步：搭新链，不立即删除旧链

- 新建基于 `collect/rank/compose/image/render/publish-*` 的完整链路
- 保留旧文件，但让新入口独立可跑
- 在测试和手动验证通过前，不直接删除旧实现

### 第二步：切流并清理

- 调度器切换到新入口
- 旧 `01-research` 到 `05-wechat` 标记废弃并移除
- 文档与运行说明同步更新

## 成功标准

以下条件同时成立时，视为本设计实现成功：

- 每日能稳定从目标仓库中筛出 `5-8` 条重点 PR
- 生成的文章具备明确开篇判断、完整引用和较强可读性
- GitHub 版本和 WeChat 版本来自同一个文章对象
- 题图是正式资产，不依赖临时目录
- GitHub 能全自动发布到 `main`
- WeChat 草稿箱能带作者、摘要、缩略图等完整元数据
- WeChat 失败时能记录待补发，而不是污染 GitHub 成果
- 关键阶段失败时不会产出模板化假成功日报
