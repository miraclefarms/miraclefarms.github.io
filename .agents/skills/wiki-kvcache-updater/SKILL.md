---
name: wiki-kvcache-updater
description: >
  遍历主站 brief / essay / reading / field-note 帖子，提取与 KV Cache 相关的最新进展、数据、论文和工程判断，
  增量更新 wiki-src/docs/kvcache/ 下的对应章节。每次更新记录版本历史与已扫描参考文献，逐步将 wiki 构建为知识图谱。
---

# Wiki KVCache 更新器

将主站 blog 内容系统性下沉到 KVCache Wiki，保持 Wiki 与主站的技术判断同步，
同时通过增量更新 + 版本管理 + 参考文献追踪，让 Wiki 成为可审计的长期知识资产。

---

## 1. 核心原则

1. **主站是 source of truth**：Wiki 内容必须能在主站文章中找到出处（essay/reading 为主，brief 为辅）。外部 PR/paper 可以作为补充引用，但不能代替主站已有判断。
2. **增量更新，不可回溯**：每次更新只扫描**上次更新之后新发布或新修改**的主站文章。已扫描过的文章记录在 `scanned_references.md` 中，下次跳过。
3. **版本可审计**：每个 wiki 文件维护自己的版本历史（仿 essay 版本表格式），全局维护 `changelog.md`。
4. **链接成图**：每轮更新时，在更新涉及的 wiki 文件之间新增/修复交叉引用链接，逐步形成知识图谱。
5. **框架变更需确认**：如需新增章节、合并章节、重组导航，必须在变更前说明理由并等待确认。

---

## 2. 文件体系

```
wiki-src/docs/kvcache/         # MkDocs 源文件（编辑区）
├── index.md                   # 综述首页
├── changelog.md               # 全局版本日志（本 skill 维护，不在 nav 中）
├── scanned_references.md      # 已扫描的主站文章清单（本 skill 维护，不在 nav 中）
├── first-principles.md        # 第一性原理
├── basics.md                  # 基础概念
├── attention-variants.md      # Attention 变体
├── sparsity.md                # KV 稀疏化
├── compression-quantization.md# KV 压缩与量化
├── runtime-architecture.md    # 运行时架构
├── storage-hierarchy.md       # 存储层级
├── paged-kv.md                # Paged KV
├── prefix-cache.md            # Prefix Cache
├── offload.md                 # KV Offload
├── lifecycle.md               # 生命周期与淘汰
├── pd-disaggregation.md       # PD 分离中的 KVCache
├── parallelism.md             # 并行切分下的 KV 形态
├── routing.md                 # 路由与亲和性
├── elasticity.md              # 弹性、故障与跨集群
├── workloads.md               # 工作负载维度
├── crossings.md               # 维度交叉
├── evaluation.md              # 评估方法
├── future.md                  # 开放问题与未来方向
├── frameworks.md              # 框架对比
├── glossary.md                # 术语表
└── references.md              # 参考资料（论文、框架文档、博客合集）

wiki/kvcache/                  # 构建输出（mkdocs build 生成，需一并提交）
```

**三个元文件（务必维护，不在 mkdocs.yml nav 中）：**

| 文件 | 用途 | Nav |
|------|------|-----|
| `changelog.md` | 记录每次 wiki 更新的日期、涉及章节、变更摘要 | 不暴露 |
| `scanned_references.md` | 记录已扫描过的主站文章路径 + 扫描日期，防止重复扫描 | 不暴露 |
| `references.md` | 汇总所有引用（论文、框架文档、博客），每次更新追加新引用 | 暴露，在 nav 末尾 |

> 元文件不在导航中是刻意设计：changelog 和 scanned_references 是维护工具链的一部分，非读者页面。`mkdocs build` 会输出 INFO 提示这两个文件未包含在 nav 中，这是预期行为，不是错误。

---

## 3. 版本管理规范

### 3.1 每个 wiki 文件内的版本表

在文件的最末尾（所有正文和交叉引用之后）维护版本历史：

```markdown
## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2026-05-14 | 框架搭建 |
| v0.2 | 2026-05-14 | 纳入主站 essay: SGLang KVCache 架构解析 |
```

**版本号规则：**
- `v0.x` — 框架搭建 + 内容初填阶段
- `v1.0` — 内容基本完整，脱离"施工中"状态
- `v1.1+` — 增量修订
- `v2.0` — 框架级重写

### 3.2 内容来源标注

新增内容末尾使用统一的来源标注格式（实践中三种可选，首选第一种）：

**首选：段落末尾来源行：**
```markdown
...技术描述结束。

来源：主站 essay [文章标题](/notes/YYYY/MM/DD/slug/)
```

**备选：多来源汇总（同一段内容引用多篇文章时）：**
```markdown
来源：主站 essay [A](/notes/.../)，主站 brief [B](/notes/.../)
```

**备选：段落前引注（引用内容为整段或多段时）：**
```markdown
> 以下内容来自主站 essay [SGLang KVCache Runtime 架构](/notes/2026/03/14/sglang-kvcache-runtime-architecture/) v1.1。
```

实际实践中，段落末尾来源行最简洁且不打断阅读流，是首选格式。

### 3.3 全局 changelog.md 格式

按 Phase 组织，每轮更新追加一个新 Phase：

```markdown
# KVCache Wiki 更新日志

## YYYY-MM-DD — Phase N: 标题

**说明：** 本轮更新的目标和上下文（1-2 句）。

**涉及章节：**（表格）

| 章节 | 版本 | 主要变更 |
|------|------|----------|
| prefix-cache.md | v0.1→v0.2 | +xxx |

**未修改章节：**（列表）
**已扫描文章：** 见 `scanned_references.md`
```

---

## 4. 参考文献追踪

### 4.1 scanned_references.md 格式

按 kind 分组，每篇文章一行。实践格式：

```markdown
# 已扫描参考文献清单

## 扫描状态说明
| 状态 | 含义 |
|------|------|
| ⏳ 待扫描 | 尚未扫描 |
| ✅ 已纳入 | 内容已提取，已更新到 wiki |
| ⏭️ 跳过 | 审查后确认无 KV Cache 相关内容（注明原因） |

## Phase N 扫描结果 (YYYY-MM-DD)

### Essay
| 文章路径 | 日期 | 扫描状态 | 扫描日期 | 纳入章节 |
|----------|------|----------|----------|----------|
| `_posts/2026-xxx.md` | 2026-xx | ✅ 已纳入 | 2026-xx | chapter-a, chapter-b |

### Reading
| ... | ... | ... | ... | ... |

### Brief
| ... | ... | ... | ... | ... |
```

**跳过条件：** 文章经 subagent 内容审查后确认无 LLM KV Cache 相关内容（如 GPU embedding 哈希表、视频扩散 KV、站点元文章），标记为 `⏭️ 跳过` 并注明原因（如"非 LLM Attention KV Cache"）。

### 4.2 扫描状态机

```
未记录 → ⏳ 待扫描 → 🔍 扫描中 → ✅ 已纳入 / ⏭️ 跳过
```

---

## 5. 内容收集流程（Subagent 驱动）

### 5.1 Step 1: 确定扫描范围

1. 读取 `scanned_references.md`，获取已扫描文章集合
2. 用 glob 获取 `_posts/*.md` 全部文章列表
3. 取差集：`待扫描 = 全部 - 已扫描`
4. **按 kind 分组，按日期排列**

### 5.2 Step 2: 分批并行扫描（4 个 Subagent）

待扫描文章按 kind 分入 4 个 subagent，同时启动：

| Subagent | 负责 kind | 任务侧重 |
|----------|----------|----------|
| **Agent 1** | essay | 深度技术判断、架构对比、生产数据 |
| **Agent 2** | reading | 论文方法、实验数据、适用条件 |
| **Agent 3** | brief | PR 动态、release 信息、生态新闻 |
| **Agent 4** | field-note + 发现 | 研究笔记 + **全局 grep 发现遗漏文章** |

Agent 4 的发现任务很重要：用 grep 搜索 `_posts/*.md` 中匹配 `kvcache|KV cache|KVCache|prefix cache|PagedAttention` 关键词的文件，对照已有清单找出遗漏文章，追加到扫描列表。

**Subagent prompt 模板（以 essay 为例）：**

```
你是主站文章扫描器，正在为 KVCache Wiki 提取内容。逐个读取以下文件，对每篇文章返回：

1. 文件路径、日期、kind
2. 主题标签（选 1-3 个，从固定标签表选择）
3. 1-3 条关键判断或数据点（每条 <100 字，含具体数字/比率/commit）
4. 引用的 PR 号、arxiv ID、GitHub 仓库
5. 文中提到的架构图或 benchmark 图表

固定主题标签表：
- attention-variants, sparsity, compression-quantization
- runtime-architecture, storage-hierarchy, paged-kv, prefix-cache, offload, lifecycle
- pd-disaggregation, parallelism, routing, elasticity
- workloads, evaluation
- frameworks (vLLM/SGLang/TRT-LLM/LMCache/Mooncake)
- cxl, agent-kvcache, long-context, multimodal

文章列表：
<paths>
```

### 5.3 Step 3: 合并扫描结果，构建内容映射表

将 4 个 subagent 的返回结果合并为统一的内容映射表：

| 主题标签 | 来源文章 | 可提取内容摘要 | 目标 wiki 章节 |
|----------|----------|----------------|----------------|
| prefix-cache | 2026-05-13-kvcache-prefix-matching-design.md | Radix Tree vs 链式哈希 vs 两阶段 Claim 对比 | prefix-cache.md |
| ... | ... | ... | ... |

按**目标 wiki 章节**聚合同一章节的所有条目，作为编辑阶段的输入。

---

## 6. Wiki 内容更新流程

### 6.1 更新策略

1. **先读后写**：打开目标 wiki 文件，理解当前结构后再编辑
2. **按章节聚合**：内容映射表中指向同一 wiki 章节的多个来源应合并编写，而非逐条追加
3. **并行编辑**：同一轮更新可以并行处理多个章节文件（使用并行的 Edit 工具调用），但每个文件的编辑需要保证一致性
4. **不在同一文件中并发编辑**：避免同一文件的两个并行编辑产生冲突

### 6.2 内容转换规则

| 来源类型 | 提取重点 | Wiki 中的形式 | 优先级 |
|----------|----------|---------------|--------|
| essay | 核心判断、架构对比、生产数据 | 正文段落 + 对比表格 | 最高 |
| reading | 论文方法、实验数字、适用边界 | 小节 + 关键数字 | 高 |
| brief | PR 动态、release 版本 | 列表 + 链接（聚合到 frameworks.md 为主） | 中 |
| field-note | 独到观察 | 引用整合 | 低 |

### 6.3 写作风格

- **客观陈述**：直接陈述事实，不写"我们/笔者认为"
- **结构化优先**：表格 > 列表 > 段落。能用表格对比的不要写长段落
- **数字前置**：关键性能数字放在句首或表头，阅读更高效
- **来源可见**：每条独立事实标注来源，让读者可以溯源到主站文章
- **中文术语一致**：KVCache（不译）、Prefill/Decode（不译）、注意力（不译为"关注"）
- **数学用 LaTeX**：行内 `$...$`，独立公式用 `$$...$$`
- **链接用标准 Markdown**：`[text](url)`

### 6.4 质量底线

每次更新保证：
- 新增内容不超过 wiki 页面的 50%（避免页面过长）
- 每个新增大段有来源标注
- 每个新增段内的数字和 PR 号与原始文章一致（可抽样核对）
- 交叉引用链接新增或修复至少 2 处（同一轮更新涉及的页面之间）

---

## 7. 交叉引用与知识图谱构建

### 7.1 每轮更新时检查的链接类型

| 链接类型 | 示例 | 放置位置 |
|----------|------|----------|
| 概念依赖 | "详见 [Paged KV](paged-kv.md)" | 正文内首次提到时 |
| "另见"引用 | "量化与 paged 的交互见 [维度交叉](crossings.md)" | 节末或页末的"关联章节" |
| 主站文章链接 | "/notes/2026/..." | 来源标注末尾 |

### 7.2 确保链接有效

1. 内部链接目标文件必须存在（mkdocs 构建会检验）
2. 内部链接使用相对于 `kvcache/` 目录的路径 → `[text](target-file.md)`
3. 主站链接使用 Jekyll permalink 格式 → `/notes/YYYY/MM/DD/slug/`

### 7.3 关联章节节

每个更新过的 wiki 页面末尾（版本表之前）应包含"关联章节"节：

```markdown
## 关联章节

- xxx 的物理基础：[Paged KV](paged-kv.md)
- 分布式 KV 池的关系：[PD 分离](pd-disaggregation.md)
- ...
```

关联章节的目标页面不一定需要同时添加反向链接（这是 best-effort 目标，实践中允许单向引用），但若同一轮更新也编辑了目标页面，则必须双向链接。

---

## 8. Brief 批量处理策略

Brief 数量大（~30 篇/月），且每篇 KV 相关内容少（通常 1-3 条 PR 动态），需要高效处理：

| 策略 | 说明 |
|------|------|
| **聚合写入** | 多个 brief 的 PR 更新聚合到 frameworks.md 的对应框架小节，不逐篇展开 |
| **按月批处理** | 每轮更新处理一个月的 brief，一次性更新 frameworks.md 和受影响的主题页 |
| **仅取标记性事件** | 跳过纯 bugfix/CI/重构类 PR，只取新功能、性能突破、架构变更类 PR |
| **记录扫描完成** | brief 扫描后仍然标记为 ✅ 已纳入（即使很多条目被过滤），避免下轮重复扫描 |

Phase 1 的经验：10 篇精选 brief 聚合处理后，frameworks.md 各框架小节获得了实质性的 PR 列表。后续 Phase 可以按月批量处理积压的 ~40 篇 brief。

---

## 9. 构建与提交

### 9.1 构建验证

MkDocs 环境位于 `wiki-src/.venv/`，构建命令：

```bash
cd wiki-src && source .venv/bin/activate && mkdocs build --strict
```

**构建输出的预期行为：**
- `INFO - Documentation built in X seconds` → 成功
- `The following pages exist but are not included in the "nav" configuration: kvcache/changelog.md, kvcache/scanned_references.md` → 预期，这两个是维护元文件
- `contains an absolute link '/notes/...', it was left as is.` → 预期，主站链接不转换为相对路径
- 任何 ERROR 或 WARNING → 必须修复后再提交

### 9.2 提交

编辑完成后：
1. 用 `git status` 确认只涉及 wiki 相关文件的变更
2. 只 add wiki 相关路径：`.agents/skills/wiki-kvcache-updater/`、`wiki-src/docs/kvcache/`
3. 构建输出 `wiki/` 在 mkdocs build 后自动生成，也需要一并 add 和提交
4. Commit message 格式：`wiki(kvcache): Phase N — <一句话摘要>`

```bash
# 提交示例
git add .agents/skills/wiki-kvcache-updater/ wiki-src/docs/kvcache/ wiki/
git commit -m "wiki(kvcache): Phase 2 — brief monthly batch, frameworks update"
git push
```

---

## 10. 初始化检查清单

首次使用本 skill 时，执行以下初始化：

- [ ] 确认 `changelog.md` 存在（不存在则创建，记录框架搭建为 Phase 0）
- [ ] 确认 `scanned_references.md` 存在（不存在则创建，按 kind 分组列出所有主站文章，状态标记为 ⏳）
- [ ] 确认所有 wiki 页面已有 `## 版本历史` 节（缺失则用 bash 脚本批量追加 v0.1）
- [ ] 执行首次内容扫描（覆盖所有已发布的相关主站文章）
- [ ] 执行首次内容更新
- [ ] 验证 `mkdocs build --strict` 通过
- [ ] `git add` 并提交

---

## 11. 自检清单

每轮更新完成后，人工确认：

- [ ] `changelog.md` 已追加本轮 Phase 记录，含涉及章节表格
- [ ] `scanned_references.md` 已更新所有本轮扫描文章的扫描状态
- [ ] 所有修改过的 wiki 文件末尾版本表已更新（`v0.x→v0.y`）
- [ ] 新增段落末尾有来源标注（`来源：主站 essay/reading/brief [标题](link)`）
- [ ] 修改的每个文件末尾有/更新了"关联章节"节
- [ ] 内部交叉链接目标文件存在，路径正确
- [ ] 主站文章链接使用 `/notes/YYYY/MM/DD/slug/` 格式
- [ ] `references.md` 已追加本轮新引用的论文（arxiv ID + 一句话描述）
- [ ] `mkdocs build --strict` 构建通过（零 ERROR，预期内 INFO 除外）
- [ ] `git status` 确认无无意中 staged 的文件（x-formatter/claude settings 等）
