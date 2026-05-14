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
wiki-src/docs/kvcache/
├── index.md                  # 综述首页
├── changelog.md              # 全局版本日志（本 skill 维护）
├── scanned_references.md     # 已扫描的主站文章清单（本 skill 维护）
├── first-principles.md       # 第一性原理
├── basics.md                 # 基础概念
├── attention-variants.md     # Attention 变体
├── sparsity.md               # KV 稀疏化
├── compression-quantization.md # KV 压缩与量化
├── runtime-architecture.md   # 运行时架构
├── storage-hierarchy.md      # 存储层级
├── paged-kv.md               # Paged KV
├── prefix-cache.md           # Prefix Cache
├── offload.md                # KV Offload
├── lifecycle.md              # 生命周期与淘汰
├── pd-disaggregation.md      # PD 分离中的 KVCache
├── parallelism.md            # 并行切分下的 KV 形态
├── routing.md                # 路由与亲和性
├── elasticity.md             # 弹性、故障与跨集群
├── workloads.md              # 工作负载维度
├── crossings.md              # 维度交叉
├── evaluation.md             # 评估方法
├── future.md                 # 开放问题与未来方向
├── frameworks.md             # 框架对比
├── glossary.md               # 术语表
└── references.md             # 参考资料（论文、框架文档、博客合集）
```

**务必维护的三个元文件：**

| 文件 | 用途 |
|------|------|
| `changelog.md` | 记录每次 wiki 更新的日期、涉及章节、变更摘要 |
| `scanned_references.md` | 记录已扫描过的主站文章路径 + 扫描日期，防止重复扫描 |
| `references.md` | 汇总所有引用（论文、框架文档、博客），每次更新追加新引用 |

---

## 3. 版本管理规范

### 3.1 每个 wiki 文件内的版本表

在文件末尾（正文之后、关联章节链接之前）维护版本历史：

```markdown
## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.1 | 2025-05-14 | 框架搭建 |
| v0.2 | 2025-05-20 | 纳入主站 essay: SGLang KVCache 架构解析 |
```

**版本号规则：**
- `v0.x` — 框架搭建 + 内容初填阶段
- `v1.0` — 内容基本完整，脱离"施工中"状态
- `v1.1+` — 增量修订
- `v2.0` — 框架级重写

### 3.2 内容更新标记

当向已有章节插入新内容时，在段落开头标注来源：

```markdown
> 主站文章 `_posts/2026-05-09-trtllm-kvcache-runtime-architecture.md`：
```

或简写为脚注形式：

```markdown
...此结论在 SGLang PR #24925 中得到了进一步验证。[^src]
[^src]: 主站 essay [SGLang KVCache Runtime 架构](/notes/2026/03/14/sglang-kvcache-runtime-architecture/) v1.1, §八.
```

### 3.3 全局 changelog.md 格式

```markdown
# KVCache Wiki 更新日志

## 2026-05-14 — Phase 0: 框架搭建

- 创建全部 23 个章节文件
- 完成综述首页与导航结构
- 各章节填充框架级内容

## 2026-05-20 — Phase 1: 初轮内容下沉

**变更范围：**
- `runtime-architecture.md` v0.1→v0.2：纳入 SGLang/vLLM/TRT-LLM 三层架构对比
- `prefix-cache.md` v0.1→v0.2：纳入 Radix Tree vs 链式哈希精确工程分析
- `offload.md` v0.1→v0.2：纳入 CXL 方案与 Beluga 实测数据

**已扫描文章：** 见 `scanned_references.md`
```

---

## 4. 参考文献追踪

### 4.1 scanned_references.md 格式

```markdown
# 已扫描参考文献清单

记录每个主站文章是否已被纳入 wiki。每次更新只扫描状态为"未扫描"的文章。

| 文章路径 | 发布日期 | kind | 扫描状态 | 扫描日期 | 纳入章节 |
|-----------|----------|------|----------|----------|----------|
| `_posts/2026-05-13-kvcache-prefix-matching-design.md` | 2026-05-13 | essay | ✅ 已纳入 | 2026-05-20 | prefix-cache, frameworks |
| `_posts/2026-05-13-cxl-kvcache-survey.md` | 2026-05-13 | essay | ⏳ 待扫描 | — | — |
```

### 4.2 扫描状态机

```
未记录 → ⏳ 待扫描 → 🔍 扫描中 → ✅ 已纳入 / ⏭️ 跳过（无 KV 相关内容）
```

**跳过条件：** 文章经内容审查后确认无 KV Cache 相关内容（如纯模型训练文章），标记为 `⏭️ 跳过` 并注明原因。

---

## 5. 内容收集流程（Subagent 驱动）

为提高效率，内容收集阶段使用 subagent 并行扫描。

### 5.1 Step 1: 确定扫描范围

1. 读取 `scanned_references.md`，获取已扫描文章集合
2. 用 glob 获取 `_posts/*.md` 全部文章列表
3. 取差集：`待扫描 = 全部 - 已扫描`
4. 按发布日期从早到晚排列

### 5.2 Step 2: 批量内容扫描（Subagent）

将待扫描文章分成 3-5 个批次，每批启动一个 general subagent 执行：

**Subagent prompt 模板：**

```
扫描以下主站文章，提取与 KV Cache 相关的内容。对每篇文章返回：

1. 文章路径、日期、kind
2. KV Cache 相关主题标签（从以下选择 1-3 个）：
   - attention-variants, sparsity, compression-quantization
   - runtime-architecture, storage-hierarchy, paged-kv, prefix-cache, offload, lifecycle
   - pd-disaggregation, parallelism, routing, elasticity
   - workloads, evaluation
   - frameworks (vLLM/SGLang/TRT-LLM/LMCache/Mooncake)
   - cxl, agent-kvcache, long-context, multimodal
3. 可提取的关键判断/数据（1-3 条，每条 <100 字）
4. 引用的 PR/论文/仓库链接
5. 是否有配图可用（架构图、benchmark 图）

文章列表：
_paths separated by newlines_
```

### 5.3 Step 3: 合并扫描结果

将所有 subagent 返回的结果合并为一个"内容映射表"：

| 主题标签 | 来源文章 | 可提取内容摘要 | 目标 wiki 章节 |
|----------|----------|----------------|----------------|
| prefix-cache | 2026-05-13-kvcache-prefix-matching-design.md | Radix Tree vs 链式哈希精确对比 | prefix-cache.md §3 |

---

## 6. Wiki 内容更新流程

### 6.1 更新的粒度

每次更新以"一个 wiki 章节"为单位，但同一轮可以更新多个章节。更新应遵循：

1. **打开目标 wiki 文件**，阅读当前内容
2. **查找内容映射表**中指向该章节的所有条目
3. **确定插入位置**：在哪个已有节下追加，或是否需要新增小节
4. **撰写内容**：用 wiki 的风格（客观、结构化、有表格/代码块）改写来源内容
5. **添加来源标注**：在新增内容末尾或以脚注形式标注来源文章
6. **更新文件内版本表**
7. **添加/修复交叉引用链接**

### 6.2 内容转换规则

| 来源类型 | 提取重点 | Wiki 中的形式 |
|----------|----------|---------------|
| essay（深度分析） | 核心判断、架构对比、工程数据 | 正文段落 + 对比表格 |
| reading（论文解读） | 论文方法、实验数据、适用条件 | "论文观察"小节 + 关键数字 |
| brief（日报） | PR 动态、release 信息 | "生态进展"小节 + 链接列表 |
| field-note | 研究笔记中的独到观察 | 引用整合到相关概念节 |

### 6.3 写作风格

- **客观、结构化**：使用表格、列表、代码块
- **避免"我们/笔者认为"**：直接陈述事实或标注来源
- **中文技术术语一致**：KVCache（不译）、Prefill/Decode（不译）、注意力（不译为"关注"）
- **数学用 LaTeX**：行内 `$...$`，独立公式用 `$$...$$`
- **引用链接**：使用 `[text](url)` 标准 Markdown 链接

---

## 7. 交叉引用与知识图谱构建

### 7.1 每轮更新时检查的链接类型

| 链接类型 | 示例 | 放置位置 |
|----------|------|----------|
| 概念依赖 | "详见 [Paged KV](paged-kv.md)" | 正文内首次提到时 |
| "另见"引用 | "量化与 paged 的交互见 [维度交叉 §3.1](crossings.md)" | 节末或页末 |
| 综述索引 | "本章属于[算法维度（横向）]，相邻章节：..." | 页首导语 |
| 主站文章链接 | "/notes/2026/.../" | 脚注或"延伸阅读"小节 |

### 7.2 确保链接有效

每次更新后，检查：
1. 新增的内部链接目标文件是否存在
2. 相对路径是否正确（所有内部链接使用相对于 `kvcache/` 目录的路径）
3. 主站链接使用 Jekyll permalink 格式：`/notes/YYYY/MM/DD/slug/`

### 7.3 双向链接

在引用目标章节中，也应添加反向引用。例如：
- `prefix-cache.md` 引用了 `paged-kv.md`
- → 在 `paged-kv.md` 中也应提及 prefix cache 的使用场景

---

## 8. 框架变更流程

当前 wiki 的四维框架（算法 / 系统 / 部署 / 工作负载）是经过设计的。如果发现框架不合理：

1. **识别问题**：具体哪个章节放错了位置？哪两个维度之间的界限模糊？哪些内容无处可放？
2. **提出方案**：说明建议的新增/合并/重组方案，以及理由
3. **等待确认**：不得在确认前执行框架级变更
4. **执行变更**：确认后同步更新：
   - 受影响的 md 文件内容
   - `mkdocs.yml` 的 nav 配置
   - `index.md` 的综述描述和阅读路径
   - `glossary.md`（如涉及术语变动）

---

## 9. 构建验证

每次更新完成后，必须验证 MkDocs 构建通过：

```bash
cd wiki-src && mkdocs build --strict
```

常见构建错误：
- 内部链接指向不存在的文件
- Markdown 语法错误（未闭合的代码块、表格格式错乱）
- mkdocs.yml 导航路径与实际文件不匹配

---

## 10. 初始化检查清单

首次使用本 skill 时，执行以下初始化：

- [ ] 确认 `changelog.md` 存在（不存在则创建，记录框架搭建为 Phase 0）
- [ ] 确认 `scanned_references.md` 存在（不存在则创建，标记所有已有主站文章的扫描状态）
- [ ] 确认所有 wiki 页面已有 `## 版本历史` 节（缺失则添加，标记为 v0.1）
- [ ] 执行首次内容扫描（覆盖所有已发布的相关主站文章）
- [ ] 执行首次内容更新
- [ ] 验证 `mkdocs build --strict` 通过

---

## 11. 自检清单

每轮更新完成后：

- [ ] `changelog.md` 已追加本轮变更记录
- [ ] `scanned_references.md` 已更新各文章扫描状态
- [ ] 所有修改过的 wiki 文件末尾版本表已更新
- [ ] 所有新增内容带有来源标注
- [ ] 内部交叉链接有效（相对路径正确）
- [ ] 主站文章链接使用 Jekyll permalink 格式
- [ ] `references.md` 已追加本轮新引用
- [ ] `mkdocs build --strict` 构建通过
