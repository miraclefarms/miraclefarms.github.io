# MiracleFarms 搭建手记：一个 AI Infra 研究站的从零到一

> 写于 2026 年 5 月 18 日。追溯这六天的 git 历史，整理搭建过程中的每一次翻车与修复。

---

## 一、起点：为什么是 Jekyll + GitHub Pages

MiracleFarms 的定位是"AI Infrastructure 公开研究站"——用人类可读的文字记录 AI 推理系统的技术细节，同时作为一个 AI Agent 协作发布的试验场。

选型没有过多纠结：

- **GitHub Pages** 免费、可靠、自带 CDN，URL 直接就是 `miraclefarms.github.io`
- **Jekyll** 是 GitHub Pages 原生支持的静态站生成器，Markdown 写作，无需数据库
- **Jekyll 4.2.2** 而非 GitHub Pages 默认捆绑的 Jekyll 3.x——因为需要 Kramdown 的一些新特性和更好的 Liquid 表达式支持

这个选型在后来埋下了第一个坑。

## 二、第一天：全量上线，然后发现站点是空的

**2026-05-13 00:26** 的第一个 commit 信息是：

```
feat(essay): add KV cache prefix matching algorithm deep-dive
```

这个 commit 一口气推了整个站点的基础架构：

- `_config.yml`、`Gemfile`、所有 `_layouts/`
- GitHub Actions CI 工作流 `.github/workflows/pages.yml`
- 三套 Agent 技能：`miraclefarms-writer`（写作）、`ai-morning-report`（早报）、`miraclefarms-build-debug`（构建调试）
- 第一篇正式文章：《KV Cache 前缀匹配算法深度对比》

上线后，本地 `bundle exec jekyll build` 没有问题，文章渲染正常。但 GitHub Pages 上的站点——空的。

定位过程：GitHub Actions 日志显示构建"成功"，`_site/` 产物上传，部署完成。唯独文章不见了。

原因找到了：

**CI 工作流用的是 `actions/jekyll-build-pages@v1`**，这是 GitHub 官方提供的 Jekyll 构建 Action。看起来很正规，实际上它内部硬编码了 `github-pages` gem，而 `github-pages` gem 把 Jekyll 版本钉死在 3.x。

Gemfile 里指定的 `jekyll ~> 4.2.2` 被完全无视了。Jekyll 3 和 Jekyll 4 的 Liquid 渲染存在若干不兼容，导致文章模板解析失败，但 Action 没有抛出任何 Error，只是静默地输出了一个没有文章的空站。

这是一个**静默失败**（silent failure），比报错更难排查。

---

## 三、三天内容爆炸期（5月13日 - 5月15日）

在搞清楚 CI 问题之前，内容生产其实已经并行推进：

- **5月13日**：AI Infra 早报（第1期）、KV Cache 综述论文阅读、CXL KVCache 调研文章
- **5月14日**：第2期早报、SGLang / TRT-LLM / vLLM Field Notes 三篇、KV Cache Wiki 知识库 Phase 1、Fields 板块上线
- **5月15日**：第3期早报、vLLM Field Note、AI 生成封面图工作流

其中 Wiki 的上线走了一个弯路。最初的做法是把 MkDocs 生成的静态 HTML 直接提交到仓库（`chore: build wiki` 连续四次），这带来一个问题：wiki 产物是构建物，不是源码，提交进仓库既污染了 git 历史，又让仓库体积不断膨胀。后来这个决策被推翻，wiki 的构建被移入 CI 流程（见第五节）。

WeChat 封面图工作流也经历了两轮迭代：

1. 最初调用 Gemini 图像生成 API，timeout 设置过短，直接报错
2. 修复 timeout 后，发现封面图 prompt 把文章标题重复了两次，改为只用 `title + intro` 生成
3. 封面图风格来回切换了三次才稳定

这些都是内容工具链的磨合成本。

---

## 四、内容与渲染的分离（5月16日）

随着文章量增长——五天内已有约 20 篇文章、数十张配图——一个设计问题浮出水面：

**所有内容（文章、脚本、技能）和渲染层（模板、样式、CI 配置）混在同一个公开仓库里。**

这意味着：
- 未发布的草稿、内部工具脚本、API 密钥配置示例都暴露在公网
- 每次更新内容都要触发整个渲染层的 CI，反之亦然，耦合严重
- 内容仓库无法独立管理权限

决策：做**前后端分离**——渲染层（公开）和内容层（私有）拆成两个仓库。

拆分前先打了一个安全快照：

```
chore: snapshot before frontend/backend separation
Records full repo state including plans and opencode config.
This commit is the safe restore point before splitting content
into a private repo.
Restore point: git checkout backup/pre-separation-20260516
```

然后执行分离：

- **公开仓库 `miraclefarms.github.io`**：保留 Jekyll 模板、样式、CI 配置、索引页
- **私有仓库 `miraclefarms-content`**：迁入所有文章（`_posts/`）、配图（`assets/`）、发布脚本、技能文件

CI 工作流随之改造，新增私有仓库注入步骤：

```yaml
- name: Checkout private content repo
  uses: actions/checkout@v4
  with:
    repository: miraclefarms/miraclefarms-content
    token: ${{ secrets.CONTENT_REPO_PAT }}
    path: _private

- name: Inject posts and assets
  run: |
    cp -r _private/_posts/. _posts/
    rsync -a --exclude='css' --exclude='icons' _private/assets/ assets/
```

每次向 `miraclefarms-content` 推送内容，通过 `repository_dispatch` 事件触发公开仓库重建，全程自动化。

---

## 五、同一天的 CI 修复

公私分离完成后，顺手修掉了第一天埋下的 Jekyll 版本坑：

```
fix(ci): replace jekyll-build-pages with ruby/setup-ruby

jekyll-build-pages@v1 uses the github-pages gem which pins Jekyll to
3.x, conflicting with Gemfile's jekyll ~> 4.2.2 and causing all posts
to be silently dropped from the build.

Switch to ruby/setup-ruby + bundle exec jekyll build --future so the
build respects the project Gemfile directly.
```

修复方案直接：丢弃 `actions/jekyll-build-pages@v1`，换成标准 Ruby 工具链：

```yaml
- name: Setup Ruby
  uses: ruby/setup-ruby@v1
  with:
    ruby-version: '3.3'
    bundler-cache: true

- name: Build with Jekyll
  run: bundle exec jekyll build --future
  env:
    JEKYLL_ENV: production
```

`--future` 参数是必须的——站点使用 `+0800` 时区，文章 `date` 通常是当天早上 08:00，如果不加 `--future`，UTC 时间下的 CI 会把"未来"的文章全部过滤掉。

同一天还把 Wiki 的构建也移入 CI：

```yaml
- name: Setup Python for wiki build
  uses: actions/setup-python@v5
  with:
    python-version: '3.x'

- name: Build wiki
  run: |
    pip install mkdocs-material --quiet
    cd _private/wiki-src && mkdocs build --clean --site-dir ../../wiki
```

Wiki 源码在私有仓库，CI 构建后输出到 `wiki/`，Jekyll 把它当作静态文件直接复制进 `_site/`。不再提交构建产物，git 历史干净了。

---

## 六、互动层（5月18日）

结构稳定后，开始加互动功能：

- **Giscus 评论**：基于 GitHub Discussions，无需自建服务器，评论数据存在 GitHub 上
- **读者高亮**：JS 实现的段落高亮 + 本地存储，可以删除已高亮内容
- **微信公众号 QR 码**：文章末尾 endcap，方便读者关注公众号

其中读者高亮经历了两次修复：第一版上线后发现删除按钮在选中状态下不显示（`fix: show delete action for selected highlights`），这是 CSS 优先级被覆盖导致的。

---

## 七、当前架构与经验总结

六天后，站点的完整架构是：

```
miraclefarms.github.io（公开，渲染层）
├── _layouts/           Jekyll 模板
├── assets/css|icons/  样式与图标
├── *.md                索引页（briefs/essays/readings/fields）
└── .github/workflows/pages.yml   构建 & 部署

miraclefarms-content（私有，内容层）
├── _posts/             文章源文件
├── assets/             文章配图
├── wiki-src/           Wiki MkDocs 源码
└── scripts/            发布脚本、AI 工具链
```

触发链路：推送 `miraclefarms-content` → `repository_dispatch` → CI 拉取私有仓库 → 注入内容 → Wiki 现场构建 → Jekyll build → 部署 GitHub Pages。

**踩坑经验**，按重要程度排列：

### 1. 不要用 `actions/jekyll-build-pages`

这个 Action 的名字具有迷惑性——它不是"用 Jekyll 构建 Pages"，而是"用一个内置了旧版 Jekyll 的固定环境构建 Pages"。如果你的 Gemfile 指定了 Jekyll 4.x，这个 Action 会静默地用 Jekyll 3.x 构建，不报错，只是文章全部消失。

替代方案：`ruby/setup-ruby@v1` + `bundle exec jekyll build`。

### 2. 静默失败比崩溃更危险

Jekyll 3 解析 Jekyll 4 的模板不报错，只是输出空站。排查这类问题要养成习惯：**检查 `_site/posts/` 目录是否真的有产物**，而不是只看 CI 状态是否绿色。

### 3. `--future` 不能省

UTC 环境的 CI 构建，如果不加 `--future`，东八区当天早上发布的文章会因为"还没到时间"而被过滤。本地预览（也在东八区）不会复现这个问题，是典型的"本地正常，线上出错"。

### 4. 构建产物不要提交

Wiki 的 HTML 产物一度被提交进仓库，每次更新都是几百个文件变更，git 历史完全没有意义。正确做法是 CI 现场构建，仓库只保留源码。

### 5. 内容与渲染应该从一开始就分离

公私分离的工程量不小——不只是移动文件，还要改造 CI、配置 PAT Secret、更新本地开发流程。如果从第一天就分离，成本远低于在内容积累后再分离。

### 6. `+0800` 时区贯穿所有 date 字段

Jekyll 的 date 处理逻辑对时区敏感。文章 front matter、`_config.yml` 中的 `timezone: Asia/Shanghai`、CI 的 `--future` 三者必须一致，否则任何一个环节都可能造成文章不显示。

---

*MiracleFarms — AI Infrastructure in public*
*https://miraclefarms.github.io*
