# MiracleFarms

MiracleFarms 是一个以 AI Infrastructure 为核心的公开研究站点，使用 Jekyll 构建，部署于 GitHub Pages。

站点地址：`https://miraclefarms.github.io`

> **仓库说明**：本仓库只包含渲染层（Jekyll 模板、样式、部署配置）。文章内容、Wiki 源码和发布工具链存放在私有仓库 `miraclefarms/miraclefarms-content`，CI 构建时自动注入。

---

## 架构

```
miraclefarms.github.io (本仓库，公开)
├── _layouts/          Jekyll 模板
├── assets/css|icons/  共享样式与图标
├── *.md               索引页（briefs.md, essays.md …）
└── .github/workflows/pages.yml  构建 & 部署

miraclefarms/miraclefarms-content (私有)
├── _posts/            文章源文件
├── assets/            文章配图
├── wiki-src/          Wiki MkDocs 源码
└── …                  发布脚本、技能等
```

每次向 `miraclefarms-content` 推送内容，CI 自动触发本仓库重建：拉取私有内容 → 构建 wiki → Jekyll build → 部署到 GitHub Pages。

---

## 本地开发

本地预览需要先从 `miraclefarms-content` 复制内容：

```bash
# 将私有仓库的 _posts/ 和 assets/ 复制到本仓库
cp -r ../miraclefarms-content/_posts/* _posts/
rsync -a --exclude='css' --exclude='icons' ../miraclefarms-content/assets/ assets/

# 启动 Jekyll 本地预览
/usr/local/Cellar/ruby/4.0.2/bin/bundle install
/usr/local/Cellar/ruby/4.0.2/bin/bundle exec jekyll serve --host 127.0.0.1 --port 4000
```

> 如遇 Ruby/Bundler 版本问题，使用 `miraclefarms-build-debug` skill 诊断。

---

## 部署

推送到 `main` 分支后，`pages.yml` 自动完成：

1. Checkout 私有内容仓库（via `CONTENT_REPO_PAT` secret）
2. 注入 `_posts/` 和 `assets/`
3. 用 `mkdocs build` 编译 `wiki-src/` → `wiki/`
4. `bundle exec jekyll build --future`
5. 部署到 GitHub Pages

也可在 Actions 页面手动触发 `workflow_dispatch`。

> **⚠️ Pages 源必须是「GitHub Actions」，不能是「Deploy from a branch」。**
> 文章内容来自私有仓库、由 `pages.yml` 注入；如果 Pages 源退回到分支模式，GitHub 内置的 `pages build and deployment` 会在每次 push 本仓库时直接构建空的 `_posts/` 并部署，**导致整站文章消失**（线上 `/notes/...` 全 404、板块页列表全空）。
> - 自检：`gh api repos/miraclefarms/miraclefarms.github.io/pages --jq .build_type` 应为 `workflow`
> - 修复：`gh api -X PUT repos/.../pages -f build_type=workflow`，再 `gh workflow run pages.yml --ref main` 重新部署
> - push **content 仓库**不受此影响（走 `repository_dispatch`，只触发自定义 workflow）

---

## Wiki

Wiki 在 `/wiki/` 子路径下提供独立技术知识库，使用 MkDocs Material 构建。

- **Wiki 入口**：`https://miraclefarms.github.io/wiki/`
- **源码位置**：私有仓库 `miraclefarms-content/wiki-src/`
- **构建方式**：CI 每次构建时从源码现场生成，不提交产物

Wiki 本地预览：

```bash
cd ../miraclefarms-content/wiki-src
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdocs serve   # http://127.0.0.1:8000/
```
