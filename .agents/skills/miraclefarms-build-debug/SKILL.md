---
name: miraclefarms-build-debug
description: Set up the local Jekyll environment and diagnose or fix MiracleFarms site build problems. Use when working in miraclefarms.github.io and you need to install Bundler/Jekyll dependencies, run `bundle exec jekyll build` or `serve`, investigate Liquid/Markdown/front matter failures in `_posts/`, or resolve output conflicts and config issues in `_config.yml`.
---

# MiracleFarms Build Debug

## Overview

Use this skill to make the repo locally buildable first, then locate content or config problems from real Jekyll output instead of guessing from static inspection.

---

## 第一步：诊断 Ruby 环境

> **平台说明**：本节描述的是 macOS（Homebrew）上的路径和版本。其他 Linux 发行版、Windows（WSL）等请参考思路，用项目级 gem 安装而非系统 gem，具体路径和版本号根据实际环境调整。

macOS 自带系统 Ruby（`/usr/bin/ruby`）版本为 2.6.10，过老，不兼容本项目 Gemfile.lock 要求的 bundler 版本。必须使用 Homebrew 安装的 Ruby。

### 诊断命令

```bash
ruby -v
bundle -v
which bundle
```

**正确环境应该显示**：
- `ruby 4.0.2`（通过 `/usr/local/Cellar/ruby/4.0.2/bin/ruby` 或 PATH 中的 `ruby`）
- bundler 版本号（如 `4.0.10`）

**错误症状**：
- `ruby -v` 显示 `2.6.10` → 系统 Ruby 在劫持 PATH
- `bundle exec jekyll build` 报错 `Could not find 'bundler' (2.6.9)` → bundler 版本不匹配
- bundler 报错 `undefined method 'untaint'` → bundler 版本与 Ruby 版本不兼容（bundler 1.x 不兼容 Ruby 4.x）

### 修复步骤

1. **确认 Homebrew Ruby 4.0.2 路径**：
   ```bash
   ls /usr/local/Cellar/ruby/4.0.2/bin/
   ```

2. **安装匹配版本的 bundler**（如果报错 `undefined method 'untaint'`）：
   ```bash
   /usr/local/Cellar/ruby/4.0.2/bin/gem install bundler
   ```
   这会安装 bundler 4.0.10，与 Ruby 4.0.2 兼容。

3. **修复 Gemfile.lock 的 BUNDLED WITH 版本**：
   编辑 `Gemfile.lock`，将 `BUNDLED WITH` 改为实际安装的 bundler 版本（如 `4.0.10`）。

4. **安装依赖到 vendor/bundle**：
   ```bash
   /usr/local/Cellar/ruby/4.0.2/bin/bundle install
   ```

5. **验证**：
   ```bash
   /usr/local/Cellar/ruby/4.0.2/bin/bundle exec jekyll build
   ```

---

## 第二步：构建验证

确认环境正常后，用 Jekyll build 作为最终依据。

```bash
/usr/local/Cellar/ruby/4.0.2/bin/bundle exec jekyll build
```

**成功标志**：输出包含 `done in X seconds`，无 `Error:` 行。

---

## 第三步：本地预览

只在 `jekyll build` 通过后才启动预览。

```bash
/usr/local/Cellar/ruby/4.0.2/bin/bundle exec jekyll serve --host 127.0.0.1 --port 4000
```

用 `curl -s http://127.0.0.1:4000/notes/2026/04/21/turboquant-vllm-sglang-trtllm-integration/ | grep -o '<title>[^<]*</title>'` 验证文章页面渲染正常。

---

## 常见构建失败

### Liquid 解析错误

`bundle exec jekyll build` 的错误是最终依据。常见失败原因：Jekyll 会把文章里的 `{% generation %}` 这样的文本当作 Liquid 标签解析，即使它在行内代码块里。

处理方式：
- 重写为纯文本，如 ``generation block``
- 或用 Liquid raw 块包裹保留原始语法

不要以为 Markdown 反引号能阻止 Liquid 解析。

### Brief 和 Essay 尾部格式

修复构建问题时顺手规范化文章结尾：
- `brief` 文章：`---` 分隔符 + `## 参考来源`
- `essay` 文章：`---` 分隔符 + `## 参考资料`

缺失分隔符或章节名不一定导致构建失败，但违反 repo 规范，发现时应修复。

### 输出路径冲突

如果 Jekyll 警告多个文件写入同一目标，先检查是否有重复页面和 permalink 冲突。本项目中 `docs/` 是规划文档目录，不应作为 Jekyll 页面渲染。

如 `docs/` 下文件与根目录 `briefs.md`、`essays.md` 等冲突，需在 `_config.yml` 中 exclude `docs`。

### 非阻塞警告

以下警告可忽略（构建仍返回 exit 0）：
- `Logger not initialized properly`
- `Jekyll::Stevenson#initialize: does not call super probably`

除非用户明确要求升级 Ruby/Jekyll 栈，否则不要花时间追这些。

---

## 验证清单

完成后按顺序确认：

1. `bundle exec jekyll build` → exit 0
2. 启动 `jekyll serve` → 端口 4000 可访问
3. 确认构建错误是 hard error 还是 warning 或 repo 规范问题
4. 报告修改了哪些文件，以及哪些已知警告可以忽略

---

## 命令速查

```bash
# 诊断 Ruby 环境
/usr/local/Cellar/ruby/4.0.2/bin/ruby -v
/usr/local/Cellar/ruby/4.0.2/bin/bundle -v

# 修复 bundler 版本不匹配（Gemfile.lock BUNDLED WITH 与实际不符）
/usr/local/Cellar/ruby/4.0.2/bin/gem install bundler
# 然后手动编辑 Gemfile.lock 中的 BUNDLED WITH 为实际版本

# 安装依赖
/usr/local/Cellar/ruby/4.0.2/bin/bundle install

# 构建
/usr/local/Cellar/ruby/4.0.2/bin/bundle exec jekyll build

# 预览
/usr/local/Cellar/ruby/4.0.2/bin/bundle exec jekyll serve --host 127.0.0.1 --port 4000
```
