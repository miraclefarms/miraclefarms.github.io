# X Publishing Handoff — 2026-05-09

记录时间：2026-05-09 00:34 +0800

## 当前结论

X 发文流程已经完成基础接入，并且成功用 `miraclefarms_io` 发出过第一条真实推文。

当前每日早报自动流水线已经加上安全闸门：默认不会运行 X 改写，也不会运行 X 推送。现有 GitHub.io 和微信公众号自动发布路径保持为默认主流程。

## 安全开关现状

本机 `.env` 当前状态：

```text
ENABLE_X_PUSH=0
X_DRY_RUN=1
X_ALLOW_DETERMINISTIC_FALLBACK=0
```

`SKIP_X` 已移除，不再作为控制开关。

`ai-morning-report/bin/run-daily.sh` 的行为：

```text
默认：只跑 Stage 1-7，包含 GitHub.io + WeChat
ENABLE_X_PUSH=1：额外开启 Stage 8，执行 X rewrite + publish
X_DRY_RUN=1：生成 X 内容但不真实发帖
X_DRY_RUN=0：允许真实发帖
```

明日自动早报不需要发 X 时，保持当前 `.env` 即可。

## 已完成内容

- 新增 `x-formatter` skill，用于把 GitHub.io 文章转写为英文 X 内容。
- 新增 `08-x-push.js`，作为早报流水线的可选 Stage 8。
- 新增 `scripts/publish-x.js`，支持单篇文章 dry-run / real post。
- 新增 `scripts/x-oauth.js`，用于 X OAuth 授权和 token 写入。
- 新增 `scripts/lib/x-publisher.js`，封装 X API 发文逻辑。
- 新增 X 相关测试，覆盖发布器、skill 契约和 `ENABLE_X_PUSH` 安全闸门。
- 更新文档 `docs/x-publishing.md` 和 `ai-morning-report/docs/SPEC.md`。
- 更新 `.env.example`，默认 `ENABLE_X_PUSH=0`。

## 已验证

以下验证已通过：

```bash
npm run test:x
bash -n ai-morning-report/bin/run-daily.sh
bash -n ai-morning-report/src/stages/01-fetch.sh
bash -n ai-morning-report/src/stages/05-publish.sh
node --check ai-morning-report/src/stages/06-wechat-format.js
node --check ai-morning-report/src/stages/07-wechat-push.js
node --check ai-morning-report/src/stages/08-x-push.js
node --check scripts/publish-x.js
node --check scripts/x-oauth.js
node --test scripts/tests/miraclefarms-writer-skill.test.js scripts/tests/wechat-config.test.js scripts/tests/wechat-render.test.js
git diff --check
```

未完成验证：

```text
bundle exec jekyll build
```

原因：当前机器未找到文档里写的 Homebrew Ruby 4.0.2 路径，系统 Ruby 2.6 又缺少 `bundler 4.0.10`。这看起来是本机 Ruby 环境问题，不是本次 X 改动引入的问题。

## 当前工作区状态

本地尚未提交的变更包括：

```text
.env.example
ai-morning-report/bin/run-daily.sh
ai-morning-report/docs/SPEC.md
ai-morning-report/src/lib/cli-adapter.js
package.json
.agents/skills/x-formatter/
ai-morning-report/src/stages/08-x-push.js
docs/x-publishing.md
scripts/lib/x-publisher.js
scripts/publish-x.js
scripts/tests/run-daily-x-gate.test.js
scripts/tests/x-format-skill.test.js
scripts/tests/x-publisher.test.js
scripts/x-oauth.js
```

本文件也是新的交接记录。

## 明天继续的建议步骤

1. 先保持 `ENABLE_X_PUSH=0`，确认早报 GitHub.io + WeChat 自动流水线正常完成。
2. 选一篇已经生成的 GitHub.io 文章，手动运行 X dry-run：

```bash
X_DRY_RUN=1 node scripts/publish-x.js _posts/YYYY-MM-DD-ai-infra-daily-brief.md
```

3. 检查英文 X 内容是否足够像 `miraclefarms` 的账号定位，而不是机械摘要。
4. 如果内容 OK，再手动真实推送单篇：

```bash
X_DRY_RUN=0 node scripts/publish-x.js _posts/YYYY-MM-DD-ai-infra-daily-brief.md
```

5. 连续几次手动确认质量稳定后，再考虑把 `.env` 改成：

```text
ENABLE_X_PUSH=1
X_DRY_RUN=0
```

6. 在完全稳定前，不建议把 X Stage 8 直接纳入每天早报默认真实推送。

