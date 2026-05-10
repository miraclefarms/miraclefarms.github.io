const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const {
  buildContentSourceUrl,
  insertCoverImageMarkdown,
  prepareMarkdownForDraft,
} = require('../src/stages/07-wechat-push');

test('prepareMarkdownForDraft moves the first H1 into the title field and removes it from the body', () => {
  const markdown = `---
wechat_variant: brief
intro: 今天关注 FP4 量化跨越可用性门槛。
---
# 今日焦点：FP4 量化跨越可用性门槛

**📅 2026-05-10**

> 今天的摘要。

---

## 推理侧

正文`;

  const prepared = prepareMarkdownForDraft(markdown);

  assert.equal(prepared.title, '今日焦点：FP4 量化跨越可用性门槛');
  assert.equal(prepared.digest, '今天关注 FP4 量化跨越可用性门槛。');
  assert.equal(prepared.author, '荔枝不耐思');
  assert.doesNotMatch(prepared.markdownContent, /^# 今日焦点/m);
});

test('insertCoverImageMarkdown places the uploaded cover below the digest blockquote', () => {
  const body = `**📅 2026-05-10**

> 今天的摘要。

---

## 推理侧

正文`;

  const rewritten = insertCoverImageMarkdown(body, 'https://mmbiz.qpic.cn/cover.jpg');

  assert.match(
    rewritten,
    /> 今天的摘要。\n\n!\[题图\]\(https:\/\/mmbiz\.qpic\.cn\/cover\.jpg\)\n\n---/,
  );
});

test('buildContentSourceUrl derives the GitHub.io permalink from the WeChat markdown path', () => {
  const wechatPath = path.join(
    '/tmp/morning-report/2026-05-10/wechat',
    '2026-05-10-ai-infra-daily-brief-wechat.md',
  );

  assert.equal(
    buildContentSourceUrl(wechatPath),
    'https://miraclefarms.github.io/notes/2026/05/10/ai-infra-daily-brief/',
  );
});
