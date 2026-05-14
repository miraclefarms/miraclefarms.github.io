const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const {
  prepareArticleBodyForPublish,
  resolveExplicitWechatFiles,
} = require('../publish-wechat');

test('prepareArticleBodyForPublish removes the first body H1 so the draft title is not repeated', () => {
  const body = [
    '# 今日焦点：量化规格开始收口',
    '',
    '**📅 2026-05-14**',
    '',
    '![题图](assets/2026-05-14/cover.png)',
    '',
    '> DeepSeek V4 与 GPT-OSS 量化路径同时收口。',
    '',
    '## 推理侧',
    '',
    '正文',
  ].join('\n');

  const prepared = prepareArticleBodyForPublish(body);

  assert.equal(prepared.removedTitle, '今日焦点：量化规格开始收口');
  assert.doesNotMatch(prepared.body, /^# 今日焦点/m);
  assert.match(prepared.body, /^\*\*📅 2026-05-14\*\*/);
  assert.match(prepared.body, /!\[题图\]\(assets\/2026-05-14\/cover\.png\)/);
});

test('resolveExplicitWechatFiles accepts repeated --file arguments', () => {
  const files = resolveExplicitWechatFiles([
    '--file',
    'docs/wechat/2026-05-08-demo-wechat.md',
    '--file=docs/wechat/2026-05-09-demo-wechat.md',
  ]);

  assert.equal(files.length, 2);
  assert.equal(path.basename(files[0]), '2026-05-08-demo-wechat.md');
  assert.equal(path.basename(files[1]), '2026-05-09-demo-wechat.md');
  assert.equal(path.isAbsolute(files[0]), true);
});
