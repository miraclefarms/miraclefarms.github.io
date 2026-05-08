const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const {
  buildFallbackThread,
  composePostWithUrl,
  normalizeThread,
  parseFrontMatter,
  parseThreadJson,
  resolvePostUrl,
} = require('../lib/x-publisher');

test('resolves MiracleFarms permalink from post filename', () => {
  const url = resolvePostUrl('/repo/_posts/2026-05-08-ai-infra-daily-brief.md');
  assert.equal(url, 'https://miraclefarms.github.io/notes/2026/05/08/ai-infra-daily-brief/');
});

test('parses simple front matter values', () => {
  const { frontMatter, body } = parseFrontMatter(`---\ntitle: "Hello"\nkind: brief\n---\nBody`);
  assert.equal(frontMatter.title, 'Hello');
  assert.equal(frontMatter.kind, 'brief');
  assert.equal(body, 'Body');
});

test('normalizes thread JSON and enforces post length', () => {
  const thread = parseThreadJson('```json\n["hello", "world"]\n```');
  assert.deepEqual(thread, ['hello', 'world']);

  const longThread = normalizeThread(['a'.repeat(400)]);
  assert.equal(longThread.length, 1);
  assert.equal(longThread[0].length, 280);
});

test('composes a post without truncating the URL', () => {
  const url = 'https://miraclefarms.github.io/notes/2026/05/08/ai-infra-daily-brief/';
  const post = composePostWithUrl('a'.repeat(400), url);
  assert.equal(post.endsWith(url), true);
  assert.equal(post.length <= 280, true);
});

test('builds fallback thread from a post', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'x-publisher-'));
  const postFile = path.join(dir, '2026-05-08-ai-infra-daily-brief.md');
  fs.writeFileSync(postFile, `---\ntitle: AI Infra 早报｜测试\nintro: 今天关注推理系统。\n---\n\n正文\n\n## 一、推理系统\n\n## 二、参考来源之外\n\n---\n\n## 参考来源\n`, 'utf8');

  const thread = buildFallbackThread(postFile);
  assert.equal(thread.length >= 2, true);
  assert.match(thread[0], /https:\/\/miraclefarms\.github\.io\/notes\/2026\/05\/08\/ai-infra-daily-brief\//);
  assert.match(thread[1], /推理系统/);
});
