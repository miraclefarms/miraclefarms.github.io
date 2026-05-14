const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const { wechatFormat } = require('../src/stages/06-wechat-format');

const projectRoot = path.resolve(__dirname, '..', '..');

test('wechatFormat archives the WeChat markdown copy separately from runtime HTML output', async (t) => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'wechat-format-'));
  const postFile = path.join(tmpDir, '2099-01-01-ai-infra-daily-brief.md');
  const coverPath = path.join(tmpDir, 'cover.png');
  const outputDir = path.join(tmpDir, 'runtime');
  const archiveDir = path.join(tmpDir, 'docs', 'wechat');
  const fixtureFile = path.join(__dirname, 'fixtures', 'write-response.md');

  fs.writeFileSync(postFile, [
    '---',
    'title: AI Infra 早报｜测试',
    'date: 2099-01-01 08:00:00 +0800',
    'author: 荔枝不耐思',
    'kind: brief',
    'category: Brief',
    'intro: 测试摘要。',
    '---',
    '',
    '正文',
  ].join('\n'), 'utf8');
  fs.writeFileSync(coverPath, Buffer.from('fake png bytes'));

  const previousCli = process.env.AI_CLI;
  const previousMockResponse = process.env.MOCK_AI_RESPONSE_FILE;
  process.env.AI_CLI = 'mock';
  process.env.MOCK_AI_RESPONSE_FILE = fixtureFile;

  t.after(() => {
    if (previousCli === undefined) delete process.env.AI_CLI;
    else process.env.AI_CLI = previousCli;
    if (previousMockResponse === undefined) delete process.env.MOCK_AI_RESPONSE_FILE;
    else process.env.MOCK_AI_RESPONSE_FILE = previousMockResponse;
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  const result = await wechatFormat('2099-01-01', postFile, outputDir, projectRoot, coverPath, archiveDir);

  const runtimeMdPath = path.join(outputDir, '2099-01-01-ai-infra-daily-brief-wechat.md');
  const runtimeHtmlPath = path.join(outputDir, '2099-01-01-ai-infra-daily-brief-wechat.html');
  const archiveMdPath = path.join(archiveDir, '2099-01-01-ai-infra-daily-brief-wechat.md');
  const archiveCoverPath = path.join(archiveDir, 'assets', '2099-01-01', 'ai-infra-daily-brief-cover.png');
  const archiveHtmlPath = path.join(archiveDir, '2099-01-01-ai-infra-daily-brief-wechat.html');

  assert.equal(result.mdPath, runtimeMdPath);
  assert.equal(result.htmlPath, runtimeHtmlPath);
  assert.equal(result.archiveMdPath, archiveMdPath);
  assert.equal(result.archiveCoverPath, archiveCoverPath);
  assert.match(fs.readFileSync(archiveMdPath, 'utf8'), /!\[题图\]\(assets\/2099-01-01\/ai-infra-daily-brief-cover\.png\)/);
  assert.equal(fs.readFileSync(archiveCoverPath, 'utf8'), 'fake png bytes');
  assert.equal(fs.existsSync(archiveHtmlPath), false);
});

test('wechatFormat moves the first body H1 into front matter before saving', async (t) => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'wechat-format-title-'));
  const postFile = path.join(tmpDir, '2099-01-02-ai-infra-daily-brief.md');
  const outputDir = path.join(tmpDir, 'runtime');
  const archiveDir = path.join(tmpDir, 'docs', 'wechat');
  const fixtureFile = path.join(tmpDir, 'wechat-response.md');

  fs.writeFileSync(postFile, [
    '---',
    'title: AI Infra 早报｜测试',
    'date: 2099-01-02 08:00:00 +0800',
    'author: 荔枝不耐思',
    'kind: brief',
    'category: Brief',
    'intro: 测试摘要。',
    '---',
    '',
    '正文',
  ].join('\n'), 'utf8');

  fs.writeFileSync(fixtureFile, [
    '---',
    'wechat_variant: brief',
    'intro: 今天关注量化与 KV offload。',
    '---',
    '',
    '# 今日焦点：量化规格开始收口',
    '',
    '**📅 2099-01-02**',
    '',
    '> DeepSeek V4 与 GPT-OSS 量化路径同时收口，今天的更新决定新格式能不能快速进入默认服务。',
  ].join('\n'), 'utf8');

  const previousCli = process.env.AI_CLI;
  const previousMockResponse = process.env.MOCK_AI_RESPONSE_FILE;
  process.env.AI_CLI = 'mock';
  process.env.MOCK_AI_RESPONSE_FILE = fixtureFile;

  t.after(() => {
    if (previousCli === undefined) delete process.env.AI_CLI;
    else process.env.AI_CLI = previousCli;
    if (previousMockResponse === undefined) delete process.env.MOCK_AI_RESPONSE_FILE;
    else process.env.MOCK_AI_RESPONSE_FILE = previousMockResponse;
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  await wechatFormat('2099-01-02', postFile, outputDir, projectRoot, '', archiveDir);

  const runtimeMd = fs.readFileSync(
    path.join(outputDir, '2099-01-02-ai-infra-daily-brief-wechat.md'),
    'utf8',
  );
  const archiveMd = fs.readFileSync(
    path.join(archiveDir, '2099-01-02-ai-infra-daily-brief-wechat.md'),
    'utf8',
  );

  assert.match(runtimeMd, /^title: 今日焦点：量化规格开始收口$/m);
  assert.doesNotMatch(runtimeMd, /^# 今日焦点/m);
  assert.match(archiveMd, /^title: 今日焦点：量化规格开始收口$/m);
  assert.doesNotMatch(archiveMd, /^# 今日焦点/m);
});
