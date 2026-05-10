const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const projectRoot = path.resolve(__dirname, '..', '..');

test('stage 04 fallback prompt asks for a Chinese infographic cover', async () => {
  const originalAiCli = process.env.AI_CLI;
  const originalOpenRouterKey = process.env.OPENROUTER_API_KEY;

  process.env.AI_CLI = 'not-supported';
  delete process.env.OPENROUTER_API_KEY;

  const { generateCover } = require('../src/stages/04-cover');
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'miraclefarms-cover-'));
  const postFile = path.join(tempDir, '2026-05-10-ai-infra-daily-brief.md');
  const assetsDir = path.join(tempDir, 'assets');

  try {
    fs.writeFileSync(postFile, [
      '---',
      'title: AI Infra 早报｜KV Cache 调度进入中文信息图时代',
      'date: 2026-05-10 08:00:00 +0800',
      'intro: 今天的核心变化是调度、缓存和路由一起进入默认路径。',
      '---',
      '',
      '今天的早报正文。',
    ].join('\n'), 'utf8');

    const result = await generateCover('2026-05-10', postFile, assetsDir, projectRoot);
    const fallbackPrompt = fs.readFileSync(path.join(assetsDir, 'cover-prompt.txt'), 'utf8');

    assert.equal(result, null);
    assert.match(fallbackPrompt, /中文信息图/);
    assert.match(fallbackPrompt, /简体中文/);
    assert.match(fallbackPrompt, /禁止出现英文长句/);
    assert.doesNotMatch(fallbackPrompt, /Create a hand-drawn|concise English|English cursive/i);
  } finally {
    if (originalAiCli === undefined) {
      delete process.env.AI_CLI;
    } else {
      process.env.AI_CLI = originalAiCli;
    }

    if (originalOpenRouterKey === undefined) {
      delete process.env.OPENROUTER_API_KEY;
    } else {
      process.env.OPENROUTER_API_KEY = originalOpenRouterKey;
    }

    fs.rmSync(tempDir, { recursive: true, force: true });
  }
});

test('stage 04 final prompt renders daily newspaper template placeholders', () => {
  const { buildFinalPrompt } = require('../src/stages/04-cover');
  const template = {
    prompt: 'Put this whole text, verbatim, into a photo of A daily morning paper with reading creases on a desk, with photos, beautiful typography design, pull quotes and brave formatting. The text: [题图提示词][当天日期]',
  };
  const prompt = buildFinalPrompt(
    '今日核心判断：推理调度进入日报头版。',
    template,
    '2026-05-11',
  );

  assert.match(prompt, /Put this whole text, verbatim, into a photo of A daily morning paper/);
  assert.match(prompt, /The text: 今日核心判断：推理调度进入日报头版。\n2026-05-11/);
  assert.doesNotMatch(prompt, /\[题图提示词\]|\[当天日期\]/);
  assert.doesNotMatch(prompt, /硬性约束：最终图片必须是中文信息图/);
});
