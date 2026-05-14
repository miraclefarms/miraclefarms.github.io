const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const projectRoot = path.resolve(__dirname, '..', '..');

test('cover prompt templates are stored in a versioned template file with a locked daily paper default', () => {
  const {
    loadWechatCoverPromptTemplates,
    resolveWechatCoverPromptTemplate,
  } = require('../lib/wechat-cover-prompts');

  const registry = loadWechatCoverPromptTemplates(projectRoot);
  const template = resolveWechatCoverPromptTemplate({}, registry);

  assert.equal(registry.version, 5);
  assert.equal(registry.defaultTemplateId, 'daily-morning-paper-v1');
  assert.equal(template.id, 'daily-morning-paper-v1');
  assert.equal(template.imageConfig.aspectRatio, '9:16');
  assert.match(template.prompt, /article brief/i);
  assert.match(template.prompt, /not literal/i);
  assert.match(template.prompt, /concise Chinese/i);
  assert.doesNotMatch(template.prompt, /verbatim|中文信息图|9:16 vertical/i);
  assert.doesNotMatch(template.prompt, /English cursive|concise English phrases/i);
});

test('daily morning paper template renders prompt text and date placeholders', () => {
  const {
    buildWechatCoverPrompt,
    loadWechatCoverPromptTemplates,
    resolveWechatCoverPromptTemplate,
  } = require('../lib/wechat-cover-prompts');

  const registry = loadWechatCoverPromptTemplates(projectRoot);
  const template = resolveWechatCoverPromptTemplate({
    wechat_cover_prompt_template: 'daily-morning-paper-v1',
  }, registry);
  const prompt = buildWechatCoverPrompt({
    frontMatter: {
      wechat_cover_prompt_template: 'daily-morning-paper-v1',
      date: '2026-05-11 08:00:00 +0800',
    },
    title: 'AI Infra 早报｜推理调度进入日报头版',
    digest: 'KV cache 与路由策略成为今天的主线。',
    bodyMarkdown: '## 一、调度路径\n\n今天的核心判断是调度面开始变成默认基础设施。',
    registry,
  });

  assert.equal(template.imageConfig.aspectRatio, '9:16');
  assert.match(prompt, /daily morning newspaper/i);
  assert.match(prompt, /article brief/i);
  assert.match(prompt, /AI Infra 早报｜推理调度进入日报头版/);
  assert.match(prompt, /KV cache 与路由策略成为今天的主线。/);
  assert.match(prompt, /2026-05-11/);
  assert.doesNotMatch(prompt, /\[题图提示词\]|\[当天日期\]/);
  assert.doesNotMatch(prompt, /Put this whole text, verbatim|The text:/);
});

test('cover prompt builder uses article content without visible prompt metadata', () => {
  const {
    buildWechatCoverPrompt,
    loadWechatCoverPromptTemplates,
  } = require('../lib/wechat-cover-prompts');

  const registry = loadWechatCoverPromptTemplates(projectRoot);
  const prompt = buildWechatCoverPrompt({
    frontMatter: {},
    title: 'Prefill-as-a-Service：LLM 推理的部署边界，为什么开始越过数据中心',
    digest: 'hybrid attention 把 KVCache 的网络成本压到一个新拐点。',
    bodyMarkdown: '## 一、为什么这不是一篇普通调度论文\n\n这篇论文真正讨论的是跨数据中心 prefill。',
    registry,
  });

  assert.match(prompt, /Prefill-as-a-Service/);
  assert.match(prompt, /hybrid attention/);
  assert.match(prompt, /跨数据中心 prefill/);
  assert.match(prompt, /article brief/i);
  assert.doesNotMatch(prompt, /9:16|竖版|中文信息图|题图提示词|硬性约束|最终图片/);
  assert.doesNotMatch(prompt, /landscape 16:9/i);
  assert.doesNotMatch(prompt, /no text, no letters/i);
  assert.doesNotMatch(prompt, /Article title:|Article summary:|Main body summary:/);
  assert.doesNotMatch(prompt, /English cursive|concise English phrases/i);
});

test('cover image request inherits aspect ratio from the locked template instead of a hardcoded fallback', () => {
  const {
    buildWechatCoverImageRequest,
    loadWechatCoverPromptTemplates,
  } = require('../lib/wechat-cover-prompts');

  const registry = loadWechatCoverPromptTemplates(projectRoot);
  const payload = buildWechatCoverImageRequest({
    frontMatter: {},
    title: 'Prefill-as-a-Service：LLM 推理的部署边界，为什么开始越过数据中心',
    digest: 'hybrid attention 把 KVCache 的网络成本压到一个新拐点。',
    bodyMarkdown: '## 一、为什么这不是一篇普通调度论文\n\n这篇论文真正讨论的是跨数据中心 prefill。',
    registry,
    model: 'openrouter/mock-image-model',
    imageSize: '1536x1024',
  });

  assert.equal(payload.model, 'openrouter/mock-image-model');
  assert.equal(payload.image_config.aspect_ratio, '9:16');
  assert.equal(payload.image_config.image_size, '1536x1024');
  assert.match(payload.messages[0].content, /article brief/i);
  assert.match(payload.messages[0].content, /concise Chinese/i);
  assert.doesNotMatch(payload.messages[0].content, /9:16|竖版|中文信息图|题图提示词|硬性约束|最终图片/);
});

test('title image insertion works even when articles no longer contain inline Chinese\/English prompt blocks', () => {
  const {
    insertGeneratedTitleImageMarkdown,
  } = require('../lib/wechat-cover-prompts');

  const markdown = [
    '# 标题',
    '',
    '**📅 2026-04-19**',
    '',
    '> 一句话摘要',
    '',
    '---',
    '',
    '正文第一段。',
  ].join('\n');

  const rewritten = insertGeneratedTitleImageMarkdown(markdown, 'assets/2026-04-19/demo-cover.png');

  assert.match(rewritten, /\*\*📅 2026-04-19\*\*\n\n!\[题图\]\(assets\/2026-04-19\/demo-cover\.png\)\n\n> 一句话摘要/);
});

test('legacy inline prompt blocks remain compatible with the new template-driven cover flow', () => {
  const {
    buildWechatCoverPrompt,
    insertGeneratedTitleImageMarkdown,
    loadWechatCoverPromptTemplates,
  } = require('../lib/wechat-cover-prompts');

  const registry = loadWechatCoverPromptTemplates(projectRoot);
  const markdown = [
    '# 旧稿标题',
    '',
    '**📅 2026-04-18**',
    '',
    '> 中文：清晨的数据中心控制室里，GPU 集群、推理网关与调度面板同时亮起，无文字，16:9',
    '>',
    '> English: A modern AI datacenter control room at dawn with GPU clusters and routing panels, no text, 16:9',
    '',
    '> 这是一篇旧格式稿件。',
  ].join('\n');

  const prompt = buildWechatCoverPrompt({
    frontMatter: {},
    title: '旧稿标题',
    digest: '这是一篇旧格式稿件。',
    bodyMarkdown: markdown,
    registry,
  });
  const rewritten = insertGeneratedTitleImageMarkdown(markdown, 'assets/2026-04-18/legacy-cover.png');

  assert.match(prompt, /旧稿标题/);
  assert.match(prompt, /article brief/i);
  assert.doesNotMatch(prompt, /旧版英文视觉备注|旧版中文视觉备注|16:9|无文字|English:/);
  assert.doesNotMatch(rewritten, /^>\s*中文[:：]/m);
  assert.doesNotMatch(rewritten, /^>\s*English[:：]/m);
  assert.match(rewritten, /!\[题图\]\(assets\/2026-04-18\/legacy-cover\.png\)/);
});

test('hook mode resolves changed WeChat articles even when their filename date is not today', () => {
  const originalExit = process.exit;
  process.exit = () => {};
  let resolveExpectedWechatFiles;

  try {
    ({ resolveExpectedWechatFiles } = require('../publish-wechat'));
  } finally {
    process.exit = originalExit;
  }

  const expected = resolveExpectedWechatFiles([
    'docs/wechat/2026-04-15-opendev-context-engineering-agent-design-wechat.md',
  ], '2026-04-19');

  assert.deepEqual(expected, [
    path.join(projectRoot, 'docs', 'wechat', '2026-04-15-opendev-context-engineering-agent-design-wechat.md'),
  ]);
});
