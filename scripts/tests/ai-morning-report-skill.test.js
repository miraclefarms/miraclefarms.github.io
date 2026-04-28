const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');

const {
  AI_MORNING_REPORT_SKILL_PATH,
  extractSkillConfig,
  loadAiMorningReportSkillConfig,
} = require('../../ai-morning-report/src/lib/skill-config');

test('ai-morning-report skill exposes a machine-readable strategy block', () => {
  const skill = fs.readFileSync(AI_MORNING_REPORT_SKILL_PATH, 'utf8');

  assert.match(skill, /codex:skill-config:start/);
  assert.match(skill, /writer_handoff/);
  assert.match(skill, /section_first_judgment/);
});

test('extractSkillConfig parses the embedded JSON config block', () => {
  const config = extractSkillConfig(`
prefix
<!-- codex:skill-config:start -->
{"repos":["example/repo"],"window":{"lookback_days":3}}
<!-- codex:skill-config:end -->
suffix
  `);

  assert.deepEqual(config, {
    repos: ['example/repo'],
    window: { lookback_days: 3 },
  });
});

test('loadAiMorningReportSkillConfig returns repo scope and narrative handoff settings', () => {
  const config = loadAiMorningReportSkillConfig();

  assert.ok(Array.isArray(config.repos));
  assert.ok(config.repos.includes('vllm-project/vllm'));
  assert.equal(config.window.timezone, 'Asia/Shanghai');
  assert.equal(config.selection.minimum_items, 5);
  assert.equal(config.writer_handoff.article_source_shape.sections.summary_required, true);
  assert.equal(config.writer_handoff.narrative_rules.item_analysis_min_sentences, 2);
});
