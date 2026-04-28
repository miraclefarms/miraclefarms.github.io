const test = require('node:test');
const assert = require('node:assert/strict');

const {
  validateResearchDocument,
  validateSelectionDocument,
  validateArticleSource,
  validatePublishRecord,
} = require('../../ai-morning-report/src/lib/article-schema');

test('validateArticleSource requires cover_prompt plus core article and wechat metadata', () => {
  const articleSource = {
    target_date: '2026-04-27',
    title: 'AI Infra 早报｜缺字段',
    intro: '今天最值得看的变化集中在推理系统和训练框架。',
    thesis: '这波变化说明系统层优化仍是 AI Infra 的主战场。',
    cover_prompt: 'A systems diagram showing default execution paths absorbing complex runtime behaviors.',
    sections: [{ heading: '一、推理系统', body: 'SGLang 与 vLLM 都在加速演进。' }],
    wechat: {
      title: 'AI Infra 早报｜缺字段',
      digest: '推理系统和训练框架更新密集。',
    },
  };

  assert.throws(
    () => validateArticleSource({
      title: articleSource.title,
      intro: articleSource.intro,
      thesis: articleSource.thesis,
      sections: articleSource.sections,
      wechat: articleSource.wechat,
    }),
    /missing required field: target_date/,
  );
  assert.throws(
    () => validateArticleSource({
      target_date: articleSource.target_date,
      title: 'AI Infra 早报｜缺字段',
      intro: articleSource.intro,
      thesis: articleSource.thesis,
      sections: articleSource.sections,
      wechat: articleSource.wechat,
    }),
    /missing required field: cover_prompt/,
  );
  assert.throws(
    () => validateArticleSource({
      target_date: articleSource.target_date,
      title: 'AI Infra 早报｜缺字段',
    }),
    /missing required field: intro/,
  );
  assert.equal(validateArticleSource(articleSource), articleSource);
});

test('validateResearchDocument standardizes on snake_case research artifact fields', () => {
  const research = {
    target_date: '2026-04-27',
    window: {
      start: '2026-04-25T00:00:00+08:00',
      end: '2026-04-27T23:59:59+08:00',
    },
    repos: [{ owner: 'sgl-project', repo: 'sglang' }],
  };

  assert.equal(validateResearchDocument(research), research);
  assert.throws(
    () => validateResearchDocument({ window: research.window, repos: research.repos }),
    /missing required field: target_date/,
  );
  assert.throws(
    () => validateResearchDocument({ target_date: research.target_date, window: research.window }),
    /missing required field: repos/,
  );
});

test('validateSelectionDocument requires target_date and selected fields', () => {
  const selection = {
    target_date: '2026-04-27',
    selected: [{ rank: 1, candidate_id: 'sglang#123' }],
  };

  assert.equal(validateSelectionDocument(selection), selection);
  assert.throws(
    () => validateSelectionDocument({ target_date: selection.target_date }),
    /missing required field: selected/,
  );
});

test('validatePublishRecord requires publish-record-compatible field names', () => {
  const publishRecord = {
    target_date: '2026-04-27',
    github: {
      status: 'published',
      commit: 'abc1234',
      published_at: '2026-04-27T08:05:00+08:00',
    },
    wechat: {
      status: 'failed',
      draft_media_id: 'draft-media-id',
      last_error: 'rate limited',
      last_attempt_at: '2026-04-27T08:10:00+08:00',
    },
  };

  assert.equal(validatePublishRecord(publishRecord), publishRecord);
  assert.throws(
    () => validatePublishRecord({ github: publishRecord.github, wechat: publishRecord.wechat }),
    /missing required field: target_date/,
  );
  assert.throws(
    () => validatePublishRecord({
      target_date: publishRecord.target_date,
      github: {},
      wechat: publishRecord.wechat,
    }),
    /missing required field: github.status/,
  );
  assert.throws(
    () => validatePublishRecord({
      target_date: publishRecord.target_date,
      github: publishRecord.github,
      wechat: {},
    }),
    /missing required field: wechat.status/,
  );
});
