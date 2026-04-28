const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const assert = require('node:assert/strict');

const { buildPipelinePaths } = require('../../ai-morning-report/src/lib/pipeline-paths');
const { runRankStage } = require('../../ai-morning-report/src/stages/02-rank');
const { runComposeStage } = require('../../ai-morning-report/src/stages/03-compose');
const {
  createPublishRecord,
  markGithubPublished,
  markWechatRetryPending,
} = require('../../ai-morning-report/src/lib/publish-record');

function createResearchDocument() {
  return {
    target_date: '2026-04-27',
    window: {
      start: '2026-04-25T00:00:00+08:00',
      end: '2026-04-27T23:59:59+08:00',
    },
    repos: [
      {
        repo: 'sgl-project/sglang',
        pull_requests: [
          {
            type: 'pull_request',
            repo: 'sgl-project/sglang',
            number: 101,
            title: 'Improve default runtime scheduler path',
            url: 'https://github.com/sgl-project/sglang/pull/101',
            merged_at: '2026-04-27T01:00:00Z',
            labels: ['runtime', 'performance'],
            body: 'Moves a critical runtime path onto the default execution path.',
            priority_score: 14,
            signals: { default_path: true, runtime: true, performance: true },
            evidence: [{ kind: 'pull_request', url: 'https://github.com/sgl-project/sglang/pull/101' }],
          },
          {
            type: 'pull_request',
            repo: 'sgl-project/sglang',
            number: 102,
            title: 'Add graph cache fallback controls',
            url: 'https://github.com/sgl-project/sglang/pull/102',
            merged_at: '2026-04-27T03:00:00Z',
            labels: ['runtime', 'stability'],
            body: 'Adds a new fallback for graph cache misses.',
            priority_score: 12,
            signals: { runtime: true, stability: true, feature: true },
            evidence: [{ kind: 'pull_request', url: 'https://github.com/sgl-project/sglang/pull/102' }],
          },
          {
            type: 'pull_request',
            repo: 'sgl-project/sglang',
            number: 103,
            title: 'Support FP8 serving path tuning',
            url: 'https://github.com/sgl-project/sglang/pull/103',
            merged_at: '2026-04-26T22:00:00Z',
            labels: ['serving'],
            body: 'Extends the serving path with FP8-aware tuning.',
            priority_score: 11,
            signals: { runtime: true, feature: true, performance: true },
            evidence: [{ kind: 'pull_request', url: 'https://github.com/sgl-project/sglang/pull/103' }],
          },
        ],
        releases: [],
        commits: [],
        failures: [],
      },
      {
        repo: 'vllm-project/vllm',
        pull_requests: [
          {
            type: 'pull_request',
            repo: 'vllm-project/vllm',
            number: 201,
            title: 'Optimize kv cache eviction on the hot path',
            url: 'https://github.com/vllm-project/vllm/pull/201',
            merged_at: '2026-04-27T05:00:00Z',
            labels: ['performance'],
            body: 'Improves kv cache eviction on the default hot path.',
            priority_score: 13,
            signals: { default_path: true, runtime: true, performance: true },
            evidence: [{ kind: 'pull_request', url: 'https://github.com/vllm-project/vllm/pull/201' }],
          },
          {
            type: 'pull_request',
            repo: 'vllm-project/vllm',
            number: 202,
            title: 'Add MoE routing diagnostics',
            url: 'https://github.com/vllm-project/vllm/pull/202',
            merged_at: '2026-04-26T14:00:00Z',
            labels: ['feature'],
            body: 'Adds diagnostics for MoE routing decisions.',
            priority_score: 10,
            signals: { runtime: true, feature: true },
            evidence: [{ kind: 'pull_request', url: 'https://github.com/vllm-project/vllm/pull/202' }],
          },
          {
            type: 'pull_request',
            repo: 'vllm-project/vllm',
            number: 203,
            title: 'Fix scheduler fallback regression',
            url: 'https://github.com/vllm-project/vllm/pull/203',
            merged_at: '2026-04-25T20:00:00Z',
            labels: ['stability'],
            body: 'Fixes a scheduler fallback regression in serving mode.',
            priority_score: 9,
            signals: { runtime: true, stability: true },
            evidence: [{ kind: 'pull_request', url: 'https://github.com/vllm-project/vllm/pull/203' }],
          },
        ],
        releases: [],
        commits: [],
        failures: [],
      },
    ],
  };
}

test('research -> selection -> article-source smoke test writes validated artifacts', async () => {
  const projectRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'ai-morning-report-pipeline-'));
  const targetDate = '2026-04-27';
  const paths = buildPipelinePaths(projectRoot, targetDate);
  const researchDocument = createResearchDocument();

  fs.mkdirSync(paths.workDir, { recursive: true });
  fs.writeFileSync(paths.researchJsonPath, `${JSON.stringify(researchDocument, null, 2)}\n`, 'utf8');

  const rankingCalls = [];
  const composeCalls = [];

  const rankingResult = await runRankStage({
    targetDate,
    projectRoot,
    repoScope: {
      selection: {
        minimum_items: 5,
        maximum_items: 8,
      },
    },
    invokeModel: async (payload) => {
      rankingCalls.push(payload);
      return {
        selected: [
          { rank: 1, candidate_id: 'sgl-project/sglang#101', why_selected: '默认路径变化', category: 'runtime', importance_score: 9.8, evidence_strength: 'high', primary_angle: '默认路径' },
          { rank: 2, candidate_id: 'vllm-project/vllm#201', why_selected: 'KV cache hot path', category: 'performance', importance_score: 9.5, evidence_strength: 'high', primary_angle: '缓存策略' },
          { rank: 3, candidate_id: 'sgl-project/sglang#102', why_selected: 'fallback controls', category: 'stability', importance_score: 9.0, evidence_strength: 'medium', primary_angle: '回退机制' },
          { rank: 4, candidate_id: 'sgl-project/sglang#103', why_selected: 'FP8 serving tuning', category: 'runtime', importance_score: 8.8, evidence_strength: 'medium', primary_angle: '服务路径' },
          { rank: 5, candidate_id: 'vllm-project/vllm#202', why_selected: 'MoE diagnostics', category: 'feature', importance_score: 8.4, evidence_strength: 'medium', primary_angle: '可观测性' },
        ],
      };
    },
  });

  assert.equal(rankingCalls[0].role, 'ranking');
  assert.equal(rankingResult.selectionDocument.selected.length, 5);
  assert.match(fs.readFileSync(paths.selectionJsonPath, 'utf8'), /candidate_id/);

  const composeResult = await runComposeStage({
    targetDate,
    projectRoot,
    invokeModel: async (payload) => {
      composeCalls.push(payload);
      return {
        target_date: '1999-01-01',
        kind: 'essay',
        author: 'Model Override',
        category: 'Essay',
        series: 'wrong-series',
        title: 'AI Infra 早报｜默认路径开始吸收复杂场景',
        intro: '今天最值得关注的变化，集中在默认执行路径的能力收敛。',
        thesis: '真正值得记住的是，复杂能力正在从特例走向默认路径。',
        cover_prompt: 'An editorial cover about default execution paths absorbing advanced runtime behavior.',
        sections: [
          {
            heading: '一、运行时默认路径',
            summary: '这组变化共同说明，复杂能力开始回流到默认执行路径。',
            items: [
              {
                title: 'SGLang runtime path',
                analysis: '默认路径吸收了新的图执行和 fallback 能力。更重要的是，这让复杂场景不再依赖旁路实现，而是开始进入主链路。',
                references: [
                  { index: 1, title: 'SGLang PR 101', url: 'https://github.com/sgl-project/sglang/pull/101' },
                ],
              },
            ],
          },
        ],
        wechat: {
          title: 'AI Infra 早报｜默认路径开始吸收复杂场景',
          digest: '复杂能力开始回流到默认执行路径。',
        },
      };
    },
  });

  assert.equal(composeCalls[0].role, 'writing');
  assert.match(composeCalls[0].prompt, /section_first_judgment/);
  assert.match(composeCalls[0].prompt, /item\.analysis/);
  assert.equal(composeResult.articleSource.title, 'AI Infra 早报｜默认路径开始吸收复杂场景');
  assert.equal(composeResult.articleSource.cover_prompt, 'An editorial cover about default execution paths absorbing advanced runtime behavior.');
  assert.equal(composeResult.articleSource.target_date, targetDate);
  assert.equal(composeResult.articleSource.wechat.digest, '复杂能力开始回流到默认执行路径。');
  assert.equal(composeResult.articleSource.author, '荔枝不耐思');
  assert.equal(composeResult.articleSource.kind, 'brief');
  assert.equal(composeResult.articleSource.category, 'Brief');
  assert.equal(composeResult.articleSource.series, 'ai-infra-daily-brief');
  assert.match(fs.readFileSync(paths.articleSourceJsonPath, 'utf8'), /默认路径开始吸收复杂场景/);
});

test('pipeline downgrade policy preserves github success when wechat publish fails', () => {
  const initial = createPublishRecord({ targetDate: '2026-04-27' });
  const afterGithub = markGithubPublished(initial, { commit: 'abc1234' });
  const finalRecord = markWechatRetryPending(afterGithub, new Error('upload timeout'));

  assert.equal(finalRecord.github.status, 'published');
  assert.equal(finalRecord.github.commit, 'abc1234');
  assert.equal(finalRecord.wechat.status, 'pending_retry');
  assert.equal(finalRecord.wechat.last_error, 'upload timeout');
});

test('new scheduler references collect/rank/compose/image/render/publish stages only', () => {
  const script = fs.readFileSync(path.join(process.cwd(), 'ai-morning-report', 'bin', 'run-daily.sh'), 'utf8');
  assert.match(script, /01-collect\.js/);
  assert.match(script, /02-rank\.js/);
  assert.match(script, /03-compose\.js/);
  assert.match(script, /04-image\.js/);
  assert.match(script, /05-render\.js/);
  assert.match(script, /06-publish-github\.js/);
  assert.match(script, /07-publish-wechat\.js/);
  assert.doesNotMatch(script, /01-research\.js/);
  assert.doesNotMatch(script, /02-write\.js/);
  assert.doesNotMatch(script, /03-images\.js/);
  assert.doesNotMatch(script, /04-publish\.js/);
  assert.doesNotMatch(script, /05-wechat\.js/);
});
