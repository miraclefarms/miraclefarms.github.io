const test = require('node:test');
const assert = require('node:assert/strict');

const {
  buildModelCandidateContext,
  buildRankingPrompt,
  limitCandidatesForModel,
} = require('../../ai-morning-report/src/stages/02-rank');
const { getTimeoutMs } = require('../../ai-morning-report/src/lib/openai');

function createCandidate(number, overrides = {}) {
  return {
    candidate_id: `repo/example#${number}`,
    repo: 'repo/example',
    number,
    title: `Candidate ${number}`,
    url: `https://github.com/repo/example/pull/${number}`,
    merged_at: `2026-04-26T0${number}:00:00Z`,
    labels: ['performance'],
    priority_score: number,
    signals: { performance: true },
    body: 'x'.repeat(2000),
    evidence: [{ kind: 'pull_request', url: `https://github.com/repo/example/pull/${number}` }],
    ...overrides,
  };
}

test('limitCandidatesForModel keeps only the highest-priority candidates while respecting per-repo caps', () => {
  const candidates = [
    createCandidate(1, { repo: 'repo/a', candidate_id: 'repo/a#1' }),
    createCandidate(5, { repo: 'repo/a', candidate_id: 'repo/a#5' }),
    createCandidate(3, { repo: 'repo/b', candidate_id: 'repo/b#3' }),
    createCandidate(2, { repo: 'repo/b', candidate_id: 'repo/b#2' }),
    createCandidate(4, { repo: 'repo/c', candidate_id: 'repo/c#4' }),
  ];
  const limited = limitCandidatesForModel(candidates, 3, 1);

  assert.deepEqual(
    limited.map(candidate => candidate.candidate_id),
    ['repo/a#5', 'repo/c#4', 'repo/b#3'],
  );
});

test('buildModelCandidateContext trims oversized PR bodies before they reach the model prompt', () => {
  const context = buildModelCandidateContext(createCandidate(1), 120);

  assert.equal(context.candidate_id, 'repo/example#1');
  assert.ok(context.why_candidate_matters.length <= 121);
  assert.equal(context.repo, 'repo/example');
  assert.equal(context.pr_number, 1);
});

test('buildRankingPrompt uses compact candidate context and exposes prefilter counts', () => {
  const prompt = buildRankingPrompt({
    research: { target_date: '2026-04-26' },
    candidates: [createCandidate(1), createCandidate(2)],
    totalCandidateCount: 173,
    minItems: 5,
    maxItems: 8,
    perRepoSoftCap: 2,
    bodyExcerptChars: 120,
  });

  assert.match(prompt, /"total_candidate_count": 173/);
  assert.match(prompt, /"candidate_count_for_model": 2/);
  assert.doesNotMatch(prompt, /"body":/);
  assert.doesNotMatch(prompt, /"url":/);
  assert.doesNotMatch(prompt, /"labels":/);
  assert.match(prompt, /"why_candidate_matters":/);
});

test('getTimeoutMs defaults to configured timeout and can be overridden by env', () => {
  const original = process.env.AI_MODEL_TIMEOUT_MS;
  delete process.env.AI_MODEL_TIMEOUT_MS;
  assert.equal(getTimeoutMs({ timeout_ms: 45000 }), 45000);

  process.env.AI_MODEL_TIMEOUT_MS = '9000';
  assert.equal(getTimeoutMs({ timeout_ms: 45000 }), 9000);

  if (original === undefined) {
    delete process.env.AI_MODEL_TIMEOUT_MS;
  } else {
    process.env.AI_MODEL_TIMEOUT_MS = original;
  }
});
