const test = require('node:test');
const assert = require('node:assert/strict');

const {
  rankCandidatePriority,
  assertNonEmptyResearch,
} = require('../../ai-morning-report/src/lib/github-research');

test('rankCandidatePriority prefers runtime and default-path pull requests', () => {
  const highPriority = rankCandidatePriority({
    labels: ['performance', 'runtime'],
    title: 'Improve default runtime scheduler path',
    signals: {
      default_path: true,
      runtime: true,
      performance: true,
    },
  });

  const lowPriority = rankCandidatePriority({
    labels: ['documentation', 'tests'],
    title: 'Refresh docs and test snapshots',
    signals: {
      docs_only: true,
      tests_only: true,
    },
  });

  assert.ok(highPriority > lowPriority);
});

test('rankCandidatePriority downranks docs, ci, and tests only changes', () => {
  const runtimeScore = rankCandidatePriority({
    labels: ['runtime'],
    title: 'Tune serving default path batching',
    signals: {
      default_path: true,
      runtime: true,
    },
  });

  const docsOnlyScore = rankCandidatePriority({
    labels: ['ci', 'docs'],
    title: 'Docs and CI touch-ups',
    signals: {
      ci_only: true,
      docs_only: true,
      tests_only: true,
    },
  });

  assert.ok(runtimeScore > docsOnlyScore);
  assert.ok(docsOnlyScore < 0);
});

test('assertNonEmptyResearch throws when no repositories produced candidates', () => {
  assert.throws(
    () => assertNonEmptyResearch([]),
    /no research candidates collected/i,
  );
});

test('assertNonEmptyResearch throws when every repository fetch failed', () => {
  assert.throws(
    () => assertNonEmptyResearch([
      { repo: 'vllm-project/vllm', pull_requests: [], failures: [{ source: 'pull_requests' }] },
      { repo: 'sgl-project/sglang', pull_requests: [], failures: [{ source: 'pull_requests' }] },
    ]),
    /all upstream collection failed/i,
  );
});

test('assertNonEmptyResearch accepts repo-organized research with at least one candidate', () => {
  const research = [
    {
      repo: 'vllm-project/vllm',
      pull_requests: [
        {
          number: 123,
          title: 'Optimize runtime default path',
          priority_score: 14,
        },
      ],
      releases: [],
      commits: [],
      failures: [],
    },
  ];

  assert.equal(assertNonEmptyResearch(research), research);
});
