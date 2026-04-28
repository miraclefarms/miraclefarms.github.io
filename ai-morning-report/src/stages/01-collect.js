const fs = require('node:fs');
const path = require('node:path');

const { validateResearchDocument } = require('../lib/article-schema');
const { buildResearchWindow, getTargetDateParts } = require('../lib/date-window');
const {
  assertNonEmptyResearch,
  collectRepoResearch,
  createGhClient,
} = require('../lib/github-research');
const { buildPipelinePaths } = require('../lib/pipeline-paths');
const { loadAiMorningReportSkillConfig } = require('../lib/skill-config');

const PROJECT_ROOT = path.resolve(__dirname, '../../..');
const DRY_RUN_REPO_DATA = {
  'sgl-project/sglang': [
    {
      number: 101,
      title: 'Improve default runtime scheduler path',
      url: 'https://github.com/sgl-project/sglang/pull/101',
      mergedAt: '2026-04-27T01:00:00Z',
      labels: [{ name: 'runtime' }, { name: 'performance' }],
      author: { login: 'alice' },
      body: 'Moves a critical runtime path onto the default execution path.',
      additions: 10,
      deletions: 2,
      changedFiles: 3,
    },
    {
      number: 102,
      title: 'Add graph cache fallback controls',
      url: 'https://github.com/sgl-project/sglang/pull/102',
      mergedAt: '2026-04-27T03:00:00Z',
      labels: [{ name: 'stability' }],
      author: { login: 'bob' },
      body: 'Adds a new fallback for graph cache misses.',
      additions: 9,
      deletions: 1,
      changedFiles: 2,
    },
    {
      number: 103,
      title: 'Support FP8 serving path tuning',
      url: 'https://github.com/sgl-project/sglang/pull/103',
      mergedAt: '2026-04-26T22:00:00Z',
      labels: [{ name: 'serving' }],
      author: { login: 'carol' },
      body: 'Extends the serving path with FP8-aware tuning.',
      additions: 15,
      deletions: 4,
      changedFiles: 5,
    },
  ],
  'vllm-project/vllm': [
    {
      number: 201,
      title: 'Optimize kv cache eviction on the hot path',
      url: 'https://github.com/vllm-project/vllm/pull/201',
      mergedAt: '2026-04-27T05:00:00Z',
      labels: [{ name: 'performance' }],
      author: { login: 'dave' },
      body: 'Improves kv cache eviction on the default hot path.',
      additions: 12,
      deletions: 3,
      changedFiles: 4,
    },
    {
      number: 202,
      title: 'Add MoE routing diagnostics',
      url: 'https://github.com/vllm-project/vllm/pull/202',
      mergedAt: '2026-04-26T14:00:00Z',
      labels: [{ name: 'feature' }],
      author: { login: 'erin' },
      body: 'Adds diagnostics for MoE routing decisions.',
      additions: 7,
      deletions: 1,
      changedFiles: 2,
    },
    {
      number: 203,
      title: 'Fix scheduler fallback regression',
      url: 'https://github.com/vllm-project/vllm/pull/203',
      mergedAt: '2026-04-25T20:00:00Z',
      labels: [{ name: 'stability' }],
      author: { login: 'frank' },
      body: 'Fixes a scheduler fallback regression in serving mode.',
      additions: 5,
      deletions: 2,
      changedFiles: 2,
    },
  ],
};

function loadRepoScope() {
  return loadAiMorningReportSkillConfig();
}

function buildWindowConfig(repoScope = {}) {
  if (repoScope.window) {
    return repoScope.window;
  }

  return {
    lookback_days: repoScope.timeWindow && repoScope.timeWindow.days
      ? repoScope.timeWindow.days
      : 3,
    timezone: repoScope.timeWindow && repoScope.timeWindow.timezone
      ? repoScope.timeWindow.timezone
      : 'Asia/Shanghai',
  };
}

function buildPriorityConfig(repoScope = {}) {
  return repoScope.candidate_priority || {};
}

function buildCollectionLimits(repoScope = {}) {
  return (repoScope.collection && repoScope.collection.limits) || {};
}

function createDryRunGhClient() {
  return {
    async listMergedPullRequests(repo) {
      return DRY_RUN_REPO_DATA[repo] || [];
    },
    async listReleases() {
      return [];
    },
    async listCommits() {
      return [];
    },
  };
}

function buildResearchDocument(targetDate, window, repoResults) {
  return validateResearchDocument({
    target_date: targetDate,
    window: {
      timezone: window.timezone,
      lookback_days: window.lookbackDays,
      start_date: window.startDate,
      end_date: window.endDate,
      start: window.start,
      end: window.end,
    },
    repos: repoResults,
  });
}

async function runCollectStage({
  targetDate,
  projectRoot = PROJECT_ROOT,
  repoScope = loadRepoScope(),
  ghClient = process.env.AI_MORNING_REPORT_DRY_RUN === '1'
    ? createDryRunGhClient()
    : createGhClient(),
} = {}) {
  const resolvedTargetDate = targetDate || getTargetDateParts().date;
  const window = buildResearchWindow(resolvedTargetDate, {
    timezone: buildWindowConfig(repoScope).timezone,
    lookbackDays: buildWindowConfig(repoScope).lookback_days,
  });
  const repoResults = [];

  for (const repo of repoScope.repos || []) {
    const repoResult = await collectRepoResearch({
      repo,
      window,
      ghClient,
      priorityConfig: buildPriorityConfig(repoScope),
      limits: buildCollectionLimits(repoScope),
    });
    repoResults.push(repoResult);
  }

  assertNonEmptyResearch(repoResults);

  const researchDocument = buildResearchDocument(resolvedTargetDate, window, repoResults);
  const paths = buildPipelinePaths(projectRoot, resolvedTargetDate);

  fs.mkdirSync(paths.workDir, { recursive: true });
  fs.writeFileSync(paths.researchJsonPath, `${JSON.stringify(researchDocument, null, 2)}\n`, 'utf8');

  return {
    outputPath: paths.researchJsonPath,
    researchDocument,
  };
}

if (require.main === module) {
  const targetDate = process.argv[2];

  runCollectStage({ targetDate })
    .then(({ outputPath }) => {
      console.log(`Research artifact written to ${outputPath}`);
    })
    .catch((error) => {
      console.error(`Collect stage failed: ${error.message}`);
      process.exit(1);
    });
}

module.exports = {
  buildResearchDocument,
  buildWindowConfig,
  createDryRunGhClient,
  loadRepoScope,
  runCollectStage,
};
