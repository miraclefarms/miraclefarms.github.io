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

const PROJECT_ROOT = path.resolve(__dirname, '../../..');
const DEFAULT_CONFIG_PATH = path.join(PROJECT_ROOT, 'ai-morning-report', 'config', 'repo-scope.json');

function loadRepoScope(configPath = DEFAULT_CONFIG_PATH) {
  return JSON.parse(fs.readFileSync(configPath, 'utf8'));
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
  ghClient = createGhClient(),
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
  loadRepoScope,
  runCollectStage,
};
