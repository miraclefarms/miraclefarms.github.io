const fs = require('node:fs');
const path = require('node:path');

const {
  requireField,
  validateResearchDocument,
  validateSelectionDocument,
} = require('../lib/article-schema');
const { buildPipelinePaths } = require('../lib/pipeline-paths');
const { invokeJsonModel } = require('../lib/openai');
const { loadAiMorningReportSkillConfig } = require('../lib/skill-config');

const PROJECT_ROOT = path.resolve(__dirname, '../../..');

function loadRepoScope() {
  return loadAiMorningReportSkillConfig();
}

function buildSelectionConfig(repoScope = {}) {
  return repoScope.selection || {};
}

function createCandidateId(candidate) {
  return `${candidate.repo}#${candidate.number}`;
}

function listPullRequestCandidates(research) {
  validateResearchDocument(research);

  return (research.repos || []).flatMap((repoResult) => (
    (repoResult.pull_requests || []).map((candidate) => ({
      ...candidate,
      candidate_id: createCandidateId(candidate),
    }))
  ));
}

function indexCandidatesById(candidates) {
  return new Map(candidates.map((candidate) => [candidate.candidate_id, candidate]));
}

function resolveCandidateId(selectedEntry) {
  if (selectedEntry.candidate_id) {
    return selectedEntry.candidate_id;
  }

  if (selectedEntry.repo && selectedEntry.pr_number) {
    return `${selectedEntry.repo}#${selectedEntry.pr_number}`;
  }

  throw new Error('selected entry must include candidate_id or repo + pr_number');
}

function normalizeSelectedEntry(selectedEntry, candidate, fallbackRank) {
  return {
    rank: selectedEntry.rank || fallbackRank,
    candidate_id: candidate.candidate_id,
    repo: candidate.repo,
    pr_number: candidate.number,
    title: candidate.title,
    url: candidate.url,
    merged_at: candidate.merged_at || null,
    priority_score: candidate.priority_score ?? null,
    why_selected: selectedEntry.why_selected || '',
    category: selectedEntry.category || 'runtime',
    importance_score: selectedEntry.importance_score ?? null,
    evidence_strength: selectedEntry.evidence_strength || null,
    primary_angle: selectedEntry.primary_angle || null,
    labels: candidate.labels || [],
    signals: candidate.signals || {},
    evidence: candidate.evidence || [],
  };
}

function buildRankingPrompt({ research, candidates, minItems, maxItems, perRepoSoftCap }) {
  return [
    '你是 AI Infra 日报的选题编辑。',
    `请基于 research.json 中的 pull_requests 选择 ${minItems} 到 ${maxItems} 条最值得写入今日早报的候选。`,
    '优先判断默认路径、runtime、serving、scheduler、KV cache、性能和稳定性变化。',
    '尽量以 pull request 为主，不要把 release 或 commit 当作主条目。',
    `同一 repo 最好不要超过 ${perRepoSoftCap} 条，除非当天确实没有更强信号。`,
    '返回 JSON 对象，包含 selected 数组。每个 selected 项必须包含 rank、candidate_id、why_selected、category、importance_score、evidence_strength、primary_angle。',
    '',
    JSON.stringify({
      target_date: research.target_date,
      candidate_count: candidates.length,
      candidates,
    }, null, 2),
  ].join('\n');
}

async function defaultInvokeModel({ role, prompt, workdir }) {
  return invokeJsonModel({ role, prompt, workdir });
}

function buildDryRunSelection(candidates, { minItems, maxItems, perRepoSoftCap }) {
  const selected = [];
  const perRepoCounts = new Map();

  for (const candidate of [...candidates].sort((left, right) => (right.priority_score || 0) - (left.priority_score || 0))) {
    const repoCount = perRepoCounts.get(candidate.repo) || 0;
    if (repoCount >= perRepoSoftCap && selected.length >= minItems) {
      continue;
    }

    selected.push({
      rank: selected.length + 1,
      candidate_id: candidate.candidate_id,
      why_selected: candidate.signals?.default_path
        ? '默认路径正在吸收更复杂的运行时能力'
        : '这条 PR 能直接改变生产中的性能、稳定性或可观测性',
      category: candidate.signals?.performance ? 'performance' : 'runtime',
      importance_score: candidate.priority_score || selected.length + 1,
      evidence_strength: 'high',
      primary_angle: candidate.signals?.default_path ? '默认路径' : '工程主线',
    });
    perRepoCounts.set(candidate.repo, repoCount + 1);

    if (selected.length >= maxItems) {
      break;
    }
  }

  return { selected };
}

async function selectCandidates({
  research,
  invokeModel = defaultInvokeModel,
  minItems = 5,
  maxItems = 8,
  perRepoSoftCap = 2,
  workdir,
  dryRun = process.env.AI_MORNING_REPORT_DRY_RUN === '1',
} = {}) {
  const candidates = listPullRequestCandidates(research);

  if (candidates.length === 0) {
    throw new Error('research document has no pull request candidates');
  }

  const response = dryRun
    ? buildDryRunSelection(candidates, { minItems, maxItems, perRepoSoftCap })
    : await invokeModel({
      role: 'ranking',
      workdir,
      prompt: buildRankingPrompt({
        research,
        candidates,
        minItems,
        maxItems,
        perRepoSoftCap,
      }),
    });

  const selected = Array.isArray(response && response.selected) ? response.selected : [];

  if (selected.length === 0) {
    throw new Error('ranking produced no selected candidates');
  }

  if (selected.length < minItems) {
    throw new Error(`ranking produced fewer than minimum items: ${selected.length}`);
  }

  if (selected.length > maxItems) {
    throw new Error(`ranking exceeded maximum items: ${selected.length}`);
  }

  const candidateMap = indexCandidatesById(candidates);
  const normalizedSelected = selected.map((selectedEntry, index) => {
    const candidateId = resolveCandidateId(selectedEntry);
    const candidate = candidateMap.get(candidateId);

    if (!candidate) {
      throw new Error(`ranking selected unknown candidate: ${candidateId}`);
    }

    return normalizeSelectedEntry(selectedEntry, candidate, index + 1);
  }).sort((left, right) => left.rank - right.rank);

  const duplicateIds = normalizedSelected.filter((entry, index, entries) => (
    entries.findIndex((other) => other.candidate_id === entry.candidate_id) !== index
  ));
  if (duplicateIds.length > 0) {
    throw new Error(`ranking selected duplicate candidate: ${duplicateIds[0].candidate_id}`);
  }

  return validateSelectionDocument({
    target_date: research.target_date,
    selected: normalizedSelected,
  });
}

async function runRankStage({
  targetDate,
  projectRoot = PROJECT_ROOT,
  repoScope = loadRepoScope(),
  invokeModel = defaultInvokeModel,
} = {}) {
  requireField(targetDate, 'targetDate');

  const paths = buildPipelinePaths(projectRoot, targetDate);
  const researchDocument = validateResearchDocument(
    JSON.parse(fs.readFileSync(paths.researchJsonPath, 'utf8')),
  );
  const selectionConfig = buildSelectionConfig(repoScope);
  const selectionDocument = await selectCandidates({
    research: researchDocument,
    invokeModel,
    minItems: selectionConfig.minimum_items || 5,
    maxItems: selectionConfig.maximum_items || 8,
    perRepoSoftCap: selectionConfig.per_repo_soft_cap || 2,
    workdir: projectRoot,
  });

  fs.mkdirSync(paths.workDir, { recursive: true });
  fs.writeFileSync(paths.selectionJsonPath, `${JSON.stringify(selectionDocument, null, 2)}\n`, 'utf8');

  return {
    outputPath: paths.selectionJsonPath,
    selectionDocument,
  };
}

if (require.main === module) {
  const targetDate = process.argv[2];

  runRankStage({ targetDate })
    .then(({ outputPath }) => {
      console.log(`Selection artifact written to ${outputPath}`);
    })
    .catch((error) => {
      console.error(`Rank stage failed: ${error.message}`);
      process.exit(1);
    });
}

module.exports = {
  buildRankingPrompt,
  buildSelectionConfig,
  createCandidateId,
  listPullRequestCandidates,
  loadRepoScope,
  runRankStage,
  selectCandidates,
};
