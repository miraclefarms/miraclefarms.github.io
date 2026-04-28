const fs = require('node:fs');
const path = require('node:path');

const {
  validateArticleSource,
  validateSelectionDocument,
} = require('../lib/article-schema');
const { invokeJsonModel } = require('../lib/openai');
const { buildPipelinePaths } = require('../lib/pipeline-paths');

const PROJECT_ROOT = path.resolve(__dirname, '../../..');

function buildComposePrompt(selection) {
  return [
    '你是 MiracleFarms 的 AI Infra 早报作者。',
    '请根据 selection.json 生成 article-source.json 所需的结构化内容。',
    '必须保持中文 brief 风格，输出 JSON 对象。',
    '返回字段至少包含 title、intro、thesis、cover_prompt、sections、wechat.title、wechat.digest。',
    'title 必须使用 `AI Infra 早报｜` 前缀。',
    'sections 应围绕今天最值得记住的主线判断组织，而不是按 repo 机械罗列。',
    '',
    JSON.stringify(selection, null, 2),
  ].join('\n');
}

async function defaultInvokeModel({ role, prompt, workdir }) {
  return invokeJsonModel({ role, prompt, workdir });
}

function createReference(item, fallbackIndex) {
  const evidence = Array.isArray(item.evidence) ? item.evidence[0] : null;
  return {
    index: fallbackIndex,
    title: item.title,
    url: evidence && evidence.url ? evidence.url : item.url,
  };
}

function buildDryRunArticleDraft(selectionDocument) {
  const selected = selectionDocument.selected || [];
  const midpoint = Math.max(1, Math.ceil(selected.length / 2));
  const firstGroup = selected.slice(0, midpoint);
  const secondGroup = selected.slice(midpoint);
  let refIndex = 1;

  const sections = [
    {
      heading: '一、默认路径开始吸收复杂能力',
      items: firstGroup.map((item) => ({
        title: item.title,
        what_changed: item.why_selected || '这条更新正在把更复杂的系统能力纳入默认路径。',
        why_it_matters: `${item.repo} 的这次变化说明，工程主线正在从“可选优化”转向“默认能力”。`,
        references: [createReference(item, refIndex++)],
      })),
    },
  ];

  if (secondGroup.length > 0) {
    sections.push({
      heading: '二、性能、稳定性与可观测性继续向主链路收敛',
      items: secondGroup.map((item) => ({
        title: item.title,
        what_changed: item.why_selected || '这条更新继续强化了生产路径上的性能或稳定性。',
        why_it_matters: `${item.repo} 不再把这些能力留给旁路方案，而是直接推向主链路。`,
        references: [createReference(item, refIndex++)],
      })),
    });
  }

  return {
    title: 'AI Infra 早报｜默认路径开始吸收复杂场景',
    intro: '今天最值得记住的变化，不是某个单点 feature，而是默认执行路径正在承接过去只能靠特例处理的复杂能力。',
    thesis: '真正的信号是：AI Infra 的默认路径正在变得更完整，复杂场景开始从旁路回流到主链路。',
    cover_prompt: 'An editorial illustration showing the default execution path of an AI infrastructure system absorbing advanced runtime, cache, and stability capabilities into the main flow.',
    sections,
    wechat: {
      title: 'AI Infra 早报｜默认路径开始吸收复杂场景',
      digest: '复杂能力开始从特例回流到默认执行路径。',
    },
  };
}

async function composeArticleSource({
  selection,
  invokeModel = defaultInvokeModel,
  workdir,
  dryRun = process.env.AI_MORNING_REPORT_DRY_RUN === '1',
} = {}) {
  const selectionDocument = validateSelectionDocument(selection);
  const articleDraft = dryRun
    ? buildDryRunArticleDraft(selectionDocument)
    : await invokeModel({
      role: 'writing',
      workdir,
      prompt: buildComposePrompt(selectionDocument),
    });

  const articleSource = validateArticleSource({
    ...articleDraft,
    target_date: selectionDocument.target_date,
    kind: 'brief',
    author: '荔枝不耐思',
    category: 'Brief',
    series: 'ai-infra-daily-brief',
    wechat: {
      ...(articleDraft.wechat || {}),
      author: '荔枝不耐思',
      thumb_strategy: 'generated-cover',
    },
  });

  return articleSource;
}

async function runComposeStage({
  targetDate,
  projectRoot = PROJECT_ROOT,
  invokeModel = defaultInvokeModel,
} = {}) {
  if (!targetDate) {
    throw new Error('targetDate is required');
  }

  const paths = buildPipelinePaths(projectRoot, targetDate);
  const selectionDocument = validateSelectionDocument(
    JSON.parse(fs.readFileSync(paths.selectionJsonPath, 'utf8')),
  );
  const articleSource = await composeArticleSource({
    selection: selectionDocument,
    invokeModel,
    workdir: projectRoot,
  });

  fs.mkdirSync(paths.workDir, { recursive: true });
  fs.writeFileSync(paths.articleSourceJsonPath, `${JSON.stringify(articleSource, null, 2)}\n`, 'utf8');

  return {
    outputPath: paths.articleSourceJsonPath,
    articleSource,
  };
}

if (require.main === module) {
  const targetDate = process.argv[2];

  runComposeStage({ targetDate })
    .then(({ outputPath }) => {
      console.log(`Article source artifact written to ${outputPath}`);
    })
    .catch((error) => {
      console.error(`Compose stage failed: ${error.message}`);
      process.exit(1);
    });
}

module.exports = {
  buildComposePrompt,
  buildDryRunArticleDraft,
  composeArticleSource,
  runComposeStage,
};
