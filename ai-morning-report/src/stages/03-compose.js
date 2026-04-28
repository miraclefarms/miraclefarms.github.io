const fs = require('node:fs');
const path = require('node:path');

const {
  validateArticleSource,
  validateSelectionDocument,
} = require('../lib/article-schema');
const { invokeJsonModel } = require('../lib/openai');
const { buildPipelinePaths } = require('../lib/pipeline-paths');
const { loadAiMorningReportSkillConfig } = require('../lib/skill-config');

const PROJECT_ROOT = path.resolve(__dirname, '../../..');

function buildComposePrompt(selection, handoffConfig = {}) {
  const sectionsConfig = handoffConfig.article_source_shape
    && handoffConfig.article_source_shape.sections
    ? handoffConfig.article_source_shape.sections
    : {};
  const narrativeRules = handoffConfig.narrative_rules || {};

  return [
    '你是 MiracleFarms 的 AI Infra 早报作者。',
    '请根据 selection.json 生成 article-source.json 所需的结构化内容。',
    '必须保持中文 brief 风格，输出 JSON 对象。',
    '返回字段至少包含 title、intro、thesis、cover_prompt、sections、wechat.title、wechat.digest。',
    'title 必须使用 `AI Infra 早报｜` 前缀。',
    'sections 应围绕今天最值得记住的主线判断组织，而不是按 repo 机械罗列。',
    '每个 section 必须先写 summary，再写 items。',
    'section.summary 应该先解释这一组变化共同说明了什么，不能只是标题改写。',
    '每个 item 必须包含 title、analysis、references。',
    `item.analysis 必须是 ${narrativeRules.item_analysis_min_sentences || 2} 到 ${narrativeRules.item_analysis_max_sentences || 4} 句的自然段，说明“做了什么、为什么重要、对主链路意味着什么”。`,
    sectionsConfig.items_require_analysis_paragraph
      ? '严禁把 item.analysis 写成列点、短语堆叠或一句话卡片。'
      : null,
    narrativeRules.section_first_judgment
      ? '先给判断，再展开证据；每节都要先讲这组变化为何值得写。'
      : null,
    narrativeRules.avoid_bullet_like_copy
      ? '不要把文章写成 bullet list 风格。'
      : null,
    '',
    JSON.stringify({ writer_handoff: handoffConfig }, null, 2),
    '',
    JSON.stringify(selection, null, 2),
  ].filter(Boolean).join('\n');
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

function buildDryRunItemAnalysis(item, angle = '工程主线') {
  const reason = item.why_selected || '它已经开始影响生产主链路上的关键行为。';
  const repo = item.repo || '这个项目';

  if (item.category === 'performance') {
    return `${repo} 的这条更新把优化点直接推到了性能敏感的主路径上，而不是停留在边缘特性里。更重要的是，${reason}，这说明团队正在优先处理真正会影响吞吐、延迟或资源利用率的系统摩擦。`;
  }

  if (item.category === 'stability') {
    return `${repo} 这次处理的不是表面上的兼容性小修，而是主链路里会反复暴露的稳定性问题。${reason}，意味着这类能力正在从“出问题再兜底”转向“默认就该可靠”。`;
  }

  if (item.category === 'feature') {
    return `${repo} 给出的不只是一个新能力开关，而是在为更复杂的生产场景补齐默认可用性。${reason}，真正值得写的是它把新能力和主链路绑定得更紧，而不是把它留在实验路径里。`;
  }

  return `${repo} 的这条更新不只是补了一个功能点，而是在重新定义 ${angle} 相关能力应该如何进入默认路径。${reason}，说明过去需要额外开关、特殊分支或人工判断才能稳定启用的能力，正在被主链路正面接住。`;
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
      summary: '今天最值得记住的信号，不是某一个特定 feature，而是越来越多复杂能力开始从旁路回流到默认执行路径。项目方不再满足于“能开开关跑起来”，而是希望这些能力直接进入主链路，变成默认可交付的系统行为。',
      items: firstGroup.map((item) => ({
        title: item.title,
        analysis: buildDryRunItemAnalysis(item, item.primary_angle || '默认路径'),
        references: [createReference(item, refIndex++)],
      })),
    },
  ];

  if (secondGroup.length > 0) {
    sections.push({
      heading: '二、性能、稳定性与可观测性继续向主链路收敛',
      summary: '另一条清晰的主线是，性能、稳定性和可观测性不再被视为互相独立的小修小补，而是在一起向生产主链路收敛。谁能把这些能力同时推到默认路径，谁就更接近真正可用的系统。',
      items: secondGroup.map((item) => ({
        title: item.title,
        analysis: buildDryRunItemAnalysis(item, item.primary_angle || '生产主链路'),
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
  skillConfig = loadAiMorningReportSkillConfig(),
} = {}) {
  const selectionDocument = validateSelectionDocument(selection);
  const handoffConfig = skillConfig.writer_handoff || {};
  const articleDraft = dryRun
    ? buildDryRunArticleDraft(selectionDocument)
    : await invokeModel({
      role: 'writing',
      workdir,
      prompt: buildComposePrompt(selectionDocument, handoffConfig),
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
  skillConfig = loadAiMorningReportSkillConfig(),
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
    skillConfig,
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
  buildDryRunItemAnalysis,
  composeArticleSource,
  runComposeStage,
};
