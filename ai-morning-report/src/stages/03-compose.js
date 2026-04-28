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

async function composeArticleSource({
  selection,
  invokeModel = defaultInvokeModel,
  workdir,
} = {}) {
  const selectionDocument = validateSelectionDocument(selection);
  const articleDraft = await invokeModel({
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
  composeArticleSource,
  runComposeStage,
};
