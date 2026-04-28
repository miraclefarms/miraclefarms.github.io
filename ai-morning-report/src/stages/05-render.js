const fs = require('node:fs');
const path = require('node:path');

const { requireField, validateArticleSource } = require('../lib/article-schema');
const { buildPipelinePaths } = require('../lib/pipeline-paths');
const {
  insertGeneratedTitleImageMarkdown,
} = require('../../../scripts/lib/wechat-cover-prompts');

const PROJECT_ROOT = path.resolve(__dirname, '../../..');

function collectReferences(article) {
  const references = [];
  const seen = new Set();

  for (const section of article.sections || []) {
    for (const item of section.items || []) {
      for (const reference of item.references || []) {
        const key = `${reference.index}:${reference.url}`;
        if (!seen.has(key)) {
          seen.add(key);
          references.push(reference);
        }
      }
    }
  }

  return references.sort((left, right) => left.index - right.index);
}

function renderReferenceSuffix(references, formatter) {
  if (!Array.isArray(references) || references.length === 0) {
    return '';
  }

  return references.map(formatter).join('');
}

function renderSectionItems(section, referenceFormatter) {
  return (section.items || []).map((item) => {
    const title = item.title ? `**${item.title}**：` : '';
    const whatChanged = item.what_changed || '';
    const whyItMatters = item.why_it_matters || '';
    const references = renderReferenceSuffix(item.references, referenceFormatter);

    return [
      `${title}${whatChanged}${references}`,
      whyItMatters,
    ].filter(Boolean).join('\n\n');
  }).join('\n\n');
}

function renderSections(article, referenceFormatter) {
  return (article.sections || []).map((section) => {
    const body = renderSectionItems(section, referenceFormatter);

    return [
      `## ${section.heading}`,
      '',
      body,
    ].join('\n');
  }).join('\n\n');
}

function renderGithubMarkdown(article) {
  const references = collectReferences(article);
  const parts = [
    '---',
    `title: ${article.title}`,
    `date: ${article.target_date} 08:00:00 +0800`,
    `author: ${article.author || '荔枝不耐思'}`,
    `kind: ${article.kind || 'brief'}`,
    `category: ${article.category || 'Brief'}`,
    article.series ? `series: ${article.series}` : null,
    `intro: ${article.intro}`,
    '---',
    '',
    article.cover_asset && article.cover_asset.path ? `![题图](${article.cover_asset.path})` : null,
    article.intro,
    '',
    article.thesis,
    '',
    renderSections(article, reference => `[[${reference.index}]](${reference.url})`),
    '',
    '---',
    '',
    '## 参考来源',
    '',
    references.map(reference => `[${reference.index}] [${reference.title}](${reference.url})`).join('\n\n'),
  ].filter(Boolean);

  return `${parts.join('\n')}\n`;
}

function renderWechatBody(article) {
  const references = collectReferences(article);

  const body = [
    `# ${article.wechat && article.wechat.title ? article.wechat.title : article.title}`,
    '',
    `**📅 ${article.target_date}**`,
    '',
    `> ${article.wechat && article.wechat.digest ? article.wechat.digest : article.intro}`,
    '',
    '---',
    '',
    article.intro,
    '',
    article.thesis,
    '',
    renderSections(article, reference => `[${reference.index}]`),
    '',
    '---',
    '',
    '## 参考',
    '',
    references.map(reference => `[${reference.index}] ${reference.title}：${reference.url}`).join('\n\n'),
  ].filter(Boolean).join('\n');

  if (article.cover_asset && article.cover_asset.path) {
    return insertGeneratedTitleImageMarkdown(body, article.cover_asset.path);
  }

  return body;
}

function renderWechatMarkdown(article) {
  const body = renderWechatBody(article);
  const author = article.wechat && article.wechat.author
    ? article.wechat.author
    : (article.author || '荔枝不耐思');

  return [
    '---',
    `title: ${article.wechat && article.wechat.title ? article.wechat.title : article.title}`,
    `date: ${article.target_date} 08:00:00 +0800`,
    `author: ${author}`,
    `kind: ${article.kind || 'brief'}`,
    `category: ${article.category || 'Brief'}`,
    `intro: ${article.wechat && article.wechat.digest ? article.wechat.digest : article.intro}`,
    '---',
    '',
    body,
    '',
  ].join('\n');
}

function renderOutputs({ article } = {}) {
  const articleSource = validateArticleSource(article);

  return {
    githubMarkdown: renderGithubMarkdown(articleSource),
    wechatMarkdown: renderWechatMarkdown(articleSource),
  };
}

async function runRenderStage({
  targetDate,
  projectRoot = PROJECT_ROOT,
} = {}) {
  requireField(targetDate, 'targetDate');

  const paths = buildPipelinePaths(projectRoot, targetDate);
  const articleSource = validateArticleSource(
    JSON.parse(fs.readFileSync(paths.articleSourceJsonPath, 'utf8')),
  );
  const { githubMarkdown, wechatMarkdown } = renderOutputs({ article: articleSource });

  fs.mkdirSync(path.dirname(paths.githubPostPath), { recursive: true });
  fs.mkdirSync(path.dirname(paths.wechatPostPath), { recursive: true });
  fs.writeFileSync(paths.githubPostPath, githubMarkdown, 'utf8');
  fs.writeFileSync(paths.wechatPostPath, wechatMarkdown, 'utf8');

  return {
    githubPostPath: paths.githubPostPath,
    wechatPostPath: paths.wechatPostPath,
    githubMarkdown,
    wechatMarkdown,
  };
}

if (require.main === module) {
  const targetDate = process.argv[2];

  runRenderStage({ targetDate })
    .then(({ githubPostPath, wechatPostPath }) => {
      console.log(`GitHub post written to ${githubPostPath}`);
      console.log(`WeChat post written to ${wechatPostPath}`);
    })
    .catch((error) => {
      console.error(`Render stage failed: ${error.message}`);
      process.exit(1);
    });
}

module.exports = {
  collectReferences,
  renderGithubMarkdown,
  renderOutputs,
  renderSections,
  renderWechatBody,
  renderWechatMarkdown,
  runRenderStage,
};
