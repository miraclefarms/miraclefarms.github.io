const path = require('node:path');

const DEFAULT_SLUG = 'ai-infra-daily-brief';

function buildPipelinePaths(projectRoot, targetDate, slug = DEFAULT_SLUG) {
  if (!projectRoot) {
    throw new Error('projectRoot is required');
  }

  if (!targetDate) {
    throw new Error('targetDate is required');
  }

  const datedSlug = `${targetDate}-${slug}`;
  const workDir = path.join(projectRoot, '.cache', 'ai-morning-report', targetDate);
  const assetDir = path.join(projectRoot, 'assets', slug, datedSlug);
  const githubPostPath = path.join(projectRoot, '_posts', `${datedSlug}.md`);
  const wechatPostPath = path.join(projectRoot, 'docs', 'wechat', `${datedSlug}-wechat.md`);

  return {
    workDir,
    researchJsonPath: path.join(workDir, 'research.json'),
    selectionJsonPath: path.join(workDir, 'selection.json'),
    articleSourceJsonPath: path.join(workDir, 'article-source.json'),
    publishRecordJsonPath: path.join(workDir, 'publish-record.json'),
    githubPostPath,
    wechatPostPath,
    assetDir,
  };
}

module.exports = {
  DEFAULT_SLUG,
  buildPipelinePaths,
};
