const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const {
  getTargetDateParts,
  buildResearchWindow,
} = require('../../ai-morning-report/src/lib/date-window');
const {
  buildPipelinePaths,
} = require('../../ai-morning-report/src/lib/pipeline-paths');

test('buildResearchWindow anchors to Asia/Shanghai natural day and 3-day lookback', () => {
  const now = new Date('2026-04-27T05:00:00+08:00');
  const parts = getTargetDateParts(now, 'Asia/Shanghai');
  const window = buildResearchWindow(parts.date, {
    timezone: 'Asia/Shanghai',
    lookbackDays: 3,
  });

  assert.equal(parts.date, '2026-04-27');
  assert.equal(window.start, '2026-04-25T00:00:00+08:00');
  assert.equal(window.end, '2026-04-27T23:59:59+08:00');
});

test('buildPipelinePaths returns explicit path-shaped contract names only', () => {
  const projectRoot = '/tmp/miraclefarms';
  const paths = buildPipelinePaths(projectRoot, '2026-04-27');

  assert.equal(paths.workDir, path.join(projectRoot, '.cache', 'ai-morning-report', '2026-04-27'));
  assert.equal(paths.researchJsonPath, path.join(paths.workDir, 'research.json'));
  assert.equal(paths.selectionJsonPath, path.join(paths.workDir, 'selection.json'));
  assert.equal(paths.articleSourceJsonPath, path.join(paths.workDir, 'article-source.json'));
  assert.equal(paths.publishRecordJsonPath, path.join(paths.workDir, 'publish-record.json'));
  assert.equal(
    paths.githubPostPath,
    path.join(projectRoot, '_posts', '2026-04-27-ai-infra-daily-brief.md'),
  );
  assert.equal(
    paths.wechatPostPath,
    path.join(projectRoot, 'docs', 'wechat', '2026-04-27-ai-infra-daily-brief-wechat.md'),
  );
  assert.equal(
    paths.assetDir,
    path.join(projectRoot, 'assets', 'ai-infra-daily-brief', '2026-04-27-ai-infra-daily-brief'),
  );
  assert.equal('researchJson' in paths, false);
  assert.equal('selectionJson' in paths, false);
  assert.equal('articleSourceJson' in paths, false);
  assert.equal('publishRecordJson' in paths, false);
  assert.equal('githubMarkdown' in paths, false);
  assert.equal('wechatMarkdown' in paths, false);
});
