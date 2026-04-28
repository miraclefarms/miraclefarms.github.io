const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const assert = require('node:assert/strict');

const {
  createPublishRecord,
  loadPublishRecord,
  markGithubPublished,
  markWechatRetryPending,
  savePublishRecord,
} = require('../../ai-morning-report/src/lib/publish-record');
const { buildPipelinePaths } = require('../../ai-morning-report/src/lib/pipeline-paths');
const { runPublishGitHubStage } = require('../../ai-morning-report/src/stages/06-publish-github');
const { runPublishWechatStage } = require('../../ai-morning-report/src/stages/07-publish-wechat');

test('markGithubPublished stores a published commit SHA', () => {
  const record = markGithubPublished(
    createPublishRecord({ targetDate: '2026-04-27' }),
    { commit: 'abc1234' },
  );
  assert.equal(record.github.status, 'published');
  assert.equal(record.github.commit, 'abc1234');
});

test('markWechatRetryPending stores the last error without clearing github success', () => {
  const record = markWechatRetryPending(
    { ...createPublishRecord({ targetDate: '2026-04-27' }), github: { status: 'published', commit: 'abc1234' } },
    new Error('upload timeout'),
  );
  assert.equal(record.github.status, 'published');
  assert.equal(record.wechat.status, 'pending_retry');
  assert.equal(record.wechat.last_error, 'upload timeout');
});

test('runPublishGitHubStage records published status via injected git executor', async () => {
  const projectRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'ai-morning-report-publish-'));
  const targetDate = '2026-04-27';
  const paths = buildPipelinePaths(projectRoot, targetDate);

  fs.mkdirSync(path.dirname(paths.githubPostPath), { recursive: true });
  fs.mkdirSync(path.dirname(paths.wechatPostPath), { recursive: true });
  fs.mkdirSync(paths.assetDir, { recursive: true });
  fs.writeFileSync(paths.githubPostPath, 'github');
  fs.writeFileSync(paths.wechatPostPath, 'wechat');
  fs.writeFileSync(path.join(paths.assetDir, 'cover.png'), 'cover');

  const gitCalls = [];
  const execGit = (args) => {
    gitCalls.push(args);
    if (args[0] === 'rev-parse') return 'deadbeef';
    if (args[0] === 'diff') {
      const error = new Error('changes present');
      error.status = 1;
      throw error;
    }
    return '';
  };

  const result = await runPublishGitHubStage({
    targetDate,
    projectRoot,
    execGit,
  });

  const saved = loadPublishRecord(paths.publishRecordJsonPath, { targetDate });
  assert.equal(result.commit, 'deadbeef');
  assert.equal(saved.github.status, 'published');
  assert.equal(saved.github.commit, 'deadbeef');
  assert.ok(gitCalls.some((args) => args[0] === 'add'));
  assert.ok(gitCalls.some((args) => args[0] === 'push'));
});

test('runPublishWechatStage records pending retry on publisher failure while preserving github success', async () => {
  const projectRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'ai-morning-report-wechat-'));
  const targetDate = '2026-04-27';
  const paths = buildPipelinePaths(projectRoot, targetDate);

  fs.mkdirSync(path.dirname(paths.publishRecordJsonPath), { recursive: true });
  savePublishRecord(
    paths.publishRecordJsonPath,
    {
      ...createPublishRecord({
        targetDate,
        githubPostPath: paths.githubPostPath,
        wechatPostPath: paths.wechatPostPath,
      }),
      github: {
        status: 'published',
        commit: 'abc1234',
      },
    },
  );

  await assert.rejects(
    () => runPublishWechatStage({
      targetDate,
      projectRoot,
      publishWechat: async () => ({ success: 0, failed: 1, fatal: 'upload timeout' }),
    }),
    /upload timeout/,
  );

  const saved = loadPublishRecord(paths.publishRecordJsonPath, { targetDate });
  assert.equal(saved.github.status, 'published');
  assert.equal(saved.wechat.status, 'pending_retry');
  assert.equal(saved.wechat.last_error, 'upload timeout');
});
