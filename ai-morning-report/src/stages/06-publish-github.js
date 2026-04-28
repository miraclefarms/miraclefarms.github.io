const fs = require('node:fs');
const path = require('node:path');
const { execFileSync } = require('node:child_process');

const { requireField } = require('../lib/article-schema');
const { buildPipelinePaths } = require('../lib/pipeline-paths');
const {
  loadPublishRecord,
  savePublishRecord,
  markGithubPublished,
  markGithubSkipped,
  markGithubDryRun,
} = require('../lib/publish-record');

const PROJECT_ROOT = path.resolve(__dirname, '../../..');

function createGitExecutor(projectRoot = PROJECT_ROOT) {
  return function execGit(args) {
    return execFileSync('git', args, {
      cwd: projectRoot,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe'],
    }).trim();
  };
}

function collectFilesForPublish(filePaths = []) {
  return filePaths.filter((filePath) => Boolean(filePath) && fs.existsSync(filePath));
}

function hasStagedChanges(execGit) {
  try {
    execGit(['diff', '--cached', '--quiet', '--exit-code']);
    return false;
  } catch (error) {
    if (error.status === 1) {
      return true;
    }
    throw error;
  }
}

function buildPublishFiles(paths) {
  return collectFilesForPublish([
    paths.githubPostPath,
    paths.wechatPostPath,
    paths.assetDir,
  ]);
}

async function runPublishGitHubStage({
  targetDate,
  projectRoot = PROJECT_ROOT,
  execGit = createGitExecutor(projectRoot),
  dryRun = process.env.AI_MORNING_REPORT_DRY_RUN === '1',
  gitRemote = process.env.AI_MORNING_REPORT_GIT_REMOTE || 'origin',
  gitTargetBranch = process.env.AI_MORNING_REPORT_GIT_TARGET_BRANCH || 'main',
} = {}) {
  requireField(targetDate, 'targetDate');

  const paths = buildPipelinePaths(projectRoot, targetDate);
  const files = buildPublishFiles(paths);

  if (files.length === 0) {
    throw new Error('no files to publish');
  }

  let publishRecord = loadPublishRecord(paths.publishRecordJsonPath, {
    targetDate,
    githubPostPath: paths.githubPostPath,
    wechatPostPath: paths.wechatPostPath,
  });

  if (dryRun) {
    publishRecord = markGithubDryRun(publishRecord, {
      postPath: paths.githubPostPath,
    });
    savePublishRecord(paths.publishRecordJsonPath, publishRecord);
    return {
      dryRun: true,
      files,
      publishRecord,
    };
  }

  execGit(['add', '--', ...files]);

  if (!hasStagedChanges(execGit)) {
    publishRecord = markGithubSkipped(publishRecord, {
      postPath: paths.githubPostPath,
    });
    savePublishRecord(paths.publishRecordJsonPath, publishRecord);
    return {
      skipped: true,
      files,
      publishRecord,
    };
  }

  execGit(['commit', '-m', `Add AI Infra morning brief for ${targetDate}`]);
  const commit = execGit(['rev-parse', 'HEAD']);
  execGit(['push', gitRemote, `HEAD:${gitTargetBranch}`]);

  publishRecord = markGithubPublished(publishRecord, {
    commit,
    postPath: paths.githubPostPath,
  });
  savePublishRecord(paths.publishRecordJsonPath, publishRecord);

  return {
    commit,
    files,
    publishRecord,
  };
}

if (require.main === module) {
  const targetDate = process.argv[2];

  runPublishGitHubStage({ targetDate })
    .then((result) => {
      if (result.dryRun) {
        console.log('GitHub publish dry-run completed');
      } else if (result.skipped) {
        console.log('GitHub publish skipped: no staged changes');
      } else {
        console.log(`GitHub publish completed at commit ${result.commit}`);
      }
    })
    .catch((error) => {
      console.error(`GitHub publish failed: ${error.message}`);
      process.exit(1);
    });
}

module.exports = {
  buildPublishFiles,
  collectFilesForPublish,
  createGitExecutor,
  hasStagedChanges,
  runPublishGitHubStage,
};
