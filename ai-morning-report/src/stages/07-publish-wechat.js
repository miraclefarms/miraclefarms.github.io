const path = require('node:path');

const { requireField } = require('../lib/article-schema');
const { buildPipelinePaths } = require('../lib/pipeline-paths');
const {
  loadPublishRecord,
  savePublishRecord,
  markWechatPublished,
  markWechatRetryPending,
  markWechatDryRun,
} = require('../lib/publish-record');
const { publishWechatDrafts } = require('../../../scripts/publish-wechat');

const PROJECT_ROOT = path.resolve(__dirname, '../../..');

async function runPublishWechatStage({
  targetDate,
  projectRoot = PROJECT_ROOT,
  publishWechat = publishWechatDrafts,
  dryRun = process.env.AI_MORNING_REPORT_DRY_RUN === '1',
} = {}) {
  requireField(targetDate, 'targetDate');

  const paths = buildPipelinePaths(projectRoot, targetDate);
  let publishRecord = loadPublishRecord(paths.publishRecordJsonPath, {
    targetDate,
    githubPostPath: paths.githubPostPath,
    wechatPostPath: paths.wechatPostPath,
  });

  if (dryRun) {
    publishRecord = markWechatDryRun(publishRecord, {
      markdownPath: paths.wechatPostPath,
    });
    savePublishRecord(paths.publishRecordJsonPath, publishRecord);
    return {
      dryRun: true,
      publishRecord,
    };
  }

  try {
    const result = await publishWechat({
      hookMode: false,
      targetDate,
      filePaths: [paths.wechatPostPath],
    });

    if (result.fatal || result.failed > 0 || result.success === 0) {
      throw new Error(result.fatal || 'WeChat draft publish did not succeed');
    }

    const draftMediaId = result.published && result.published[0]
      ? result.published[0].media_id || null
      : null;

    publishRecord = markWechatPublished(publishRecord, {
      markdownPath: paths.wechatPostPath,
      draftMediaId,
    });
    savePublishRecord(paths.publishRecordJsonPath, publishRecord);

    return {
      result,
      publishRecord,
    };
  } catch (error) {
    publishRecord = markWechatRetryPending(publishRecord, error, {
      markdownPath: paths.wechatPostPath,
    });
    savePublishRecord(paths.publishRecordJsonPath, publishRecord);
    throw error;
  }
}

if (require.main === module) {
  const targetDate = process.argv[2];

  runPublishWechatStage({ targetDate })
    .then((result) => {
      if (result.dryRun) {
        console.log('WeChat publish dry-run completed');
      } else {
        console.log('WeChat draft publish completed');
      }
    })
    .catch((error) => {
      console.error(`WeChat publish failed: ${error.message}`);
      process.exit(1);
    });
}

module.exports = {
  runPublishWechatStage,
};
