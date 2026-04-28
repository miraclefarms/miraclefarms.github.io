const fs = require('node:fs');
const path = require('node:path');

const { validatePublishRecord } = require('./article-schema');

function createPublishRecord({ targetDate, githubPostPath = null, wechatPostPath = null } = {}) {
  if (!targetDate) {
    throw new Error('targetDate is required');
  }

  return validatePublishRecord({
    target_date: targetDate,
    github: {
      status: 'pending',
      post_path: githubPostPath,
    },
    wechat: {
      status: 'pending',
      markdown_path: wechatPostPath,
    },
  });
}

function loadPublishRecord(recordPath, defaults = {}) {
  if (!recordPath) {
    throw new Error('recordPath is required');
  }

  if (!fs.existsSync(recordPath)) {
    return createPublishRecord(defaults);
  }

  return validatePublishRecord(JSON.parse(fs.readFileSync(recordPath, 'utf8')));
}

function savePublishRecord(recordPath, record) {
  if (!recordPath) {
    throw new Error('recordPath is required');
  }

  fs.mkdirSync(path.dirname(recordPath), { recursive: true });
  fs.writeFileSync(
    recordPath,
    `${JSON.stringify(validatePublishRecord(record), null, 2)}\n`,
    'utf8',
  );

  return record;
}

function markGithubPublished(record, { commit, postPath = null } = {}) {
  return validatePublishRecord({
    ...record,
    github: {
      ...(record.github || {}),
      status: 'published',
      commit,
      post_path: postPath ?? record.github?.post_path ?? null,
      published_at: new Date().toISOString(),
    },
  });
}

function markGithubSkipped(record, { postPath = null, reason = 'no_changes' } = {}) {
  return validatePublishRecord({
    ...record,
    github: {
      ...(record.github || {}),
      status: 'skipped',
      reason,
      post_path: postPath ?? record.github?.post_path ?? null,
      published_at: record.github?.published_at || null,
    },
  });
}

function markGithubDryRun(record, { postPath = null } = {}) {
  return validatePublishRecord({
    ...record,
    github: {
      ...(record.github || {}),
      status: 'dry_run',
      post_path: postPath ?? record.github?.post_path ?? null,
      published_at: new Date().toISOString(),
    },
  });
}

function markWechatPublished(record, { markdownPath = null, draftMediaId = null } = {}) {
  return validatePublishRecord({
    ...record,
    wechat: {
      ...(record.wechat || {}),
      status: 'published',
      markdown_path: markdownPath ?? record.wechat?.markdown_path ?? null,
      draft_media_id: draftMediaId,
      last_error: null,
      last_attempt_at: new Date().toISOString(),
    },
  });
}

function markWechatRetryPending(record, error, { markdownPath = null } = {}) {
  return validatePublishRecord({
    ...record,
    wechat: {
      ...(record.wechat || {}),
      status: 'pending_retry',
      markdown_path: markdownPath ?? record.wechat?.markdown_path ?? null,
      draft_media_id: null,
      last_error: error.message,
      last_attempt_at: new Date().toISOString(),
    },
  });
}

function markWechatDryRun(record, { markdownPath = null } = {}) {
  return validatePublishRecord({
    ...record,
    wechat: {
      ...(record.wechat || {}),
      status: 'dry_run',
      markdown_path: markdownPath ?? record.wechat?.markdown_path ?? null,
      last_error: null,
      last_attempt_at: new Date().toISOString(),
    },
  });
}

module.exports = {
  createPublishRecord,
  loadPublishRecord,
  savePublishRecord,
  markGithubPublished,
  markGithubSkipped,
  markGithubDryRun,
  markWechatPublished,
  markWechatRetryPending,
  markWechatDryRun,
};
