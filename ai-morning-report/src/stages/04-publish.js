const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const PROJECT_ROOT = path.resolve(__dirname, '../../..');

function isGitDirty() {
  try {
    const status = execSync('git status --porcelain', { cwd: PROJECT_ROOT, encoding: 'utf8' });
    return status.trim().length > 0;
  } catch {
    return false;
  }
}

function getFilesToAdd(date) {
  try {
    const status = execSync('git status --porcelain -uall', { cwd: PROJECT_ROOT, encoding: 'utf8' });
    const allUntracked = status.split('\n').filter(line => line.startsWith('??')).map(line => line.slice(3).trim());

    const files = [];
    const morningReportDir = `morning-report/${date}`;

    for (const f of allUntracked) {
      if (f.includes(morningReportDir) ||
          f.includes(`_posts/${date}`) ||
          f.includes(`docs/wechat/${date}`)) {
        files.push(f);
      }
    }

    return files;
  } catch {
    return [];
  }
}

async function publishGitHub({ date, postPath, wechatPostPath, commitMessage }) {
  const files = getFilesToAdd(date);

  if (files.length === 0) {
    console.log('No files to publish for', date);
    return { published: false, message: 'No files to publish' };
  }

  const msg = commitMessage || `Add AI Infra morning brief for ${date}`;

  try {
    for (const f of files) {
      execSync(`git add "${f}"`, { cwd: PROJECT_ROOT });
    }
    execSync(`git commit -m "${msg}"`, { cwd: PROJECT_ROOT });
    execSync('git push', { cwd: PROJECT_ROOT });
    return { published: true, files, message: `Published ${files.length} file(s)` };
  } catch (err) {
    throw new Error(`Git publish failed: ${err.message}`);
  }
}

if (require.main === module) {
  const date = process.argv[2] || new Date().toISOString().slice(0, 10);
  const postPath = process.argv[3];
  const wechatPostPath = process.argv[4];

  const allPaths = [postPath, wechatPostPath].filter(Boolean);
  if (allPaths.some(p => !fs.existsSync(p))) {
    console.error('One or more post files not found');
    process.exit(1);
  }

  publishGitHub({ date, postPath, wechatPostPath })
    .then((result) => console.log('GitHub publish result:', JSON.stringify(result)))
    .catch((err) => {
      console.error('GitHub publish failed:', err.message);
      process.exit(1);
    });
}

module.exports = { publishGitHub, isGitDirty, getFilesToAdd };