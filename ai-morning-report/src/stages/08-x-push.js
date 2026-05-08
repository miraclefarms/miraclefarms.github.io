/**
 * Stage 8: Generate an English X thread and publish it.
 *
 * Usage: node 08-x-push.js <date> <post-file> <project-root>
 *
 * Requires env to actually post:
 *   X_USER_ACCESS_TOKEN  OAuth 2.0 user access token with tweet.write
 * Optional:
 *   X_USERNAME           Used only for logging the final URL
 *   X_DRY_RUN=1          Generate and print, but do not post
 *   X_POST_MODE=single   Publish only the first post instead of a reply thread
 *   X_AI_TIMEOUT_MS      Timeout for AI thread generation; defaults to 90000
 *   X_FORCE_POST=1       Repost even if the same post was already recorded
 */

const fs = require('fs');
const path = require('path');
const dotenv = require('dotenv');
const { runAI } = require('../lib/cli-adapter');
const {
  buildFallbackThread,
  getPostFingerprint,
  loadPublishRecord,
  parseThreadJson,
  publishThread,
  resolvePostUrl,
  savePublishRecord,
} = require('../../../scripts/lib/x-publisher');

function resolveXFormatterSkill(projectRoot) {
  const candidates = [
    path.join(projectRoot, '.agents', 'skills', 'x-formatter', 'SKILL.md'),
    path.join(projectRoot, '.codex', 'skills', 'x-formatter', 'SKILL.md'),
    path.join(projectRoot, '.claude', 'skills', 'x-formatter', 'SKILL.md'),
  ];

  const skillPath = candidates.find((candidate) => fs.existsSync(candidate));
  if (!skillPath) {
    throw new Error('x-formatter skill not found');
  }
  return skillPath;
}

function buildXThreadPrompt({ date, postContent, postUrl }) {
  return `Use the injected x-formatter skill to condense and rewrite this MiracleFarms GitHub.io article into an English-only X thread.

Output JSON only: a JSON array of strings.

Hard constraints:
- English-only; do not output Chinese.
- Condense the source article for X; do not translate sentence by sentence.
- First array item must include this canonical URL: ${postUrl}
- Every array item must be <= 260 characters.
- Do not invent facts beyond the source post.

Date: ${date}

<githubio-post>
${postContent}
</githubio-post>`;
}

async function generateThreadWithAI(date, postFile, projectRoot) {
  const postContent = fs.readFileSync(postFile, 'utf8');
  const postUrl = resolvePostUrl(postFile, process.env.X_SITE_URL || 'https://miraclefarms.github.io');
  const prompt = buildXThreadPrompt({ date, postContent, postUrl });
  const skillPath = resolveXFormatterSkill(projectRoot);

  const timeoutMs = Number.parseInt(process.env.X_AI_TIMEOUT_MS || '90000', 10);
  const raw = await runAI({ prompt, skillPath, timeoutMs });
  return parseThreadJson(raw);
}

async function pushX(date, postFile, projectRoot) {
  if (!fs.existsSync(postFile)) {
    throw new Error(`Post file not found: ${postFile}`);
  }

  dotenv.config({ path: path.join(projectRoot, '.env') });

  let thread;
  try {
    console.log('[x-push] Generating English thread with AI...');
    thread = await generateThreadWithAI(date, postFile, projectRoot);
  } catch (err) {
    if (process.env.X_ALLOW_DETERMINISTIC_FALLBACK === '1') {
      console.log(`[x-push] AI generation failed, using deterministic fallback: ${err.message}`);
      thread = buildFallbackThread(postFile);
    } else {
      throw new Error(`AI English X rewrite failed: ${err.message}`);
    }
  }

  const postMode = process.env.X_POST_MODE === 'single' ? 'single' : 'thread';
  if (postMode === 'single') {
    thread = [thread[0]];
  }

  console.log(`[x-push] ${postMode === 'single' ? 'Single post' : 'Thread'} preview:`);
  thread.forEach((post, index) => {
    console.log(`\n--- ${index + 1}/${thread.length} (${post.length} chars) ---\n${post}`);
  });

  if (process.env.X_DRY_RUN === '1') {
    console.log('\n[x-push] Dry run only; nothing was posted.');
    return { dryRun: true, thread };
  }

  const fingerprint = getPostFingerprint(postFile);
  const record = loadPublishRecord(projectRoot);
  if (process.env.X_FORCE_POST !== '1' && record[postFile] && record[postFile].fingerprint === fingerprint) {
    console.log(`[x-push] Already posted: ${record[postFile].firstPostUrl}`);
    return { skipped: true, thread };
  }

  const results = await publishThread(thread);
  const username = process.env.X_USERNAME || '';
  const firstPostUrl = username
    ? `https://x.com/${username.replace(/^@/, '')}/status/${results[0].id}`
    : `https://x.com/i/web/status/${results[0].id}`;

  record[postFile] = {
    fingerprint,
    postedAt: new Date().toISOString(),
    firstPostUrl,
    postIds: results.map((item) => item.id),
  };
  savePublishRecord(projectRoot, record);

  console.log(`[x-push] Posted thread: ${firstPostUrl}`);
  return { firstPostUrl, thread };
}

if (require.main === module) {
  const [date, postFile, projectRoot] = process.argv.slice(2);
  if (!date || !postFile || !projectRoot) {
    console.error('Usage: node 08-x-push.js <date> <post-file> <project-root>');
    process.exit(1);
  }

  pushX(date, postFile, projectRoot)
    .then(() => console.log('[x-push] Done'))
    .catch((err) => {
      console.error('[x-push] Failed:', err.message);
      process.exit(1);
    });
}

module.exports = {
  buildXThreadPrompt,
  generateThreadWithAI,
  pushX,
  resolveXFormatterSkill,
};
