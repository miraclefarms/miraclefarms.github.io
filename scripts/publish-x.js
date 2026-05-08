#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const dotenv = require('dotenv');
const {
  buildFallbackThread,
  getPostFingerprint,
  loadPublishRecord,
  parseThreadJson,
  publishThread,
  savePublishRecord,
} = require('./lib/x-publisher');
const {
  generateThreadWithAI,
} = require('../ai-morning-report/src/stages/08-x-push');

const PROJECT_ROOT = path.join(__dirname, '..');
dotenv.config({ path: path.join(PROJECT_ROOT, '.env') });

function usage() {
  console.error('Usage: node scripts/publish-x.js <post-file> [--thread <thread-json-file>] [--dry-run] [--force] [--single]');
}

function parseArgs(argv) {
  const args = { postFile: null, threadFile: null, dryRun: false, force: false, single: false };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--dry-run') {
      args.dryRun = true;
    } else if (arg === '--force') {
      args.force = true;
    } else if (arg === '--single') {
      args.single = true;
    } else if (arg === '--thread') {
      args.threadFile = argv[i + 1];
      i += 1;
    } else if (!args.postFile) {
      args.postFile = arg;
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  return args;
}

function resolveDateFromPostFile(postFile) {
  const match = path.basename(postFile).match(/^(\d{4}-\d{2}-\d{2})-/);
  if (!match) {
    throw new Error(`Cannot derive date from post filename: ${postFile}`);
  }
  return match[1];
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.postFile) {
    usage();
    process.exit(1);
  }

  const postFile = path.resolve(PROJECT_ROOT, args.postFile);
  if (!fs.existsSync(postFile)) {
    throw new Error(`Post file not found: ${postFile}`);
  }

  let generatedThread;
  if (args.threadFile) {
    generatedThread = parseThreadJson(fs.readFileSync(path.resolve(PROJECT_ROOT, args.threadFile), 'utf8'));
  } else {
    try {
      console.log('[x-publish] Generating English thread with AI...');
      generatedThread = await generateThreadWithAI(resolveDateFromPostFile(postFile), postFile, PROJECT_ROOT);
    } catch (err) {
      if (process.env.X_ALLOW_DETERMINISTIC_FALLBACK === '1') {
        console.log(`[x-publish] AI generation failed, using deterministic fallback: ${err.message}`);
        generatedThread = buildFallbackThread(postFile);
      } else {
        throw new Error(`AI English X rewrite failed: ${err.message}`);
      }
    }
  }
  const postMode = args.single || process.env.X_POST_MODE === 'single' ? 'single' : 'thread';
  const thread = postMode === 'single' ? [generatedThread[0]] : generatedThread;

  console.log(`[x-publish] ${postMode === 'single' ? 'Single post' : 'Thread'} preview:`);
  thread.forEach((post, index) => {
    console.log(`\n--- ${index + 1}/${thread.length} (${post.length} chars) ---\n${post}`);
  });

  if (args.dryRun || process.env.X_DRY_RUN === '1') {
    console.log('\n[x-publish] Dry run only; nothing was posted.');
    return;
  }

  const fingerprint = getPostFingerprint(postFile);
  const record = loadPublishRecord(PROJECT_ROOT);
  if (!args.force && process.env.X_FORCE_POST !== '1' && record[postFile] && record[postFile].fingerprint === fingerprint) {
    console.log(`[x-publish] Already posted: ${record[postFile].firstPostUrl}`);
    return;
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
  savePublishRecord(PROJECT_ROOT, record);

  console.log(`[x-publish] Posted thread: ${firstPostUrl}`);
}

main().catch((err) => {
  console.error('[x-publish] Failed:', err.message);
  process.exit(1);
});
