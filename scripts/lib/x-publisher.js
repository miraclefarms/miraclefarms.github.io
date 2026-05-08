const fs = require('fs');
const path = require('path');
const https = require('https');
const crypto = require('crypto');

const DEFAULT_SITE_URL = 'https://miraclefarms.github.io';
const MAX_POST_LENGTH = 280;
const SAFE_POST_LENGTH = 260;

function parseFrontMatter(content) {
  const match = content.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
  if (!match) {
    return { frontMatter: {}, body: content };
  }

  const frontMatter = {};
  for (const line of match[1].split('\n')) {
    const colon = line.indexOf(':');
    if (colon <= 0) {
      continue;
    }
    const key = line.slice(0, colon).trim();
    let value = line.slice(colon + 1).trim();
    value = value.replace(/^['"]|['"]$/g, '');
    frontMatter[key] = value;
  }

  return { frontMatter, body: match[2] };
}

function resolvePostUrl(postFile, siteUrl = DEFAULT_SITE_URL) {
  const basename = path.basename(postFile, path.extname(postFile));
  const match = basename.match(/^(\d{4})-(\d{2})-(\d{2})-(.+)$/);
  if (!match) {
    throw new Error(`Cannot derive post URL from filename: ${postFile}`);
  }

  const [, year, month, day, slug] = match;
  return `${siteUrl.replace(/\/+$/, '')}/notes/${year}/${month}/${day}/${slug}/`;
}

function extractHeadings(body) {
  return body
    .split('\n')
    .map((line) => line.match(/^##\s+(.+?)\s*$/))
    .filter(Boolean)
    .map((match) => match[1].trim())
    .filter((heading) => heading !== '参考来源')
    .map((heading) => heading.replace(/^[一二三四五六七八九十]+、/, '').trim());
}

function stripMarkdown(input) {
  return input
    .replace(/\[\[?\d+\]?\]\([^)]+\)/g, '')
    .replace(/<a\s+href="[^"]+">\[\d+\]<\/a>/g, '')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/[*_`>#]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function truncatePost(text, maxLength = SAFE_POST_LENGTH) {
  const normalized = text.replace(/\s+\n/g, '\n').trim();
  if (normalized.length <= maxLength) {
    return normalized;
  }
  return `${normalized.slice(0, Math.max(0, maxLength - 1)).trimEnd()}…`;
}

function composePostWithUrl(text, url, maxLength = MAX_POST_LENGTH) {
  const normalizedUrl = url.trim();
  const separator = '\n\n';
  const availableTextLength = maxLength - normalizedUrl.length - separator.length;
  if (availableTextLength < 20) {
    throw new Error(`URL is too long for a post: ${normalizedUrl}`);
  }

  return `${truncatePost(text, availableTextLength)}${separator}${normalizedUrl}`;
}

function buildFallbackThread(postFile, options = {}) {
  const siteUrl = options.siteUrl || process.env.X_SITE_URL || DEFAULT_SITE_URL;
  const content = fs.readFileSync(postFile, 'utf8');
  const { frontMatter, body } = parseFrontMatter(content);
  const url = resolvePostUrl(postFile, siteUrl);
  const title = frontMatter.title || path.basename(postFile, path.extname(postFile));
  const intro = stripMarkdown(frontMatter.intro || '');
  const headings = extractHeadings(body).slice(0, 4);

  const posts = [
    composePostWithUrl(`New MiracleFarms note:\n${title}\n\n${intro}`, url),
  ];

  if (headings.length > 0) {
    posts.push(truncatePost(`What it covers:\n${headings.map((heading) => `- ${heading}`).join('\n')}`));
  }

  posts.push(truncatePost('MiracleFarms tracks AI infrastructure: LLM systems, inference engines, GPU clusters, agents, and the business of compute.'));

  return normalizeThread(posts);
}

function normalizeThread(posts, maxLength = MAX_POST_LENGTH) {
  if (!Array.isArray(posts)) {
    throw new Error('Thread must be an array of strings');
  }

  const normalized = posts
    .map((post) => String(post || '').replace(/\r\n/g, '\n').trim())
    .filter(Boolean)
    .map((post) => truncatePost(post, maxLength));

  if (normalized.length === 0) {
    throw new Error('Thread is empty');
  }

  return normalized;
}

function parseThreadJson(raw) {
  const cleaned = raw
    .replace(/^\s*```(?:json)?\s*/i, '')
    .replace(/\s*```\s*$/i, '')
    .trim();
  const parsed = JSON.parse(cleaned);
  return normalizeThread(parsed);
}

function requestJson(method, url, token, payload) {
  return new Promise((resolve, reject) => {
    const body = payload == null ? null : Buffer.from(JSON.stringify(payload), 'utf8');
    const parsedUrl = new URL(url);
    const headers = {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    };
    if (body) {
      headers['Content-Length'] = body.length;
    }

    const req = https.request({
      hostname: parsedUrl.hostname,
      path: `${parsedUrl.pathname}${parsedUrl.search}`,
      method,
      headers,
    }, (res) => {
      let data = '';
      res.on('data', (chunk) => { data += chunk; });
      res.on('end', () => {
        let json = {};
        try {
          json = data ? JSON.parse(data) : {};
        } catch {
          json = { raw: data };
        }

        if (res.statusCode < 200 || res.statusCode >= 300) {
          const detail = json.detail || json.title || json.error || data || `HTTP ${res.statusCode}`;
          reject(new Error(`X API ${res.statusCode}: ${detail}`));
          return;
        }

        resolve(json);
      });
    });
    req.on('error', reject);
    req.setTimeout(30000, () => req.destroy(new Error('X API request timed out')));
    if (body) {
      req.write(body);
    }
    req.end();
  });
}

async function createPost(text, options = {}) {
  const token = options.token || process.env.X_USER_ACCESS_TOKEN;
  const apiBaseUrl = options.apiBaseUrl || process.env.X_API_BASE_URL || 'https://api.x.com';
  if (!token) {
    throw new Error('X_USER_ACCESS_TOKEN is required to publish to X');
  }

  const payload = { text };
  if (options.replyToTweetId) {
    payload.reply = { in_reply_to_tweet_id: options.replyToTweetId };
  }

  return requestJson('POST', `${apiBaseUrl.replace(/\/+$/, '')}/2/tweets`, token, payload);
}

async function publishThread(posts, options = {}) {
  const normalized = normalizeThread(posts);
  const results = [];
  let replyToTweetId = null;

  for (const text of normalized) {
    const result = await createPost(text, { ...options, replyToTweetId });
    const tweetId = result && result.data && result.data.id;
    if (!tweetId) {
      throw new Error(`X API did not return a post id: ${JSON.stringify(result).slice(0, 200)}`);
    }
    results.push({ id: tweetId, text });
    replyToTweetId = tweetId;
  }

  return results;
}

function resolveGitDir(projectRoot) {
  const explicitGitDir = process.env.GIT_DIR;
  if (explicitGitDir) {
    return path.isAbsolute(explicitGitDir) ? explicitGitDir : path.join(projectRoot, explicitGitDir);
  }
  return path.join(projectRoot, '.git');
}

function getRecordFile(projectRoot) {
  return path.join(resolveGitDir(projectRoot), 'x-publish-record.local.json');
}

function loadPublishRecord(projectRoot) {
  const recordFile = getRecordFile(projectRoot);
  if (!fs.existsSync(recordFile)) {
    return {};
  }
  return JSON.parse(fs.readFileSync(recordFile, 'utf8'));
}

function savePublishRecord(projectRoot, record) {
  const recordFile = getRecordFile(projectRoot);
  fs.mkdirSync(path.dirname(recordFile), { recursive: true });
  fs.writeFileSync(recordFile, `${JSON.stringify(record, null, 2)}\n`, 'utf8');
}

function getPostFingerprint(postFile) {
  const content = fs.readFileSync(postFile, 'utf8');
  return crypto.createHash('sha256').update(content).digest('hex');
}

module.exports = {
  MAX_POST_LENGTH,
  buildFallbackThread,
  composePostWithUrl,
  createPost,
  getPostFingerprint,
  loadPublishRecord,
  normalizeThread,
  parseFrontMatter,
  parseThreadJson,
  publishThread,
  resolvePostUrl,
  savePublishRecord,
};
