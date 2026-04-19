#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { execFileSync } = require('child_process');
const axios = require('axios');
const FormData = require('form-data');
const juice = require('juice');
const { marked } = require('marked');
const dotenv = require('dotenv');

const PROJECT_ROOT = path.join(__dirname, '..');
const WECHAT_API_BASE = 'https://api.weixin.qq.com/cgi-bin';
const OPENROUTER_API_BASE = 'https://openrouter.ai/api/v1';
const LOG_FILE = path.join(PROJECT_ROOT, 'docs', 'wechat-publish.log');
const DEFAULT_THUMB_IMAGE = path.join(PROJECT_ROOT, 'assets', 'icons', 'favicon.png');
const TRACKED_RECORD_FILE = path.join(__dirname, 'wechat-publish-record.json');
const DEFAULT_OPENROUTER_IMAGE_MODEL = 'google/gemini-2.5-flash-image';
const DEFAULT_OPENROUTER_IMAGE_SIZE = '1K';

function resolveGitDir() {
  let gitDir = process.env.GIT_DIR;

  if (!gitDir) {
    try {
      gitDir = execFileSync('git', ['rev-parse', '--git-dir'], {
        cwd: PROJECT_ROOT,
        encoding: 'utf8',
      }).trim();
    } catch (err) {
      gitDir = path.join(PROJECT_ROOT, '.git');
    }
  }

  if (!path.isAbsolute(gitDir)) {
    gitDir = path.join(PROJECT_ROOT, gitDir);
  }

  return gitDir;
}

const LOCAL_RECORD_FILE = path.join(resolveGitDir(), 'wechat-publish-record.local.json');
const GENERATED_IMAGE_DIR = path.join(resolveGitDir(), 'wechat-generated-images');

function log(message) {
  const timestamp = new Date().toISOString().replace('T', ' ').substring(0, 19);
  const entry = `[${timestamp}] ${message}\n`;
  fs.appendFileSync(LOG_FILE, entry);
  console.error(entry.trim());
}

function parseFrontMatter(content) {
  const match = content.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
  if (!match) return { frontMatter: {}, body: content };

  const frontMatter = {};
  match[1].split('\n').forEach(line => {
    const [key, ...valueParts] = line.split(':');
    if (key && valueParts.length > 0) {
      frontMatter[key.trim()] = valueParts.join(':').trim();
    }
  });

  return { frontMatter, body: match[2] };
}

function getImageContentType(imagePath) {
  switch (path.extname(imagePath).toLowerCase()) {
    case '.jpg':
    case '.jpeg':
      return 'image/jpeg';
    case '.webp':
      return 'image/webp';
    case '.gif':
      return 'image/gif';
    default:
      return 'image/png';
  }
}

function getImageExtensionForMimeType(mimeType) {
  switch (mimeType.toLowerCase()) {
    case 'image/jpeg':
      return 'jpg';
    case 'image/webp':
      return 'webp';
    case 'image/gif':
      return 'gif';
    default:
      return 'png';
  }
}

function parseDataUrlImage(dataUrl) {
  const match = dataUrl.match(/^data:([^;]+);base64,(.+)$/);
  if (!match) {
    throw new Error('OpenRouter returned an unsupported image payload');
  }

  return {
    mimeType: match[1],
    buffer: Buffer.from(match[2], 'base64'),
  };
}

function extractTitleImagePrompts(markdown) {
  const lines = markdown.split('\n');

  for (let i = 0; i < lines.length; i++) {
    const chineseMatch = lines[i].match(/^>\s*中文[:：]\s*(.+?)\s*$/);
    if (!chineseMatch) {
      continue;
    }

    let cursor = i + 1;
    while (cursor < lines.length && /^>\s*$/.test(lines[cursor])) {
      cursor++;
    }

    if (cursor >= lines.length) {
      return null;
    }

    const englishMatch = lines[cursor].match(/^>\s*English[:：]\s*(.+?)\s*$/i);
    if (!englishMatch) {
      return null;
    }

    return {
      chinese: chineseMatch[1].trim(),
      english: englishMatch[1].trim(),
      startLine: i,
      endLine: cursor,
    };
  }

  return null;
}

function injectTitleImageIntoMarkdown(markdown, imageUrl) {
  const prompts = extractTitleImagePrompts(markdown);
  if (!prompts) {
    return markdown;
  }

  const lines = markdown.split('\n');
  lines.splice(prompts.startLine, prompts.endLine - prompts.startLine + 1, `![题图](${imageUrl})`);
  return lines.join('\n');
}

function buildOpenRouterImagePrompt({ title, digest, prompts }) {
  const parts = [
    'Create a polished editorial cover image for a WeChat AI infrastructure newsletter article.',
    `Article title: ${title}`,
  ];

  if (digest) {
    parts.push(`Article summary: ${digest}`);
  }

  if (prompts.english) {
    parts.push(`Primary visual brief: ${prompts.english}`);
  }

  if (prompts.chinese) {
    parts.push(`Chinese visual brief for reference: ${prompts.chinese}`);
  }

  parts.push('Requirements: landscape 16:9 cover image, no text, no letters, no logos, no watermark, no UI chrome, cinematic but credible technology editorial style, strong focal subject, clean composition, suitable for both article hero image and thumbnail.');

  return parts.join('\n');
}

async function generateImageWithOpenRouter({ title, digest, prompts }) {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    throw new Error('OPENROUTER_API_KEY not found in .env');
  }

  const model = process.env.OPENROUTER_IMAGE_MODEL || DEFAULT_OPENROUTER_IMAGE_MODEL;
  const imageSize = process.env.OPENROUTER_IMAGE_SIZE || DEFAULT_OPENROUTER_IMAGE_SIZE;
  const payload = {
    model,
    messages: [
      {
        role: 'user',
        content: buildOpenRouterImagePrompt({ title, digest, prompts }),
      },
    ],
    modalities: ['image', 'text'],
    stream: false,
    image_config: {
      aspect_ratio: '16:9',
      image_size: imageSize,
    },
  };
  const headers = {
    Authorization: `Bearer ${apiKey}`,
    'Content-Type': 'application/json',
    'HTTP-Referer': process.env.OPENROUTER_HTTP_REFERER || 'https://miraclefarms.github.io',
    'X-Title': process.env.OPENROUTER_X_TITLE || 'MiracleFarms WeChat Publisher',
  };

  const response = await axios.post(`${OPENROUTER_API_BASE}/chat/completions`, payload, {
    headers,
    timeout: 180000,
  });
  const message = response.data && Array.isArray(response.data.choices)
    ? response.data.choices[0] && response.data.choices[0].message
    : null;
  const firstImage = message && Array.isArray(message.images) ? message.images[0] : null;
  const imageUrl = firstImage && firstImage.image_url ? firstImage.image_url.url : null;

  if (!imageUrl) {
    throw new Error(`OpenRouter did not return an image for model ${model}`);
  }

  return parseDataUrlImage(imageUrl);
}

async function ensureGeneratedTitleImage({ filePath, contentHash, title, digest, prompts }) {
  fs.mkdirSync(GENERATED_IMAGE_DIR, { recursive: true });

  const cachePrefix = `${path.basename(filePath, path.extname(filePath))}-${contentHash.slice(0, 12)}`;
  const existingFile = fs.readdirSync(GENERATED_IMAGE_DIR).find(file => file.startsWith(`${cachePrefix}.`));
  if (existingFile) {
    return path.join(GENERATED_IMAGE_DIR, existingFile);
  }

  const generated = await generateImageWithOpenRouter({ title, digest, prompts });
  const ext = getImageExtensionForMimeType(generated.mimeType);
  const targetPath = path.join(GENERATED_IMAGE_DIR, `${cachePrefix}.${ext}`);
  fs.writeFileSync(targetPath, generated.buffer);
  return targetPath;
}

function extractMarkdownTitle(markdown) {
  const lines = markdown.split('\n');
  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) {
      continue;
    }
    if (line.startsWith('# ')) {
      return line.slice(2).trim();
    }
  }
  return '';
}

function resolveArticleTitle(frontMatter, body) {
  return frontMatter.title || extractMarkdownTitle(body) || 'Untitled';
}

function normalizeContentForHash(content) {
  if (!content.startsWith('---\n')) {
    return content;
  }

  const end = content.indexOf('\n---\n', 4);
  if (end === -1) {
    return content;
  }

  const frontMatterBlock = content.slice(4, end);
  const body = content.slice(end + 5);
  const normalizedFrontMatter = frontMatterBlock
    .split('\n')
    .filter(line => !line.startsWith('wechat_published:'))
    .join('\n');

  if (!normalizedFrontMatter.trim()) {
    return body;
  }

  return `---\n${normalizedFrontMatter}\n---\n${body}`;
}

function computeContentHash(content) {
  return crypto
    .createHash('sha256')
    .update(normalizeContentForHash(content), 'utf8')
    .digest('hex');
}

function computeLegacyContentHash(content) {
  let normalized = content;

  if (content.startsWith('---\n')) {
    const end = content.indexOf('\n---\n', 4);
    if (end !== -1) {
      const frontMatterBlock = content.slice(4, end);
      const body = content.slice(end + 5);
      const normalizedFrontMatter = frontMatterBlock
        .split('\n')
        .filter(line => !line.startsWith('wechat_published:'))
        .join('\n');
      normalized = `---\n${normalizedFrontMatter}\n---\n${body}`;
    }
  }

  return crypto
    .createHash('sha256')
    .update(normalized, 'utf8')
    .digest('hex');
}

function normalizeRecord(record) {
  if (!record || typeof record !== 'object') {
    return { version: 1, sent: [] };
  }

  return {
    version: record.version || 1,
    sent: Array.isArray(record.sent) ? record.sent : [],
  };
}

function loadRecordFile(recordFile) {
  if (!fs.existsSync(recordFile)) {
    return null;
  }

  try {
    return normalizeRecord(JSON.parse(fs.readFileSync(recordFile, 'utf8')));
  } catch (err) {
    log(`Failed to parse publish record ${recordFile}: ${err.message}`);
    return null;
  }
}

function mergeRecords(...records) {
  const merged = { version: 1, sent: [] };
  const seen = new Set();

  records.filter(Boolean).forEach(record => {
    normalizeRecord(record).sent.forEach(entry => {
      if (!entry || typeof entry !== 'object') {
        return;
      }

      const file = entry.file || '';
      const contentHash = entry.content_hash || '';
      const dedupeKey = `${file}::${contentHash}`;

      if (seen.has(dedupeKey)) {
        return;
      }

      seen.add(dedupeKey);
      merged.sent.push(entry);
    });
  });

  return merged;
}

function loadPublishRecord() {
  const trackedRecord = loadRecordFile(TRACKED_RECORD_FILE);
  const localRecord = loadRecordFile(LOCAL_RECORD_FILE);
  const mergedRecord = mergeRecords(trackedRecord, localRecord);

  if (!localRecord && mergedRecord.sent.length > 0) {
    savePublishRecord(mergedRecord);
  }

  return mergedRecord;
}

function savePublishRecord(record) {
  fs.mkdirSync(path.dirname(LOCAL_RECORD_FILE), { recursive: true });
  fs.writeFileSync(LOCAL_RECORD_FILE, JSON.stringify(record, null, 2) + '\n');
}

function hasBeenPublished(record, contentHashes) {
  const hashSet = new Set(Array.isArray(contentHashes) ? contentHashes : [contentHashes]);
  return record.sent.some(entry => hashSet.has(entry.content_hash));
}

function recordPublished(record, filePath, contentHash, title, response) {
  const relativePath = path.relative(PROJECT_ROOT, filePath);
  record.sent.push({
    file: relativePath,
    title,
    content_hash: contentHash,
    published_at: new Date().toISOString(),
    media_id: response && response.media_id ? response.media_id : null,
  });
  savePublishRecord(record);
}

async function getAccessToken(appid, appsecret) {
  const url = `${WECHAT_API_BASE}/token?grant_type=client_credential&appid=${appid}&secret=${appsecret}`;
  const response = await axios.get(url);
  if (response.data.errcode) {
    throw new Error(`access_token failed: ${response.data.errcode} - ${response.data.errmsg}`);
  }
  return response.data.access_token;
}

const DOOCS_CSS = `
section {
  font-family: -apple-system-font, BlinkMacSystemFont, Helvetica Neue, PingFang SC, Hiragino Sans GB, Microsoft YaHei UI, Microsoft YaHei, Arial, sans-serif;
  font-size: 16px;
  line-height: 1.75;
  color: #24292f;
  max-width: 100%;
  word-break: break-word;
}
h1 { display: table; margin: 2em auto 1em; padding: 0 1em; border-bottom: 2px solid #009B77; font-size: 1.3em; text-align: center; color: #009B77; }
h2 { display: table; margin: 1.5em auto 1em; padding: 0.3em 0.8em; background: #009B77; color: #fff; font-size: 1.2em; text-align: center; border-radius: 4px; }
h3 { padding-left: 8px; border-left: 3px solid #009B77; margin: 1.5em 8px 0.75em 0; font-size: 1.1em; color: #009B77; }
h4, h5, h6 { margin: 1.5em 8px 0.5em; color: #009B77; font-size: 1em; }
p { margin: 1em 0; letter-spacing: 0.05em; }
blockquote { font-style: normal; padding: 0.8em 1em; border-left: 4px solid #009B77; border-radius: 4px; background: #f6f8fa; margin: 1em 0; color: #576b95; }
blockquote p { margin: 0.5em 0; }
code { font-size: 90%; color: #d14; background: rgba(27, 31, 35, 0.05); padding: 2px 6px; border-radius: 4px; font-family: Consolas, Monaco, Andale Mono, monospace; }
pre { background: #f6f8fa; border-radius: 8px; overflow-x: auto; margin: 1em 0; padding: 0; border: 1px solid rgba(0,0,0,0.1); }
pre code { background: none; padding: 1em; color: inherit; border-radius: 0; display: block; }
a { color: #576b95; text-decoration: none; }
strong { color: #009B77; font-weight: bold; }
ul { list-style: circle; padding-left: 1.5em; margin: 1em 0; }
ol { padding-left: 1.5em; margin: 1em 0; list-style: decimal; }
li { margin: 0.3em 0; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 90%; }
th, td { border: 1px solid #dfdfdf; padding: 0.5em 0.75em; text-align: left; }
th { background: rgba(0,0,0,0.03); font-weight: 600; }
img { display: block; max-width: 100%; margin: 1em auto; border-radius: 4px; }
hr { border-style: solid; border-width: 2px 0 0; border-color: rgba(0,0,0,0.1); height: 0.4em; margin: 2em 0; }
`;

function markdownToHtml(markdown) {
  marked.setOptions({
    gfm: true,
    breaks: true,
  });
  const bodyHtml = marked.parse(markdown);
  const htmlWithStyleTag = '<style>' + DOOCS_CSS + '</style><section>' + bodyHtml + '</section>';
  return juice(htmlWithStyleTag);
}

async function uploadImage(accessToken, imagePath) {
  const formData = new FormData();
  formData.append('media', fs.createReadStream(imagePath), {
    filename: path.basename(imagePath),
    contentType: getImageContentType(imagePath),
  });

  const url = `${WECHAT_API_BASE}/material/add_material?access_token=${accessToken}&type=image`;
  const response = await axios.post(url, formData, {
    headers: formData.getHeaders(),
  });
  if (response.data.errcode) {
    throw new Error(`material/add_material failed: ${response.data.errcode} - ${response.data.errmsg}`);
  }
  return response.data.media_id;
}

async function uploadContentImage(accessToken, imagePath) {
  const formData = new FormData();
  formData.append('media', fs.createReadStream(imagePath), {
    filename: path.basename(imagePath),
    contentType: getImageContentType(imagePath),
  });

  const url = `${WECHAT_API_BASE}/media/uploadimg?access_token=${accessToken}`;
  const response = await axios.post(url, formData, {
    headers: formData.getHeaders(),
  });
  if (response.data.errcode) {
    throw new Error(`media/uploadimg failed: ${response.data.errcode} - ${response.data.errmsg}`);
  }
  if (!response.data.url) {
    throw new Error('media/uploadimg did not return a content image URL');
  }
  return response.data.url;
}

async function addDraft(accessToken, article) {
  const url = `${WECHAT_API_BASE}/draft/add?access_token=${accessToken}`;
  const payload = {
    articles: [
      {
        title: article.title,
        author: article.author,
        digest: article.digest,
        content: article.content,
        content_source_url: '',
        thumb_media_id: article.thumb_media_id,
        need_open_comment: 1,
        only_fans_can_comment: 0,
      },
    ],
  };
  const response = await axios.post(url, payload);
  if (response.data.errcode) {
    throw new Error(`add_draft failed: ${response.data.errcode} - ${response.data.errmsg}`);
  }
  return response.data;
}

function getTargetDate() {
  if (process.env.WECHAT_TARGET_DATE) {
    return process.env.WECHAT_TARGET_DATE;
  }

  return new Intl.DateTimeFormat('en-CA', {
    timeZone: 'Asia/Shanghai',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).format(new Date());
}

function runGit(args, allowFailure = false) {
  try {
    return execFileSync('git', args, {
      cwd: PROJECT_ROOT,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe'],
    }).trim();
  } catch (err) {
    if (allowFailure) {
      return '';
    }
    throw err;
  }
}

function listFilesInPush() {
  const upstreamRef = runGit(
    ['rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{upstream}'],
    true,
  ) || 'origin/main';
  const diffOutput = runGit(['diff', '--name-only', `${upstreamRef}..HEAD`], true);

  return diffOutput
    .split('\n')
    .map(line => line.trim())
    .filter(Boolean);
}

function resolveExpectedWechatFiles(changedFiles, targetDate) {
  const expectedFiles = new Set();

  changedFiles.forEach(file => {
    if (file.startsWith(`docs/wechat/${targetDate}-`) && file.endsWith('-wechat.md')) {
      expectedFiles.add(path.join(PROJECT_ROOT, file));
      return;
    }

    if (!file.startsWith(`_posts/${targetDate}-ai-infra-daily-brief-`) || !file.endsWith('.md')) {
      return;
    }

    const basename = path.basename(file, '.md');
    const slug = basename.slice(`${targetDate}-`.length);
    expectedFiles.add(path.join(PROJECT_ROOT, 'docs', 'wechat', `${targetDate}-${slug}-wechat.md`));
  });

  return Array.from(expectedFiles).sort();
}

function listWechatFilesForDate(targetDate) {
  const wechatDir = path.join(PROJECT_ROOT, 'docs', 'wechat');
  if (!fs.existsSync(wechatDir)) {
    return [];
  }

  return fs.readdirSync(wechatDir)
    .filter(file => file.startsWith(`${targetDate}-`) && file.endsWith('-wechat.md'))
    .sort()
    .map(file => path.join(wechatDir, file));
}

function scanWechatDir(record, files) {
  const unpublished = [];

  files.forEach(filePath => {
    const content = fs.readFileSync(filePath, 'utf-8');
    const { frontMatter } = parseFrontMatter(content);
    const contentHash = computeContentHash(content);
    const legacyContentHash = computeLegacyContentHash(content);
    const publishedInRecord = hasBeenPublished(record, [contentHash, legacyContentHash]);

    if (publishedInRecord) {
      return;
    }

    if (frontMatter.wechat_published === 'true') {
      log(`Warning: ${path.basename(filePath)} is marked wechat_published=true but no matching publish record was found. Treating it as unpublished.`);
    }

    unpublished.push({ filePath, content, contentHash });
  });

  return unpublished;
}

async function main() {
  dotenv.config();
  const hookMode = process.argv.includes('--hook');
  const targetDate = getTargetDate();
  const changedFiles = hookMode ? listFilesInPush() : [];
  const expectedWechatFiles = hookMode ? resolveExpectedWechatFiles(changedFiles, targetDate) : [];
  const publishExpected = hookMode && expectedWechatFiles.length > 0;
  const publishRecord = loadPublishRecord();
  const targetWechatFiles = publishExpected ? expectedWechatFiles : listWechatFilesForDate(targetDate);
  const missingExpectedFiles = expectedWechatFiles.filter(filePath => !fs.existsSync(filePath));
  const unpublished = scanWechatDir(publishRecord, targetWechatFiles.filter(filePath => fs.existsSync(filePath)));

  if (publishExpected && missingExpectedFiles.length > 0) {
    return {
      success: 0,
      failed: missingExpectedFiles.length,
      fatal: `Expected WeChat article file(s) missing: ${missingExpectedFiles.map(filePath => path.relative(PROJECT_ROOT, filePath)).join(', ')}`,
      publishExpected,
      targetDate,
    };
  }

  if (unpublished.length === 0) {
    console.error(`No unpublished wechat articles found for ${targetDate}.`);
    return {
      success: 0,
      failed: 0,
      publishExpected,
      targetDate,
    };
  }

  const appid = process.env.WECHAT_APPID;
  const appsecret = process.env.WECHAT_APPSECRET;
  const thumbMediaId = process.env.WECHAT_THUMB_MEDIA_ID;

  if (!appid || !appsecret) {
    const message = 'WECHAT_APPID or WECHAT_APPSECRET not found in .env';
    log(message);
    return {
      success: 0,
      failed: unpublished.length,
      fatal: message,
      publishExpected,
      targetDate,
    };
  }

  let accessToken;
  try {
    accessToken = await getAccessToken(appid, appsecret);
  } catch (err) {
    const message = `Failed to get access_token: ${err.message}`;
    log(message);
    return {
      success: 0,
      failed: unpublished.length,
      fatal: message,
      publishExpected,
      targetDate,
    };
  }

  let effectiveThumbMediaId = thumbMediaId || null;
  if (!effectiveThumbMediaId) {
    log('WECHAT_THUMB_MEDIA_ID not found in .env. Will upload the default image only when an article needs a fallback thumb.');
  }

  async function ensureDefaultThumbMediaId() {
    if (effectiveThumbMediaId) {
      return effectiveThumbMediaId;
    }

    try {
      effectiveThumbMediaId = await uploadImage(accessToken, DEFAULT_THUMB_IMAGE);
      console.error(`📤 Uploaded default thumb image, media_id: ${effectiveThumbMediaId}`);
      return effectiveThumbMediaId;
    } catch (err) {
      throw new Error(`Failed to upload default thumb image: ${err.message}`);
    }
  }

  const results = {
    success: 0,
    failed: 0,
    publishExpected,
    targetDate,
  };

  for (const { filePath, content, contentHash } of unpublished) {
    const fileName = path.basename(filePath);
    try {
      const { frontMatter, body } = parseFrontMatter(content);
      const title = resolveArticleTitle(frontMatter, body);
      const author = frontMatter.author || '';
      const digest = frontMatter.intro || '';
      let articleBody = body;
      let articleThumbMediaId = effectiveThumbMediaId;
      const titleImagePrompts = extractTitleImagePrompts(body);

      if (titleImagePrompts) {
        const generatedImagePath = await ensureGeneratedTitleImage({
          filePath,
          contentHash,
          title,
          digest,
          prompts: titleImagePrompts,
        });
        const contentImageUrl = await uploadContentImage(accessToken, generatedImagePath);
        articleThumbMediaId = await uploadImage(accessToken, generatedImagePath);
        articleBody = injectTitleImageIntoMarkdown(body, contentImageUrl);
        console.error(`🖼️ Generated and attached title image for ${fileName}`);
      } else {
        articleThumbMediaId = await ensureDefaultThumbMediaId();
      }

      const htmlContent = markdownToHtml(articleBody);

      const response = await addDraft(accessToken, { title, author, digest, content: htmlContent, thumb_media_id: articleThumbMediaId });
      recordPublished(publishRecord, filePath, contentHash, title, response);

      console.error(`✅ Published: ${fileName}`);
      results.success++;
    } catch (err) {
      log(`${fileName}\nReason: ${err.message}\n---`);
      results.failed++;
    }
  }

  return results;
}

main().then(results => {
  if (results.fatal) {
    console.error(`❌ ${results.fatal}`);
  }

  if (results.success > 0 || results.failed > 0) {
    const msg = results.failed === 0 && !results.fatal
      ? `✅ Published ${results.success} article(s) to WeChat draft`
      : results.success === 0
        ? `❌ All ${results.failed} article(s) failed (see docs/wechat-publish.log)`
        : `⚠️ ${results.success} succeeded, ${results.failed} failed (see docs/wechat-publish.log)`;
    console.error(msg);
  }

  const shouldFail =
    Boolean(results.fatal) ||
    results.failed > 0;

  process.exit(shouldFail ? 1 : 0);
}).catch(err => {
  log(`Unexpected error: ${err.message}`);
  console.error(`❌ Unexpected error: ${err.message}`);
  process.exit(1);
});
