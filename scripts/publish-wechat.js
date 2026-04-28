#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { execFileSync } = require('child_process');
const axios = require('axios');
const FormData = require('form-data');
const dotenv = require('dotenv');
const {
  renderWechatHtml,
  resolveMarkdownImagePath,
  resolveWechatRenderProfile,
  rewriteMarkdownImageUrls,
} = require('./lib/wechat-render');
const {
  buildWechatCoverImageRequest,
  insertGeneratedTitleImageMarkdown,
  loadWechatCoverPromptTemplates,
} = require('./lib/wechat-cover-prompts');
const {
  loadWechatConfig,
} = require('./lib/wechat-config');

const PROJECT_ROOT = path.join(__dirname, '..');
const CONFIG = loadWechatConfig(PROJECT_ROOT);
const COVER_PROMPT_REGISTRY = loadWechatCoverPromptTemplates(PROJECT_ROOT);
const LOG_FILE = path.join(PROJECT_ROOT, 'docs', 'wechat-publish.log');
const TRACKED_RECORD_FILE = path.join(__dirname, 'wechat-publish-record.json');

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
const WECHAT_ASSET_ROOT = path.join(PROJECT_ROOT, 'docs', 'wechat', 'assets');

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

function slugifyEnglishText(input) {
  return input
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .replace(/-{2,}/g, '-');
}

function parseWechatArticleIdentity(filePath) {
  const basename = path.basename(filePath, path.extname(filePath));
  const match = basename.match(/^(\d{4}-\d{2}-\d{2})-(.+?)(?:-wechat)?$/);

  if (!match) {
    return {
      date: getTargetDate(),
      slug: slugifyEnglishText(basename) || 'wechat-article',
    };
  }

  const normalizedSlug = match[2].replace(/^ai-infra-daily-brief-/, '');

  return {
    date: match[1],
    slug: slugifyEnglishText(normalizedSlug) || 'wechat-article',
  };
}

function getGeneratedImageBasePath(filePath) {
  const identity = parseWechatArticleIdentity(filePath);
  return path.join(WECHAT_ASSET_ROOT, identity.date, `${identity.slug}-cover`);
}

function findGeneratedImageVariant(basePath) {
  const candidates = ['png', 'jpg', 'jpeg', 'webp', 'gif'];
  return candidates
    .map(ext => `${basePath}.${ext}`)
    .find(candidate => fs.existsSync(candidate)) || null;
}

function clearGeneratedImageVariants(basePath) {
  const candidates = ['png', 'jpg', 'jpeg', 'webp', 'gif'];
  candidates.forEach(ext => {
    const candidate = `${basePath}.${ext}`;
    if (fs.existsSync(candidate)) {
      fs.unlinkSync(candidate);
    }
  });
}

function extractTitleImageMarkdown(markdown) {
  const lines = markdown.split('\n');
  const separatorIndex = lines.findIndex(line => line.trim() === '---');
  const searchEnd = separatorIndex === -1 ? Math.min(lines.length, 20) : separatorIndex;

  for (let i = 0; i < searchEnd; i++) {
    const match = lines[i].match(/^!\[([^\]]*)\]\(([^)]+)\)\s*$/);
    if (!match) {
      continue;
    }

    return {
      alt: match[1].trim(),
      src: match[2].trim(),
      startLine: i,
      endLine: i,
    };
  }

  return null;
}

function replaceTitleImageMarkdown(markdown, nextSrc, altText = '题图') {
  const currentImage = extractTitleImageMarkdown(markdown);
  if (!currentImage) {
    return markdown;
  }

  const lines = markdown.split('\n');
  const alt = currentImage.alt || altText;
  lines.splice(currentImage.startLine, currentImage.endLine - currentImage.startLine + 1, `![${alt}](${nextSrc})`);
  return lines.join('\n');
}

function replaceMarkdownBody(content, newBody) {
  const match = content.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
  if (!match) {
    return newBody;
  }

  return `---\n${match[1]}\n---\n${newBody}`;
}

function toMarkdownRelativePath(fromFilePath, targetPath) {
  return path.relative(path.dirname(fromFilePath), targetPath).split(path.sep).join('/');
}

async function generateImageWithOpenRouter({ frontMatter, title, digest, bodyMarkdown }) {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    throw new Error('OPENROUTER_API_KEY not found in .env');
  }

  const model = CONFIG.openrouterImageModel;
  const imageSize = CONFIG.openrouterImageSize;
  const payload = buildWechatCoverImageRequest({
    frontMatter,
    title,
    digest,
    bodyMarkdown,
    registry: COVER_PROMPT_REGISTRY,
    model,
    imageSize,
  });
  const headers = {
    Authorization: `Bearer ${apiKey}`,
    'Content-Type': 'application/json',
    'HTTP-Referer': CONFIG.openrouterHttpReferer,
    'X-Title': CONFIG.openrouterXTitle,
  };

  const response = await axios.post(`${CONFIG.openrouterApiBaseUrl}/chat/completions`, payload, {
    headers,
    timeout: CONFIG.openrouterImageTimeoutMs,
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

async function ensureGeneratedTitleImage({ filePath, frontMatter, title, digest, bodyMarkdown }) {
  const articleStat = fs.statSync(filePath);
  const basePath = getGeneratedImageBasePath(filePath);
  const targetDir = path.dirname(basePath);
  const existingFile = findGeneratedImageVariant(basePath);

  fs.mkdirSync(targetDir, { recursive: true });

  if (existingFile) {
    const imageStat = fs.statSync(existingFile);
    if (imageStat.mtimeMs >= articleStat.mtimeMs) {
      return existingFile;
    }
  }

  const generated = await generateImageWithOpenRouter({ frontMatter, title, digest, bodyMarkdown });
  const ext = getImageExtensionForMimeType(generated.mimeType);
  const targetPath = `${basePath}.${ext}`;
  clearGeneratedImageVariants(basePath);
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
  const url = `${CONFIG.wechatApiBaseUrl}/token?grant_type=client_credential&appid=${appid}&secret=${appsecret}`;
  const response = await axios.get(url);
  if (response.data.errcode) {
    throw new Error(`access_token failed: ${response.data.errcode} - ${response.data.errmsg}`);
  }
  return response.data.access_token;
}

async function uploadImage(accessToken, imagePath) {
  const formData = new FormData();
  formData.append('media', fs.createReadStream(imagePath), {
    filename: path.basename(imagePath),
    contentType: getImageContentType(imagePath),
  });

  const url = `${CONFIG.wechatApiBaseUrl}/material/add_material?access_token=${accessToken}&type=image`;
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

  const url = `${CONFIG.wechatApiBaseUrl}/media/uploadimg?access_token=${accessToken}`;
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
  const url = `${CONFIG.wechatApiBaseUrl}/draft/add?access_token=${accessToken}`;
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
    timeZone: CONFIG.wechatTargetTimezone,
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
    if (file.startsWith('docs/wechat/') && file.endsWith('-wechat.md')) {
      expectedFiles.add(path.join(PROJECT_ROOT, file));
      return;
    }

    const briefPostMatch = file.match(/^_posts\/(\d{4}-\d{2}-\d{2})-(ai-infra-daily-brief-.+)\.md$/);
    if (!briefPostMatch) {
      return;
    }

    const [, fileDate, slug] = briefPostMatch;
    expectedFiles.add(path.join(PROJECT_ROOT, 'docs', 'wechat', `${fileDate}-${slug}-wechat.md`));
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

async function main(options = {}) {
  dotenv.config();
  const hookMode = options.hookMode ?? process.argv.includes('--hook');
  const targetDate = options.targetDate || getTargetDate();
  const explicitFiles = Array.isArray(options.filePaths) ? options.filePaths : null;
  const changedFiles = hookMode ? listFilesInPush() : [];
  const expectedWechatFiles = explicitFiles || (hookMode ? resolveExpectedWechatFiles(changedFiles, targetDate) : []);
  const publishExpected = explicitFiles ? expectedWechatFiles.length > 0 : hookMode && expectedWechatFiles.length > 0;
  const publishRecord = loadPublishRecord();
  const targetWechatFiles = explicitFiles
    ? expectedWechatFiles
    : (publishExpected ? expectedWechatFiles : listWechatFilesForDate(targetDate));
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
      effectiveThumbMediaId = await uploadImage(accessToken, CONFIG.defaultThumbImagePath);
      console.error(`📤 Uploaded default thumb image, media_id: ${effectiveThumbMediaId}`);
      return effectiveThumbMediaId;
    } catch (err) {
      throw new Error(`Failed to upload default thumb image: ${err.message}`);
    }
  }

  const results = {
    success: 0,
    failed: 0,
    published: [],
    publishExpected,
    targetDate,
  };

  for (const { filePath, content, contentHash } of unpublished) {
    const fileName = path.basename(filePath);
    try {
      const { frontMatter, body } = parseFrontMatter(content);
      const renderProfile = resolveWechatRenderProfile(frontMatter);
      const title = resolveArticleTitle(frontMatter, body);
      const author = frontMatter.author || '';
      const digest = frontMatter.intro || '';
      let articleBody = body;
      let persistedContent = content;
      let persistedContentHash = contentHash;
      let articleThumbMediaId = effectiveThumbMediaId;
      let titleImage = extractTitleImageMarkdown(articleBody);

      if (!titleImage) {
        const generatedImagePath = await ensureGeneratedTitleImage({
          filePath,
          frontMatter,
          title,
          digest,
          bodyMarkdown: body,
        });
        const markdownImagePath = toMarkdownRelativePath(filePath, generatedImagePath);
        articleBody = insertGeneratedTitleImageMarkdown(body, markdownImagePath);
        persistedContent = replaceMarkdownBody(content, articleBody);
        fs.writeFileSync(filePath, persistedContent);
        persistedContentHash = computeContentHash(persistedContent);
        console.error(`🖼️ Generated and persisted title image for ${fileName}`);
        titleImage = extractTitleImageMarkdown(articleBody);
      }

      if (titleImage) {
        const localImagePath = resolveMarkdownImagePath(filePath, titleImage.src, PROJECT_ROOT);
        if (!localImagePath || !fs.existsSync(localImagePath)) {
          throw new Error(`Title image path could not be resolved: ${titleImage.src}`);
        }

        const contentImageUrl = await uploadContentImage(accessToken, localImagePath);
        articleThumbMediaId = await uploadImage(accessToken, localImagePath);
        articleBody = replaceTitleImageMarkdown(articleBody, contentImageUrl, titleImage.alt || '题图');
        console.error(`🖼️ Attached title image for ${fileName}`);
      } else {
        articleThumbMediaId = await ensureDefaultThumbMediaId();
      }

      if (renderProfile.keepInlineImages) {
        const rewrittenImages = await rewriteMarkdownImageUrls({
          markdown: articleBody,
          articleFilePath: filePath,
          projectRoot: PROJECT_ROOT,
          uploadImage: imagePath => uploadContentImage(accessToken, imagePath),
        });
        articleBody = rewrittenImages.markdown;
      }

      const htmlContent = renderWechatHtml(articleBody, { themeName: renderProfile.themeName });

      const response = await addDraft(accessToken, { title, author, digest, content: htmlContent, thumb_media_id: articleThumbMediaId });
      recordPublished(publishRecord, filePath, persistedContentHash, title, response);
      results.published.push({
        filePath,
        title,
        media_id: response.media_id || null,
      });

      console.error(`✅ Published: ${fileName}`);
      results.success++;
    } catch (err) {
      log(`${fileName}\nReason: ${err.message}\n---`);
      results.failed++;
    }
  }

  return results;
}

function runCli() {
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
}

module.exports = {
  getTargetDate,
  main,
  publishWechatDrafts: main,
  resolveExpectedWechatFiles,
  runCli,
};

if (require.main === module) {
  runCli();
}
