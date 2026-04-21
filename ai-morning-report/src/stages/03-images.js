const fs = require('fs');
const path = require('path');
const axios = require('axios');
const PROJECT_ROOT = path.resolve(__dirname, '../../..');
const { loadWechatConfig } = require(path.join(PROJECT_ROOT, 'scripts/lib/wechat-config'));
const {
  buildWechatCoverImageRequest,
  insertGeneratedTitleImageMarkdown,
  loadWechatCoverPromptTemplates,
} = require(path.join(PROJECT_ROOT, 'scripts/lib/wechat-cover-prompts'));

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

function parseDataUrlImage(dataUrl) {
  const match = dataUrl.match(/^data:([^;]+);base64,(.+)$/);
  if (!match) throw new Error('Invalid data URL');
  return { mimeType: match[1], buffer: Buffer.from(match[2], 'base64') };
}

function getImageExtensionForMimeType(mimeType) {
  switch (mimeType.toLowerCase()) {
    case 'image/jpeg': return 'jpg';
    case 'image/webp': return 'webp';
    case 'image/gif': return 'gif';
    default: return 'png';
  }
}

async function generateCoverImage({ markdown, outputDir }) {
  const config = loadWechatConfig(PROJECT_ROOT);
  const templates = loadWechatCoverPromptTemplates(PROJECT_ROOT);
  const { frontMatter, body } = parseFrontMatter(markdown);
  const title = frontMatter.title || '';
  const digest = frontMatter.intro || '';

  const titleMatch = body.match(/^#\s+(.+)$/m);
  const articleTitle = title || (titleMatch ? titleMatch[1] : 'AI Infra Daily Brief');

  const promptPayload = buildWechatCoverImageRequest({
    frontMatter,
    title: articleTitle,
    digest,
    bodyMarkdown: body,
    registry: templates,
    model: config.openrouterImageModel,
    imageSize: config.openrouterImageSize,
  });

  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) throw new Error('OPENROUTER_API_KEY not found');

  const response = await axios.post(
    `${config.openrouterApiBaseUrl}/chat/completions`,
    promptPayload,
    {
      headers: {
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
        'HTTP-Referer': config.openrouterHttpReferer,
        'X-Title': config.openrouterXTitle,
      },
      timeout: config.openrouterImageTimeoutMs,
    }
  );

  const message = response.data?.choices?.[0]?.message;
  const firstImage = message?.images?.[0];
  const imageUrl = firstImage?.image_url?.url;

  if (!imageUrl) throw new Error('No image returned from OpenRouter');

  const parsed = parseDataUrlImage(imageUrl);
  const ext = getImageExtensionForMimeType(parsed.mimeType);
  const coverPath = path.join(outputDir, `cover.${ext}`);
  fs.writeFileSync(coverPath, parsed.buffer);

  const relativePath = `/assets/morning-report/${path.basename(outputDir)}/cover.${ext}`;
  const updatedMarkdown = insertGeneratedTitleImageMarkdown(markdown, relativePath);
  return { coverPath, coverExt: ext, updatedMarkdown };
}

async function processWechatArticle({ date, wechatPostPath, assetsDir }) {
  if (!fs.existsSync(wechatPostPath)) {
    throw new Error(`Wechat post not found: ${wechatPostPath}`);
  }

  fs.mkdirSync(assetsDir, { recursive: true });

  const markdown = fs.readFileSync(wechatPostPath, 'utf8');
  const { coverPath, updatedMarkdown } = await generateCoverImage({
    markdown,
    outputDir: assetsDir,
  });

  fs.writeFileSync(wechatPostPath, updatedMarkdown, 'utf8');

  return { coverPath, wechatPostPath };
}

async function fetchBodyImage({ url, targetPath }) {
  try {
    const response = await axios.get(url, { responseType: 'arraybuffer', timeout: 15000 });
    const contentType = response.headers['content-type'] || 'image/png';
    const ext = contentType.includes('jpeg') || contentType.includes('jpg') ? 'jpg'
      : contentType.includes('webp') ? 'webp'
      : contentType.includes('gif') ? 'gif'
      : 'png';
    const finalPath = targetPath.endsWith(ext) ? targetPath : `${targetPath}.${ext}`;
    fs.writeFileSync(finalPath, Buffer.from(response.data));
    return finalPath;
  } catch (err) {
    console.warn(`Failed to fetch image from ${url}:`, err.message);
    return null;
  }
}

async function processGitHubPostImages({ postPath, assetsDir }) {
  if (!fs.existsSync(postPath)) {
    return { fetchedImages: [] };
  }

  fs.mkdirSync(assetsDir, { recursive: true });

  const markdown = fs.readFileSync(postPath, 'utf8');
  const imageCandidates = [];

  const imageRegex = /!\[([^\]]*)\]\(([^)]+)\)/g;
  let match;
  while ((match = imageRegex.exec(markdown)) !== null) {
    const url = match[2];
    if (url.startsWith('http') && !url.includes('openrouter')) {
      imageCandidates.push({ alt: match[1], url });
    }
  }

  const fetchedImages = [];
  for (const candidate of imageCandidates) {
    const filename = candidate.url.split('/').pop() || `img-${Date.now()}.png`;
    const targetPath = path.join(assetsDir, filename);
    const result = await fetchBodyImage({ url: candidate.url, targetPath });
    if (result) {
      fetchedImages.push(result);
    }
  }

  return { fetchedImages };
}

if (require.main === module) {
  const date = process.argv[2] || new Date().toISOString().slice(0, 10);
  const wechatPostPath = process.argv[3];
  const assetsDir = process.argv[4] || `/tmp/morning-report/${date}/assets`;

  if (!wechatPostPath) {
    console.error('Usage: node 03-images.js <date> <wechat-post-path> [assets-dir]');
    process.exit(1);
  }

  processWechatArticle({ date, wechatPostPath, assetsDir })
    .then((result) => console.log('Wechat cover generated:', JSON.stringify(result)))
    .catch((err) => {
      console.error('Wechat cover generation failed:', err.message);
      process.exit(1);
    });
}

module.exports = { generateCoverImage, processWechatArticle, fetchBodyImage, processGitHubPostImages };