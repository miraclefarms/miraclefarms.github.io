const fs = require('node:fs');
const path = require('node:path');

const axios = require('axios');

const { validateArticleSource, requireField } = require('../lib/article-schema');
const { buildPipelinePaths } = require('../lib/pipeline-paths');
const { loadWechatConfig } = require('../../../scripts/lib/wechat-config');

const PROJECT_ROOT = path.resolve(__dirname, '../../..');

function parseDataUrlImage(dataUrl) {
  const match = String(dataUrl || '').match(/^data:([^;]+);base64,(.+)$/);
  if (!match) {
    throw new Error('image response must be a data URL');
  }

  return {
    mimeType: match[1],
    buffer: Buffer.from(match[2], 'base64'),
  };
}

function getImageExtensionForMimeType(mimeType) {
  switch (String(mimeType || '').toLowerCase()) {
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

function toRepoAssetPath(projectRoot, absolutePath) {
  const relativePath = path.relative(projectRoot, absolutePath);

  if (!relativePath || relativePath.startsWith('..')) {
    throw new Error(`asset path must stay inside the repo: ${absolutePath}`);
  }

  return `/${relativePath.split(path.sep).join('/')}`;
}

function normalizeGeneratedImage(result) {
  if (Buffer.isBuffer(result)) {
    return {
      mimeType: 'image/png',
      buffer: result,
    };
  }

  if (typeof result === 'string') {
    return parseDataUrlImage(result);
  }

  if (result && Buffer.isBuffer(result.buffer)) {
    return {
      mimeType: result.mimeType || 'image/png',
      buffer: result.buffer,
    };
  }

  const imageUrl = result && result.choices
    && result.choices[0]
    && result.choices[0].message
    && Array.isArray(result.choices[0].message.images)
    && result.choices[0].message.images[0]
    && result.choices[0].message.images[0].image_url
    ? result.choices[0].message.images[0].image_url.url
    : null;

  if (imageUrl) {
    return parseDataUrlImage(imageUrl);
  }

  throw new Error('generateImage must return a Buffer, data URL, or OpenRouter image payload');
}

async function defaultGenerateImage(prompt, { projectRoot }) {
  requireField(prompt, 'cover_prompt');

  const config = loadWechatConfig(projectRoot);
  const apiKey = process.env.OPENROUTER_API_KEY;

  if (!apiKey) {
    throw new Error('OPENROUTER_API_KEY not found');
  }

  const response = await axios.post(
    `${config.openrouterApiBaseUrl}/chat/completions`,
    {
      model: config.openrouterImageModel,
      messages: [
        {
          role: 'user',
          content: prompt,
        },
      ],
      modalities: ['image', 'text'],
      stream: false,
      image_config: {
        aspect_ratio: '9:16',
        image_size: config.openrouterImageSize,
      },
    },
    {
      headers: {
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
        'HTTP-Referer': config.openrouterHttpReferer,
        'X-Title': config.openrouterXTitle,
      },
      timeout: config.openrouterImageTimeoutMs,
    },
  );

  return response.data;
}

async function generateCoverAsset({
  article,
  generateImage = defaultGenerateImage,
  assetDir,
  projectRoot = PROJECT_ROOT,
} = {}) {
  const articleSource = validateArticleSource(article);
  requireField(assetDir, 'assetDir');

  fs.mkdirSync(assetDir, { recursive: true });

  const imageResult = await generateImage(articleSource.cover_prompt, {
    article: articleSource,
    assetDir,
    projectRoot,
  });
  const { buffer, mimeType } = normalizeGeneratedImage(imageResult);
  const coverExtension = getImageExtensionForMimeType(mimeType);
  const coverPath = path.join(assetDir, `cover.${coverExtension}`);

  fs.writeFileSync(coverPath, buffer);

  return {
    article: {
      ...articleSource,
      cover_asset: {
        status: 'generated',
        path: toRepoAssetPath(projectRoot, coverPath),
        mime_type: mimeType,
        generated_at: new Date().toISOString(),
      },
    },
    coverPath,
  };
}

async function runImageStage({
  targetDate,
  projectRoot = PROJECT_ROOT,
  generateImage = defaultGenerateImage,
} = {}) {
  requireField(targetDate, 'targetDate');

  const paths = buildPipelinePaths(projectRoot, targetDate);
  const articleSource = validateArticleSource(
    JSON.parse(fs.readFileSync(paths.articleSourceJsonPath, 'utf8')),
  );
  const { article, coverPath } = await generateCoverAsset({
    article: articleSource,
    generateImage,
    assetDir: paths.assetDir,
    projectRoot,
  });

  fs.mkdirSync(paths.workDir, { recursive: true });
  fs.writeFileSync(paths.articleSourceJsonPath, `${JSON.stringify(article, null, 2)}\n`, 'utf8');

  return {
    outputPath: paths.articleSourceJsonPath,
    coverPath,
    articleSource: article,
  };
}

if (require.main === module) {
  const targetDate = process.argv[2];

  runImageStage({ targetDate })
    .then(({ outputPath, coverPath }) => {
      console.log(`Article source updated at ${outputPath}`);
      console.log(`Cover asset written to ${coverPath}`);
    })
    .catch((error) => {
      console.error(`Image stage failed: ${error.message}`);
      process.exit(1);
    });
}

module.exports = {
  defaultGenerateImage,
  generateCoverAsset,
  getImageExtensionForMimeType,
  normalizeGeneratedImage,
  parseDataUrlImage,
  runImageStage,
  toRepoAssetPath,
};
