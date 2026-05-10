/**
 * Stage 4: Cover image generation (non-blocking).
 *
 * Three steps:
 *   A. AI generates a cover prompt based on the full article
 *   B. OpenRouter generates the image using prompt + template
 *   C. Cover image is inserted into the article
 *
 * Usage: node 04-cover.js <date> <post-file> <assets-dir> [project-root]
 *
 * Intermediate files:
 *   <assets-dir>/cover-prompt.txt  — AI-generated image prompt
 *   <assets-dir>/cover.{ext}       — generated cover image
 */

const fs = require('fs');
const path = require('path');
const https = require('https');

const CHINESE_INFOGRAPHIC_GUARDRAILS = [
  '硬性约束：最终图片必须是中文信息图。',
  '所有可见文字必须使用简体中文；允许 AI、GPU、KV、LLM、PR、API 等必要技术缩写。',
  '禁止出现英文长句、英文花体、拼音、水印、Logo、乱码或无意义字符。',
].join('\n');

function parseFrontMatter(content) {
  const match = content.match(/^---\n([\s\S]*?)\n---/);
  if (!match) return {};
  const fm = {};
  for (const line of match[1].split('\n')) {
    const colon = line.indexOf(':');
    if (colon > 0) fm[line.slice(0, colon).trim()] = line.slice(colon + 1).trim();
  }
  return fm;
}

function extractBody(content) {
  const match = content.match(/^---\n[\s\S]*?\n---\n([\s\S]*)$/);
  return match ? match[1].trim() : content;
}

async function httpPost(url, headers, body) {
  return new Promise((resolve, reject) => {
    const urlObj = new URL(url);
    const options = {
      hostname: urlObj.hostname,
      path: urlObj.pathname + urlObj.search,
      method: 'POST',
      headers,
    };
    const req = https.request(options, (res) => {
      let data = '';
      res.on('data', (d) => { data += d; });
      res.on('end', () => {
        try { resolve({ status: res.statusCode, body: JSON.parse(data) }); }
        catch { resolve({ status: res.statusCode, body: data }); }
      });
    });
    req.on('error', reject);
    req.setTimeout(120000, () => { req.destroy(new Error('timeout')); });
    req.write(body);
    req.end();
  });
}

function parseDataUrl(dataUrl) {
  const m = dataUrl.match(/^data:([^;]+);base64,(.+)$/);
  if (!m) throw new Error('Not a data URL');
  return { mimeType: m[1], buffer: Buffer.from(m[2], 'base64') };
}

function extForMime(mime) {
  if (mime.includes('jpeg') || mime.includes('jpg')) return 'jpg';
  if (mime.includes('webp')) return 'webp';
  if (mime.includes('gif')) return 'gif';
  return 'png';
}

function loadImageTemplate(projectRoot) {
  const templatePath = path.join(projectRoot, 'scripts', 'config', 'wechat-cover-prompt-templates.json');
  if (!fs.existsSync(templatePath)) return null;
  const registry = JSON.parse(fs.readFileSync(templatePath, 'utf8'));
  const templateId = registry.defaultTemplateId;
  return registry.templates[templateId] || null;
}

async function generateCoverPrompt(date, postContent, projectRoot) {
  let runAI;
  try {
    ({ runAI } = require('../lib/cli-adapter'));
  } catch {
    console.warn('[cover] cli-adapter not available, using fallback prompt');
    return null;
  }

  const body = extractBody(postContent);
  const prompt = `你是一个技术配图提示词专家。请阅读以下 AI Infra 早报全文，生成一段用于 AI 图片生成的提示词。

**重要约束：不要读取任何文件，不要创建任何文件，不要使用任何工具。只需要将提示词作为纯文本输出。**

要求：
- 提示词用中文撰写
- 控制在 500 字符以内
- 突出当天最重要的 PR 内容或整体技术趋势
- 生成 9:16 竖版中文信息图，而不是英文信息图
- 画面中的标题、分区标签、说明短语必须使用简体中文
- 可保留 AI、GPU、KV、LLM、PR、API 等必要技术缩写，但禁止出现英文长句、英文花体、拼音、乱码、水印或 Logo
- 风格：米白/米色纸张质感背景，手绘插画风格，红黑配色点缀，留白充足
- 不要描述具体的人脸

以下为今日早报正文：

<article>
${body}
</article>

直接输出提示词，不要任何前缀或解释。`;

  console.log('[cover] Step A: AI generating cover prompt...');
  try {
    const result = await runAI({ prompt });
    return result.trim();
  } catch (err) {
    console.warn(`[cover] AI prompt generation failed: ${err.message}`);
    return null;
  }
}

function buildFinalPrompt(aiPrompt, template) {
  const parts = [];

  if (template && template.prompt) {
    parts.push(template.prompt);
  }

  parts.push(CHINESE_INFOGRAPHIC_GUARDRAILS);

  if (aiPrompt) {
    parts.push(`文章主题提示：\n${aiPrompt}`);
  }

  return parts.join('\n\n');
}

async function generateViaOpenRouter(prompt) {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) throw new Error('OPENROUTER_API_KEY not set');

  const payload = JSON.stringify({
    model: process.env.COVER_IMAGE_MODEL || process.env.OPENROUTER_IMAGE_MODEL || 'google/gemini-2.5-flash-image',
    messages: [{ role: 'user', content: prompt }],
    modalities: ['image', 'text'],
  });

  const { status, body } = await httpPost(
    'https://openrouter.ai/api/v1/chat/completions',
    {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      'HTTP-Referer': 'https://miraclefarms.github.io',
      'X-Title': 'MiracleFarms AI Morning Report',
    },
    payload
  );

  if (status !== 200) throw new Error(`OpenRouter HTTP ${status}: ${JSON.stringify(body).slice(0, 200)}`);

  const imgUrl = body?.choices?.[0]?.message?.images?.[0]?.image_url?.url;
  if (!imgUrl) throw new Error('No image in OpenRouter response');
  return imgUrl;
}

function insertCoverIntoArticle(postFile, date, coverExt) {
  const content = fs.readFileSync(postFile, 'utf8');
  const coverMd = `![题图](/assets/${date}-ai-infra-daily-brief/cover.${coverExt})`;

  if (content.includes(coverMd)) {
    console.log('[cover] Cover already in article, skipping insertion');
    return;
  }

  const fmMatch = content.match(/^(---\n[\s\S]*?\n---)\n/);
  if (!fmMatch) {
    console.warn('[cover] Cannot find front matter boundary, skipping insertion');
    return;
  }

  const updated = content.replace(
    /^(---\n[\s\S]*?\n---)\n/,
    `$1\n\n${coverMd}\n\n`
  );
  fs.writeFileSync(postFile, updated, 'utf8');
  console.log('[cover] Inserted cover image into article');
}

function copyCoverToAssets(coverPath, date, projectRoot) {
  if (!projectRoot || !coverPath) return;
  const targetDir = path.join(projectRoot, 'assets', `${date}-ai-infra-daily-brief`);
  fs.mkdirSync(targetDir, { recursive: true });
  const ext = path.extname(coverPath).slice(1);
  const targetPath = path.join(targetDir, `cover.${ext}`);
  fs.copyFileSync(coverPath, targetPath);
  console.log(`[cover] Copied to project assets: ${targetPath}`);
  return ext;
}

async function generateCover(date, postFile, assetsDir, projectRoot) {
  if (!fs.existsSync(postFile)) {
    console.warn(`[cover] Post file not found: ${postFile}, skipping`);
    return null;
  }

  const postContent = fs.readFileSync(postFile, 'utf8');
  fs.mkdirSync(assetsDir, { recursive: true });

  try {
    // Step A: AI generates cover prompt
    let aiPrompt = await generateCoverPrompt(date, postContent, projectRoot);

    if (!aiPrompt) {
      const fm = parseFrontMatter(postContent);
      const cleanTitle = (fm.title || '').replace(/AI Infra 早报｜/, '').trim();
      aiPrompt = `生成 9:16 竖版中文信息图题图。主题："${cleanTitle}"。画面使用米白或浅米色纸张质感背景，温暖克制的手绘技术编辑风格，红黑色重点标注，分成 2-4 个信息区块。标题和区块短语必须使用简体中文；允许 AI、GPU、KV、LLM、PR、API 等必要技术缩写；禁止出现英文长句、英文花体、拼音、水印、Logo、乱码或无意义字符。使用简单手绘图标、箭头、缓存块、芯片或数据流示意，不要具体人脸。`;
      console.log('[cover] Using fallback prompt');
    }

    const promptPath = path.join(assetsDir, 'cover-prompt.txt');
    fs.writeFileSync(promptPath, aiPrompt, 'utf8');
    console.log(`[cover] Prompt saved: ${promptPath}`);

    // Step B: Generate image via OpenRouter
    const template = projectRoot ? loadImageTemplate(projectRoot) : null;
    const finalPrompt = buildFinalPrompt(aiPrompt, template);

    console.log('[cover] Step B: Generating cover image...');
    const imgUrl = await generateViaOpenRouter(finalPrompt);

    let buffer, ext;
    if (imgUrl.startsWith('data:')) {
      const parsed = parseDataUrl(imgUrl);
      buffer = parsed.buffer;
      ext = extForMime(parsed.mimeType);
    } else {
      throw new Error(`Unexpected non-data URL response: ${imgUrl.slice(0, 80)}`);
    }

    const coverPath = path.join(assetsDir, `cover.${ext}`);
    fs.writeFileSync(coverPath, buffer);
    console.log(`[cover] Saved: ${coverPath}`);

    // Step C: Copy to project assets and insert into article
    if (projectRoot) {
      copyCoverToAssets(coverPath, date, projectRoot);
      insertCoverIntoArticle(postFile, date, ext);
    }

    return coverPath;
  } catch (err) {
    console.warn(`[cover] Image generation failed (non-blocking): ${err.message}`);
    return null;
  }
}

if (require.main === module) {
  const [date, postFile, assetsDir, projectRoot] = process.argv.slice(2);
  if (!date || !postFile || !assetsDir) {
    console.error('Usage: node 04-cover.js <date> <post-file> <assets-dir> [project-root]');
    process.exit(1);
  }
  generateCover(date, postFile, assetsDir, projectRoot || '')
    .then((p) => {
      if (p) console.log(`[cover] Cover: ${p}`);
      else console.log('[cover] No cover generated, proceeding without.');
      process.exit(0);
    })
    .catch((err) => {
      console.warn(`[cover] Unexpected error: ${err.message}`);
      process.exit(0);
    });
}

module.exports = { generateCover };
