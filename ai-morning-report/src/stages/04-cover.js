/**
 * Stage 4: Cover image generation (non-blocking).
 *
 * Usage: node 04-cover.js <date> <post-file> <assets-dir>
 *
 * On success: writes cover.{ext} to assets-dir, exits 0.
 * On failure: logs warning, exits 0 (non-blocking — no cover is fine).
 */

const fs = require('fs');
const path = require('path');
const https = require('https');

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

function buildCoverPrompt(title, intro) {
  const cleanTitle = title.replace(/AI Infra 早报｜/, '').trim();
  return `Create a clean, modern tech infographic card for a Chinese AI infrastructure newsletter.
Theme: dark background with emerald green (#2e9e6b) accents.
Topic: "${cleanTitle}".
Style: minimalist, professional, hand-drawn diagram aesthetic.
Include subtle circuit or neural network motifs. No text overlays.
Aspect ratio: 3:2 (landscape, suitable for WeChat cover).`;
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
    req.setTimeout(60000, () => { req.destroy(new Error('timeout')); });
    req.write(body);
    req.end();
  });
}

async function generateViaOpenRouter(prompt) {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) throw new Error('OPENROUTER_API_KEY not set');

  const payload = JSON.stringify({
    model: process.env.COVER_IMAGE_MODEL || 'google/gemini-2.0-flash-exp:free',
    messages: [{ role: 'user', content: prompt }],
    max_tokens: 1,
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

  if (status !== 200) throw new Error(`OpenRouter HTTP ${status}`);

  const imgUrl = body?.choices?.[0]?.message?.images?.[0]?.image_url?.url;
  if (!imgUrl) throw new Error('No image in OpenRouter response');
  return imgUrl;
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

async function generateCover(date, postFile, assetsDir) {
  if (!fs.existsSync(postFile)) {
    console.warn(`[cover] Post file not found: ${postFile}, skipping`);
    return null;
  }

  const content = fs.readFileSync(postFile, 'utf8');
  const fm = parseFrontMatter(content);
  const title = fm.title || `AI Infra 早报｜${date}`;
  const intro = fm.intro || '';

  const prompt = buildCoverPrompt(title, intro);

  console.log('[cover] Generating cover image...');
  fs.mkdirSync(assetsDir, { recursive: true });

  try {
    const imgUrl = await generateViaOpenRouter(prompt);

    let buffer, ext;
    if (imgUrl.startsWith('data:')) {
      const parsed = parseDataUrl(imgUrl);
      buffer = parsed.buffer;
      ext = extForMime(parsed.mimeType);
    } else {
      // Regular URL — shouldn't happen with current models but handle gracefully
      throw new Error(`Unexpected non-data URL response: ${imgUrl.slice(0, 80)}`);
    }

    const coverPath = path.join(assetsDir, `cover.${ext}`);
    fs.writeFileSync(coverPath, buffer);
    console.log(`[cover] Saved: ${coverPath}`);
    return coverPath;
  } catch (err) {
    console.warn(`[cover] Image generation failed (non-blocking): ${err.message}`);
    return null;
  }
}

if (require.main === module) {
  const [date, postFile, assetsDir] = process.argv.slice(2);
  if (!date || !postFile || !assetsDir) {
    console.error('Usage: node 04-cover.js <date> <post-file> <assets-dir>');
    process.exit(1);
  }
  generateCover(date, postFile, assetsDir)
    .then((p) => {
      if (p) console.log(`[cover] Cover: ${p}`);
      else console.log('[cover] No cover generated, proceeding without.');
      process.exit(0);
    })
    .catch((err) => {
      console.warn(`[cover] Unexpected error: ${err.message}`);
      process.exit(0); // always exit 0 — non-blocking
    });
}

module.exports = { generateCover };
