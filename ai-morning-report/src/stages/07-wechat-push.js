/**
 * Stage 7: Push WeChat draft to official account draft box.
 *
 * Usage: node 07-wechat-push.js <wechat-md-path> <cover-image-path|"">
 *
 * Requires env: WECHAT_APPID, WECHAT_APPSECRET
 * Optional env: WECHAT_THUMB_MEDIA_ID (fallback cover if no generated image)
 */

const fs = require('fs');
const path = require('path');
const https = require('https');
const dotenv = require('dotenv');

// Load .env from project root (two levels up from src/stages/)
const PROJECT_ROOT = path.resolve(__dirname, '../../..');
dotenv.config({ path: path.join(PROJECT_ROOT, '.env') });

function parseFrontMatter(content) {
  const match = content.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
  if (!match) return { fm: {}, body: content };
  const fm = {};
  for (const line of match[1].split('\n')) {
    const colon = line.indexOf(':');
    if (colon > 0) fm[line.slice(0, colon).trim()] = line.slice(colon + 1).trim();
  }
  return { fm, body: match[2] };
}

function wechatRequest(method, urlPath, accessToken, body, isForm) {
  return new Promise((resolve, reject) => {
    const base = 'https://api.weixin.qq.com';
    const fullPath = `${urlPath}${urlPath.includes('?') ? '&' : '?'}access_token=${accessToken}`;
    const options = {
      hostname: 'api.weixin.qq.com',
      path: fullPath,
      method,
      headers: isForm ? body.getHeaders() : { 'Content-Type': 'application/json' },
    };
    const req = https.request(options, (res) => {
      let data = '';
      res.on('data', (d) => { data += d; });
      res.on('end', () => {
        try { resolve(JSON.parse(data)); }
        catch { resolve({ raw: data }); }
      });
    });
    req.on('error', reject);
    req.setTimeout(30000, () => req.destroy(new Error('timeout')));
    if (isForm) body.pipe(req);
    else { req.write(typeof body === 'string' ? body : JSON.stringify(body)); req.end(); }
  });
}

async function getAccessToken(appid, appsecret) {
  const res = await wechatRequest(
    'GET',
    `/cgi-bin/token?grant_type=client_credential&appid=${appid}&secret=${appsecret}`,
    '', null, false
  );
  if (res.errcode) throw new Error(`access_token: ${res.errcode} ${res.errmsg}`);
  return res.access_token;
}

async function uploadCoverImage(accessToken, imagePath) {
  const FormData = require('form-data');
  const mime = require('mime-types');
  const form = new FormData();
  form.append('media', fs.createReadStream(imagePath), {
    filename: path.basename(imagePath),
    contentType: mime.lookup(imagePath) || 'image/png',
  });
  const res = await wechatRequest('POST', '/cgi-bin/material/add_material?type=image', accessToken, form, true);
  if (res.errcode) throw new Error(`upload cover: ${res.errcode} ${res.errmsg}`);
  return res.media_id;
}

async function addDraft(accessToken, article) {
  const res = await wechatRequest('POST', '/cgi-bin/draft/add', accessToken, {
    articles: [{
      title: article.title,
      author: article.author,
      digest: article.digest,
      content: article.content,
      content_source_url: '',
      thumb_media_id: article.thumb_media_id,
      need_open_comment: 1,
      only_fans_can_comment: 0,
    }],
  }, false);
  if (res.errcode) throw new Error(`add_draft: ${res.errcode} ${res.errmsg}`);
  return res;
}

async function pushWechatDraft(wechatMdPath, coverImagePath) {
  const appid = process.env.WECHAT_APPID;
  const appsecret = process.env.WECHAT_APPSECRET;
  if (!appid || !appsecret) throw new Error('WECHAT_APPID or WECHAT_APPSECRET missing');

  // Read the HTML version if available (preferred for WeChat), else use markdown body
  const htmlPath = wechatMdPath.replace(/\.md$/, '.html');
  const mdContent = fs.readFileSync(wechatMdPath, 'utf8');
  const { fm } = parseFrontMatter(mdContent);

  const htmlContent = fs.existsSync(htmlPath)
    ? fs.readFileSync(htmlPath, 'utf8')
    : mdContent; // fallback: raw markdown (renders poorly but doesn't fail)

  const title = fm.title || 'AI Infra 早报';
  const author = fm.author || '荔枝不耐思';
  const digest = fm.intro || '';

  console.log('[wechat-push] Getting access token...');
  const accessToken = await getAccessToken(appid, appsecret);

  let thumbMediaId = process.env.WECHAT_THUMB_MEDIA_ID || null;

  const hasCover = coverImagePath && fs.existsSync(coverImagePath);
  if (hasCover) {
    console.log('[wechat-push] Uploading cover image...');
    thumbMediaId = await uploadCoverImage(accessToken, coverImagePath);
  } else {
    console.log('[wechat-push] No cover image — using fallback thumb_media_id');
  }

  if (!thumbMediaId) {
    throw new Error('No thumb_media_id available. Set WECHAT_THUMB_MEDIA_ID or provide a cover image.');
  }

  console.log('[wechat-push] Adding draft...');
  const result = await addDraft(accessToken, { title, author, digest, content: htmlContent, thumb_media_id: thumbMediaId });
  console.log(`[wechat-push] Draft created: media_id=${result.media_id}`);
  return result;
}

if (require.main === module) {
  const [wechatMdPath, coverImagePath] = process.argv.slice(2);
  if (!wechatMdPath) {
    console.error('Usage: node 07-wechat-push.js <wechat-md-path> [cover-image-path]');
    process.exit(1);
  }
  pushWechatDraft(wechatMdPath, coverImagePath || '')
    .then(() => console.log('[wechat-push] Done'))
    .catch((err) => {
      console.error('[wechat-push] Failed:', err.message);
      process.exit(1);
    });
}

module.exports = { pushWechatDraft };
