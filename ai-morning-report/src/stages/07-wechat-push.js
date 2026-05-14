/**
 * Stage 7: Push WeChat draft to official account draft box.
 *
 * Usage: node 07-wechat-push.js <wechat-md-path> <cover-image-path|"">
 *
 * Requires env: WECHAT_APPID, WECHAT_APPSECRET
 * Optional env: WECHAT_THUMB_MEDIA_ID (fallback cover if no generated image)
 * Optional env: WECHAT_CONTENT_SOURCE_BASE_URL (defaults to https://miraclefarms.github.io)
 */

const fs = require('fs');
const path = require('path');
const https = require('https');
const dotenv = require('dotenv');
const { execFileSync } = require('child_process');
const os = require('os');
const { render } = require('../lib/wechat-renderer');

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

function stringifyFrontMatter(fm) {
  const lines = Object.entries(fm).map(([key, value]) => `${key}: ${value}`);
  return `---\n${lines.join('\n')}\n---\n`;
}

function normalizeFrontMatterValue(value) {
  if (value == null) return '';
  const str = String(value).trim();
  const quoted = str.match(/^(['"])([\s\S]*)\1$/);
  return quoted ? quoted[2] : str;
}

function removeFirstH1(body) {
  const lines = body.split('\n');
  const index = lines.findIndex((line) => /^#\s+(.+?)\s*$/.test(line));
  if (index < 0) return { title: '', body };

  const title = lines[index].replace(/^#\s+/, '').trim();
  lines.splice(index, 1);
  while (lines[index] === '') lines.splice(index, 1);

  return { title, body: lines.join('\n').replace(/^\n+/, '') };
}

function prepareMarkdownForDraft(markdownContent) {
  const { fm, body } = parseFrontMatter(markdownContent);
  const cleaned = removeFirstH1(body);
  const title = cleaned.title || normalizeFrontMatterValue(fm.title) || 'AI Infra 早报';
  const author = normalizeFrontMatterValue(fm.author) || '荔枝不耐思';
  const digest = normalizeFrontMatterValue(fm.intro) || '';
  const frontMatter = stringifyFrontMatter(fm);

  return {
    fm,
    title,
    author,
    digest,
    body: cleaned.body,
    markdownContent: `${frontMatter}${cleaned.body}`,
  };
}

function removeTopImageMarkdown(body) {
  const firstSectionIndex = body.search(/^##\s+/m);
  const head = firstSectionIndex >= 0 ? body.slice(0, firstSectionIndex) : body;
  const tail = firstSectionIndex >= 0 ? body.slice(firstSectionIndex) : '';
  const cleanedHead = head.replace(/(^|\n)!\[[^\]]*\]\([^)]+\)\n{0,2}/, '$1');
  return `${cleanedHead}${tail}`;
}

function insertCoverImageMarkdown(body, contentImageUrl) {
  if (!contentImageUrl) return body;

  const imageMarkdown = `![题图](${contentImageUrl})`;
  const nextBody = removeTopImageMarkdown(body);
  if (nextBody.includes(imageMarkdown)) return nextBody;

  const quoteMatch = nextBody.match(/(^> .+(?:\n> .+)*)\n{2,}/m);
  if (quoteMatch) {
    return nextBody.replace(quoteMatch[0], `${quoteMatch[1]}\n\n${imageMarkdown}\n\n`);
  }

  const dateMatch = nextBody.match(/^\*\*📅 .+\*\*\n{2,}/m);
  if (dateMatch) {
    return nextBody.replace(dateMatch[0], `${dateMatch[0]}${imageMarkdown}\n\n`);
  }

  return `${imageMarkdown}\n\n${nextBody}`;
}

function buildContentSourceUrl(wechatMdPath, baseUrl = process.env.WECHAT_CONTENT_SOURCE_BASE_URL || 'https://miraclefarms.github.io') {
  const basename = path.basename(wechatMdPath, '.md');
  const match = basename.match(/^(\d{4})-(\d{2})-(\d{2})-(.+?)(?:-wechat)?$/);
  if (!match) return '';

  const [, year, month, day, slug] = match;
  const cleanBaseUrl = baseUrl.replace(/\/+$/, '');
  return `${cleanBaseUrl}/notes/${year}/${month}/${day}/${slug}/`;
}

function wechatRequest(method, urlPath, accessToken, body, isForm) {
  return new Promise((resolve, reject) => {
    const fullPath = `${urlPath}${urlPath.includes('?') ? '&' : '?'}access_token=${accessToken}`;
    const headers = isForm ? body.getHeaders() : { 'Content-Type': 'application/json; charset=utf-8' };

    let bodyBuf = null;
    if (!isForm && body != null) {
      const str = typeof body === 'string' ? body : JSON.stringify(body);
      bodyBuf = Buffer.from(str, 'utf8');
      headers['Content-Length'] = bodyBuf.length;
    }

    const options = {
      hostname: 'api.weixin.qq.com',
      path: fullPath,
      method,
      headers,
    };
    const req = https.request(options, (res) => {
      let data = '';
      res.on('data', (d) => { data += d; });
      res.on('end', () => {
        try { resolve(JSON.parse(data)); }
        catch { resolve({ raw: data, status: res.statusCode }); }
      });
    });
    req.on('error', reject);
    req.setTimeout(30000, () => req.destroy(new Error('timeout')));
    if (isForm) body.pipe(req);
    else if (bodyBuf) { req.write(bodyBuf); req.end(); }
    else { req.end(); }
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
  const res = await new Promise((resolve, reject) => {
    form.submit({
      protocol: 'https:',
      host: 'api.weixin.qq.com',
      path: `/cgi-bin/material/add_material?type=image&access_token=${accessToken}`,
      method: 'POST',
    }, (err, response) => {
      if (err) return reject(err);
      let data = '';
      response.on('data', (d) => { data += d; });
      response.on('end', () => {
        try { resolve(JSON.parse(data)); }
        catch { resolve({ raw: data }); }
      });
      response.on('error', reject);
    });
  });
  if (res.errcode) throw new Error(`upload cover: ${res.errcode} ${res.errmsg}`);
  if (!res.media_id) throw new Error(`upload cover: no media_id in response: ${JSON.stringify(res).slice(0, 200)}`);
  return res.media_id;
}

async function uploadContentImage(accessToken, imagePath) {
  const FormData = require('form-data');
  const mime = require('mime-types');
  const form = new FormData();
  form.append('media', fs.createReadStream(imagePath), {
    filename: path.basename(imagePath),
    contentType: mime.lookup(imagePath) || 'image/jpeg',
  });
  const res = await new Promise((resolve, reject) => {
    form.submit({
      protocol: 'https:',
      host: 'api.weixin.qq.com',
      path: `/cgi-bin/media/uploadimg?access_token=${accessToken}`,
      method: 'POST',
    }, (err, response) => {
      if (err) return reject(err);
      let data = '';
      response.on('data', (d) => { data += d; });
      response.on('end', () => {
        try { resolve(JSON.parse(data)); }
        catch { resolve({ raw: data }); }
      });
      response.on('error', reject);
    });
  });
  if (res.errcode) throw new Error(`upload content image: ${res.errcode} ${res.errmsg}`);
  if (!res.url) throw new Error(`upload content image: no url in response: ${JSON.stringify(res).slice(0, 200)}`);
  return res.url;
}

function ensureWechatContentImageFile(imagePath) {
  const ext = path.extname(imagePath).toLowerCase();
  const size = fs.statSync(imagePath).size;
  const isSupportedType = ['.jpg', '.jpeg', '.png'].includes(ext);
  if (isSupportedType && size < 1024 * 1024) return imagePath;

  const target = path.join(os.tmpdir(), `wechat-content-${Date.now()}-${path.basename(imagePath, ext)}.jpg`);
  const attempts = [
    { width: 900, quality: 82 },
    { width: 720, quality: 78 },
    { width: 640, quality: 72 },
  ];

  for (const attempt of attempts) {
    try {
      execFileSync('/usr/bin/sips', [
        '-s', 'format', 'jpeg',
        '-s', 'formatOptions', String(attempt.quality),
        '--resampleWidth', String(attempt.width),
        imagePath,
        '--out', target,
      ], { stdio: 'ignore' });
      if (fs.existsSync(target) && fs.statSync(target).size < 1024 * 1024) return target;
    } catch {
      break;
    }
  }

  throw new Error('Cover image is not suitable for WeChat body upload. Use jpg/png under 1MB or install macOS sips.');
}

async function addDraft(accessToken, article) {
  const res = await wechatRequest('POST', '/cgi-bin/draft/add', accessToken, {
    articles: [{
      title: article.title,
      author: article.author,
      digest: article.digest,
      content: article.content,
      content_source_url: article.content_source_url || '',
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

  const mdContent = fs.readFileSync(wechatMdPath, 'utf8');
  const prepared = prepareMarkdownForDraft(mdContent);

  console.log('[wechat-push] Getting access token...');
  const accessToken = await getAccessToken(appid, appsecret);

  let thumbMediaId = process.env.WECHAT_THUMB_MEDIA_ID || null;
  let contentImageUrl = '';

  const hasCover = coverImagePath && fs.existsSync(coverImagePath);
  if (hasCover) {
    console.log('[wechat-push] Uploading cover image...');
    thumbMediaId = await uploadCoverImage(accessToken, coverImagePath);
    console.log('[wechat-push] Uploading body image...');
    contentImageUrl = await uploadContentImage(accessToken, ensureWechatContentImageFile(coverImagePath));
  } else {
    console.log('[wechat-push] No cover image — using fallback thumb_media_id');
  }

  if (!thumbMediaId) {
    throw new Error('No thumb_media_id available. Set WECHAT_THUMB_MEDIA_ID or provide a cover image.');
  }

  const body = insertCoverImageMarkdown(prepared.body, contentImageUrl);
  const markdownForRender = `${stringifyFrontMatter(prepared.fm)}${body}`;
  const htmlContent = render(markdownForRender, null);
  const contentSourceUrl = buildContentSourceUrl(wechatMdPath);

  console.log('[wechat-push] Adding draft...');
  const result = await addDraft(accessToken, {
    title: prepared.title,
    author: prepared.author,
    digest: prepared.digest,
    content: htmlContent,
    content_source_url: contentSourceUrl,
    thumb_media_id: thumbMediaId,
  });
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
      if (/40164|invalid ip|not in whitelist|IP.*白名单/i.test(err.message)) {
        console.error('[wechat-push] 当前出口 IP 不在公众号后台白名单。添加白名单后，重新执行上面的 07-wechat-push 命令即可。');
      }
      process.exit(1);
    });
}

module.exports = {
  buildContentSourceUrl,
  insertCoverImageMarkdown,
  prepareMarkdownForDraft,
  pushWechatDraft,
};
