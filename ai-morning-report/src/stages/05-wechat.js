const path = require('path');
const axios = require('axios');
const FormData = require('form-data');
const { loadWechatConfig } = require('../../scripts/lib/wechat-config');

const PROJECT_ROOT = path.resolve(__dirname, '../../..');

async function getAccessToken(appid, appsecret) {
  const config = loadWechatConfig(PROJECT_ROOT);
  const url = `${config.wechatApiBaseUrl}/token?grant_type=client_credential&appid=${appid}&secret=${appsecret}`;
  const response = await axios.get(url);
  if (response.data.errcode) {
    throw new Error(`access_token failed: ${response.data.errcode} - ${response.data.errmsg}`);
  }
  return response.data.access_token;
}

async function uploadImage(accessToken, imagePath) {
  const config = loadWechatConfig(PROJECT_ROOT);
  const formData = new FormData();
  formData.append('media', require('fs').createReadStream(imagePath), {
    filename: path.basename(imagePath),
    contentType: require('mime-types').lookup(imagePath) || 'image/png',
  });

  const url = `${config.wechatApiBaseUrl}/material/add_material?access_token=${accessToken}&type=image`;
  const response = await axios.post(url, formData, {
    headers: formData.getHeaders(),
  });
  if (response.data.errcode) {
    throw new Error(`material/add_material failed: ${response.data.errcode} - ${response.data.errmsg}`);
  }
  return response.data.media_id;
}

async function uploadContentImage(accessToken, imagePath) {
  const config = loadWechatConfig(PROJECT_ROOT);
  const formData = new FormData();
  formData.append('media', require('fs').createReadStream(imagePath), {
    filename: path.basename(imagePath),
    contentType: require('mime-types').lookup(imagePath) || 'image/png',
  });

  const url = `${config.wechatApiBaseUrl}/media/uploadimg?access_token=${accessToken}`;
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
  const config = loadWechatConfig(PROJECT_ROOT);
  const url = `${config.wechatApiBaseUrl}/draft/add?access_token=${accessToken}`;
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

async function pushWechatDraft({ wechatPostPath, assetsDir }) {
  const appid = process.env.WECHAT_APPID;
  const appsecret = process.env.WECHAT_APPSECRET;
  const thumbMediaId = process.env.WECHAT_THUMB_MEDIA_ID;

  if (!appid || !appsecret) {
    throw new Error('WECHAT_APPID or WECHAT_APPSECRET not found in .env');
  }

  const accessToken = await getAccessToken(appid, appsecret);
  const fs = require('fs');
  const content = fs.readFileSync(wechatPostPath, 'utf8');
  const { frontMatter, body } = parseFrontMatter(content);
  const title = frontMatter.title || 'Untitled';
  const author = frontMatter.author || '';
  const digest = frontMatter.intro || '';

  let effectiveThumbMediaId = thumbMediaId || null;
  const coverPath = path.join(assetsDir, 'cover.png');

  if (fs.existsSync(coverPath)) {
    effectiveThumbMediaId = await uploadImage(accessToken, coverPath);
  }

  const htmlContent = body;

  const result = await addDraft(accessToken, {
    title,
    author,
    digest,
    content: htmlContent,
    thumb_media_id: effectiveThumbMediaId,
  });

  return { success: true, media_id: result.media_id };
}

if (require.main === module) {
  const wechatPostPath = process.argv[2];
  const assetsDir = process.argv[3] || '/tmp/morning-report/assets';

  if (!wechatPostPath) {
    console.error('Usage: node 05-wechat.js <wechat-post-path> [assets-dir]');
    process.exit(1);
  }

  pushWechatDraft({ wechatPostPath, assetsDir })
    .then((result) => console.log('WeChat draft pushed:', JSON.stringify(result)))
    .catch((err) => {
      console.error('WeChat draft push failed:', err.message);
      process.exit(1);
    });
}

module.exports = { pushWechatDraft, getAccessToken };