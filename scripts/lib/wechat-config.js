const path = require('node:path');

function resolvePathFromProjectRoot(projectRoot, relativeOrAbsolutePath) {
  if (path.isAbsolute(relativeOrAbsolutePath)) {
    return relativeOrAbsolutePath;
  }

  return path.join(projectRoot, relativeOrAbsolutePath);
}

function parseIntegerEnv(rawValue, fallback) {
  if (rawValue === undefined || rawValue === null || rawValue === '') {
    return fallback;
  }

  const parsed = Number.parseInt(rawValue, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function loadWechatConfig(projectRoot) {
  return {
    wechatApiBaseUrl: process.env.WECHAT_API_BASE_URL || 'https://api.weixin.qq.com/cgi-bin',
    openrouterApiBaseUrl: process.env.OPENROUTER_API_BASE_URL || 'https://openrouter.ai/api/v1',
    defaultThumbImagePath: resolvePathFromProjectRoot(
      projectRoot,
      process.env.WECHAT_DEFAULT_THUMB_IMAGE || 'assets/icons/favicon.png',
    ),
    openrouterImageModel: process.env.OPENROUTER_IMAGE_MODEL || 'google/gemini-2.5-flash-image',
    openrouterImageSize: process.env.OPENROUTER_IMAGE_SIZE || '1K',
    openrouterHttpReferer: process.env.OPENROUTER_HTTP_REFERER || 'https://miraclefarms.github.io',
    openrouterXTitle: process.env.OPENROUTER_X_TITLE || 'MiracleFarms WeChat Publisher',
    openrouterImageTimeoutMs: parseIntegerEnv(process.env.OPENROUTER_IMAGE_TIMEOUT_MS, 180000),
    wechatTargetTimezone: process.env.WECHAT_TARGET_TIMEZONE || 'Asia/Shanghai',
  };
}

module.exports = {
  loadWechatConfig,
};
