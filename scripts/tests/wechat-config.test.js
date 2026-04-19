const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const projectRoot = path.resolve(__dirname, '..', '..');
const envExamplePath = path.join(projectRoot, '.env.example');

test('wechat runtime config is centralized and env-driven', async () => {
  const {
    loadWechatConfig,
  } = require('../lib/wechat-config');

  const originalEnv = {
    WECHAT_API_BASE_URL: process.env.WECHAT_API_BASE_URL,
    OPENROUTER_API_BASE_URL: process.env.OPENROUTER_API_BASE_URL,
    WECHAT_DEFAULT_THUMB_IMAGE: process.env.WECHAT_DEFAULT_THUMB_IMAGE,
    OPENROUTER_IMAGE_MODEL: process.env.OPENROUTER_IMAGE_MODEL,
    OPENROUTER_IMAGE_SIZE: process.env.OPENROUTER_IMAGE_SIZE,
    OPENROUTER_HTTP_REFERER: process.env.OPENROUTER_HTTP_REFERER,
    OPENROUTER_X_TITLE: process.env.OPENROUTER_X_TITLE,
    OPENROUTER_IMAGE_TIMEOUT_MS: process.env.OPENROUTER_IMAGE_TIMEOUT_MS,
    WECHAT_TARGET_TIMEZONE: process.env.WECHAT_TARGET_TIMEZONE,
  };

  process.env.WECHAT_API_BASE_URL = 'https://wechat.example.test';
  process.env.OPENROUTER_API_BASE_URL = 'https://openrouter.example.test';
  process.env.WECHAT_DEFAULT_THUMB_IMAGE = 'assets/icons/custom-thumb.png';
  process.env.OPENROUTER_IMAGE_MODEL = 'openrouter/custom-image-model';
  process.env.OPENROUTER_IMAGE_SIZE = '2K';
  process.env.OPENROUTER_HTTP_REFERER = 'https://example.test';
  process.env.OPENROUTER_X_TITLE = 'Example Publisher';
  process.env.OPENROUTER_IMAGE_TIMEOUT_MS = '456789';
  process.env.WECHAT_TARGET_TIMEZONE = 'UTC';

  try {
    const config = loadWechatConfig(projectRoot);

    assert.equal(config.wechatApiBaseUrl, 'https://wechat.example.test');
    assert.equal(config.openrouterApiBaseUrl, 'https://openrouter.example.test');
    assert.equal(
      config.defaultThumbImagePath,
      path.join(projectRoot, 'assets', 'icons', 'custom-thumb.png'),
    );
    assert.equal(config.openrouterImageModel, 'openrouter/custom-image-model');
    assert.equal(config.openrouterImageSize, '2K');
    assert.equal(config.openrouterHttpReferer, 'https://example.test');
    assert.equal(config.openrouterXTitle, 'Example Publisher');
    assert.equal(config.openrouterImageTimeoutMs, 456789);
    assert.equal(config.wechatTargetTimezone, 'UTC');
  } finally {
    for (const [key, value] of Object.entries(originalEnv)) {
      if (value === undefined) {
        delete process.env[key];
      } else {
        process.env[key] = value;
      }
    }
  }
});

test('.env.example documents the runtime parameters needed by the WeChat publisher', () => {
  assert.equal(fs.existsSync(envExamplePath), true, '.env.example should exist');

  const envExample = fs.readFileSync(envExamplePath, 'utf8');
  const requiredKeys = [
    'WECHAT_APPID',
    'WECHAT_APPSECRET',
    'WECHAT_THUMB_MEDIA_ID',
    'WECHAT_API_BASE_URL',
    'WECHAT_DEFAULT_THUMB_IMAGE',
    'WECHAT_TARGET_TIMEZONE',
    'OPENROUTER_API_KEY',
    'OPENROUTER_API_BASE_URL',
    'OPENROUTER_IMAGE_MODEL',
    'OPENROUTER_IMAGE_SIZE',
    'OPENROUTER_IMAGE_TIMEOUT_MS',
    'OPENROUTER_HTTP_REFERER',
    'OPENROUTER_X_TITLE',
  ];

  for (const key of requiredKeys) {
    assert.match(envExample, new RegExp(`^${key}=`, 'm'));
  }
});
