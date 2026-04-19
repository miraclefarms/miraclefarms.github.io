const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const {
  resolveWechatRenderProfile,
  renderWechatHtml,
  rewriteMarkdownImageUrls,
} = require('../lib/wechat-render');

test('essay-longform WeChat articles resolve to the classic blue profile', () => {
  const essayProfile = resolveWechatRenderProfile({ wechat_variant: 'essay-longform' });
  const briefProfile = resolveWechatRenderProfile({});

  assert.equal(essayProfile.variant, 'essay-longform');
  assert.equal(essayProfile.themeName, 'classic-blue');
  assert.equal(essayProfile.keepInlineImages, true);

  assert.equal(briefProfile.variant, 'brief-daily');
  assert.equal(briefProfile.themeName, 'emerald-green');
});

test('renderWechatHtml applies the classic blue theme for essay long-form articles', () => {
  const html = renderWechatHtml('# 标题\n\n## 小节\n\n> 引导语', { themeName: 'classic-blue' });

  assert.match(html, /#2F5EA7/i);
  assert.doesNotMatch(html, /#009B77/i);
});

test('rewriteMarkdownImageUrls uploads all local markdown images, not only the title image', async () => {
  const articleFilePath = '/repo/docs/wechat/2026-04-19-demo-wechat.md';
  const markdown = [
    '![题图](assets/2026-04-19/demo-cover.png)',
    '',
    '正文第一段。',
    '',
    '![图 1](../../assets/demo-post/fig-1-architecture.png)',
    '',
    '![远端图](https://example.com/remote.png)',
  ].join('\n');

  const uploaded = [];
  const result = await rewriteMarkdownImageUrls({
    markdown,
    articleFilePath,
    projectRoot: '/repo',
    uploadImage: async imagePath => {
      uploaded.push(imagePath);
      return `https://wx.example/${path.basename(imagePath)}`;
    },
  });

  assert.deepEqual(uploaded, [
    '/repo/docs/wechat/assets/2026-04-19/demo-cover.png',
    '/repo/assets/demo-post/fig-1-architecture.png',
  ]);
  assert.match(result.markdown, /https:\/\/wx\.example\/demo-cover\.png/);
  assert.match(result.markdown, /https:\/\/wx\.example\/fig-1-architecture\.png/);
  assert.match(result.markdown, /https:\/\/example\.com\/remote\.png/);
});
