const path = require('node:path');
const juice = require('juice');
const { marked } = require('marked');

const THEMES = {
  'emerald-green': {
    accent: '#009B77',
    accentDeep: '#00795d',
    blockquoteText: '#576b95',
  },
  'classic-blue': {
    accent: '#2F5EA7',
    accentDeep: '#1F4B8F',
    blockquoteText: '#405F8C',
  },
};

function resolveWechatRenderProfile(frontMatter = {}) {
  if (frontMatter.wechat_variant === 'essay-longform') {
    return {
      variant: 'essay-longform',
      themeName: 'classic-blue',
      keepInlineImages: true,
    };
  }

  return {
    variant: 'brief-daily',
    themeName: 'emerald-green',
    keepInlineImages: false,
  };
}

function buildWechatCss(themeName = 'emerald-green') {
  const theme = THEMES[themeName] || THEMES['emerald-green'];

  return `
section {
  font-family: -apple-system-font, BlinkMacSystemFont, Helvetica Neue, PingFang SC, Hiragino Sans GB, Microsoft YaHei UI, Microsoft YaHei, Arial, sans-serif;
  font-size: 16px;
  line-height: 1.75;
  color: #24292f;
  max-width: 100%;
  word-break: break-word;
}
h1 { display: table; margin: 2em auto 1em; padding: 0 1em; border-bottom: 2px solid ${theme.accent}; font-size: 1.3em; text-align: center; color: ${theme.accent}; }
h2 { display: table; margin: 1.5em auto 1em; padding: 0.3em 0.8em; background: ${theme.accent}; color: #fff; font-size: 1.2em; text-align: center; border-radius: 4px; }
h3 { padding-left: 8px; border-left: 3px solid ${theme.accent}; margin: 1.5em 8px 0.75em 0; font-size: 1.1em; color: ${theme.accent}; }
h4, h5, h6 { margin: 1.5em 8px 0.5em; color: ${theme.accent}; font-size: 1em; }
p { margin: 1em 0; letter-spacing: 0.05em; }
blockquote { font-style: normal; padding: 0.8em 1em; border-left: 4px solid ${theme.accent}; border-radius: 4px; background: #f6f8fa; margin: 1em 0; color: ${theme.blockquoteText}; }
blockquote p { margin: 0.5em 0; }
code { font-size: 90%; color: #d14; background: rgba(27, 31, 35, 0.05); padding: 2px 6px; border-radius: 4px; font-family: Consolas, Monaco, Andale Mono, monospace; }
pre { background: #f6f8fa; border-radius: 8px; overflow-x: auto; margin: 1em 0; padding: 0; border: 1px solid rgba(0,0,0,0.1); }
pre code { background: none; padding: 1em; color: inherit; border-radius: 0; display: block; }
a { color: ${theme.blockquoteText}; text-decoration: none; }
strong { color: ${theme.accent}; font-weight: bold; }
ul { list-style: circle; padding-left: 1.5em; margin: 1em 0; }
ol { padding-left: 1.5em; margin: 1em 0; list-style: decimal; }
li { margin: 0.3em 0; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 90%; }
th, td { border: 1px solid #dfdfdf; padding: 0.5em 0.75em; text-align: left; }
th { background: rgba(0,0,0,0.03); font-weight: 600; }
img { display: block; max-width: 100%; margin: 1em auto; border-radius: 4px; }
hr { border-style: solid; border-width: 2px 0 0; border-color: rgba(0,0,0,0.1); height: 0.4em; margin: 2em 0; }
`;
}

function renderWechatHtml(markdown, { themeName = 'emerald-green' } = {}) {
  marked.setOptions({
    gfm: true,
    breaks: true,
  });

  const bodyHtml = marked.parse(markdown);
  const htmlWithStyleTag = `<style>${buildWechatCss(themeName)}</style><section>${bodyHtml}</section>`;
  return juice(htmlWithStyleTag);
}

function resolveMarkdownImagePath(articleFilePath, imageSrc, projectRoot) {
  if (!imageSrc || /^https?:\/\//i.test(imageSrc) || /^data:/i.test(imageSrc)) {
    return null;
  }

  if (imageSrc.startsWith('/')) {
    return path.join(projectRoot, imageSrc.replace(/^\/+/, ''));
  }

  return path.resolve(path.dirname(articleFilePath), imageSrc);
}

async function rewriteMarkdownImageUrls({
  markdown,
  articleFilePath,
  projectRoot,
  uploadImage,
}) {
  if (typeof uploadImage !== 'function') {
    throw new Error('uploadImage callback is required');
  }

  const uploadCache = new Map();
  const imagePattern = /!\[([^\]]*)\]\(([^)]+)\)/g;
  let rewritten = '';
  let cursor = 0;
  let match;

  while ((match = imagePattern.exec(markdown)) !== null) {
    const [fullMatch, altText, src] = match;
    const resolvedPath = resolveMarkdownImagePath(articleFilePath, src.trim(), projectRoot);

    rewritten += markdown.slice(cursor, match.index);

    if (!resolvedPath) {
      rewritten += fullMatch;
      cursor = match.index + fullMatch.length;
      continue;
    }

    let uploadedUrl = uploadCache.get(resolvedPath);
    if (!uploadedUrl) {
      uploadedUrl = await uploadImage(resolvedPath);
      uploadCache.set(resolvedPath, uploadedUrl);
    }

    rewritten += `![${altText}](${uploadedUrl})`;
    cursor = match.index + fullMatch.length;
  }

  rewritten += markdown.slice(cursor);

  return {
    markdown: rewritten,
  };
}

module.exports = {
  renderWechatHtml,
  resolveMarkdownImagePath,
  resolveWechatRenderProfile,
  rewriteMarkdownImageUrls,
};
