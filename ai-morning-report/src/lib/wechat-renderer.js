/**
 * Converts WeChat-format Markdown to styled HTML.
 *
 * Flow:
 *   1. Parse front matter to determine wechat_variant
 *   2. Select CSS theme from themes.json
 *   3. Render markdown → HTML via marked
 *   4. Resolve CSS var() references to static values
 *   5. Wrap in <style> block + #output container
 */

const fs = require('fs');
const path = require('path');

const THEMES_DIR = path.join(__dirname, '../../wechat-themes');
const THEMES_JSON = path.join(THEMES_DIR, 'themes.json');

function parseFrontMatter(content) {
  const match = content.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
  if (!match) return { fm: {}, body: content };
  const fm = {};
  for (const line of match[1].split('\n')) {
    const colon = line.indexOf(':');
    if (colon > 0) {
      const key = line.slice(0, colon).trim();
      const val = line.slice(colon + 1).trim();
      fm[key] = val;
    }
  }
  return { fm, body: match[2] };
}

function selectTheme(wechatVariant) {
  const config = JSON.parse(fs.readFileSync(THEMES_JSON, 'utf8'));
  for (const [key, theme] of Object.entries(config.themes)) {
    if (theme.wechatVariants.includes(wechatVariant)) {
      return { key, theme };
    }
  }
  const defaultKey = config.default;
  return { key: defaultKey, theme: config.themes[defaultKey] };
}

function loadCSS(themeFiles) {
  return themeFiles
    .map((f) => fs.readFileSync(path.join(THEMES_DIR, f), 'utf8'))
    .join('\n');
}

/**
 * Resolve CSS var() references to their literal values.
 * Handles :root { --foo: bar } declarations and replaces var(--foo) usages.
 * Pure string manipulation, no DOM required.
 */
function resolveCSSVars(css) {
  const vars = {};
  const rootBlock = css.match(/:root\s*\{([^}]+)\}/s);
  if (rootBlock) {
    for (const line of rootBlock[1].split('\n')) {
      const m = line.match(/--([^:]+):\s*(.+?);/);
      if (m) vars[`--${m[1].trim()}`] = m[2].trim();
    }
  }
  let resolved = css;
  for (const [name, value] of Object.entries(vars)) {
    resolved = resolved.replaceAll(`var(${name})`, value);
  }
  // Remove :root block itself (not needed after resolution)
  resolved = resolved.replace(/:root\s*\{[^}]+\}/s, '');
  return resolved;
}

/**
 * Very small markdown renderer (no external dependency required at build time).
 * Handles the subset used in WeChat posts: headings, bold, blockquotes,
 * paragraphs, lists, hr, code, images, inline code.
 *
 * Falls back to a proper marked install if available.
 */
function renderMarkdown(md) {
  try {
    // Use marked if available
    const { marked } = require('marked');
    return marked.parse(md);
  } catch {
    // Minimal fallback renderer
    return minimalRender(md);
  }
}

function minimalRender(md) {
  let html = md
    // Escape HTML entities in code blocks first
    .replace(/```[\s\S]*?```/g, (block) =>
      `<pre><code>${block.replace(/```\w*\n?/, '').replace(/```$/, '').replace(/</g, '&lt;').replace(/>/g, '&gt;')}</code></pre>`
    )
    // Inline code
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    // Images
    .replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img alt="$1" src="$2">')
    // Headings
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    // Bold
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    // Italic
    .replace(/\*([^*]+)\*/g, '<em>$1</em>')
    // HR
    .replace(/^---$/gm, '<hr>')
    // Blockquote
    .replace(/^> (.+)$/gm, '<blockquote><p>$1</p></blockquote>')
    // UL
    .replace(/^- (.+)$/gm, '<li>$1</li>')
    // Paragraphs (lines not already wrapped)
    .split('\n\n')
    .map((block) => {
      if (/^<(h[1-6]|blockquote|pre|ul|ol|hr|li)/.test(block.trim())) return block;
      if (block.trim().startsWith('<li>')) return `<ul>${block}</ul>`;
      return `<p>${block.trim()}</p>`;
    })
    .join('\n');

  return html;
}

/**
 * @param {string} markdownContent - Full markdown including front matter
 * @param {string} [coverImagePath] - Absolute path to cover image (optional)
 * @returns {string} - Complete HTML string ready for WeChat editor
 */
function render(markdownContent, coverImagePath) {
  const { fm, body } = parseFrontMatter(markdownContent);
  const wechatVariant = fm.wechat_variant || 'brief';

  const { theme } = selectTheme(wechatVariant);
  const rawCSS = loadCSS(theme.files);
  const resolvedCSS = resolveCSSVars(rawCSS);

  let fullBody = body;
  if (coverImagePath && fs.existsSync(coverImagePath)) {
    const ext = path.extname(coverImagePath).slice(1);
    const imgData = fs.readFileSync(coverImagePath);
    const b64 = imgData.toString('base64');
    const dataUrl = `data:image/${ext === 'jpg' ? 'jpeg' : ext};base64,${b64}`;
    fullBody = `![封面图](${dataUrl})\n\n${body}`;
  }

  const bodyHTML = renderMarkdown(fullBody);
  const styledHTML = `<style>${resolvedCSS}</style>\n<div id="output">\n${bodyHTML}\n</div>`;

  try {
    const juice = require('juice');
    return juice(styledHTML, { removeStyleTags: true });
  } catch {
    return styledHTML;
  }
}

module.exports = { render, parseFrontMatter, selectTheme, resolveCSSVars };
