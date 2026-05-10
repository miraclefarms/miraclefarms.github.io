/**
 * Stage 6: WeChat formatting — two-step.
 *
 *   Step A: AI rewrites GitHub.io markdown → WeChat markdown (wechat-formatter skill)
 *   Step B: wechat-renderer converts WeChat markdown → styled HTML
 *
 * Usage: node 06-wechat-format.js <date> <post-file> <output-dir> <project-root> [cover-image-path] [archive-dir]
 *
 * Outputs:
 *   <output-dir>/<date>-ai-infra-daily-brief-wechat.md   (WeChat markdown)
 *   <output-dir>/<date>-ai-infra-daily-brief-wechat.html (Styled HTML for pasting)
 *   <archive-dir>/<date>-ai-infra-daily-brief-wechat.md  (optional repository copy)
 */

const fs = require('fs');
const path = require('path');
const { runAI, findSkill } = require('../lib/cli-adapter');
const { render } = require('../lib/wechat-renderer');

function slugifyEnglishText(input) {
  return input
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .replace(/-{2,}/g, '-');
}

function parseWechatArchiveIdentity(filePath, fallbackDate) {
  const basename = path.basename(filePath, path.extname(filePath));
  const match = basename.match(/^(\d{4}-\d{2}-\d{2})-(.+?)(?:-wechat)?$/);
  if (!match) {
    return {
      date: fallbackDate,
      slug: slugifyEnglishText(basename) || 'wechat-article',
    };
  }

  const normalizedSlug = match[2].replace(/^ai-infra-daily-brief-/, '');
  return {
    date: match[1],
    slug: slugifyEnglishText(normalizedSlug) || 'wechat-article',
  };
}

function insertTitleImageFallback(markdown, imageUrl) {
  return `![题图](${imageUrl})\n\n${markdown}`;
}

function loadTitleImageInserter(projectRoot) {
  try {
    const coverPrompts = require(path.join(projectRoot, 'scripts', 'lib', 'wechat-cover-prompts'));
    return coverPrompts.insertGeneratedTitleImageMarkdown || insertTitleImageFallback;
  } catch {
    return insertTitleImageFallback;
  }
}

function insertArchiveCoverMarkdown(markdown, imageUrl, projectRoot) {
  const insertTitleImage = loadTitleImageInserter(projectRoot);
  const match = markdown.match(/^(---\n[\s\S]*?\n---\n)([\s\S]*)$/);
  if (!match) return insertTitleImage(markdown, imageUrl);
  return `${match[1]}${insertTitleImage(match[2], imageUrl)}`;
}

function copyCoverToArchive(date, archiveDir, archiveMdPath, coverImagePath) {
  if (!coverImagePath || !fs.existsSync(coverImagePath)) return null;

  const ext = path.extname(coverImagePath).slice(1) || 'png';
  const identity = parseWechatArchiveIdentity(archiveMdPath, date);
  const assetDir = path.join(archiveDir, 'assets', identity.date);
  const filename = `${identity.slug}-cover.${ext}`;
  const archiveCoverPath = path.join(assetDir, filename);

  fs.mkdirSync(assetDir, { recursive: true });
  fs.copyFileSync(coverImagePath, archiveCoverPath);

  return {
    archiveCoverPath,
    markdownPath: path.posix.join('assets', identity.date, filename),
  };
}

async function wechatFormat(date, postFile, outputDir, projectRoot, coverImagePath, archiveDir) {
  if (!fs.existsSync(postFile)) {
    throw new Error(`Post file not found: ${postFile}`);
  }

  const postContent = fs.readFileSync(postFile, 'utf8');
  const skillPath = findSkill('wechat-formatter', projectRoot);

  // Step A: AI semantic rewrite
  const prompt = `请将以下 GitHub.io 版日报改写为微信公众号格式。

**重要约束：不要读取任何文件，不要创建任何文件，不要使用任何工具。只需要将改写后的 markdown 作为纯文本输出。输出第一行必须是 ---。**

报道日期：${date}

<github-post>
${postContent}
</github-post>

按照上面 wechat-formatter skill 的规则输出完整 markdown（含 front matter）。直接开始，不要任何前缀。`;

  console.log('[wechat-format] Step A: AI rewriting for WeChat...');
  const rawMd = await runAI({ prompt, skillPath });
  const wechatMd = rawMd
    .replace(/\r\n/g, '\n')
    .replace(/^```[a-z]*\n/, '')
    .replace(/\n```\s*$/, '\n');

  fs.mkdirSync(outputDir, { recursive: true });
  const mdPath = path.join(outputDir, `${date}-ai-infra-daily-brief-wechat.md`);
  fs.writeFileSync(mdPath, wechatMd, 'utf8');
  console.log(`[wechat-format] WeChat markdown saved: ${mdPath}`);

  let archiveMdPath = null;
  let archiveCoverPath = null;
  if (archiveDir) {
    fs.mkdirSync(archiveDir, { recursive: true });
    archiveMdPath = path.join(archiveDir, `${date}-ai-infra-daily-brief-wechat.md`);
    let archiveMd = wechatMd;
    const archivedCover = copyCoverToArchive(date, archiveDir, archiveMdPath, coverImagePath);
    if (archivedCover) {
      archiveCoverPath = archivedCover.archiveCoverPath;
      archiveMd = insertArchiveCoverMarkdown(archiveMd, archivedCover.markdownPath, projectRoot);
      console.log(`[wechat-format] WeChat cover archived: ${archiveCoverPath}`);
    }
    fs.writeFileSync(archiveMdPath, archiveMd, 'utf8');
    console.log(`[wechat-format] WeChat markdown archived: ${archiveMdPath}`);
  }

  // Step B: CSS rendering
  console.log('[wechat-format] Step B: Rendering to styled HTML...');
  const html = render(wechatMd, coverImagePath || null);

  const htmlPath = path.join(outputDir, `${date}-ai-infra-daily-brief-wechat.html`);
  fs.writeFileSync(htmlPath, html, 'utf8');
  console.log(`[wechat-format] Styled HTML saved: ${htmlPath}`);

  return { mdPath, htmlPath, archiveMdPath, archiveCoverPath };
}

if (require.main === module) {
  const [date, postFile, outputDir, projectRoot, coverImagePath, archiveDir] = process.argv.slice(2);
  if (!date || !postFile || !outputDir || !projectRoot) {
    console.error('Usage: node 06-wechat-format.js <date> <post-file> <output-dir> <project-root> [cover-image-path] [archive-dir]');
    process.exit(1);
  }
  wechatFormat(date, postFile, outputDir, projectRoot, coverImagePath, archiveDir)
    .then(({ mdPath, htmlPath, archiveMdPath, archiveCoverPath }) => console.log('[wechat-format] Done:', mdPath, htmlPath, archiveMdPath || '', archiveCoverPath || ''))
    .catch((err) => {
      console.error('[wechat-format] Failed:', err.message);
      process.exit(1);
    });
}

module.exports = { wechatFormat };
