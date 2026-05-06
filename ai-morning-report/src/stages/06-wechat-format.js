/**
 * Stage 6: WeChat formatting — two-step.
 *
 *   Step A: AI rewrites GitHub.io markdown → WeChat markdown (wechat-formatter skill)
 *   Step B: wechat-renderer converts WeChat markdown → styled HTML
 *
 * Usage: node 06-wechat-format.js <date> <post-file> <output-dir> <project-root> [cover-image-path]
 *
 * Outputs:
 *   <output-dir>/<date>-ai-infra-daily-brief-wechat.md   (WeChat markdown)
 *   <output-dir>/<date>-ai-infra-daily-brief-wechat.html (Styled HTML for pasting)
 */

const fs = require('fs');
const path = require('path');
const { runAI, findSkill } = require('../lib/cli-adapter');
const { render } = require('../lib/wechat-renderer');

async function wechatFormat(date, postFile, outputDir, projectRoot, coverImagePath) {
  if (!fs.existsSync(postFile)) {
    throw new Error(`Post file not found: ${postFile}`);
  }

  const postContent = fs.readFileSync(postFile, 'utf8');
  const skillPath = findSkill('wechat-formatter', projectRoot);

  // Step A: AI semantic rewrite
  const prompt = `请将以下 GitHub.io 版日报改写为微信公众号格式。

报道日期：${date}

<github-post>
${postContent}
</github-post>

按照上面 wechat-formatter skill 的规则输出完整 markdown（含 front matter）。直接开始，不要任何前缀。`;

  console.log('[wechat-format] Step A: AI rewriting for WeChat...');
  const wechatMd = await runAI({ prompt, skillPath });

  fs.mkdirSync(outputDir, { recursive: true });
  const mdPath = path.join(outputDir, `${date}-ai-infra-daily-brief-wechat.md`);
  fs.writeFileSync(mdPath, wechatMd, 'utf8');
  console.log(`[wechat-format] WeChat markdown saved: ${mdPath}`);

  // Step B: CSS rendering
  console.log('[wechat-format] Step B: Rendering to styled HTML...');
  const html = render(wechatMd, coverImagePath || null);

  const htmlPath = path.join(outputDir, `${date}-ai-infra-daily-brief-wechat.html`);
  fs.writeFileSync(htmlPath, html, 'utf8');
  console.log(`[wechat-format] Styled HTML saved: ${htmlPath}`);

  return { mdPath, htmlPath };
}

if (require.main === module) {
  const [date, postFile, outputDir, projectRoot, coverImagePath] = process.argv.slice(2);
  if (!date || !postFile || !outputDir || !projectRoot) {
    console.error('Usage: node 06-wechat-format.js <date> <post-file> <output-dir> <project-root> [cover-image-path]');
    process.exit(1);
  }
  wechatFormat(date, postFile, outputDir, projectRoot, coverImagePath)
    .then(({ mdPath, htmlPath }) => console.log('[wechat-format] Done:', mdPath, htmlPath))
    .catch((err) => {
      console.error('[wechat-format] Failed:', err.message);
      process.exit(1);
    });
}

module.exports = { wechatFormat };
