/**
 * Stage 3: AI writing — convert material package into a publishable GitHub.io brief.
 *
 * Usage: node 03-write.js <date> <material-file> <project-root>
 *
 * Gate: validates front matter (title, date, intro required).
 */

const fs = require('fs');
const path = require('path');
const { runAI, findSkill } = require('../lib/cli-adapter');

const REQUIRED_FM_FIELDS = ['title', 'date', 'intro'];

function validateFrontMatter(content) {
  const match = content.match(/^---\n([\s\S]*?)\n---/);
  if (!match) throw new Error('No front matter found in AI output');
  const fm = match[1];
  for (const field of REQUIRED_FM_FIELDS) {
    if (!new RegExp(`^${field}:`, 'm').test(fm)) {
      throw new Error(`Missing front matter field: ${field}`);
    }
  }
}

async function write(date, materialFile, projectRoot) {
  if (!fs.existsSync(materialFile)) {
    throw new Error(`Material file not found: ${materialFile}`);
  }

  const material = fs.readFileSync(materialFile, 'utf8');
  const skillPath = findSkill('miraclefarms-writer', projectRoot);

  const prompt = `你是 MiracleFarms AI Infra 日报的技术编辑。今天的报道日期是 ${date}。

以下是今日的素材包（已经过筛选和聚类）：

<material>
${material}
</material>

请按照上面 miraclefarms-writer skill 的要求，生成一篇完整的 GitHub.io 版 AI Infra 早报（kind: brief）。

**输出要求：**
- 直接输出完整 markdown（含 front matter），不要任何前缀说明
- front matter 必须包含：title、date（${date} 08:00:00 +0800）、author（荔枝不耐思）、kind（brief）、category（Brief）、series（ai-infra-daily-brief）、intro
- 正文结构：开头综述段（无 H2）→ H2 章节（中文数字编号：一、二、三、）→ 可选"今天真正值得记住的判断"→ 分隔线 → ## 参考来源
- 引用格式：正文用 [[N]](url)，参考来源用 [N] [标题](url)
- 所有 URL 必须来自素材包中的真实链接，不要编造
- 重点内容加粗
- 控制在 800-1500 字`;

  console.log('[write] Calling AI...');
  const result = await runAI({ prompt, skillPath });

  validateFrontMatter(result);

  const postsDir = path.join(projectRoot, '_posts');
  fs.mkdirSync(postsDir, { recursive: true });
  const outputPath = path.join(postsDir, `${date}-ai-infra-daily-brief.md`);
  fs.writeFileSync(outputPath, result, 'utf8');

  const wordCount = result.length;
  console.log(`[write] Brief saved: ${outputPath} (~${wordCount} chars)`);
  return outputPath;
}

if (require.main === module) {
  const [date, materialFile, projectRoot] = process.argv.slice(2);
  if (!date || !materialFile || !projectRoot) {
    console.error('Usage: node 03-write.js <date> <material-file> <project-root>');
    process.exit(1);
  }
  write(date, materialFile, projectRoot).catch((err) => {
    console.error('[write] Failed:', err.message);
    process.exit(1);
  });
}

module.exports = { write };
