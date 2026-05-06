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

function extractAndValidate(raw) {
  // Normalize line endings
  let text = raw.replace(/\r\n/g, '\n');

  // Strip markdown code fence if AI wrapped the entire output (e.g. ```yaml or ```markdown)
  text = text.replace(/^```[a-z]*\n/, '').replace(/\n```\s*$/, '\n');

  // Find front matter start (---) — skip any preamble text
  const fmStart = text.search(/(?:^|\n)---\n/);
  if (fmStart === -1) {
    const preview = text.slice(0, 300).replace(/\n/g, '↵');
    throw new Error(`No front matter found in AI output. First 300 chars: ${preview}`);
  }

  // Slice from the --- line (handle the leading \n from the regex)
  const content = text.slice(fmStart).replace(/^\n/, '');
  const match = content.match(/^---\n([\s\S]*?)\n---/);
  if (!match) throw new Error('Malformed front matter in AI output');

  const fm = match[1];
  for (const field of REQUIRED_FM_FIELDS) {
    if (!new RegExp(`^${field}:`, 'm').test(fm)) {
      throw new Error(`Missing front matter field: ${field}`);
    }
  }
  return content;
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

**输出必须直接以 --- 开始（YAML front matter），不要任何前置说明、不要"好的"、不要代码块包裹。**

front matter 必须包含：title、date（${date} 08:00:00 +0800）、author（荔枝不耐思）、kind（brief）、category（Brief）、series（ai-infra-daily-brief）、intro
正文结构：开头综述段（无 H2）→ H2 章节（中文数字编号：一、二、三、）→ 可选"今天真正值得记住的判断"→ 分隔线 → ## 参考来源
引用格式：正文用 [[N]](url)，参考来源用 [N] [标题](url)
所有 URL 必须来自素材包中的真实链接，不要编造。重点内容加粗。控制在 800-1500 字。`;

  console.log('[write] Calling AI...');
  const raw = await runAI({ prompt, skillPath });
  const result = extractAndValidate(raw);

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
