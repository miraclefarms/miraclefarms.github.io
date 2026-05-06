/**
 * Stage 2: AI analysis — filter, cluster, deduplicate raw data into a material package.
 *
 * Usage: node 02-analyze.js <date> <raw-data-file> <output-file> <project-root>
 *
 * Gate: fails if < 2 distinct themes found in output.
 */

const fs = require('fs');
const path = require('path');
const { runAI, findSkill } = require('../lib/cli-adapter');

const MIN_THEMES = 2;

function getRecentBriefSummaries(projectRoot, count = 3) {
  const postsDir = path.join(projectRoot, '_posts');
  if (!fs.existsSync(postsDir)) return '';
  const briefs = fs.readdirSync(postsDir)
    .filter((f) => f.includes('ai-infra-daily-brief') && f.endsWith('.md'))
    .sort()
    .slice(-count);

  return briefs.map((f) => {
    const content = fs.readFileSync(path.join(postsDir, f), 'utf8');
    const titleMatch = content.match(/^title:\s*(.+)$/m);
    const introMatch = content.match(/^intro:\s*(.+)$/m);
    const dateMatch = f.match(/^(\d{4}-\d{2}-\d{2})/);
    return `- ${dateMatch?.[1] || f}: ${titleMatch?.[1] || ''} — ${introMatch?.[1] || ''}`;
  }).join('\n');
}

function countThemes(output) {
  return (output.match(/^## /gm) || []).length;
}

async function analyze(date, rawDataFile, outputFile, projectRoot) {
  if (!fs.existsSync(rawDataFile)) {
    throw new Error(`Raw data file not found: ${rawDataFile}`);
  }

  const rawData = fs.readFileSync(rawDataFile, 'utf8');
  const recentSummaries = getRecentBriefSummaries(projectRoot);
  const skillPath = findSkill('ai-morning-report', projectRoot);

  const prompt = `你是 AI Infra 领域的技术编辑。今天的报道日期是 ${date}。

以下是从 AI Infra 相关 repo 抓取的原始数据（最近 3 天）：

<raw-data>
${rawData}
</raw-data>

最近已发布的日报（用于去重，避免重复写同一件事）：

<recent-briefs>
${recentSummaries || '（暂无）'}
</recent-briefs>

请按照上面 AI Morning Report skill 的要求，输出结构化素材包。

**重要约束：**
- 不要读取任何文件，不要创建任何文件，不要使用任何工具
- 只需要将内容作为纯文本输出到 stdout

**输出格式要求：**
- 以 \`# 素材包 — ${date}\` 开头
- 每个主题写一个 H2（## 主题名称），包含：主线判断（为什么重要）、证据列表（PR/release/commit，含完整 URL）
- 最后一个 H2 写：\`## 今日主线判断\`，用 1-2 句话说明今天最值得写的角度
- 直接输出 markdown，不要说"好的"或任何前缀
- 控制在 3-6 个主题

**重要：**
- 只写有真实 PR/release 证据的事，不要生成无 URL 支撑的判断
- URL 必须来自原始数据中出现的真实链接`;

  console.log('[analyze] Calling AI...');
  const raw = await runAI({ prompt, skillPath });
  // Normalize + strip code fences + preamble before first heading
  let text = raw.replace(/\r\n/g, '\n').replace(/^```[a-z]*\n/, '').replace(/\n```\s*$/, '\n');
  const headingStart = text.search(/(?:^|\n)# /);
  const result = headingStart > 0 ? text.slice(headingStart).replace(/^\n/, '') : text;

  const themeCount = countThemes(result);
  if (themeCount < MIN_THEMES) {
    throw new Error(
      `[analyze] Gate failed: only ${themeCount} theme(s) found (minimum ${MIN_THEMES}). Raw AI output:\n${result.slice(0, 500)}`
    );
  }

  fs.writeFileSync(outputFile, result, 'utf8');
  console.log(`[analyze] Material package saved: ${outputFile} (${themeCount} themes)`);
}

if (require.main === module) {
  const [date, rawDataFile, outputFile, projectRoot] = process.argv.slice(2);
  if (!date || !rawDataFile || !outputFile || !projectRoot) {
    console.error('Usage: node 02-analyze.js <date> <raw-data-file> <output-file> <project-root>');
    process.exit(1);
  }
  analyze(date, rawDataFile, outputFile, projectRoot).catch((err) => {
    console.error('[analyze] Failed:', err.message);
    process.exit(1);
  });
}

module.exports = { analyze };
