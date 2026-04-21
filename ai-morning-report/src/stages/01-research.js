const path = require('path');
const { runOpencode, getModel, getProjectRoot } = require('../lib/openai');
const repoScope = require('../../config/repo-scope.json');

function getDateRange() {
  const end = new Date();
  const start = new Date();
  start.setDate(start.getDate() - repoScope.timeWindow.days);
  const fmt = (d) => d.toISOString().slice(0, 10);
  return { start: fmt(start), end: fmt(end) };
}

async function runResearch(date) {
  const { start, end } = getDateRange();
  const repos = repoScope.repos.join(', ');
  const model = getModel();

  const prompt = `你是 AI Infra 日报调研员。调用 ai-morning-report skill，执行以下任务：

1. 调研日期：${date}（北京时间）
2. 时间窗口：${start} 至 ${end}（最近三天）
3. 追踪仓库：${repos}

请按 ai-morning-report skill 的流程：
1. 确定范围（默认时间窗口 + repo 列表）
2. 采集候选更新（关注 release、merged PR、高影响 commit、官方博客）
3. 过滤低价值噪音（降权纯文档、机械重构、单测补丁）
4. 聚类成"今天的几件事"（3-6 个主题）
5. 检查最近日报去重
6. 输出结构化素材包

输出素材包应包含：
- 报道日期
- 主题判断（1-2 个主线）
- 证据列表（标题、URL、为何重要、属于哪个主题）
- 候选图片（如果有）

注意：只负责素材整理，不要调用 miraclefarms-writer。`;

  const workdir = getProjectRoot();
  const result = await runOpencode({ prompt, skill: 'ai-morning-report', workdir, model });
  return result.stdout;
}

async function saveResearchOutput(date, output, workDir) {
  const outputPath = path.join(workDir, 'research-output.md');
  const fs = require('fs');
  fs.writeFileSync(outputPath, output, 'utf8');
  return outputPath;
}

if (require.main === module) {
  const date = process.argv[2] || new Date().toISOString().slice(0, 10);
  const workDir = path.join('/tmp/morning-report', date);
  require('fs').mkdirSync(workDir, { recursive: true });

  runResearch(date)
    .then((output) => saveResearchOutput(date, output, workDir))
    .then((path) => console.log('Research output saved to:', path))
    .catch((err) => {
      console.error('Research failed:', err.message);
      process.exit(1);
    });
}

module.exports = { runResearch, saveResearchOutput };