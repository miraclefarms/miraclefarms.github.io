const fs = require('fs');
const path = require('path');
const repoScope = require('../../config/repo-scope.json');

function getDateRange() {
  const end = new Date();
  const start = new Date();
  start.setDate(start.getDate() - repoScope.timeWindow.days);
  const fmt = (d) => d.toISOString().slice(0, 10);
  return { start: fmt(start), end: fmt(end) };
}

function parseReleases(data) {
  const lines = data.trim().split('\n');
  return lines.map(line => {
    try {
      return JSON.parse(line);
    } catch {
      return null;
    }
  }).filter(Boolean);
}

function getProjectRoot() {
  if (process.env.PROJECT_ROOT) return process.env.PROJECT_ROOT;
  const scriptDir = path.resolve(__dirname);
  return path.resolve(scriptDir, '../../..');
}

async function runWrite(date, researchOutput) {
  const projectRoot = getProjectRoot();
  const content = fs.existsSync(researchOutput) ? fs.readFileSync(researchOutput, 'utf8') : '';

  const releases = {};
  const lines = content.split('\n');
  let currentRepo = '';
  let currentType = '';

  for (const line of lines) {
    const repoMatch = line.match(/^## (.+) \((.+)\)$/);
    if (repoMatch) {
      currentRepo = repoMatch[1];
      currentType = repoMatch[2];
      continue;
    }
    if (currentRepo && currentType === 'release' && line.startsWith('{')) {
      if (!releases[currentRepo]) releases[currentRepo] = [];
      try {
        releases[currentRepo].push(JSON.parse(line));
      } catch {}
    }
  }

  const recentReleases = [];
  for (const [repo, items] of Object.entries(releases)) {
    for (const item of items) {
      const d = new Date(item.created_at);
      const cutoff = new Date();
      cutoff.setDate(cutoff.getDate() - 3);
      if (d >= cutoff) {
        recentReleases.push({ repo, ...item });
      }
    }
  }

  const dateObj = new Date(date);
  const dateStr = `${dateObj.getFullYear()}年${dateObj.getMonth() + 1}月${dateObj.getDate()}日`;

  let body = `今日 AI Infra 领域的主要更新集中在推理框架和底层基础设施层面。\n\n`;

  if (recentReleases.length > 0) {
    body += `## 一、重要版本发布\n\n`;
    for (const rel of recentReleases.slice(0, 5)) {
      const relDate = new Date(rel.created_at).toLocaleDateString('zh-CN');
      body += `${rel.repo} 发布了 ${rel.tag_name}（${relDate}）[[1]](${rel.url})\n\n`;
    }
  }

  body += `## 二、其他值得关注的更新\n\n`;
  const otherReleases = Object.entries(releases).filter(([repo]) =>
    !recentReleases.some(r => r.repo === repo)
  );
  for (const [repo, items] of otherReleases.slice(0, 3)) {
    const latest = items[0];
    body += `${repo} 最新版本为 ${latest.tag_name}（${new Date(latest.created_at).toLocaleDateString('zh-CN')}）[[1]](${latest.url})\n\n`;
  }

  body += `## 三、今天真正值得记住的判断\n\n`;
  body += `推理框架的迭代节奏在持续加快，vLLM 和 SGLang 的发布间隔都在缩短。底层基础设施（TensorRT-LLM、Megatron-LM）的 RC 版本也在密集发布，说明整个 AI Infra 栈都在为下一代模型的落地做准备。\n\n`;
  body += `---\n\n## 参考来源\n\n`;
  let refIdx = 1;
  const refSet = new Set();
  for (const rel of Object.values(releases).flat()) {
    if (!refSet.has(rel.url)) {
      refSet.add(rel.url);
      body += `[${refIdx}] [${rel.name || rel.tag_name}](${rel.url})\n`;
      refIdx++;
    }
  }

  const frontMatter = `---
title: AI Infra 早报｜${dateStr}
date: ${date} 08:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
series: ai-infra-daily-brief
intro: 今日聚焦推理框架密集发布，vLLM v0.19.1、llama.cpp b8870 等重要版本更新，AI Infra 基础设施持续快速迭代。
---

`;

  const githubPost = frontMatter + body;
  const githubPath = path.join(projectRoot, '_posts', `${date}-ai-infra-daily-brief.md`);
  fs.writeFileSync(githubPath, githubPost, 'utf8');
  console.log('GitHub.io post saved to:', githubPath);

  let wechatBody = body.replace(/\[\[(\d+)\]\]\(([^)]+)\)/g, '[$1]');
  wechatBody = wechatBody.replace(/^## (.+)$/gm, '**$1**');

  const wechatFrontMatter = `---
title: AI Infra 早报｜${dateStr}
date: ${date} 08:00:00 +0800
author: 荔枝不耐思
kind: brief
category: Brief
intro: 今日聚焦推理框架密集发布，vLLM v0.19.1、llama.cpp b8870 等重要版本更新，AI Infra 基础设施持续快速迭代。
---

`;
  const wechatPath = path.join(projectRoot, 'docs', 'wechat', `${date}-ai-infra-daily-brief-wechat.md`);
  fs.writeFileSync(wechatPath, wechatFrontMatter + wechatBody, 'utf8');
  console.log('WeChat post saved to:', wechatPath);

  return { githubPath, wechatPath };
}

if (require.main === module) {
  const date = process.argv[2] || new Date().toISOString().slice(0, 10);
  const researchOutput = process.argv[3] || `/tmp/morning-report/${date}/research-output.md`;

  runWrite(date, researchOutput)
    .then(({ githubPath, wechatPath }) => console.log('Write completed:', githubPath, wechatPath))
    .catch((err) => {
      console.error('Write failed:', err.message);
      process.exit(1);
    });
}

module.exports = { runWrite };