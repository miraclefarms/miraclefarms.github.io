const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const repoScope = require('../../config/repo-scope.json');

function getDateRange() {
  const end = new Date();
  const start = new Date();
  start.setDate(start.getDate() - repoScope.timeWindow.days);
  const fmt = (d) => d.toISOString().slice(0, 10);
  return { start: fmt(start), end: fmt(end) };
}

function fetchRepoData(repo) {
  const results = [];

  try {
    const releases = execSync(
      `gh api repos/${repo}/releases --jq '.[0:5] | .[] | {tag_name: .tag_name, name: .name, created_at: .created_at, url: .html_url}' 2>/dev/null`,
      { encoding: 'utf8', timeout: 15000 }
    );
    if (releases && releases.trim()) {
      results.push({ type: 'release', repo, data: releases.trim() });
    }
  } catch (e) {
    // no releases
  }

  try {
    const since = getDateRange().start;
    const commits = execSync(
      `gh api repos/${repo}/commits --since "${since}" --jq '.[0:15] | .[] | {sha: .sha[0:7], message: (.commit.message | split("\n")[0]), author: .commit.author.name, date: .commit.author.date, url: .html_url}' 2>/dev/null`,
      { encoding: 'utf8', timeout: 15000 }
    );
    if (commits && commits.trim()) {
      results.push({ type: 'commit', repo, data: commits.trim() });
    }
  } catch (e) {
    // no commits
  }

  return results;
}

async function runResearch(date) {
  const { start, end } = getDateRange();
  const repos = repoScope.repos;

  const rawData = [];
  for (const repo of repos) {
    try {
      const data = fetchRepoData(repo);
      rawData.push(...data);
    } catch (e) {
      console.error(`Error fetching ${repo}:`, e.message);
    }
  }

  let output = `# AI Infra 日报素材 - ${date}\n\n`;
  output += `时间窗口：${start} 至 ${end}\n\n`;

  for (const item of rawData) {
    output += `## ${item.repo} (${item.type})\n\n`;
    output += item.data + '\n\n';
  }

  return output;
}

async function saveResearchOutput(date, output, workDir) {
  const outputPath = path.join(workDir, 'research-output.md');
  fs.writeFileSync(outputPath, output, 'utf8');
  return outputPath;
}

if (require.main === module) {
  const date = process.argv[2] || new Date().toISOString().slice(0, 10);
  const workDir = path.join('/tmp/morning-report', date);
  fs.mkdirSync(workDir, { recursive: true });

  runResearch(date)
    .then((output) => saveResearchOutput(date, output, workDir))
    .then((filePath) => console.log('Research output saved to:', filePath))
    .catch((err) => {
      console.error('Research failed:', err.message);
      process.exit(1);
    });
}

module.exports = { runResearch, saveResearchOutput };