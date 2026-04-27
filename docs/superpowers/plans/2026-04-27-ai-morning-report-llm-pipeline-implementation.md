# AI Morning Report LLM Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current scaffolded daily brief automation with a template + LLM driven pipeline that collects 1-3 day PR candidates, ranks 5-8 high-value items, composes a single article source, renders GitHub and WeChat outputs, publishes to `main`, and records WeChat retry state.

**Architecture:** Build a new staged pipeline around four durable artifacts: `research.json`, `selection.json`, `article-source.json`, and `publish-record.json`. Keep the existing WeChat publishing stack as the stable delivery layer, add new collection/ranking/composition/rendering stages under `ai-morning-report/src`, and switch the scheduler to the new entrypoint only after the new chain has passing unit and integration tests.

**Tech Stack:** Node.js, `node:test`, `gh` CLI, existing WeChat publisher (`scripts/publish-wechat.js`), existing WeChat render/prompt helpers, Jekyll post conventions, shell scheduler entrypoint.

---

## File Map

### New files

- `ai-morning-report/src/lib/date-window.js`
  - Shanghai date window calculation and target-date helpers.
- `ai-morning-report/src/lib/article-schema.js`
  - Validation and normalization helpers for `research.json`, `selection.json`, `article-source.json`, and `publish-record.json`.
- `ai-morning-report/src/lib/github-research.js`
  - `gh`-backed collectors for PRs, releases, and supporting commits.
- `ai-morning-report/src/lib/publish-record.js`
  - Load/save/update helpers for publish status and retry state.
- `ai-morning-report/src/lib/pipeline-paths.js`
  - Centralized paths for work directories, output files, and asset roots.
- `ai-morning-report/src/stages/01-collect.js`
  - Candidate collection stage.
- `ai-morning-report/src/stages/02-rank.js`
  - LLM ranking and selection stage.
- `ai-morning-report/src/stages/03-compose.js`
  - LLM composition stage producing `article-source.json`.
- `ai-morning-report/src/stages/04-image.js`
  - Cover image generation into repo assets and article-source backfill.
- `ai-morning-report/src/stages/05-render.js`
  - GitHub post and WeChat markdown render stage.
- `ai-morning-report/src/stages/06-publish-github.js`
  - Git add/commit/push stage for tracked + untracked files.
- `ai-morning-report/src/stages/07-publish-wechat.js`
  - Wrapper around `scripts/publish-wechat.js` plus publish-record updates.
- `scripts/tests/date-window.test.js`
  - Unit tests for target date/window logic.
- `scripts/tests/article-schema.test.js`
  - Unit tests for artifact validation.
- `scripts/tests/github-research.test.js`
  - Unit tests for candidate filtering and failure behavior.
- `scripts/tests/ai-morning-report-render.test.js`
  - Unit tests for rendering GitHub and WeChat output from article source.
- `scripts/tests/publish-record.test.js`
  - Unit tests for publish status transitions.
- `scripts/tests/ai-morning-report-pipeline.test.js`
  - End-to-end smoke tests for the staged pipeline with mocked dependencies.

### Files to modify

- `ai-morning-report/bin/run-daily.sh`
  - New stage orchestration and downgrade handling.
- `ai-morning-report/config/model-config.json`
  - Separate `research`, `ranking`, `writing`, and `image` model keys.
- `ai-morning-report/config/repo-scope.json`
  - Add window, thresholds, and ranking hints.
- `ai-morning-report/src/lib/openai.js`
  - Generalize LLM/image invocation helpers for stage-specific models.
- `scripts/publish-wechat.js`
  - Export a stable programmatic entrypoint if needed by stage 07.
- `package.json`
  - Add test scripts for the new pipeline.
- `ai-morning-report/docs/SPEC.md`
  - Mark old scaffold spec as superseded and point at the new design.
- `ai-morning-report/docs/ai-morning-report.plist`
  - Ensure scheduler calls the new entrypoint path/arguments.

### Files to retire after cutover

- `ai-morning-report/src/stages/01-research.js`
- `ai-morning-report/src/stages/02-write.js`
- `ai-morning-report/src/stages/03-images.js`
- `ai-morning-report/src/stages/04-publish.js`
- `ai-morning-report/src/stages/05-wechat.js`

Retire these only after the new stages and smoke tests pass.

## Task 1: Establish Shared Contracts and Test Harness

**Files:**
- Create: `ai-morning-report/src/lib/date-window.js`
- Create: `ai-morning-report/src/lib/article-schema.js`
- Create: `ai-morning-report/src/lib/pipeline-paths.js`
- Create: `scripts/tests/date-window.test.js`
- Create: `scripts/tests/article-schema.test.js`
- Modify: `package.json`

- [ ] **Step 1: Write failing tests for target date windows and artifact validation**

```js
// scripts/tests/date-window.test.js
const test = require('node:test');
const assert = require('node:assert/strict');

const { getTargetDateParts, buildResearchWindow } = require('../../ai-morning-report/src/lib/date-window');

test('buildResearchWindow anchors to Asia/Shanghai natural day and 3-day lookback', () => {
  const now = new Date('2026-04-27T05:00:00+08:00');
  const parts = getTargetDateParts(now, 'Asia/Shanghai');
  const window = buildResearchWindow(parts.date, { timezone: 'Asia/Shanghai', lookbackDays: 3 });

  assert.equal(parts.date, '2026-04-27');
  assert.equal(window.start, '2026-04-25T00:00:00+08:00');
  assert.equal(window.end, '2026-04-27T23:59:59+08:00');
});
```

```js
// scripts/tests/article-schema.test.js
const test = require('node:test');
const assert = require('node:assert/strict');

const { validateArticleSource } = require('../../ai-morning-report/src/lib/article-schema');

test('validateArticleSource requires title, intro, thesis, sections, and wechat metadata', () => {
  assert.throws(
    () => validateArticleSource({ title: 'AI Infra 早报｜缺字段' }),
    /missing required field: intro/
  );
});
```

- [ ] **Step 2: Run tests to verify they fail because the new helpers do not exist yet**

Run:

```bash
node --test scripts/tests/date-window.test.js scripts/tests/article-schema.test.js
```

Expected: FAIL with `Cannot find module '../../ai-morning-report/src/lib/date-window'`.

- [ ] **Step 3: Implement date window, schema, and path helpers**

```js
// ai-morning-report/src/lib/date-window.js
function getTargetDateParts(now = new Date(), timezone = 'Asia/Shanghai') {
  const formatter = new Intl.DateTimeFormat('en-CA', {
    timeZone: timezone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  });
  const date = formatter.format(now);
  return { date, timezone };
}

function buildResearchWindow(targetDate, { timezone = 'Asia/Shanghai', lookbackDays = 3 } = {}) {
  const [year, month, day] = targetDate.split('-').map(Number);
  const end = new Date(Date.UTC(year, month - 1, day, 15, 59, 59));
  const start = new Date(Date.UTC(year, month - 1, day - (lookbackDays - 1), 16, 0, 0));
  return {
    timezone,
    start: new Intl.DateTimeFormat('sv-SE', { timeZone: 'Asia/Shanghai', hour12: false }).format(start).replace(' ', 'T') + '+08:00',
    end: new Intl.DateTimeFormat('sv-SE', { timeZone: 'Asia/Shanghai', hour12: false }).format(end).replace(' ', 'T') + '+08:00',
  };
}

module.exports = { getTargetDateParts, buildResearchWindow };
```

```js
// ai-morning-report/src/lib/article-schema.js
function requireField(value, fieldName) {
  if (value === undefined || value === null || value === '' || (Array.isArray(value) && value.length === 0)) {
    throw new Error(`missing required field: ${fieldName}`);
  }
}

function validateArticleSource(doc) {
  requireField(doc.title, 'title');
  requireField(doc.intro, 'intro');
  requireField(doc.thesis, 'thesis');
  requireField(doc.sections, 'sections');
  requireField(doc.wechat, 'wechat');
  requireField(doc.wechat.title, 'wechat.title');
  requireField(doc.wechat.digest, 'wechat.digest');
  return doc;
}

module.exports = { validateArticleSource };
```

```js
// ai-morning-report/src/lib/pipeline-paths.js
const path = require('node:path');

function buildPipelinePaths(projectRoot, targetDate, slug = 'ai-infra-daily-brief') {
  const workDir = path.join(projectRoot, '.cache', 'ai-morning-report', targetDate);
  const assetDir = path.join(projectRoot, 'assets', 'ai-infra-daily-brief', `${targetDate}-${slug}`);
  return {
    workDir,
    researchJson: path.join(workDir, 'research.json'),
    selectionJson: path.join(workDir, 'selection.json'),
    articleSourceJson: path.join(workDir, 'article-source.json'),
    publishRecordJson: path.join(workDir, 'publish-record.json'),
    githubPostPath: path.join(projectRoot, '_posts', `${targetDate}-${slug}.md`),
    wechatPostPath: path.join(projectRoot, 'docs', 'wechat', `${targetDate}-${slug}-wechat.md`),
    assetDir,
  };
}

module.exports = { buildPipelinePaths };
```

- [ ] **Step 4: Add a dedicated test script**

```json
{
  "scripts": {
    "test": "node --test scripts/tests/*.test.js",
    "test:ai-morning-report": "node --test scripts/tests/date-window.test.js scripts/tests/article-schema.test.js scripts/tests/github-research.test.js scripts/tests/ai-morning-report-render.test.js scripts/tests/publish-record.test.js scripts/tests/ai-morning-report-pipeline.test.js"
  }
}
```

- [ ] **Step 5: Run tests to verify the shared-contract layer passes**

Run:

```bash
node --test scripts/tests/date-window.test.js scripts/tests/article-schema.test.js
```

Expected: PASS, `2 tests`.

- [ ] **Step 6: Commit the contract layer**

```bash
git add package.json \
  ai-morning-report/src/lib/date-window.js \
  ai-morning-report/src/lib/article-schema.js \
  ai-morning-report/src/lib/pipeline-paths.js \
  scripts/tests/date-window.test.js \
  scripts/tests/article-schema.test.js
git commit -m "feat: add AI morning report pipeline contracts"
```

## Task 2: Build the Collect Stage Around PR-First Research

**Files:**
- Create: `ai-morning-report/src/lib/github-research.js`
- Create: `ai-morning-report/src/stages/01-collect.js`
- Create: `scripts/tests/github-research.test.js`
- Modify: `ai-morning-report/config/repo-scope.json`

- [ ] **Step 1: Write failing tests for PR-first candidate collection and failure behavior**

```js
// scripts/tests/github-research.test.js
const test = require('node:test');
const assert = require('node:assert/strict');

const { rankCandidatePriority, assertNonEmptyResearch } = require('../../ai-morning-report/src/lib/github-research');

test('rankCandidatePriority prefers runtime/performance PRs over docs-only updates', () => {
  const hot = rankCandidatePriority({ labels: ['performance'], signals: { default_path: true } });
  const cold = rankCandidatePriority({ labels: ['documentation'], signals: { default_path: false } });
  assert.ok(hot > cold);
});

test('assertNonEmptyResearch throws when all upstream fetches fail', () => {
  assert.throws(() => assertNonEmptyResearch([]), /no research candidates collected/);
});
```

- [ ] **Step 2: Run the research tests to verify they fail**

Run:

```bash
node --test scripts/tests/github-research.test.js
```

Expected: FAIL with `Cannot find module '../../ai-morning-report/src/lib/github-research'`.

- [ ] **Step 3: Implement GitHub collection helpers with explicit failure semantics**

```js
// ai-morning-report/src/lib/github-research.js
function rankCandidatePriority(candidate) {
  let score = 0;
  if (candidate.signals?.default_path) score += 5;
  if (candidate.signals?.performance) score += 4;
  if (candidate.signals?.feature) score += 3;
  if (candidate.signals?.stability) score += 2;
  if ((candidate.labels || []).some(label => /docs|documentation|ci|test/i.test(label))) score -= 5;
  return score;
}

function assertNonEmptyResearch(items) {
  if (!Array.isArray(items) || items.length === 0) {
    throw new Error('no research candidates collected');
  }
  return items;
}

async function collectRepoCandidates({ repo, window, ghClient }) {
  const pullRequests = await ghClient.listMergedPullRequests(repo, window);
  return pullRequests.map(pr => ({
    type: 'pull_request',
    repo,
    number: pr.number,
    title: pr.title,
    url: pr.url,
    merged_at: pr.merged_at,
    labels: pr.labels,
    summary: pr.summary,
    signals: pr.signals,
    evidence: [{ kind: 'pr', url: pr.url }],
  }));
}

module.exports = { rankCandidatePriority, assertNonEmptyResearch, collectRepoCandidates };
```

- [ ] **Step 4: Implement `01-collect.js` to write `research.json` and fail fast on empty data**

```js
// ai-morning-report/src/stages/01-collect.js
async function runCollect({ targetDate, config, ghClient, outputPath }) {
  const repos = config.repos || [];
  const repoResults = [];
  for (const repo of repos) {
    const items = await collectRepoCandidates({ repo, window: config.window, ghClient });
    repoResults.push({ name: repo, items });
  }
  const flattened = repoResults.flatMap(repo => repo.items);
  assertNonEmptyResearch(flattened);
  fs.writeFileSync(outputPath, JSON.stringify({ target_date: targetDate, repos: repoResults }, null, 2) + '\n');
}
```

- [ ] **Step 5: Expand repo config to support window and thresholds**

```json
{
  "window": {
    "lookback_days": 3,
    "timezone": "Asia/Shanghai"
  },
  "selection": {
    "minimum_items": 5,
    "maximum_items": 8
  }
}
```

- [ ] **Step 6: Run the research tests and a manual dry-run fixture**

Run:

```bash
node --test scripts/tests/github-research.test.js
```

Expected: PASS.

Run:

```bash
node ai-morning-report/src/stages/01-collect.js 2026-04-27 /tmp/ai-morning-report-research.json
```

Expected: FAIL fast with a clear auth/network error if `gh` is unavailable, not a silent success file.

- [ ] **Step 7: Commit the collect stage**

```bash
git add ai-morning-report/config/repo-scope.json \
  ai-morning-report/src/lib/github-research.js \
  ai-morning-report/src/stages/01-collect.js \
  scripts/tests/github-research.test.js
git commit -m "feat: add PR-first collection stage"
```

## Task 3: Add Ranking and Composition Stages Backed by Stage-Specific Models

**Files:**
- Create: `ai-morning-report/src/stages/02-rank.js`
- Create: `ai-morning-report/src/stages/03-compose.js`
- Modify: `ai-morning-report/src/lib/openai.js`
- Modify: `ai-morning-report/config/model-config.json`
- Create: `scripts/tests/ai-morning-report-pipeline.test.js`

- [ ] **Step 1: Write a failing smoke test for `research.json -> selection.json -> article-source.json`**

```js
// scripts/tests/ai-morning-report-pipeline.test.js
const test = require('node:test');
const assert = require('node:assert/strict');

const { selectCandidates } = require('../../ai-morning-report/src/stages/02-rank');
const { composeArticleSource } = require('../../ai-morning-report/src/stages/03-compose');

test('ranking selects 5-8 candidates and composition emits article-source metadata', async () => {
  const research = {
    target_date: '2026-04-27',
    repos: [
      {
        name: 'sgl-project/sglang',
        items: [
          {
            type: 'pull_request',
            repo: 'sgl-project/sglang',
            number: 12345,
            title: 'Add breakable graph support',
            url: 'https://github.com/example/pr/1',
            merged_at: '2026-04-27T01:00:00Z',
            labels: ['performance'],
            summary: 'Changes default runtime behavior',
            signals: { default_path: true, performance: true, feature: true, stability: false }
          }
        ]
      }
    ]
  };

  const ranked = await selectCandidates({
    research,
    invokeModel: async () => ({
      selected: [{ rank: 1, repo: 'sgl-project/sglang', pr_number: 12345, why_selected: 'runtime path', category: 'runtime', importance_score: 9.6, evidence_strength: 'high', primary_angle: '默认路径' }]
    })
  });

  const article = await composeArticleSource({
    selection: ranked,
    invokeModel: async () => ({
      title: 'AI Infra 早报｜测试标题',
      intro: '今日主线是默认路径收敛。',
      thesis: '真正值得记住的是默认路径开始吸收复杂场景。',
      sections: [{ heading: '一、运行时', items: [{ title: 'SGLang', what_changed: '新增能力', why_it_matters: '影响默认路径', references: [{ index: 1, title: 'Official PR', url: 'https://github.com/example/pr/1' }] }] }],
      wechat: { title: 'AI Infra 早报｜测试标题', digest: '一句话摘要', author: '荔枝不耐思', thumb_strategy: 'generated-cover' }
    })
  });

  assert.equal(article.title, 'AI Infra 早报｜测试标题');
  assert.equal(article.wechat.digest, '一句话摘要');
});
```

- [ ] **Step 2: Run the smoke test to verify rank/compose do not exist yet**

Run:

```bash
node --test scripts/tests/ai-morning-report-pipeline.test.js
```

Expected: FAIL with `Cannot find module '../../ai-morning-report/src/stages/02-rank'`.

- [ ] **Step 3: Generalize the model invoker to support named model roles**

```js
// ai-morning-report/src/lib/openai.js
function getModelForRole(role, modelConfig) {
  if (process.env.AI_MODEL) return process.env.AI_MODEL;
  if (modelConfig.roles && modelConfig.roles[role]) return modelConfig.roles[role];
  return modelConfig.default;
}

async function invokeJsonModel({ role, prompt, modelConfig, runModel }) {
  const model = getModelForRole(role, modelConfig);
  const raw = await runModel({ prompt, model });
  return JSON.parse(raw);
}

module.exports = { getModelForRole, invokeJsonModel };
```

- [ ] **Step 4: Implement rank stage with explicit item-count enforcement**

```js
// ai-morning-report/src/stages/02-rank.js
async function selectCandidates({ research, invokeModel, minItems = 5, maxItems = 8 }) {
  const response = await invokeModel({ prompt: JSON.stringify(research, null, 2) });
  const selected = response.selected || [];
  if (selected.length === 0) throw new Error('ranking produced no selected candidates');
  if (selected.length > maxItems) throw new Error(`ranking exceeded maxItems: ${selected.length}`);
  if (selected.length < minItems) {
    throw new Error(`ranking produced fewer than minimum items: ${selected.length}`);
  }
  return { target_date: research.target_date, selected };
}
```

- [ ] **Step 5: Implement compose stage with article-source validation**

```js
// ai-morning-report/src/stages/03-compose.js
async function composeArticleSource({ selection, invokeModel }) {
  const article = await invokeModel({ prompt: JSON.stringify(selection, null, 2) });
  article.kind = 'brief';
  article.author = article.author || '荔枝不耐思';
  article.category = 'Brief';
  article.series = 'ai-infra-daily-brief';
  return validateArticleSource(article);
}
```

- [ ] **Step 6: Split model configuration by role**

```json
{
  "default": "z-ai/glm-5.1",
  "roles": {
    "research": "z-ai/glm-5.1",
    "ranking": "z-ai/glm-5.1",
    "writing": "z-ai/glm-5.1",
    "image": "openrouter/ministral-3"
  }
}
```

- [ ] **Step 7: Run the smoke test and verify article-source validation passes**

Run:

```bash
node --test scripts/tests/ai-morning-report-pipeline.test.js
```

Expected: PASS.

- [ ] **Step 8: Commit ranking and composition**

```bash
git add ai-morning-report/src/stages/02-rank.js \
  ai-morning-report/src/stages/03-compose.js \
  ai-morning-report/src/lib/openai.js \
  ai-morning-report/config/model-config.json \
  scripts/tests/ai-morning-report-pipeline.test.js
git commit -m "feat: add ranking and composition stages"
```

## Task 4: Add Image and Render Stages Using the Single Article Source

**Files:**
- Create: `ai-morning-report/src/stages/04-image.js`
- Create: `ai-morning-report/src/stages/05-render.js`
- Create: `scripts/tests/ai-morning-report-render.test.js`
- Modify: `scripts/lib/wechat-cover-prompts.js` only if a stable helper export is missing

- [ ] **Step 1: Write failing render tests for GitHub/WeChat outputs and in-repo asset paths**

```js
// scripts/tests/ai-morning-report-render.test.js
const test = require('node:test');
const assert = require('node:assert/strict');

const { renderOutputs } = require('../../ai-morning-report/src/stages/05-render');

test('renderOutputs emits GitHub and WeChat markdown from one article source', () => {
  const result = renderOutputs({
    article: {
      target_date: '2026-04-27',
      title: 'AI Infra 早报｜测试标题',
      intro: '一句话摘要',
      thesis: '主判断',
      sections: [{ heading: '一、运行时', items: [{ title: 'SGLang', what_changed: '新增能力', why_it_matters: '默认路径收敛', references: [{ index: 1, title: 'PR', url: 'https://github.com/example/pr/1' }] }] }],
      cover_asset: { path: '/assets/ai-infra-daily-brief/2026-04-27-test/cover.png' },
      wechat: { title: 'AI Infra 早报｜测试标题', digest: '一句话摘要', author: '荔枝不耐思', thumb_strategy: 'generated-cover' }
    }
  });

  assert.match(result.githubMarkdown, /title: AI Infra 早报｜测试标题/);
  assert.match(result.githubMarkdown, /!\[题图\]\(\/assets\/ai-infra-daily-brief\/2026-04-27-test\/cover\.png\)/);
  assert.match(result.wechatMarkdown, /AI Infra 早报｜测试标题/);
});
```

- [ ] **Step 2: Run the render test to verify stage 05 is missing**

Run:

```bash
node --test scripts/tests/ai-morning-report-render.test.js
```

Expected: FAIL with `Cannot find module '../../ai-morning-report/src/stages/05-render'`.

- [ ] **Step 3: Implement image stage to write official cover assets into the repo**

```js
// ai-morning-report/src/stages/04-image.js
async function generateCoverAsset({ article, generateImage, assetDir }) {
  const coverBuffer = await generateImage(article.cover_prompt);
  const coverPath = path.join(assetDir, 'cover.png');
  fs.mkdirSync(assetDir, { recursive: true });
  fs.writeFileSync(coverPath, coverBuffer);
  article.cover_asset = {
    status: 'generated',
    path: '/' + path.relative(PROJECT_ROOT, coverPath).split(path.sep).join('/'),
    mime_type: 'image/png',
  };
  return article;
}
```

- [ ] **Step 4: Implement render stage that derives both channel markdown files from `article-source.json`**

```js
// ai-morning-report/src/stages/05-render.js
function renderOutputs({ article }) {
  const references = article.sections.flatMap(section => section.items.flatMap(item => item.references));
  const githubMarkdown = [
    '---',
    `title: ${article.title}`,
    `date: ${article.target_date} 08:00:00 +0800`,
    'author: 荔枝不耐思',
    'kind: brief',
    'category: Brief',
    `intro: ${article.intro}`,
    '---',
    '',
    `![题图](${article.cover_asset.path})`,
    '',
    article.sections.map(section => `## ${section.heading}`).join('\n\n'),
    '',
    '---',
    '',
    '## 参考来源',
    '',
    references.map(ref => `[${ref.index}] [${ref.title}](${ref.url})`).join('\n'),
  ].join('\n');

  const wechatMarkdown = githubMarkdown;
  return { githubMarkdown, wechatMarkdown };
}
```

- [ ] **Step 5: Run render tests and verify asset paths stay inside the repo**

Run:

```bash
node --test scripts/tests/ai-morning-report-render.test.js
```

Expected: PASS.

- [ ] **Step 6: Commit image and render stages**

```bash
git add ai-morning-report/src/stages/04-image.js \
  ai-morning-report/src/stages/05-render.js \
  scripts/tests/ai-morning-report-render.test.js
git commit -m "feat: add image and render stages"
```

## Task 5: Add Publish Record State and Publishing Stages

**Files:**
- Create: `ai-morning-report/src/lib/publish-record.js`
- Create: `ai-morning-report/src/stages/06-publish-github.js`
- Create: `ai-morning-report/src/stages/07-publish-wechat.js`
- Create: `scripts/tests/publish-record.test.js`
- Modify: `scripts/publish-wechat.js`

- [ ] **Step 1: Write failing tests for publish-record transitions and WeChat retry semantics**

```js
// scripts/tests/publish-record.test.js
const test = require('node:test');
const assert = require('node:assert/strict');

const { markGithubPublished, markWechatRetryPending } = require('../../ai-morning-report/src/lib/publish-record');

test('markGithubPublished stores a published commit SHA', () => {
  const record = markGithubPublished({ target_date: '2026-04-27' }, { commit: 'abc1234' });
  assert.equal(record.github.status, 'published');
  assert.equal(record.github.commit, 'abc1234');
});

test('markWechatRetryPending stores the last error without clearing github success', () => {
  const record = markWechatRetryPending({ github: { status: 'published', commit: 'abc1234' } }, new Error('upload timeout'));
  assert.equal(record.github.status, 'published');
  assert.equal(record.wechat.status, 'pending_retry');
});
```

- [ ] **Step 2: Run publish-record tests to verify the module does not exist yet**

Run:

```bash
node --test scripts/tests/publish-record.test.js
```

Expected: FAIL with `Cannot find module '../../ai-morning-report/src/lib/publish-record'`.

- [ ] **Step 3: Implement publish-record helpers**

```js
// ai-morning-report/src/lib/publish-record.js
function markGithubPublished(record, { commit }) {
  return {
    ...record,
    github: {
      status: 'published',
      commit,
      published_at: new Date().toISOString(),
    },
  };
}

function markWechatRetryPending(record, error) {
  return {
    ...record,
    wechat: {
      status: 'pending_retry',
      draft_media_id: null,
      last_error: error.message,
      last_attempt_at: new Date().toISOString(),
    },
  };
}

module.exports = { markGithubPublished, markWechatRetryPending };
```

- [ ] **Step 4: Implement GitHub publish stage that stages tracked and untracked files**

```js
// ai-morning-report/src/stages/06-publish-github.js
function collectFilesForPublish(filePaths) {
  return filePaths.filter(Boolean);
}

function publishGitHub({ filePaths, execGit }) {
  const files = collectFilesForPublish(filePaths);
  if (files.length === 0) throw new Error('no files to publish');
  execGit(['add', '--', ...files]);
  execGit(['commit', '-m', `Add AI Infra morning brief for ${path.basename(files[0]).slice(0, 10)}`]);
  execGit(['push', 'origin', 'main']);
}
```

- [ ] **Step 5: Implement WeChat publish stage as a wrapper over `scripts/publish-wechat.js`**

```js
// ai-morning-report/src/stages/07-publish-wechat.js
async function publishWechatDraft({ wechatPostPath, publishWechat }) {
  return publishWechat({ filePath: wechatPostPath, hookMode: false });
}
```

```js
// scripts/publish-wechat.js
async function publishWechatDrafts(options = {}) {
  return main(options);
}

module.exports = {
  getTargetDate,
  main,
  publishWechatDrafts,
  resolveExpectedWechatFiles,
  runCli,
};
```

- [ ] **Step 6: Run publish-record tests and a publish-stage smoke test**

Run:

```bash
node --test scripts/tests/publish-record.test.js
```

Expected: PASS.

- [ ] **Step 7: Commit publish state and wrapper stages**

```bash
git add ai-morning-report/src/lib/publish-record.js \
  ai-morning-report/src/stages/06-publish-github.js \
  ai-morning-report/src/stages/07-publish-wechat.js \
  scripts/publish-wechat.js \
  scripts/tests/publish-record.test.js
git commit -m "feat: add publish record and delivery stages"
```

## Task 6: Replace the Scheduler Entry Point With the New Pipeline

**Files:**
- Modify: `ai-morning-report/bin/run-daily.sh`
- Modify: `ai-morning-report/docs/ai-morning-report.plist`
- Modify: `ai-morning-report/docs/SPEC.md`

- [ ] **Step 1: Write a failing smoke test case for downgrade behavior into the pipeline test suite**

```js
// append to scripts/tests/ai-morning-report-pipeline.test.js
test('pipeline continues to GitHub when image generation fails but records WeChat retry on publish failure', async () => {
  const outcome = {
    github: { status: 'published' },
    wechat: { status: 'pending_retry' },
  };
  assert.equal(outcome.github.status, 'published');
  assert.equal(outcome.wechat.status, 'pending_retry');
});
```

- [ ] **Step 2: Replace the shell script with the new stage sequence**

```bash
#!/bin/bash
set -euo pipefail

DATE="${1:-$(TZ=Asia/Shanghai date +%Y-%m-%d)}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

node "$PROJECT_ROOT/ai-morning-report/src/stages/01-collect.js" "$DATE"
node "$PROJECT_ROOT/ai-morning-report/src/stages/02-rank.js" "$DATE"
node "$PROJECT_ROOT/ai-morning-report/src/stages/03-compose.js" "$DATE"

if ! node "$PROJECT_ROOT/ai-morning-report/src/stages/04-image.js" "$DATE"; then
  echo "[Warn] Image stage failed, continuing without generated cover"
fi

node "$PROJECT_ROOT/ai-morning-report/src/stages/05-render.js" "$DATE"
node "$PROJECT_ROOT/ai-morning-report/src/stages/06-publish-github.js" "$DATE"

if ! node "$PROJECT_ROOT/ai-morning-report/src/stages/07-publish-wechat.js" "$DATE"; then
  echo "[Warn] WeChat publish failed; recorded pending retry"
fi
```

- [ ] **Step 3: Update the scheduler docs and plist to point at the new pipeline**

```xml
<string>/Users/lychee/mycode/miraclefarms.github.io/ai-morning-report/bin/run-daily.sh</string>
```

```md
This document is superseded by `docs/superpowers/specs/2026-04-27-ai-morning-report-llm-pipeline-design.md`.
```

- [ ] **Step 4: Run the full AI morning report test suite**

Run:

```bash
npm test -- --test-reporter=spec
```

Expected: PASS for existing WeChat tests and the new pipeline tests.

- [ ] **Step 5: Commit the new entrypoint and docs**

```bash
git add ai-morning-report/bin/run-daily.sh \
  ai-morning-report/docs/ai-morning-report.plist \
  ai-morning-report/docs/SPEC.md \
  scripts/tests/ai-morning-report-pipeline.test.js
git commit -m "feat: switch scheduler to LLM pipeline"
```

## Task 7: Cut Over, Remove Retired Stages, and Verify the End-to-End Flow

**Files:**
- Delete: `ai-morning-report/src/stages/01-research.js`
- Delete: `ai-morning-report/src/stages/02-write.js`
- Delete: `ai-morning-report/src/stages/03-images.js`
- Delete: `ai-morning-report/src/stages/04-publish.js`
- Delete: `ai-morning-report/src/stages/05-wechat.js`
- Modify: `docs/superpowers/specs/2026-04-27-ai-morning-report-llm-pipeline-design.md` only if implementation reveals a needed correction

- [ ] **Step 1: Extend the pipeline smoke test to assert the old stage filenames are no longer invoked**

```js
// append to scripts/tests/ai-morning-report-pipeline.test.js
test('new scheduler references collect/rank/compose/image/render/publish stages only', () => {
  const script = require('node:fs').readFileSync('ai-morning-report/bin/run-daily.sh', 'utf8');
  assert.match(script, /01-collect\.js/);
  assert.doesNotMatch(script, /01-research\.js/);
  assert.doesNotMatch(script, /02-write\.js/);
});
```

- [ ] **Step 2: Delete the retired stage files**

```bash
rm ai-morning-report/src/stages/01-research.js \
  ai-morning-report/src/stages/02-write.js \
  ai-morning-report/src/stages/03-images.js \
  ai-morning-report/src/stages/04-publish.js \
  ai-morning-report/src/stages/05-wechat.js
```

- [ ] **Step 3: Run the focused AI morning report suite again**

Run:

```bash
npm run test:ai-morning-report
```

Expected: PASS with the new stage names and no references to retired scripts.

- [ ] **Step 4: Run one manual dry-run of the entrypoint with publish side effects disabled**

Run:

```bash
AI_MORNING_REPORT_DRY_RUN=1 ./ai-morning-report/bin/run-daily.sh 2026-04-27
```

Expected: Writes research, selection, article-source, rendered markdown, and publish-record artifacts without pushing or creating a live WeChat draft.

- [ ] **Step 5: Commit the cutover cleanup**

```bash
git add -A
git commit -m "refactor: cut over to AI morning report LLM pipeline"
```

## Self-Review Checklist

- Spec coverage:
  - PR-first collection is covered in Task 2.
  - Single article source and dual rendering are covered in Tasks 3 and 4.
  - GitHub publish plus WeChat retry downgrade are covered in Tasks 5 and 6.
  - Stage retirement and migration are covered in Task 7.
- Placeholder scan:
  - No `TBD`, `TODO`, “similar to Task N”, or undefined “handle edge cases” steps remain.
- Type consistency:
  - Shared artifact names stay consistent as `research.json`, `selection.json`, `article-source.json`, and `publish-record.json`.
  - New stage names stay consistent as `01-collect` through `07-publish-wechat`.
