const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const assert = require('node:assert/strict');

const { buildPipelinePaths } = require('../../ai-morning-report/src/lib/pipeline-paths');
const { runImageStage } = require('../../ai-morning-report/src/stages/04-image');
const {
  renderOutputs,
  runRenderStage,
} = require('../../ai-morning-report/src/stages/05-render');

function createArticleSource(overrides = {}) {
  return {
    target_date: '2026-04-27',
    title: 'AI Infra 早报｜测试标题',
    intro: '一句话摘要',
    thesis: '默认执行路径正在吸收过去只能靠旁路处理的复杂能力。',
    cover_prompt: 'An editorial cover showing the default execution path absorbing advanced runtime capabilities.',
    sections: [
      {
        heading: '一、运行时默认路径',
        summary: '这组变化共同说明，复杂能力正在进入默认执行路径。',
        items: [
          {
            title: 'SGLang runtime path',
            analysis: '默认路径吸收了新的调度和 fallback 能力。复杂场景因此不再依赖旁路实现，而是更像主链路的一部分。',
            references: [
              { index: 1, title: 'SGLang PR', url: 'https://github.com/example/pr/1' },
            ],
          },
        ],
      },
      {
        heading: '二、缓存与稳定性',
        summary: '另一条主线是性能和稳定性能力继续向生产默认路径收敛。',
        items: [
          {
            title: 'vLLM kv cache eviction',
            analysis: '缓存淘汰策略开始对 hot path 做更细粒度控制。这让性能优化更容易进入默认配置，而不是停留在特定调优场景里。',
            references: [
              { index: 2, title: 'vLLM PR', url: 'https://github.com/example/pr/2' },
            ],
          },
        ],
      },
    ],
    wechat: {
      title: 'AI Infra 早报｜测试标题',
      digest: '默认路径开始吸收复杂场景。',
      author: '荔枝不耐思',
      thumb_strategy: 'generated-cover',
    },
    ...overrides,
  };
}

test('runImageStage writes cover assets under repo-backed assetDir and backfills article-source', async () => {
  const projectRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'ai-morning-report-render-'));
  const article = createArticleSource();
  const paths = buildPipelinePaths(projectRoot, article.target_date);

  fs.mkdirSync(paths.workDir, { recursive: true });
  fs.writeFileSync(paths.articleSourceJsonPath, `${JSON.stringify(article, null, 2)}\n`, 'utf8');

  const result = await runImageStage({
    targetDate: article.target_date,
    projectRoot,
    generateImage: async (prompt) => {
      assert.equal(prompt, article.cover_prompt);
      return Buffer.from('fake-cover-binary');
    },
  });

  const generatedCoverPath = path.join(paths.assetDir, 'cover.png');
  const savedArticle = JSON.parse(fs.readFileSync(paths.articleSourceJsonPath, 'utf8'));

  assert.equal(result.outputPath, paths.articleSourceJsonPath);
  assert.equal(result.coverPath, generatedCoverPath);
  assert.equal(savedArticle.cover_asset.status, 'generated');
  assert.equal(savedArticle.cover_asset.path, '/assets/ai-infra-daily-brief/2026-04-27-ai-infra-daily-brief/cover.png');
  assert.equal(savedArticle.cover_asset.mime_type, 'image/png');
  assert.ok(fs.existsSync(generatedCoverPath));
  assert.ok(savedArticle.cover_asset.path.startsWith('/assets/'));
  assert.ok(!savedArticle.cover_asset.path.startsWith('/tmp/'));
});

test('renderOutputs emits GitHub and WeChat markdown from one article source', () => {
  const article = createArticleSource({
    cover_asset: {
      status: 'generated',
      path: '/assets/ai-infra-daily-brief/2026-04-27-ai-infra-daily-brief/cover.png',
      mime_type: 'image/png',
    },
  });

  const result = renderOutputs({ article });

  assert.match(result.githubMarkdown, /title: AI Infra 早报｜测试标题/);
  assert.match(result.githubMarkdown, /!\[题图\]\(\/assets\/ai-infra-daily-brief\/2026-04-27-ai-infra-daily-brief\/cover\.png\)/);
  assert.match(result.githubMarkdown, /## 参考来源/);
  assert.match(result.githubMarkdown, /\[\[1\]\]\(https:\/\/github\.com\/example\/pr\/1\)/);
  assert.match(result.githubMarkdown, /这组变化共同说明，复杂能力正在进入默认执行路径。/);
  assert.match(result.githubMarkdown, /\*\*SGLang runtime path\*\*\[\[1\]\]\(https:\/\/github\.com\/example\/pr\/1\)：默认路径吸收了新的调度和 fallback 能力。/);
  assert.match(result.wechatMarkdown, /^# AI Infra 早报｜测试标题/m);
  assert.match(result.wechatMarkdown, /\*\*📅 2026-04-27\*\*/);
  assert.match(result.wechatMarkdown, /!\[题图\]\(\/assets\/ai-infra-daily-brief\/2026-04-27-ai-infra-daily-brief\/cover\.png\)/);
  assert.match(result.wechatMarkdown, /> 默认路径开始吸收复杂场景。/);
  assert.match(result.wechatMarkdown, /这组变化共同说明，复杂能力正在进入默认执行路径。/);
  assert.match(result.wechatMarkdown, /\*\*SGLang runtime path\*\*\[1\]：默认路径吸收了新的调度和 fallback 能力。/);
  assert.match(result.wechatMarkdown, /## 参考/);
  assert.doesNotMatch(result.wechatMarkdown, /## 参考来源/);
  assert.match(result.wechatMarkdown, /\[1\] SGLang PR：https:\/\/github\.com\/example\/pr\/1/);
  assert.doesNotMatch(result.wechatMarkdown, /\[1\]\(https:\/\/github\.com\/example\/pr\/1\)/);
});

test('runRenderStage writes GitHub and WeChat markdown files from the article-source artifact', async () => {
  const projectRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'ai-morning-report-render-'));
  const article = createArticleSource({
    cover_asset: {
      status: 'generated',
      path: '/assets/ai-infra-daily-brief/2026-04-27-ai-infra-daily-brief/cover.png',
      mime_type: 'image/png',
    },
  });
  const paths = buildPipelinePaths(projectRoot, article.target_date);

  fs.mkdirSync(paths.workDir, { recursive: true });
  fs.writeFileSync(paths.articleSourceJsonPath, `${JSON.stringify(article, null, 2)}\n`, 'utf8');

  const result = await runRenderStage({
    targetDate: article.target_date,
    projectRoot,
  });

  assert.equal(result.githubPostPath, paths.githubPostPath);
  assert.equal(result.wechatPostPath, paths.wechatPostPath);
  assert.ok(fs.existsSync(paths.githubPostPath));
  assert.ok(fs.existsSync(paths.wechatPostPath));
  assert.match(fs.readFileSync(paths.githubPostPath, 'utf8'), /AI Infra 早报｜测试标题/);
  assert.match(fs.readFileSync(paths.wechatPostPath, 'utf8'), /\*\*📅 2026-04-27\*\*/);
});
