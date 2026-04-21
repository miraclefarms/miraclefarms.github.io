const path = require('path');
const { runOpencode, getModel, getProjectRoot } = require('../lib/openai');

async function runWrite(date, researchOutput, outputType) {
  const model = getModel();

  const prompt = `你是 MiracleFarms 技术博主。调用 miraclefarms-writer skill，执行以下任务：

1. 报道日期：${date}
2. 输出类型：${outputType}（GitHub.io 早报 + 微信公众号草稿）
3. 素材来源：${researchOutput}

请按 miraclefarms-writer skill 的流程：
1. 判断文章类型（${outputType} → brief）
2. 调研（读取 research output 中的素材）
3. 应用写作原则
4. 按格式规则输出（读取 references/brief-format.md）
5. 如有图片，读取 references/image-handling.md 处理
6. 如需微信公众号版本，读取 references/wechat-format.md

输出要求：
- GitHub.io 版本保存到 _posts/YYYY-MM-DD-ai-infra-daily-brief.md
- 微信公众号版本保存到 docs/wechat/YYYY-MM-DD-ai-infra-daily-brief-wechat.md
- 不要主动 commit 或 push

完成后报告输出文件路径。`;

  const workdir = getProjectRoot();
  const result = await runOpencode({ prompt, skill: 'miraclefarms-writer', workdir, model });
  return result.stdout;
}

if (require.main === module) {
  const date = process.argv[2] || new Date().toISOString().slice(0, 10);
  const researchOutput = process.argv[3] || `/tmp/morning-report/${date}/research-output.md`;
  const outputType = process.argv[4] || 'brief';

  runWrite(date, researchOutput, outputType)
    .then((output) => console.log('Write completed:\n', output))
    .catch((err) => {
      console.error('Write failed:', err.message);
      process.exit(1);
    });
}

module.exports = { runWrite };