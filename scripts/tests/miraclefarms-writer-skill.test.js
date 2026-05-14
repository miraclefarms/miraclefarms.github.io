const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const projectRoot = path.resolve(__dirname, '..', '..');
const skillPath = path.join(projectRoot, '.codex', 'skills', 'miraclefarms-writer', 'SKILL.md');
const wechatFormatPath = path.join(
  projectRoot,
  '.codex',
  'skills',
  'miraclefarms-writer',
  'references',
  'wechat-format.md',
);

function read(filePath) {
  return fs.readFileSync(filePath, 'utf8');
}

test('miraclefarms-writer explicitly calls out essay-to-WeChat as a distinct long-form path', () => {
  const skill = read(skillPath);

  assert.match(
    skill,
    /essay.*微信公众号技术长文|微信公众号技术长文.*essay|essay 改写成微信公众号技术长文/u,
  );
  assert.match(
    skill,
    /与 brief .*复用.*路径.*但.*配图.*配色.*单独/u,
  );
});

test('wechat-format documents a dedicated essay template with classic blue styling', () => {
  const wechatFormat = read(wechatFormatPath);

  assert.match(wechatFormat, /## 文章结构模板（Essay 类型）/u);
  assert.match(wechatFormat, /经典蓝|classic blue/u);
  assert.match(wechatFormat, /wechat_variant:\s*essay-longform/u);
});

test('wechat-format requires essay WeChat versions to keep the same inline figures as the GitHub essay', () => {
  const wechatFormat = read(wechatFormatPath);

  assert.match(
    wechatFormat,
    /与 GitHub\.io 版 essay 使用同一组配图|保留与 essay 相同的正文配图|不光是题图/u,
  );
});

test('wechat-format requires brief WeChat bodies to avoid repeating the title', () => {
  const wechatFormat = read(wechatFormatPath);

  assert.match(wechatFormat, /Brief.*front matter.*title|title.*front matter.*Brief/u);
  assert.match(wechatFormat, /正文 body 禁止重复标题|body 禁止重复标题/u);
  assert.doesNotMatch(wechatFormat, /```markdown\n# 今日焦点/u);
});

test('wechat-format strengthens the brief lead with current-topic tension', () => {
  const wechatFormat = read(wechatFormatPath);

  assert.match(wechatFormat, /导语.*热点|热点.*导语/u);
  assert.match(wechatFormat, /DeepSeek|GPT-OSS|Blackwell|新模型|新硬件/u);
  assert.match(wechatFormat, /为什么今天值得读|为什么读者现在需要关心/u);
});

test('wechat-format references the versioned cover prompt template file and the current locked default template', () => {
  const wechatFormat = read(wechatFormatPath);

  assert.match(wechatFormat, /scripts\/config\/wechat-cover-prompt-templates\.json/u);
  assert.match(wechatFormat, /daily-morning-paper-v1/u);
  assert.match(wechatFormat, /当前默认锁定这一套模板|当前先锁定这一个模板/u);
});
