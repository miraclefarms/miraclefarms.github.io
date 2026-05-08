const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const projectRoot = path.resolve(__dirname, '..', '..');
const skillPath = path.join(projectRoot, '.agents', 'skills', 'x-formatter', 'SKILL.md');

test('x-formatter skill documents English-only condensed rewriting from GitHub.io posts', () => {
  assert.equal(fs.existsSync(skillPath), true, 'x-formatter skill should exist');

  const skill = fs.readFileSync(skillPath, 'utf8');
  assert.match(skill, /English-only/i);
  assert.match(skill, /GitHub\.io/i);
  assert.match(skill, /condens/i);
  assert.match(skill, /JSON array/i);
  assert.match(skill, /Do not translate sentence by sentence/i);
});

test('x-push builds a model prompt around the x-formatter skill contract', () => {
  const {
    buildXThreadPrompt,
    resolveXFormatterSkill,
  } = require('../../ai-morning-report/src/stages/08-x-push');

  const resolvedSkillPath = resolveXFormatterSkill(projectRoot);
  assert.equal(resolvedSkillPath, skillPath);

  const prompt = buildXThreadPrompt({
    date: '2026-05-08',
    postContent: '---\ntitle: 测试\n---\n中文正文',
    postUrl: 'https://miraclefarms.github.io/notes/2026/05/08/test/',
  });

  assert.match(prompt, /https:\/\/miraclefarms\.github\.io\/notes\/2026\/05\/08\/test\//);
  assert.match(prompt, /JSON array/);
  assert.match(prompt, /English-only/i);
  assert.match(prompt, /do not output Chinese/i);
  assert.match(prompt, /condense/i);
});
