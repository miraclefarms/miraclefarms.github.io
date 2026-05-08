const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const projectRoot = path.resolve(__dirname, '..', '..');

test('daily pipeline keeps X rewrite and publishing behind an explicit opt-in flag', () => {
  const runDaily = fs.readFileSync(path.join(projectRoot, 'ai-morning-report', 'bin', 'run-daily.sh'), 'utf8');

  assert.match(runDaily, /\$\{ENABLE_X_PUSH:-0\}/);
  assert.doesNotMatch(runDaily, /SKIP_X/);
  assert.match(runDaily, /node "\$\{STAGES_DIR\}\/08-x-push\.js"/);
});

test('X setup docs and env example use ENABLE_X_PUSH instead of legacy SKIP_X', () => {
  const files = [
    '.env.example',
    'docs/x-publishing.md',
    'ai-morning-report/docs/SPEC.md',
    'scripts/x-oauth.js',
  ];

  for (const file of files) {
    const content = fs.readFileSync(path.join(projectRoot, file), 'utf8');
    assert.match(content, /ENABLE_X_PUSH/);
    assert.doesNotMatch(content, /SKIP_X/, `${file} should not mention legacy SKIP_X`);
  }
});
