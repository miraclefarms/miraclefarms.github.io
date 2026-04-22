const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const modelConfig = require('../../config/model-config.json');

function getModel() {
  return process.env.AI_MODEL || modelConfig.default;
}

function getProjectRoot() {
  return path.resolve(__dirname, '../../..');
}

function getSkillPath(skillName) {
  const skillRoots = [
    path.join(getProjectRoot(), '.codex/skills', skillName),
    path.join(getProjectRoot(), '.claude/skills', skillName),
  ];
  for (const root of skillRoots) {
    if (fs.existsSync(root)) return root;
  }
  return null;
}

function runOpencode({ prompt, skill, workdir, model }) {
  return new Promise((resolve, reject) => {
    const args = ['run'];

    if (model) {
      args.push('--model', model);
    }

    if (workdir) {
      args.push('--dir', workdir);
    }

    args.push(prompt);

    const child = spawn('opencode', args, {
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env },
      cwd: workdir || undefined,
    });

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    child.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    child.on('close', (code) => {
      if (code === 0) {
        resolve({ stdout, stderr });
      } else {
        reject(new Error(`opencode exited with code ${code}\nstdout: ${stdout}\nstderr: ${stderr}`));
      }
    });

    child.on('error', (err) => {
      reject(err);
    });
  });
}

module.exports = { runOpencode, getModel, getProjectRoot, getSkillPath };