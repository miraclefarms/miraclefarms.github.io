const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const modelConfig = require('../../config/model-config.json');

function getRoleEnvVar(role) {
  if (!role) {
    return null;
  }

  return `AI_MODEL_${String(role).toUpperCase().replace(/[^A-Z0-9]+/g, '_')}`;
}

function getModelForRole(role, config = modelConfig) {
  if (process.env.AI_MODEL) {
    return process.env.AI_MODEL;
  }

  const roleEnvVar = getRoleEnvVar(role);
  if (roleEnvVar && process.env[roleEnvVar]) {
    return process.env[roleEnvVar];
  }

  if (config.roles && role && config.roles[role]) {
    return config.roles[role];
  }

  return config.default;
}

function getModel(role) {
  return getModelForRole(role, modelConfig);
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

function extractJsonString(raw) {
  const trimmed = String(raw || '').trim();

  if (!trimmed) {
    throw new Error('model returned empty output');
  }

  const fencedMatch = trimmed.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  if (fencedMatch) {
    return fencedMatch[1].trim();
  }

  return trimmed;
}

function parseJsonResponse(raw) {
  const jsonString = extractJsonString(raw);

  try {
    return JSON.parse(jsonString);
  } catch (error) {
    throw new Error(`failed to parse model JSON response: ${error.message}`);
  }
}

async function invokeJsonModel({
  role,
  prompt,
  workdir,
  skill,
  model,
  config = modelConfig,
  runModel = runOpencode,
} = {}) {
  const response = await runModel({
    prompt,
    skill,
    workdir,
    role,
    model: model || getModelForRole(role, config),
  });

  if (response && typeof response === 'object' && !('stdout' in response) && !Buffer.isBuffer(response)) {
    return response;
  }

  const rawOutput = response && typeof response === 'object' && 'stdout' in response
    ? response.stdout
    : response;

  return parseJsonResponse(rawOutput);
}

module.exports = {
  runOpencode,
  getModel,
  getModelForRole,
  invokeJsonModel,
  parseJsonResponse,
  getProjectRoot,
  getSkillPath,
};
