const axios = require('axios');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const modelConfig = require('../../config/model-config.json');
const { loadWechatConfig } = require('../../../scripts/lib/wechat-config');

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

function getTimeoutMs(config = modelConfig) {
  if (process.env.AI_MODEL_TIMEOUT_MS) {
    return Number(process.env.AI_MODEL_TIMEOUT_MS);
  }

  return Number(config.timeout_ms || 120000);
}

function getProjectRoot() {
  return path.resolve(__dirname, '../../..');
}

function isOpenRouterModel(model) {
  return typeof model === 'string' && model.startsWith('openrouter/');
}

function getOpenRouterModelName(model) {
  return String(model || '').replace(/^openrouter\//, '');
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

function runOpencode({ prompt, skill, workdir, model, timeoutMs = getTimeoutMs(modelConfig) }) {
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

    const timeoutHandle = Number.isFinite(timeoutMs) && timeoutMs > 0
      ? setTimeout(() => {
        child.kill('SIGTERM');
        reject(new Error(`opencode timed out after ${timeoutMs}ms`));
      }, timeoutMs)
      : null;

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    child.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    child.on('close', (code) => {
      if (timeoutHandle) {
        clearTimeout(timeoutHandle);
      }
      if (code === 0) {
        resolve({ stdout, stderr });
      } else {
        reject(new Error(`opencode exited with code ${code}\nstdout: ${stdout}\nstderr: ${stderr}`));
      }
    });

    child.on('error', (err) => {
      if (timeoutHandle) {
        clearTimeout(timeoutHandle);
      }
      reject(err);
    });
  });
}

async function runOpenRouterModel({
  prompt,
  model,
  timeoutMs = getTimeoutMs(modelConfig),
  projectRoot = getProjectRoot(),
}) {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    throw new Error('OPENROUTER_API_KEY not found');
  }

  const config = loadWechatConfig(projectRoot);
  const response = await axios.post(
    `${config.openrouterApiBaseUrl}/chat/completions`,
    {
      model: getOpenRouterModelName(model),
      messages: [
        {
          role: 'system',
          content: 'Return only valid JSON that matches the requested schema. Do not include markdown fences or extra commentary.',
        },
        {
          role: 'user',
          content: prompt,
        },
      ],
      temperature: 0.2,
      response_format: {
        type: 'json_object',
      },
    },
    {
      headers: {
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
        'HTTP-Referer': config.openrouterHttpReferer,
        'X-Title': config.openrouterXTitle,
      },
      timeout: timeoutMs,
    },
  );

  const message = response.data
    && Array.isArray(response.data.choices)
    && response.data.choices[0]
    && response.data.choices[0].message
    ? response.data.choices[0].message
    : null;

  if (!message) {
    throw new Error('OpenRouter did not return a message payload');
  }

  if (typeof message.content === 'string') {
    return { stdout: message.content, stderr: '' };
  }

  if (Array.isArray(message.content)) {
    const textContent = message.content
      .filter(part => part && part.type === 'text' && part.text)
      .map(part => part.text)
      .join('\n');

    if (textContent) {
      return { stdout: textContent, stderr: '' };
    }
  }

  throw new Error('OpenRouter returned no text content');
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
  runModel,
} = {}) {
  const resolvedModel = model || getModelForRole(role, config);
  const modelRunner = runModel || (isOpenRouterModel(resolvedModel) ? runOpenRouterModel : runOpencode);
  const response = await modelRunner({
    prompt,
    skill,
    workdir,
    role,
    model: resolvedModel,
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
  runOpenRouterModel,
  getModel,
  getModelForRole,
  getTimeoutMs,
  getOpenRouterModelName,
  invokeJsonModel,
  isOpenRouterModel,
  parseJsonResponse,
  getProjectRoot,
  getSkillPath,
};
