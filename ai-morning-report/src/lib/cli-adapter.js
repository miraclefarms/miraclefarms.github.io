/**
 * Multi-CLI adapter for AI calls.
 *
 * Supports claude / opencode / codex, selected by AI_CLI env var.
 * Falls back in order: claude → opencode → codex.
 * Skills are injected as plain text prepended to the user prompt.
 */

const { spawn, execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const SUPPORTED = ['claude', 'opencode', 'codex'];

function detectCLI() {
  const preferred = process.env.AI_CLI;
  if (preferred) {
    if (!SUPPORTED.includes(preferred)) {
      throw new Error(`AI_CLI=${preferred} is not supported. Use: ${SUPPORTED.join(', ')}`);
    }
    return preferred;
  }
  for (const cli of SUPPORTED) {
    try {
      execSync(`which ${cli}`, { stdio: 'ignore' });
      return cli;
    } catch {
      // not found, try next
    }
  }
  throw new Error(`No supported AI CLI found. Install one of: ${SUPPORTED.join(', ')}`);
}

function buildPrompt(skillContent, userPrompt) {
  if (!skillContent) return userPrompt;
  return `${skillContent.trim()}\n\n---\n\n${userPrompt.trim()}`;
}

function spawnCLI(cli, fullPrompt, model) {
  return new Promise((resolve, reject) => {
    let args;

    if (cli === 'claude') {
      // Claude Code non-interactive mode
      args = ['--print', '--output-format', 'text'];
      if (model) args.push('--model', model);
    } else if (cli === 'opencode') {
      args = ['run'];
      if (model) args.push('--model', model);
    } else if (cli === 'codex') {
      args = ['--quiet'];
      if (model) args.push('--model', model);
    }

    const child = spawn(cli, args, {
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env },
    });

    child.stdin.write(fullPrompt);
    child.stdin.end();

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (d) => { stdout += d.toString(); });
    child.stderr.on('data', (d) => { stderr += d.toString(); });

    child.on('close', (code) => {
      if (code === 0) {
        resolve(stdout.trim());
      } else {
        reject(new Error(`${cli} exited ${code}\nstderr: ${stderr.slice(0, 500)}`));
      }
    });

    child.on('error', reject);
  });
}

/**
 * @param {object} opts
 * @param {string} opts.prompt        - User-facing prompt (task description + context)
 * @param {string} [opts.skillPath]   - Absolute path to a SKILL.md to inject as system context
 * @param {string} [opts.model]       - Model override (optional)
 * @param {string} [opts.cli]         - CLI override; defaults to AI_CLI env or auto-detection
 * @returns {Promise<string>}         - stdout from the AI CLI
 */
async function runAI({ prompt, skillPath, model, cli: cliOverride } = {}) {
  const cli = cliOverride || detectCLI();

  let skillContent = '';
  if (skillPath) {
    if (!fs.existsSync(skillPath)) {
      throw new Error(`Skill not found: ${skillPath}`);
    }
    skillContent = fs.readFileSync(skillPath, 'utf8');
  }

  const fullPrompt = buildPrompt(skillContent, prompt);
  return spawnCLI(cli, fullPrompt, model);
}

function findSkill(skillName, projectRoot) {
  const candidates = [
    path.join(projectRoot, '.codex', 'skills', skillName, 'SKILL.md'),
    path.join(projectRoot, '.claude', 'skills', skillName, 'SKILL.md'),
  ];
  for (const p of candidates) {
    if (fs.existsSync(p)) return p;
  }
  throw new Error(`Skill "${skillName}" not found in .codex/skills or .claude/skills`);
}

module.exports = { runAI, findSkill, detectCLI };
