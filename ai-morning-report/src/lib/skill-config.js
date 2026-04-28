const fs = require('node:fs');
const path = require('node:path');

const PROJECT_ROOT = path.resolve(__dirname, '../../..');
const AI_MORNING_REPORT_SKILL_PATH = path.join(
  PROJECT_ROOT,
  '.codex',
  'skills',
  'ai-morning-report',
  'SKILL.md',
);

const SKILL_CONFIG_START = '<!-- codex:skill-config:start -->';
const SKILL_CONFIG_END = '<!-- codex:skill-config:end -->';

function extractSkillConfig(markdown) {
  const source = String(markdown || '');
  const startIndex = source.indexOf(SKILL_CONFIG_START);
  const endIndex = source.indexOf(SKILL_CONFIG_END);

  if (startIndex === -1 || endIndex === -1 || endIndex <= startIndex) {
    throw new Error('skill config markers not found');
  }

  const jsonString = source
    .slice(startIndex + SKILL_CONFIG_START.length, endIndex)
    .trim();

  if (!jsonString) {
    throw new Error('skill config block is empty');
  }

  try {
    return JSON.parse(jsonString);
  } catch (error) {
    throw new Error(`failed to parse skill config JSON: ${error.message}`);
  }
}

function loadSkillConfig(skillPath) {
  if (!skillPath) {
    throw new Error('skillPath is required');
  }

  return extractSkillConfig(fs.readFileSync(skillPath, 'utf8'));
}

function loadAiMorningReportSkillConfig(skillPath = AI_MORNING_REPORT_SKILL_PATH) {
  return loadSkillConfig(skillPath);
}

module.exports = {
  AI_MORNING_REPORT_SKILL_PATH,
  SKILL_CONFIG_END,
  SKILL_CONFIG_START,
  extractSkillConfig,
  loadAiMorningReportSkillConfig,
  loadSkillConfig,
};
