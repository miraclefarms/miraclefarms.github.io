const fs = require('node:fs');
const path = require('node:path');

const TEMPLATE_FILE = path.join('scripts', 'config', 'wechat-cover-prompt-templates.json');

function loadWechatCoverPromptTemplates(projectRoot) {
  const registryPath = path.join(projectRoot, TEMPLATE_FILE);
  return JSON.parse(fs.readFileSync(registryPath, 'utf8'));
}

function resolveWechatCoverPromptTemplate(frontMatter = {}, registry) {
  const templateId = frontMatter.wechat_cover_prompt_template || registry.defaultTemplateId;
  const template = registry.templates[templateId];

  if (!template) {
    throw new Error(`Unknown WeChat cover prompt template: ${templateId}`);
  }

  return template;
}

function resolveWechatCoverImageConfig(frontMatter = {}, registry) {
  const template = resolveWechatCoverPromptTemplate(frontMatter, registry);
  return {
    aspectRatio: template.imageConfig?.aspectRatio || '9:16',
  };
}

function stripMarkdown(markdown) {
  return markdown
    .replace(/^---[\s\S]*?---\n?/m, '')
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`[^`]*`/g, ' ')
    .replace(/!\[[^\]]*\]\([^)]+\)/g, ' ')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/^>\s?/gm, '')
    .replace(/^#{1,6}\s*/gm, '')
    .replace(/[*_~]/g, '')
    .replace(/\n+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function summarizeMainBodyForPrompt(bodyMarkdown) {
  const plain = stripMarkdown(bodyMarkdown);
  if (!plain) {
    return '';
  }

  return plain.slice(0, 1200);
}

function extractLegacyTitleImagePrompts(markdown) {
  const lines = markdown.split('\n');

  for (let i = 0; i < lines.length; i++) {
    const chineseMatch = lines[i].match(/^>\s*中文[:：]\s*(.+?)\s*$/);
    if (!chineseMatch) {
      continue;
    }

    let cursor = i + 1;
    while (cursor < lines.length && /^>\s*$/.test(lines[cursor])) {
      cursor++;
    }

    if (cursor >= lines.length) {
      return null;
    }

    const englishMatch = lines[cursor].match(/^>\s*English[:：]\s*(.+?)\s*$/i);
    if (!englishMatch) {
      return null;
    }

    return {
      chinese: chineseMatch[1].trim(),
      english: englishMatch[1].trim(),
      startLine: i,
      endLine: cursor,
    };
  }

  return null;
}

function buildWechatCoverPrompt({
  frontMatter = {},
  title,
  digest,
  bodyMarkdown,
  registry,
}) {
  const template = resolveWechatCoverPromptTemplate(frontMatter, registry);
  const mainBodySummary = summarizeMainBodyForPrompt(bodyMarkdown);
  const legacyPrompts = extractLegacyTitleImagePrompts(bodyMarkdown);
  const parts = [template.prompt];

  if (title) {
    parts.push(`Article title: ${title}`);
  }

  if (digest) {
    parts.push(`Article summary: ${digest}`);
  }

  if (mainBodySummary) {
    parts.push(`Main body summary: ${mainBodySummary}`);
  }

  if (legacyPrompts?.english) {
    parts.push(`Legacy article-specific visual note: ${legacyPrompts.english}`);
  }

  if (legacyPrompts?.chinese) {
    parts.push(`Legacy Chinese visual note: ${legacyPrompts.chinese}`);
  }

  return parts.join('\n');
}

function buildWechatCoverImageRequest({
  frontMatter = {},
  title,
  digest,
  bodyMarkdown,
  registry,
  model,
  imageSize,
}) {
  const imageConfig = resolveWechatCoverImageConfig(frontMatter, registry);
  return {
    model,
    messages: [
      {
        role: 'user',
        content: buildWechatCoverPrompt({
          frontMatter,
          title,
          digest,
          bodyMarkdown,
          registry,
        }),
      },
    ],
    modalities: ['image', 'text'],
    stream: false,
    image_config: {
      aspect_ratio: imageConfig.aspectRatio,
      image_size: imageSize,
    },
  };
}

function insertGeneratedTitleImageMarkdown(markdown, imageUrl) {
  const legacyPrompts = extractLegacyTitleImagePrompts(markdown);
  const lines = markdown.split('\n');

  if (legacyPrompts) {
    lines.splice(
      legacyPrompts.startLine,
      legacyPrompts.endLine - legacyPrompts.startLine + 1,
      `![题图](${imageUrl})`,
    );
    return lines.join('\n');
  }

  const dateLineIndex = lines.findIndex(line => /^\*\*📅\s+\d{4}-\d{2}-\d{2}\*\*\s*$/.test(line.trim()));
  if (dateLineIndex !== -1) {
    lines.splice(dateLineIndex + 2, 0, `![题图](${imageUrl})`, '');
    return lines.join('\n');
  }

  const firstHeadingIndex = lines.findIndex(line => /^#\s+/.test(line.trim()));
  if (firstHeadingIndex !== -1) {
    lines.splice(firstHeadingIndex + 2, 0, `![题图](${imageUrl})`, '');
    return lines.join('\n');
  }

  return `![题图](${imageUrl})\n\n${markdown}`;
}

module.exports = {
  buildWechatCoverImageRequest,
  buildWechatCoverPrompt,
  extractLegacyTitleImagePrompts,
  insertGeneratedTitleImageMarkdown,
  loadWechatCoverPromptTemplates,
  resolveWechatCoverImageConfig,
  resolveWechatCoverPromptTemplate,
};
