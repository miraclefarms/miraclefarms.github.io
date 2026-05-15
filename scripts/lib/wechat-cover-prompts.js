const fs = require('node:fs');
const path = require('node:path');

const TEMPLATE_FILE = path.join('scripts', 'config', 'wechat-cover-prompt-templates.json');

const PROMPT_METADATA_PATTERNS = [
  /9\s*:\s*16/giu,
  /16\s*:\s*9/giu,
  /竖版/giu,
  /横版/giu,
  /中文信息图/giu,
  /信息图/giu,
  /题图提示词/giu,
  /提示词/giu,
  /硬性约束/giu,
  /最终图片/giu,
];

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
  const plain = stripMarkdown(removeLegacyTitleImagePromptBlocks(bodyMarkdown));
  if (!plain) {
    return '';
  }

  return plain.slice(0, 1200);
}

function removeLegacyTitleImagePromptBlocks(markdown) {
  const legacyPrompts = extractLegacyTitleImagePrompts(markdown);
  if (!legacyPrompts) {
    return markdown;
  }

  const lines = markdown.split('\n');
  lines.splice(
    legacyPrompts.startLine,
    legacyPrompts.endLine - legacyPrompts.startLine + 1,
  );
  return lines.join('\n');
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

function templateHasContentSlots(templatePrompt = '') {
  return /\[题图提示词\]|\[当天日期\]/.test(templatePrompt);
}

function resolveCoverPromptDate({ date, frontMatter = {}, bodyMarkdown = '' } = {}) {
  const candidates = [
    date,
    frontMatter.date,
    bodyMarkdown.match(/\*\*📅\s*(\d{4}-\d{2}-\d{2})\*\*/)?.[1],
  ];

  for (const candidate of candidates) {
    if (!candidate) {
      continue;
    }

    const match = String(candidate).match(/\d{4}-\d{2}-\d{2}/);
    if (match) {
      return match[0];
    }
  }

  return '';
}

function buildCoverPromptText({ title, digest }) {
  return sanitizeCoverPromptText([title, digest]
    .filter(Boolean)
    .join('\n'));
}

function sanitizeCoverPromptText(input = '') {
  const lines = String(input)
    .replace(/\r\n/g, '\n')
    .split('\n')
    .map(line => line.trim())
    .filter(Boolean)
    .map((line) => {
      let next = line;
      for (const pattern of PROMPT_METADATA_PATTERNS) {
        next = next.replace(pattern, '');
      }
      return next
        .replace(/^[。；，、\s:：-]+/, '')
        .replace(/\s{2,}/g, ' ')
        .trim();
    })
    .filter(Boolean)
    .filter(line => !/^(生成|请|要求|禁止|不要|必须|允许|控制在|直接输出)/u.test(line));

  return lines.join('\n').trim();
}

function renderCoverPromptTemplate(templatePrompt, { coverPromptText = '', date = '' } = {}) {
  const dateText = date ? `\n${date}` : '';
  return templatePrompt
    .replace(/\[题图提示词\]/g, sanitizeCoverPromptText(coverPromptText))
    .replace(/\[当天日期\]/g, dateText)
    .trim();
}

function buildWechatCoverPrompt({
  frontMatter = {},
  title,
  digest,
  bodyMarkdown,
  registry,
  date,
}) {
  const template = resolveWechatCoverPromptTemplate(frontMatter, registry);
  const legacyPrompts = extractLegacyTitleImagePrompts(bodyMarkdown);
  const coverPromptText = buildCoverPromptText({ title, digest });
  const promptDate = resolveCoverPromptDate({ date, frontMatter, bodyMarkdown });

  if (templateHasContentSlots(template.prompt)) {
    return renderCoverPromptTemplate(template.prompt, {
      coverPromptText,
      date: promptDate,
    });
  }

  const parts = [
    template.prompt,
    '以下上下文仅用于理解文章主题和信息层级；最终画面文字必须使用简体中文，不要照搬上下文标签。',
  ];

  if (title) {
    parts.push(`文章标题：${title}`);
  }

  if (digest) {
    parts.push(`文章摘要：${digest}`);
  }

  if (legacyPrompts?.english) {
    parts.push(`旧版英文视觉备注（只提取场景含义，禁止照搬英文文字）：${legacyPrompts.english}`);
  }

  if (legacyPrompts?.chinese) {
    parts.push(`旧版中文视觉备注：${legacyPrompts.chinese}`);
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
  date,
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
          date,
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
  renderCoverPromptTemplate,
  resolveWechatCoverImageConfig,
  resolveWechatCoverPromptTemplate,
  sanitizeCoverPromptText,
};
