function isBlank(value) {
  return (
    value === undefined
    || value === null
    || value === ''
    || (Array.isArray(value) && value.length === 0)
  );
}

function requireField(value, fieldName) {
  if (isBlank(value)) {
    throw new Error(`missing required field: ${fieldName}`);
  }
}

function requireObject(value, fieldName) {
  requireField(value, fieldName);

  if (typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`expected object: ${fieldName}`);
  }
}

function validateResearchDocument(doc) {
  requireObject(doc, 'research');
  requireField(doc.target_date, 'target_date');
  requireField(doc.window, 'window');
  requireField(doc.repos, 'repos');
  return doc;
}

function validateSelectionDocument(doc) {
  requireObject(doc, 'selection');
  requireField(doc.target_date, 'target_date');
  requireField(doc.selected, 'selected');
  return doc;
}

function validateArticleSource(doc) {
  requireObject(doc, 'articleSource');
  requireField(doc.target_date, 'target_date');
  requireField(doc.title, 'title');
  requireField(doc.intro, 'intro');
  requireField(doc.thesis, 'thesis');
  requireField(doc.cover_prompt, 'cover_prompt');
  requireField(doc.sections, 'sections');
  requireObject(doc.wechat, 'wechat');
  requireField(doc.wechat.title, 'wechat.title');
  requireField(doc.wechat.digest, 'wechat.digest');
  return doc;
}

function validatePublishRecord(doc) {
  requireObject(doc, 'publishRecord');
  requireField(doc.target_date, 'target_date');
  requireObject(doc.github, 'github');
  requireField(doc.github.status, 'github.status');
  requireObject(doc.wechat, 'wechat');
  requireField(doc.wechat.status, 'wechat.status');
  return doc;
}

module.exports = {
  requireField,
  validateResearchDocument,
  validateSelectionDocument,
  validateArticleSource,
  validatePublishRecord,
};
