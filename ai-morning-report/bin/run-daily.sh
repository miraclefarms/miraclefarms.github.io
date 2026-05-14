#!/bin/bash
# AI Morning Report — Daily orchestration script
# Called by launchd at 05:30 Asia/Shanghai
#
# Environment variables:
#   AI_CLI          claude|opencode|codex  (defaults to opencode if unset)
#   OPENROUTER_API_KEY                     (for cover image, optional)
#   WECHAT_APPID / WECHAT_APPSECRET        (required for WeChat push)
#   WECHAT_THUMB_MEDIA_ID                  (fallback cover if no generated image)
#   WECHAT_CONTENT_SOURCE_BASE_URL         (original-link base URL; defaults to GitHub Pages)
#   SKIP_WECHAT=1                          (skip WeChat stages)
#   ENABLE_X_PUSH=1                        (enable optional X rewrite + publishing; default off)
#   X_DRY_RUN=1                            (generate X thread but do not post)

set -uo pipefail

DATE=$(TZ="Asia/Shanghai" date +%Y-%m-%d)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"          # ai-morning-report/
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"     # repo root
LOG_DIR="/tmp/morning-report/logs"
LOG_FILE="${LOG_DIR}/${DATE}.log"

mkdir -p "${LOG_DIR}"

# Rotate: keep last 7 days
find "${LOG_DIR}" -name '*.log' -mtime +7 -delete 2>/dev/null || true

exec >> "${LOG_FILE}" 2>&1

# Load .env from project root
if [ -f "${PROJECT_ROOT}/.env" ]; then
  set -a; source "${PROJECT_ROOT}/.env"; set +a
fi
AI_CLI="${AI_CLI:-opencode}"
export AI_CLI
STAGES_DIR="${REPORT_DIR}/src/stages"
CONFIG_DIR="${REPORT_DIR}/config"
WORK_DIR="/tmp/morning-report/${DATE}"
WECHAT_OUT_DIR="${WORK_DIR}/wechat"
WECHAT_ARCHIVE_DIR="${PROJECT_ROOT}/docs/wechat"
WECHAT_ARCHIVE_ASSET_DIR="${WECHAT_ARCHIVE_DIR}/assets/${DATE}"
WECHAT_MD_PATH="${WECHAT_ARCHIVE_DIR}/${DATE}-ai-infra-daily-brief-wechat.md"

POST_FILE="${PROJECT_ROOT}/_posts/${DATE}-ai-infra-daily-brief.md"
REPO_SCOPE="${CONFIG_DIR}/repo-scope.json"

mkdir -p "${WORK_DIR}"

log() { echo "[$(TZ="Asia/Shanghai" date '+%H:%M:%S')] $*"; }

commit_wechat_archive() {
  local archive_file="${1:?}"
  local pathspecs=("${archive_file}")

  git -C "${PROJECT_ROOT}" add "${archive_file}"
  if [ -d "${WECHAT_ARCHIVE_ASSET_DIR}" ]; then
    git -C "${PROJECT_ROOT}" add "${WECHAT_ARCHIVE_ASSET_DIR}"
    pathspecs+=("${WECHAT_ARCHIVE_ASSET_DIR}")
  fi

  if git -C "${PROJECT_ROOT}" diff --cached --quiet -- "${pathspecs[@]}"; then
    log "No WeChat archive changes to commit."
    return 0
  fi

  git -C "${PROJECT_ROOT}" commit -m "Archive ${DATE} WeChat draft" -- "${pathspecs[@]}"
  git -C "${PROJECT_ROOT}" push
}

log_wechat_retry_command() {
  log "添加 IP 白名单后可手动重新执行："
  log "cd \"${PROJECT_ROOT}\" && node \"${STAGES_DIR}/07-wechat-push.js\" \"${WECHAT_MD_PATH}\" \"${COVER_PATH}\" && git -C \"${PROJECT_ROOT}\" add \"${WECHAT_MD_PATH}\" && if [ -d \"${WECHAT_ARCHIVE_ASSET_DIR}\" ]; then git -C \"${PROJECT_ROOT}\" add \"${WECHAT_ARCHIVE_ASSET_DIR}\"; fi && if ! git -C \"${PROJECT_ROOT}\" diff --cached --quiet; then git -C \"${PROJECT_ROOT}\" commit -m \"Archive ${DATE} WeChat draft\"; fi && git -C \"${PROJECT_ROOT}\" push"
}

FINAL_STATUS="SUCCESS"
TOTAL_STAGES=7
if [ "${ENABLE_X_PUSH:-0}" = "1" ]; then
  TOTAL_STAGES=8
fi

log "=== AI Morning Report ${DATE} ==="
log "AI_CLI: ${AI_CLI}"

# ── Stage 1: Fetch raw data ────────────────────────────────────────────────
log "[1/${TOTAL_STAGES}] Fetching repo data..."
if ! bash "${STAGES_DIR}/01-fetch.sh" \
    "${DATE}" \
    "${WORK_DIR}/raw-data.md" \
    "${REPO_SCOPE}"; then
  log "FAILED at stage 1 (fetch)"
  FINAL_STATUS="FAILED"
  log "=== ${FINAL_STATUS} ==="
  exit 1
fi

# ── Stage 2: AI analysis ───────────────────────────────────────────────────
log "[2/${TOTAL_STAGES}] Analyzing (AI call 1/3)..."
if ! node "${STAGES_DIR}/02-analyze.js" \
    "${DATE}" \
    "${WORK_DIR}/raw-data.md" \
    "${WORK_DIR}/material.md" \
    "${PROJECT_ROOT}"; then
  log "FAILED at stage 2 (analyze)"
  FINAL_STATUS="FAILED"
  log "=== ${FINAL_STATUS} ==="
  exit 1
fi

# ── Stage 3: AI writing ────────────────────────────────────────────────────
log "[3/${TOTAL_STAGES}] Writing brief (AI call 2/3)..."
if ! node "${STAGES_DIR}/03-write.js" \
    "${DATE}" \
    "${WORK_DIR}/material.md" \
    "${PROJECT_ROOT}"; then
  log "FAILED at stage 3 (write)"
  FINAL_STATUS="FAILED"
  log "=== ${FINAL_STATUS} ==="
  exit 1
fi

# ── Stage 4: Cover image (non-blocking) ───────────────────────────────────
log "[4/${TOTAL_STAGES}] Generating cover image (AI call 3/3)..."
COVER_PATH=""
node "${STAGES_DIR}/04-cover.js" \
  "${DATE}" \
  "${POST_FILE}" \
  "${WORK_DIR}/assets" \
  "${PROJECT_ROOT}" || true
for ext in png jpg webp; do
  if [ -f "${WORK_DIR}/assets/cover.${ext}" ]; then
    COVER_PATH="${WORK_DIR}/assets/cover.${ext}"
    break
  fi
done
[ -n "${COVER_PATH}" ] && log "Cover: ${COVER_PATH}" || log "No cover generated, proceeding without."

# ── Stage 5: Publish to GitHub.io ─────────────────────────────────────────
log "[5/${TOTAL_STAGES}] Publishing to GitHub.io..."
if ! bash "${STAGES_DIR}/05-publish.sh" "${DATE}" "${PROJECT_ROOT}"; then
  log "FAILED at stage 5 (publish)"
  FINAL_STATUS="FAILED"
  log "=== ${FINAL_STATUS} ==="
  exit 1
fi

# ── Stage 6: WeChat formatting ────────────────────────────────────────────
if [ "${SKIP_WECHAT:-0}" = "1" ]; then
  log "[6/${TOTAL_STAGES}] Skipping WeChat formatting (SKIP_WECHAT=1)"
  log "[7/${TOTAL_STAGES}] Skipping WeChat push"
else
  log "[6/${TOTAL_STAGES}] Formatting for WeChat..."
  if ! node "${STAGES_DIR}/06-wechat-format.js" \
      "${DATE}" \
      "${POST_FILE}" \
      "${WECHAT_OUT_DIR}" \
      "${PROJECT_ROOT}" \
      "${COVER_PATH}" \
      "${WECHAT_ARCHIVE_DIR}"; then
    log "FAILED at stage 6 (wechat-format)"
    FINAL_STATUS="FAILED"
    log "=== ${FINAL_STATUS} ==="
    exit 1
  fi

  # ── Stage 7: WeChat push ────────────────────────────────────────────────
  log "[7/${TOTAL_STAGES}] Pushing WeChat draft..."
  if ! node "${STAGES_DIR}/07-wechat-push.js" \
      "${WECHAT_MD_PATH}" \
      "${COVER_PATH}"; then
    log "FAILED at stage 7 (wechat-push)"
    log_wechat_retry_command
    FINAL_STATUS="FAILED"
    log "=== ${FINAL_STATUS} ==="
    exit 1
  fi

  log "[7/${TOTAL_STAGES}] Persisting WeChat archive..."
  if ! commit_wechat_archive "${WECHAT_MD_PATH}"; then
    log "FAILED at stage 7 (wechat-archive)"
    FINAL_STATUS="FAILED"
    log "=== ${FINAL_STATUS} ==="
    exit 1
  fi
fi

# ── Stage 8: X publishing (optional, non-blocking) ────────────────────────
if [ "${ENABLE_X_PUSH:-0}" = "1" ]; then
  log "[8/8] Publishing X thread..."
  if ! node "${STAGES_DIR}/08-x-push.js" \
      "${DATE}" \
      "${POST_FILE}" \
      "${PROJECT_ROOT}"; then
    log "WARNING: stage 8 failed (x-push), keeping GitHub.io/WeChat result"
  fi
else
  log "X publish disabled (ENABLE_X_PUSH is not 1)"
fi

log "=== ${FINAL_STATUS} ==="
