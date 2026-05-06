#!/bin/bash
# AI Morning Report — Daily orchestration script
# Called by launchd at 05:30 Asia/Shanghai
#
# Environment variables:
#   AI_CLI          claude|opencode|codex  (auto-detected if unset)
#   OPENROUTER_API_KEY                     (for cover image, optional)
#   WECHAT_APPID / WECHAT_APPSECRET        (required for WeChat push)
#   WECHAT_THUMB_MEDIA_ID                  (fallback cover if no generated image)
#   SKIP_WECHAT=1                          (skip WeChat stages)

set -euo pipefail

DATE=$(TZ="Asia/Shanghai" date +%Y-%m-%d)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"          # ai-morning-report/
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"     # repo root
STAGES_DIR="${REPORT_DIR}/src/stages"
CONFIG_DIR="${REPORT_DIR}/config"
WORK_DIR="/tmp/morning-report/${DATE}"
WECHAT_OUT_DIR="${PROJECT_ROOT}/docs/wechat"

POST_FILE="${PROJECT_ROOT}/_posts/${DATE}-ai-infra-daily-brief.md"
REPO_SCOPE="${CONFIG_DIR}/repo-scope.json"

mkdir -p "${WORK_DIR}"

log() { echo "[$(TZ="Asia/Shanghai" date '+%H:%M:%S')] $*"; }

log "=== AI Morning Report ${DATE} ==="
log "AI_CLI: ${AI_CLI:-auto}"

# ── Stage 1: Fetch raw data ────────────────────────────────────────────────
log "[1/7] Fetching repo data..."
bash "${STAGES_DIR}/01-fetch.sh" \
  "${DATE}" \
  "${WORK_DIR}/raw-data.md" \
  "${REPO_SCOPE}"

# ── Stage 2: AI analysis ───────────────────────────────────────────────────
log "[2/7] Analyzing (AI call 1/2)..."
node "${STAGES_DIR}/02-analyze.js" \
  "${DATE}" \
  "${WORK_DIR}/raw-data.md" \
  "${WORK_DIR}/material.md" \
  "${PROJECT_ROOT}"

# ── Stage 3: AI writing ────────────────────────────────────────────────────
log "[3/7] Writing brief (AI call 2/2)..."
node "${STAGES_DIR}/03-write.js" \
  "${DATE}" \
  "${WORK_DIR}/material.md" \
  "${PROJECT_ROOT}"

# ── Stage 4: Cover image (non-blocking) ───────────────────────────────────
log "[4/7] Generating cover image (optional)..."
COVER_PATH=""
node "${STAGES_DIR}/04-cover.js" \
  "${DATE}" \
  "${POST_FILE}" \
  "${WORK_DIR}/assets" || true
# Detect generated cover (any extension)
for ext in png jpg webp; do
  if [ -f "${WORK_DIR}/assets/cover.${ext}" ]; then
    COVER_PATH="${WORK_DIR}/assets/cover.${ext}"
    break
  fi
done
[ -n "${COVER_PATH}" ] && log "Cover: ${COVER_PATH}" || log "No cover generated, proceeding without."

# ── Stage 5: Publish to GitHub.io ─────────────────────────────────────────
log "[5/7] Publishing to GitHub.io..."
bash "${STAGES_DIR}/05-publish.sh" "${DATE}" "${PROJECT_ROOT}"

# ── Stage 6: WeChat formatting ────────────────────────────────────────────
if [ "${SKIP_WECHAT:-0}" = "1" ]; then
  log "[6/7] Skipping WeChat (SKIP_WECHAT=1)"
  log "[7/7] Skipping WeChat push"
  log "=== Done (GitHub.io only) ==="
  exit 0
fi

log "[6/7] Formatting for WeChat..."
node "${STAGES_DIR}/06-wechat-format.js" \
  "${DATE}" \
  "${POST_FILE}" \
  "${WECHAT_OUT_DIR}" \
  "${PROJECT_ROOT}" \
  "${COVER_PATH}"

# ── Stage 7: WeChat push ──────────────────────────────────────────────────
log "[7/7] Pushing WeChat draft..."
node "${STAGES_DIR}/07-wechat-push.js" \
  "${WECHAT_OUT_DIR}/${DATE}-ai-infra-daily-brief-wechat.md" \
  "${COVER_PATH}"

log "=== Done ==="
