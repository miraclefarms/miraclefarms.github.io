#!/bin/bash

set -euo pipefail

DATE="${1:-$(TZ=Asia/Shanghai date +%Y-%m-%d)}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGES_DIR="${PROJECT_ROOT}/ai-morning-report/src/stages"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting AI Morning Report for ${DATE}"

echo "[Stage 1] Collecting research candidates..."
node "${STAGES_DIR}/01-collect.js" "${DATE}"

echo "[Stage 2] Ranking candidates..."
node "${STAGES_DIR}/02-rank.js" "${DATE}"

echo "[Stage 3] Composing article source..."
node "${STAGES_DIR}/03-compose.js" "${DATE}"

echo "[Stage 4] Generating cover asset..."
if ! node "${STAGES_DIR}/04-image.js" "${DATE}"; then
  echo "[Warn] Image stage failed; continuing without generated cover asset"
fi

echo "[Stage 5] Rendering GitHub and WeChat markdown..."
node "${STAGES_DIR}/05-render.js" "${DATE}"

echo "[Stage 6] Publishing to GitHub..."
node "${STAGES_DIR}/06-publish-github.js" "${DATE}"

echo "[Stage 7] Publishing WeChat draft..."
if ! node "${STAGES_DIR}/07-publish-wechat.js" "${DATE}"; then
  echo "[Warn] WeChat publish failed; publish-record.json marked pending_retry"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] AI Morning Report completed for ${DATE}"
