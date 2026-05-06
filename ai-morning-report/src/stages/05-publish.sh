#!/bin/bash
# Stage 5: Publish GitHub.io post via git push.
# Usage: ./05-publish.sh <date> <project-root>
# Gate: fails if push fails (blocks WeChat stage).

set -euo pipefail

DATE="${1:?Usage: $0 <date> <project-root>}"
PROJECT_ROOT="${2:?}"

POST_FILE="${PROJECT_ROOT}/_posts/${DATE}-ai-infra-daily-brief.md"
ASSETS_DIR="${PROJECT_ROOT}/assets/${DATE}-ai-infra-daily-brief"

if [ ! -f "$POST_FILE" ]; then
  echo "[publish] ERROR: Post file not found: $POST_FILE" >&2
  exit 1
fi

cd "$PROJECT_ROOT"

FILES_TO_ADD=("$POST_FILE")
if [ -d "$ASSETS_DIR" ]; then
  FILES_TO_ADD+=("$ASSETS_DIR")
fi

git add "${FILES_TO_ADD[@]}"

if git diff --cached --quiet; then
  echo "[publish] Nothing to commit (post may already exist)"
  exit 0
fi

git commit -m "Add ${DATE} AI Infra morning brief"
git push

echo "[publish] Pushed: $POST_FILE"
