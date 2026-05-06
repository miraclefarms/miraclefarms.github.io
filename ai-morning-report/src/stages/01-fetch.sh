#!/bin/bash
# Stage 1: Fetch raw data from GitHub repos (no AI, pure gh CLI)
# Usage: ./01-fetch.sh <date> <output-file> <repo-scope-json>

set -euo pipefail

DATE="${1:?Usage: $0 <date> <output-file> <repo-scope-json>}"
OUTPUT_FILE="${2:?}"
REPO_SCOPE_JSON="${3:?}"

# 3 days ago in ISO 8601 (macOS + Linux compatible)
SINCE=$(date -v-3d +%Y-%m-%dT00:00:00Z 2>/dev/null || date -d "3 days ago" +%Y-%m-%dT00:00:00Z)
SINCE_DATE="${SINCE%T*}"

REPOS=$(jq -r '.repos[]' "$REPO_SCOPE_JSON")
REPO_COUNT=$(echo "$REPOS" | wc -l | tr -d ' ')
echo "[fetch] Fetching $REPO_COUNT repos since $SINCE_DATE" >&2

TMPDIR_FETCH=$(mktemp -d)
trap 'rm -rf "$TMPDIR_FETCH"' EXIT

fetch_one() {
  local repo="$1"
  local since="$2"
  local tmpdir="$3"
  local safe="${repo//\//_}"
  local out="${tmpdir}/${safe}.md"

  {
    echo "## ${repo}"
    echo ""

    # Merged PRs via gh api (filter by merged_at >= since)
    echo "### Merged PRs"
    local pr_json
    pr_json=$(gh api "repos/${repo}/pulls?state=closed&per_page=30&sort=updated&direction=desc" 2>/dev/null) || pr_json="[]"
    local prs
    prs=$(echo "$pr_json" | jq -r \
      --arg since "$since" \
      '[.[] | select(.merged_at != null and .merged_at >= $since)] |
       .[] | "- [#\(.number)] \(.title)\n  merged: \(.merged_at)\n  url: \(.html_url)\n  body_excerpt: \(.body // "" | gsub("[\r\n]+";" ") | .[0:200])"' \
      2>/dev/null) || prs=""
    if [ -n "$prs" ]; then
      echo "$prs"
    else
      echo "(no merged PRs in window)"
    fi

    echo ""
    echo "### Releases"
    local rel_json
    rel_json=$(gh api "repos/${repo}/releases?per_page=10" 2>/dev/null) || rel_json="[]"
    local rels
    rels=$(echo "$rel_json" | jq -r \
      --arg since "$since" \
      '[.[] | select(.published_at >= $since)] |
       .[] | "- \(.tag_name): \(.name)\n  published: \(.published_at)\n  url: \(.html_url)\n  notes: \(.body // "" | gsub("[\r\n]+";" ") | .[0:300])"' \
      2>/dev/null) || rels=""
    if [ -n "$rels" ]; then
      echo "$rels"
    else
      echo "(no releases in window)"
    fi

    echo ""
    echo "---"
    echo ""
  } > "$out"
  echo "[fetch] done: $repo" >&2
}

# Launch all repos in parallel (background jobs, no export -f needed)
pids=()
for repo in $REPOS; do
  fetch_one "$repo" "$SINCE" "$TMPDIR_FETCH" &
  pids+=($!)
done

for pid in "${pids[@]}"; do
  wait "$pid" || true
done

# Assemble in original repo order
{
  echo "# AI Infra Raw Data — ${DATE}"
  echo ""
  echo "Time window: ${SINCE_DATE} to ${DATE} (Asia/Shanghai)"
  echo ""
  for repo in $REPOS; do
    safe="${repo//\//_}"
    f="${TMPDIR_FETCH}/${safe}.md"
    [ -f "$f" ] && cat "$f"
  done
} > "$OUTPUT_FILE"

LINE_COUNT=$(wc -l < "$OUTPUT_FILE" | tr -d ' ')
echo "[fetch] Output: $OUTPUT_FILE ($LINE_COUNT lines)" >&2
