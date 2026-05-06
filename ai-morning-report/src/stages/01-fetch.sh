#!/bin/bash
# Stage 1: Fetch raw data from GitHub repos (no AI, pure gh CLI)
# Usage: ./01-fetch.sh <date> <output-file> <repo-scope-json>
# Output: raw-data.md with merged PRs + releases + recent commits per repo

set -euo pipefail

DATE="${1:?Usage: $0 <date> <output-file> <repo-scope-json>}"
OUTPUT_FILE="${2:?}"
REPO_SCOPE_JSON="${3:?}"

SINCE=$(date -v-3d +%Y-%m-%dT00:00:00Z 2>/dev/null || date -d "3 days ago" +%Y-%m-%dT00:00:00Z)

REPOS=$(jq -r '.repos[]' "$REPO_SCOPE_JSON")
REPO_COUNT=$(echo "$REPOS" | wc -l | tr -d ' ')

echo "[fetch] Fetching $REPO_COUNT repos since $SINCE" >&2

TMPDIR_FETCH=$(mktemp -d)
trap 'rm -rf "$TMPDIR_FETCH"' EXIT

fetch_repo() {
  local repo="$1"
  local out="$2"
  local safe_name="${repo//\//_}"
  local repo_file="${out}/${safe_name}.md"

  {
    echo "## ${repo}"
    echo ""

    # Merged PRs (most important signal)
    echo "### Merged PRs"
    gh pr list --repo "$repo" \
      --state merged \
      --search "merged:>=${SINCE%T*}" \
      --limit 20 \
      --json number,title,mergedAt,url,labels,body \
      --jq '.[] | "- [#\(.number)] \(.title)\n  merged: \(.mergedAt)\n  url: \(.url)\n  labels: \([.labels[].name] | join(", "))\n  body_excerpt: \(.body | gsub("\r\n";" ") | gsub("\n";" ") | .[0:200])\n"' \
      2>/dev/null || echo "(no merged PRs)"

    echo ""
    echo "### Releases"
    gh api "repos/${repo}/releases" \
      --jq "[.[] | select(.published_at >= \"${SINCE}\")] | .[] | \"- \(.tag_name): \(.name)\n  published: \(.published_at)\n  url: \(.html_url)\n  body_excerpt: \(.body | gsub(\"\r\n\";\" \") | gsub(\"\n\";\" \") | .[0:300])\n\"" \
      2>/dev/null || echo "(no releases)"

    echo ""
    echo "### Recent Commits"
    gh api "repos/${repo}/commits?since=${SINCE}&per_page=10" \
      --jq '.[] | "- \(.sha[0:7]) \(.commit.message | split("\n")[0])\n  author: \(.commit.author.name)\n  date: \(.commit.author.date)\n  url: \(.html_url)\n"' \
      2>/dev/null || echo "(no commits)"

    echo ""
    echo "---"
    echo ""
  } > "$repo_file" 2>/dev/null
  echo "[fetch] done: $repo" >&2
}

export -f fetch_repo
export SINCE

# Parallel fetch (up to 6 concurrent)
echo "$REPOS" | xargs -P 6 -I{} bash -c 'fetch_repo "$@"' _ {} "$TMPDIR_FETCH"

# Assemble output
{
  echo "# AI Infra Raw Data — ${DATE}"
  echo ""
  echo "Time window: ${SINCE%T*} to ${DATE} (Asia/Shanghai)"
  echo ""
  for repo in $REPOS; do
    safe_name="${repo//\//_}"
    repo_file="${TMPDIR_FETCH}/${safe_name}.md"
    if [ -f "$repo_file" ]; then
      cat "$repo_file"
    fi
  done
} > "$OUTPUT_FILE"

LINE_COUNT=$(wc -l < "$OUTPUT_FILE" | tr -d ' ')
echo "[fetch] Output: $OUTPUT_FILE ($LINE_COUNT lines)" >&2
