#!/bin/bash
# Stage 1: Fetch raw data from GitHub repos (no AI, pure gh CLI)
# Usage: ./01-fetch.sh <date> <output-file> <repo-scope-json>

set -uo pipefail  # no -e: we handle errors explicitly per command

DATE="${1:?Usage: $0 <date> <output-file> <repo-scope-json>}"
OUTPUT_FILE="${2:?}"
REPO_SCOPE_JSON="${3:?}"

# ── Dependency checks ──────────────────────────────────────────────────────
if ! command -v gh >/dev/null 2>&1; then
  echo "[fetch] ERROR: 'gh' not in PATH. Install: https://cli.github.com" >&2
  exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "[fetch] ERROR: 'jq' not in PATH. Install: brew install jq" >&2
  exit 1
fi
if ! gh auth status >/dev/null 2>&1; then
  echo "[fetch] ERROR: gh not authenticated. Run: gh auth login" >&2
  exit 1
fi

# ── Date range ────────────────────────────────────────────────────────────
SINCE=$(date -v-3d +%Y-%m-%dT00:00:00Z 2>/dev/null \
     || date -d "3 days ago" +%Y-%m-%dT00:00:00Z)
SINCE_DATE="${SINCE%T*}"

REPOS=$(jq -r '.repos[]' "$REPO_SCOPE_JSON")
REPO_COUNT=$(echo "$REPOS" | wc -l | tr -d ' ')
echo "[fetch] $REPO_COUNT repos | window: $SINCE_DATE → $DATE" >&2

# ── Fetch each repo sequentially with visible errors ──────────────────────
fetch_repo() {
  local repo="$1"
  echo "[fetch] → $repo" >&2

  echo "## ${repo}"
  echo ""

  # Merged PRs
  echo "### Merged PRs"
  local pr_err pr_json exit_code
  pr_json=$(gh api "repos/${repo}/pulls?state=closed&per_page=50&sort=updated&direction=desc" 2>/tmp/gh-stderr-$$) \
    && exit_code=0 || exit_code=$?
  pr_err=$(cat /tmp/gh-stderr-$$ 2>/dev/null); rm -f /tmp/gh-stderr-$$

  if [ $exit_code -ne 0 ]; then
    echo "(gh api error: ${pr_err:-exit $exit_code})"
  elif [ -z "$pr_json" ] || [ "$pr_json" = "[]" ]; then
    echo "(repo returned no closed PRs)"
  else
    local prs
    prs=$(echo "$pr_json" | jq -r --arg since "$SINCE" \
      '[.[] | select(.merged_at != null and .merged_at >= $since)] |
       .[] | "- [#\(.number)] \(.title)\n  merged: \(.merged_at)\n  url: \(.html_url)\n  summary: \(.body // "" | gsub("[\r\n]+"; " ") | .[0:200])"' \
      2>/dev/null) || prs=""
    if [ -n "$prs" ]; then
      echo "$prs"
    else
      echo "(no merged PRs in window $SINCE_DATE)"
    fi
  fi

  echo ""

  # Releases
  echo "### Releases"
  local rel_err rel_json rel_exit
  rel_json=$(gh api "repos/${repo}/releases?per_page=10" 2>/tmp/gh-stderr-$$) \
    && rel_exit=0 || rel_exit=$?
  rel_err=$(cat /tmp/gh-stderr-$$ 2>/dev/null); rm -f /tmp/gh-stderr-$$

  if [ $rel_exit -ne 0 ]; then
    echo "(gh api error: ${rel_err:-exit $rel_exit})"
  else
    local rels
    rels=$(echo "$rel_json" | jq -r --arg since "$SINCE" \
      '[.[] | select(.published_at != null and .published_at >= $since)] |
       .[] | "- \(.tag_name): \(.name)\n  published: \(.published_at)\n  url: \(.html_url)\n  notes: \(.body // "" | gsub("[\r\n]+"; " ") | .[0:300])"' \
      2>/dev/null) || rels=""
    if [ -n "$rels" ]; then
      echo "$rels"
    else
      echo "(no releases in window $SINCE_DATE)"
    fi
  fi

  echo ""
  echo "---"
  echo ""
  echo "[fetch] done: $repo" >&2
}

# Run sequentially, write to output file
{
  echo "# AI Infra Raw Data — ${DATE}"
  echo "Time window: ${SINCE_DATE} to ${DATE} (Asia/Shanghai)"
  echo ""

  for repo in $REPOS; do
    fetch_repo "$repo"
  done
} > "$OUTPUT_FILE"

LINE_COUNT=$(wc -l < "$OUTPUT_FILE" | tr -d ' ')
echo "[fetch] Output: $OUTPUT_FILE ($LINE_COUNT lines)" >&2

# Gate: warn if every repo returned errors or no data
if ! grep -q "^- \[#" "$OUTPUT_FILE" && ! grep -q "^- v" "$OUTPUT_FILE"; then
  echo "[fetch] WARN: no PR or release data found in output — check gh auth or date window" >&2
fi
