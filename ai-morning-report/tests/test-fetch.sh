#!/bin/bash
# Unit test for Stage 1 (01-fetch.sh)
# Tests: gh availability, auth, API response, jq date filter, full fetch output

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGES_DIR="${REPO_ROOT}/ai-morning-report/src/stages"
CONFIG="${REPO_ROOT}/ai-morning-report/config/repo-scope.json"

PASS=0; FAIL=0
ok()   { echo "  ✓ $1"; PASS=$((PASS+1)); }
fail() { echo "  ✗ $1"; echo "    $2" >&2; FAIL=$((FAIL+1)); }

echo "=== Unit Test: 01-fetch.sh ==="

# Test 1: gh in PATH
echo ""
echo "[ Test 1: gh available ]"
if command -v gh >/dev/null 2>&1; then
  ok "gh found at $(command -v gh)"
else
  fail "gh not in PATH" "Install: https://cli.github.com"
fi

# Test 2: gh auth
echo ""
echo "[ Test 2: gh auth ]"
if gh auth status >/dev/null 2>&1; then
  ok "gh authenticated"
else
  fail "gh not authenticated" "Run: gh auth login"
fi

# Test 3: gh api returns valid JSON for one repo
echo ""
echo "[ Test 3: gh api returns JSON ]"
result=$(gh api "repos/vllm-project/vllm/pulls?state=closed&per_page=3&sort=updated&direction=desc" 2>&1)
if echo "$result" | jq 'type' 2>/dev/null | grep -q '"array"'; then
  count=$(echo "$result" | jq 'length')
  ok "vllm pulls API returned array of $count items"
else
  fail "gh api did not return JSON array" "${result:0:200}"
fi

# Test 4: jq date filter works
echo ""
echo "[ Test 4: jq date filter ]"
SINCE_TEST="2020-01-01T00:00:00Z"
sample='[{"merged_at":"2025-01-15T10:00:00Z","number":1},{"merged_at":null,"number":2},{"merged_at":"2019-12-31T00:00:00Z","number":3}]'
filtered=$(echo "$sample" | jq --arg since "$SINCE_TEST" \
  '[.[] | select(.merged_at != null and .merged_at >= $since)] | length' 2>/dev/null)
if [ "$filtered" = "1" ]; then
  ok "jq date filter: correctly found 1 item out of 3"
else
  fail "jq date filter returned $filtered, expected 1" ""
fi

# Test 5: full fetch script produces real data
echo ""
echo "[ Test 5: full fetch produces real data ]"
TMP_OUT=$(mktemp)
TMP_SCOPE=$(mktemp)
trap 'rm -f "$TMP_OUT" "$TMP_SCOPE"' EXIT

# Use only vllm + sglang for speed
cat > "$TMP_SCOPE" <<'JSON'
{
  "repos": ["vllm-project/vllm", "sgl-project/sglang"],
  "timeWindow": { "days": 7, "timezone": "Asia/Shanghai" }
}
JSON

TODAY=$(date +%Y-%m-%d)
if bash "${STAGES_DIR}/01-fetch.sh" "$TODAY" "$TMP_OUT" "$TMP_SCOPE" 2>&1; then
  lines=$(wc -l < "$TMP_OUT" | tr -d ' ')
  ok "fetch script ran without error ($lines lines)"
  if grep -q "^- \[#" "$TMP_OUT"; then
    pr_count=$(grep -c "^- \[#" "$TMP_OUT" || true)
    ok "found $pr_count merged PR(s) in output"
  else
    fail "no merged PRs found (check date window or repo activity)" \
      "$(grep 'gh api error\|no merged' "$TMP_OUT" | head -5)"
  fi
else
  fail "fetch script exited with error" "$(cat "$TMP_OUT" | head -10)"
fi

# Summary
echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[ $FAIL -eq 0 ]
