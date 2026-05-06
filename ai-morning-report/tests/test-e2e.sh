#!/bin/bash
# End-to-end test with mocked AI responses (AI_CLI=mock)
# Tests the full pipeline shape without real AI calls or git push.
#
# Usage: bash tests/test-e2e.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGES="${REPO_ROOT}/ai-morning-report/src/stages"
FIXTURES="${REPO_ROOT}/ai-morning-report/tests/fixtures"
TEST_DATE="2099-01-01"
WORK_DIR="/tmp/morning-report-test/${TEST_DATE}"
POST_FILE="${REPO_ROOT}/_posts/${TEST_DATE}-ai-infra-daily-brief.md"

PASS=0; FAIL=0
ok()   { echo "  ✓ $1"; PASS=$((PASS+1)); }
fail() { echo "  ✗ $1"; echo "    $2" >&2; FAIL=$((FAIL+1)); }

cleanup() {
  rm -f "$POST_FILE"
  rm -rf "$WORK_DIR"
}
trap cleanup EXIT

mkdir -p "$WORK_DIR/assets"

echo "=== End-to-End Test (AI_CLI=mock, date=$TEST_DATE) ==="

# ── Stage 2: analyze with fixture response ─────────────────────────────────
echo ""
echo "[ Stage 2: analyze (mock AI) ]"

# Provide a minimal raw-data.md with real-looking data
cat > "${WORK_DIR}/raw-data.md" <<'MD'
# AI Infra Raw Data — 2099-01-01
Time window: 2098-12-29 to 2099-01-01

## vllm-project/vllm

### Merged PRs
- [#35461] Add probabilistic rejection sampling support in MRV2
  merged: 2099-01-01T06:12:00Z
  url: https://github.com/vllm-project/vllm/pull/35461

### Releases
(no releases in window)

---

## sgl-project/sglang

### Merged PRs
- [#19248] Add Elastic NIXL-EP communication path
  merged: 2099-01-01T08:33:00Z
  url: https://github.com/sgl-project/sglang/pull/19248

### Releases
(no releases in window)

---
MD

export AI_CLI=mock
export MOCK_AI_RESPONSE_FILE="${FIXTURES}/analyze-response.md"

if node "${STAGES}/02-analyze.js" \
    "$TEST_DATE" \
    "${WORK_DIR}/raw-data.md" \
    "${WORK_DIR}/material.md" \
    "$REPO_ROOT" 2>&1; then
  if [ -f "${WORK_DIR}/material.md" ]; then
    themes=$(grep -c "^## " "${WORK_DIR}/material.md" || true)
    ok "material.md created ($themes H2 sections)"
  else
    fail "material.md not created" ""
  fi
else
  fail "02-analyze.js exited with error" ""
fi

# ── Stage 3: write with fixture response ───────────────────────────────────
echo ""
echo "[ Stage 3: write (mock AI) ]"

export MOCK_AI_RESPONSE_FILE="${FIXTURES}/write-response.md"

if node "${STAGES}/03-write.js" \
    "$TEST_DATE" \
    "${WORK_DIR}/material.md" \
    "$REPO_ROOT" 2>&1; then
  if [ -f "$POST_FILE" ]; then
    ok "post file created: $(basename "$POST_FILE")"
    # Check required front matter fields
    for field in title date author kind category intro; do
      if grep -q "^${field}:" "$POST_FILE"; then
        ok "front matter has: $field"
      else
        fail "front matter missing: $field" ""
      fi
    done
    # Check body structure
    if grep -q "^## " "$POST_FILE"; then
      ok "post has H2 sections"
    else
      fail "post has no H2 sections" ""
    fi
    if grep -q "参考来源" "$POST_FILE"; then
      ok "post has 参考来源 section"
    else
      fail "post missing 参考来源 section" ""
    fi
  else
    fail "post file not found at: $POST_FILE" ""
  fi
else
  fail "03-write.js exited with error" ""
fi

# ── Stage 4: cover (non-blocking, no API key needed) ─────────────────────
echo ""
echo "[ Stage 4: cover image (non-blocking) ]"
unset OPENROUTER_API_KEY 2>/dev/null || true
if node "${STAGES}/04-cover.js" "$TEST_DATE" "$POST_FILE" "${WORK_DIR}/assets" 2>&1; then
  ok "04-cover.js exited 0 (non-blocking even without API key)"
else
  fail "04-cover.js exited non-zero (should be non-blocking)" ""
fi

# ── Stage 6: wechat-format with mock AI ───────────────────────────────────
echo ""
echo "[ Stage 6: wechat-format (mock AI) ]"
export MOCK_AI_RESPONSE_FILE="${FIXTURES}/write-response.md"
WECHAT_OUT_DIR="${WORK_DIR}/wechat"
mkdir -p "$WECHAT_OUT_DIR"

if node "${STAGES}/06-wechat-format.js" \
    "$TEST_DATE" \
    "$POST_FILE" \
    "$WECHAT_OUT_DIR" \
    "$REPO_ROOT" 2>&1; then
  md_out="${WECHAT_OUT_DIR}/${TEST_DATE}-ai-infra-daily-brief-wechat.md"
  html_out="${WECHAT_OUT_DIR}/${TEST_DATE}-ai-infra-daily-brief-wechat.html"
  if [ -f "$md_out" ]; then
    ok "WeChat markdown created"
  else
    fail "WeChat markdown not found" ""
  fi
  if [ -f "$html_out" ]; then
    ok "WeChat HTML created"
    if grep -q 'style="' "$html_out"; then
      ok "HTML has inline styles"
    else
      fail "HTML missing inline styles" ""
    fi
  else
    fail "WeChat HTML not found" ""
  fi
else
  fail "06-wechat-format.js exited with error" ""
fi

# Summary
echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[ $FAIL -eq 0 ]
