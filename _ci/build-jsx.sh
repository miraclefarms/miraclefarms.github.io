#!/usr/bin/env bash
# Precompile JSX -> plain JS (IIFE-wrapped, classic React.createElement) so the
# browser never needs @babel/standalone or type="text/babel". Output is same-origin
# under assets/js/ and committed for local `jekyll serve`; CI also re-runs this.
#
# Why IIFE: shared.jsx / tweaks-panel.jsx share components ONLY via Object.assign(window,…);
# each section page overrides window.ALL_* with live Liquid data. Wrapping every compiled
# file in an IIFE (esbuild --bundle --format=iife) keeps top-level const/function from
# leaking into the global lexical scope and shadowing those window overrides.
set -euo pipefail
cd "$(dirname "$0")/.."

EB="npx --yes esbuild@0.25.0"
COMMON=(--bundle --format=iife --loader:.jsx=jsx --target=es2018)

mkdir -p assets/js/pages

$EB assets/shared.jsx "${COMMON[@]}" --outfile=assets/js/shared.js
$EB tweaks-panel.jsx  "${COMMON[@]}" --outfile=assets/js/tweaks-panel.js

for p in home briefs readings fields essays foundations; do
  $EB "_jsx/pages/$p.jsx" "${COMMON[@]}" --outfile="assets/js/pages/$p.js"
done

echo "build-jsx: compiled shared.js, tweaks-panel.js, and 6 page bundles."
