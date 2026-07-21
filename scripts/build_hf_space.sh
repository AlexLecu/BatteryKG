#!/usr/bin/env bash
# Build the Hugging Face Space bundle in hf_space/ — the Streamlit demo running
# exclusively in snapshot mode (no Neo4j): app + trained artifacts + the static
# KG snapshot + the two reports the Pipeline Status page reads. Kept slim: no
# src/, no tests, no datasets.
#
# Usage:  scripts/build_hf_space.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/hf_space"
SNAPSHOT="$ROOT/data/kg_snapshots/app_snapshot.json"

if [ ! -f "$SNAPSHOT" ]; then
  echo "ERROR: $SNAPSHOT missing — run 'python -m scripts.gen_app_snapshot'" \
       "against a live KG first." >&2
  exit 1
fi

rm -rf "$DEST"
mkdir -p "$DEST/data/kg_snapshots" "$DEST/outputs"

RSYNC="rsync -a --exclude=__pycache__ --exclude=.DS_Store"
$RSYNC "$ROOT/app/" "$DEST/app/"
rm -f "$DEST/app/Dockerfile"                     # Space uses the streamlit SDK
rm -f "$DEST"/app/pages/*🔬*.py                  # demo Space: no research pages
N_PAGES=$(ls "$DEST"/app/pages/*.py | wc -l | tr -d ' ')
if [ "$N_PAGES" != "4" ]; then
  echo "ERROR: expected exactly 4 demo pages in hf_space, found $N_PAGES:" >&2
  ls "$DEST"/app/pages/ >&2
  exit 1
fi
$RSYNC "$ROOT/.streamlit/" "$DEST/.streamlit/"
cp "$SNAPSHOT" "$DEST/data/kg_snapshots/"
for f in experiment_01_prediction.md experiment_02_extraction.md; do
  cp "$ROOT/outputs/$f" "$DEST/outputs/"
done

# slim runtime deps (subset of requirements.txt the app actually imports;
# neo4j stays so NEO4J_* Space secrets can switch it to a live DB e.g. Aura)
cat > "$DEST/requirements.txt" <<'REQS'
numpy
pandas
pyarrow
matplotlib
streamlit
pyvis
xgboost
scikit-learn
neo4j
REQS

cat > "$DEST/README.md" <<'CARD'
---
title: BatteryKG
emoji: 🔋
colorFrom: blue
colorTo: gray
sdk: streamlit
app_file: app/main.py
pinned: false
license: mit
---

# 🔋 BatteryKG — interactive demo

We check what battery makers promise against what independent tests measure —
and our predictor refuses to guess when it shouldn't.

This Space runs in **snapshot mode**: knowledge-graph answers are served from a
static snapshot recorded from the live BatteryKG graph (same rows, no mocking);
predictions run live through the trained model artifacts.

Code, data provenance, and paper: https://github.com/AlexLecu/BatteryKG
CARD

# same fail-hard hygiene guards as the public release build
LEAKS=$(find "$DEST" \( -name "*.pdf" -o -name ".env" -o -iname ".claude*" \
        -o -iname "CLAUDE.md" -o -iname ".cursor*" -o -iname ".aider*" \
        -o -iname ".idea" -o -name "DECISIONS.md" \) | grep -v '^$' || true)
if [ -n "$LEAKS" ]; then
  echo "ERROR: excluded files found in hf_space/:" >&2; echo "$LEAKS" >&2
  exit 1
fi
if grep -rnE "gsk_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}" "$DEST" 2>/dev/null; then
  echo "ERROR: key-shaped string found in hf_space/." >&2
  exit 1
fi

echo "==> Done. Space bundle at $DEST ($(du -sh "$DEST" | cut -f1))"
echo "    Deploy: push the CONTENTS of hf_space/ to your Space repo."
