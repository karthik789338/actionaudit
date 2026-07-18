#!/usr/bin/env bash

set -euo pipefail

python -m compileall -q \
  src \
  scripts \
  tests

python -m pytest -q

python scripts/phase6a_final_claim_manifest.py \
  --output-dir artifacts/phase6a_final_claim_manifest

python scripts/phase6b_manuscript_tables_figures.py \
  --output-dir artifacts/phase6b_manuscript_tables_figures
