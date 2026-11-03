#!/usr/bin/env bash
# Run all detconf stages in order, then rebuild results/summary.md.
# Uses cached predictions/attack images where their metadata matches; add --force to a
# stage below to recompute it. Expects the venv from the README (.venv).
set -euo pipefail
cd "$(dirname "$0")"

if [ -x .venv/Scripts/python ]; then PY=.venv/Scripts/python; else PY=.venv/bin/python; fi

"$PY" run.py --stage baseline
"$PY" run.py --stage shift
"$PY" run.py --stage reliability
"$PY" run.py --stage attack
"$PY" summarize.py
