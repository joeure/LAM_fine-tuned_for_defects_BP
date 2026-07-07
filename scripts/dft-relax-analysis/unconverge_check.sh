#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DFT_ROOT="${DFT_ROOT:-$REPO_ROOT/data/raw-vasp/BP_1014}"
OUT_CSV="${OUT_CSV:-$REPO_ROOT/results/generated/unconverged_report.csv}"

python unconverge_analyze.py "$DFT_ROOT" "$OUT_CSV" --only-unconverged
