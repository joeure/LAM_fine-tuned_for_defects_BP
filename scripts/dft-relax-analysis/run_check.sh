#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DFT_ROOT="${DFT_ROOT:-$REPO_ROOT/data/raw-vasp/BP_1014}"
OUT_CSV="${OUT_CSV:-$REPO_ROOT/results/generated/vasp_scan.csv}"

python scan_vasp_convergence.py \
  --dft_dir "$DFT_ROOT" \
  --out_csv "$OUT_CSV" \
  --warn_missing
