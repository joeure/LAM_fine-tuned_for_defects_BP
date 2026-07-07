#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DFT_ROOT="${DFT_ROOT:-$REPO_ROOT/data/raw-vasp/BP_1014}"
OOD_CSV="${OOD_CSV:-$REPO_ROOT/metadata/ood_per_cif.csv}"
OUT_PNG="${OUT_PNG:-vasp_scan_fmax_vs_ood.png}"
OUT_CSV="${OUT_CSV:-$REPO_ROOT/results/generated/vasp_scan_fmax_vs_ood.csv}"

python scan_vasp_fmax_vs_ood.py \
  --dft_dir "$DFT_ROOT" \
  --ood_csv "$OOD_CSV" \
  --out "$OUT_PNG" \
  --save_csv "$OUT_CSV"
