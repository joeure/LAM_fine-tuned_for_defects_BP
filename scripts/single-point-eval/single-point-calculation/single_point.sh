#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
DFT_ROOT="${DFT_ROOT:-$REPO_ROOT/data/raw-vasp/BP_1014}"
META_CSV="${META_CSV:-$REPO_ROOT/metadata/BP_defects.csv}"
OUT_ROOT="${OUT_ROOT:-$SCRIPT_DIR/prepare_space}"
MODEL_SPEC="${MODEL_SPEC:-DPA3-OMAT:deepmd:$REPO_ROOT/models/fine-tuned/frozen-ckpt_DPA3-OMat24.pth}"

python prep_sp_from_dft.py \
  --dft_dir "$DFT_ROOT" \
  --meta_csv "$META_CSV" \
  --system_name BP_spin_500 \
  --out_root "$OUT_ROOT" \
  --models "$MODEL_SPEC"
