#!/usr/bin/env bash

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
# Quiet noisy TF logs (optional)
export TF_CPP_MIN_LOG_LEVEL=2

# Keep TF from grabbing all GPU memory (optional but nice)
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Keep oneDNN on unless you specifically need CPU determinism
# export TF_ENABLE_ONEDNN_OPTS=1   # default is effectively on; set 0 to disable

# DeePMD threading (tune to your CPU):
export DP_INTRA_OP_PARALLELISM_THREADS="${DP_INTRA_OP_PARALLELISM_THREADS:-1}"
export DP_INTER_OP_PARALLELISM_THREADS="${DP_INTER_OP_PARALLELISM_THREADS:-1}"

# Avoid BLAS oversubscription
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

set -euo pipefail

# Where run.py lives (edit if needed)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNPY="${SCRIPT_DIR}/run.py"

# Models to run
# MODELS=("DPA3")
MODELS=("DPA3-OMAT")

# One timestamp so the three logs are grouped
STAMP="$(date +%Y%m%d_%H%M%S)"

for m in "${MODELS[@]}"; do
  # log_basename can be anything; keep it unique per model
  LOG_BASENAME="sp_${m}_${STAMP}.log"
  echo "[INFO] Running ${m}  ->  log: ${LOG_BASENAME}"
  python3 "${RUNPY}" --model "${m}" --log_basename "${LOG_BASENAME}"
done

echo "[DONE] All models finished."
