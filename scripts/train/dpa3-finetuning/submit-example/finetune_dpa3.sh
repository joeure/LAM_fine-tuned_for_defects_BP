#!/usr/bin/env bash
# finetune_dpa3_min.sh — minimal DeepMD-PT finetune/continue runner + diagnostics
# Examples:
#   Fresh finetune from foundation:  --model DPA-3.1-3M.pt
#   Continue training (resume):      --resume auto
#   New phase from your ckpt:        --init-from ckpt/ckpt-00075000.pt

set -euo pipefail

MODEL=""         # foundation .pt
INPUT_JSON=""
TRAIN_LIST=""
VAL_LIST=""
OUTDIR="runs/dpa3_ft_$(date +%Y%m%d-%H%M%S)"
BRANCH=""
GPU="0"
TYPEMAP="N,P"
LOG=""
SESSION=""
RESUME=""        # "auto" or path to ckpt-xxxxx.pt  (uses --restart)
INIT_FROM=""     # path to your ckpt .pt            (uses --finetune)

export DP_INTRA_OP_PARALLELISM_THREADS="${DP_INTRA_OP_PARALLELISM_THREADS:-1}"
export DP_INTER_OP_PARALLELISM_THREADS="${DP_INTER_OP_PARALLELISM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

# ---------- tiny helpers / traps ----------
ts() { date +"%Y-%m-%d %H:%M:%S%z"; }
on_err() {
  local rc=$?
  echo "[ERR] $(date +"%Y-%m-%d %H:%M:%S%z") rc=$rc at line ${BASH_LINENO[0]} while running: ${BASH_COMMAND}" >&2
  echo "[ERR] PWD=$PWD" >&2
  if [[ -n "${LOG:-}" && -f "$LOG" ]]; then
    echo "[ERR] --- tail of $LOG ---" >&2
    tail -n 60 "$LOG" >&2
  fi
  exit $rc
}
on_exit() {
  local rc=$?
  echo "[EXIT] $(ts) rc=$rc"
}
trap on_err ERR
trap on_exit EXIT
set -o errtrace

# ---------- args ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)      MODEL="$2"; shift 2;;
    --input)      INPUT_JSON="$2"; shift 2;;
    --train-list) TRAIN_LIST="$2"; shift 2;;
    --val-list)   VAL_LIST="$2"; shift 2;;
    --out)        OUTDIR="$2"; shift 2;;
    --branch)     BRANCH="$2"; shift 2;;
    --gpu)        GPU="$2"; shift 2;;
    --type-map)   TYPEMAP="$2"; shift 2;;
    --resume)     RESUME="$2"; shift 2;;
    --init-from)  INIT_FROM="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

command -v dp >/dev/null 2>&1 || { echo "ERROR: 'dp' not in PATH"; exit 2; }
[[ -f "$INPUT_JSON" ]] || { echo "ERROR: input.json not found: $INPUT_JSON"; exit 2; }
[[ -f "$TRAIN_LIST" ]] || { echo "ERROR: train list not found: $TRAIN_LIST"; exit 2; }
[[ -f "$VAL_LIST"   ]] || { echo "ERROR: val list not found: $VAL_LIST";   exit 2; }

# mode validation: exactly one of {model,resume,init-from}
n=0; [[ -n "$MODEL" ]] && ((++n)); [[ -n "$RESUME" ]] && ((++n)); [[ -n "$INIT_FROM" ]] && ((++n))
if [[ $n -ne 1 ]]; then
  echo "ERROR: choose exactly one of --model | --resume | --init-from"; exit 2
fi

mkdir -p "$OUTDIR"/{logs}
cp -f "$INPUT_JSON" "$OUTDIR"/input.json
cp -f "$TRAIN_LIST" "$OUTDIR"/train_systems.txt
cp -f "$VAL_LIST"   "$OUTDIR"/val_systems.txt
if [ -d "ckpt" ]; then
    echo "Directory ckpt found. Copying to '$OUTDIR'..."
    # Copy the directory recursively, preserving attributes
    cp -a "ckpt" "$OUTDIR"
    if [ $? -eq 0 ]; then
        echo "Directory ckpt successfully copied to '$OUTDIR'."
    else
        echo "Error: Failed to copy directory ckpt to '$OUTDIR'."
    fi
else
    echo 'ckpt directory does not exist.'
fi
[[ -n "$MODEL" ]] && cp -f "$MODEL" "$OUTDIR"/

export CUDA_VISIBLE_DEVICES="$GPU"
cd "$OUTDIR"

# --- make sure logs/ exists in the CURRENT dir before tee'ing ---
mkdir -p logs

# one set of log paths; pre-create so tee never fails
LOG="logs/train_$(date +%Y%m%d-%H%M%S).log"
SESSION="logs/session_$(date +%Y%m%d-%H%M%S).log"
: > "$LOG"
: > "$SESSION"

# session header
{
  echo "[INFO] $(date +"%Y-%m-%d %H:%M:%S%z") starting in $PWD"
  echo "[INFO] dp version: $(dp --version 2>/dev/null || echo 'unknown')"
  echo "[INFO] python: $(python -V 2>&1)"
  echo "[INFO] torch:  $(python - <<'PY' 2>/dev/null || true
import sys
try:
    import torch
    print(getattr(torch,'__version__','<no torch>'))
except Exception:
    print('<no torch>')
PY
)"
  echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi -L; else echo "[INFO] nvidia-smi not available"; fi
  echo "[INFO] THREADS: OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS OPENBLAS=$OPENBLAS_NUM_THREADS DP_INTRA=$DP_INTRA_OP_PARALLELISM_THREADS DP_INTER=$DP_INTER_OP_PARALLELISM_THREADS"
} | tee -a "$SESSION"


# Expand @files into explicit arrays (no JSON editing beyond this)
python - "$PWD/input.json" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]); j = json.load(open(p))
def read_list(fname):
    path = pathlib.Path(fname)
    lines = [ln.strip() for ln in open(path, encoding="utf-8") if ln.strip()]
    missing = [x for x in lines if not pathlib.Path(x).exists()]
    if missing: raise SystemExit("[expand] some system paths do not exist:\n" + "\n".join(missing[:10]))
    return lines
j.setdefault("training", {}); j["training"].setdefault("training_data", {}); j["training"].setdefault("validation_data", {})
j["training"]["training_data"]["systems"]   = read_list("train_systems.txt")
j["training"]["validation_data"]["systems"] = read_list("val_systems.txt")
json.dump(j, open(p, "w"), indent=2)
print("[expand] systems expanded OK")
PY

# Show key JSON fields (helps debug silent mismatches)
python - <<'PY' | tee -a "$SESSION"
import json, pathlib
j=json.load(open("input.json"))
tr=j.get("training",{}); lr=j.get("learning_rate",{}); ls=j.get("loss",{})
print("[JSON] training.numb_steps =", tr.get("numb_steps"))
print("[JSON] lr:", {k:lr.get(k) for k in ("type","start_lr","stop_lr","decay_steps")})
print("[JSON] loss:", {k:ls.get(k) for k in ("type","start_pref_e","limit_pref_e","start_pref_f","limit_pref_f","start_pref_v","limit_pref_v")})
print("[JSON] batches:", "train", tr.get("training_data",{}).get("batch_size"), "val", tr.get("validation_data",{}).get("batch_size"))
PY

# Preview lists
echo "[INFO] train systems: $(wc -l < train_systems.txt)"; head -n 3 train_systems.txt || true
echo "[INFO] val   systems: $(wc -l < val_systems.txt)";   head -n 3 val_systems.txt   || true

echo "==== DPA-3 run ===="
[[ -n "$MODEL"    ]] && echo "Mode: finetune (foundation)  : $(basename "$MODEL")"
[[ -n "$INIT_FROM" ]] && echo "Mode: init from your ckpt    : $INIT_FROM"
[[ -n "$RESUME"    ]] && echo "Mode: resume from checkpoint : $RESUME"
echo "Input JSON: input.json"
echo "Branch    : ${BRANCH:-<none>}"
echo "Out dir   : $OUTDIR"

BRANCH_ARG=(); [[ -n "$BRANCH" ]] && BRANCH_ARG=(--model-branch "$BRANCH")

# Resolve checkpoint if resuming
if [[ -n "$RESUME" ]]; then
  ckpt="$RESUME"
  if [[ "$RESUME" == "auto" ]]; then
    ckpt=$(ls -1 ckpt/*.pt 2>/dev/null | sort | tail -n1 || true)
  fi
  [[ -z "${ckpt:-}" ]] && { echo "ERROR: --resume wanted but no ckpt/*.pt found"; exit 2; }
  [[ -f "$ckpt" ]] || { echo "ERROR: resume checkpoint not found: $ckpt"; exit 2; }
  echo "[INFO] resume checkpoint: $ckpt" | tee -a "$SESSION"
fi

# ---------- launch dp and capture real exit code ----------
set -x
if [[ -n "$RESUME" ]]; then
  dp --pytorch train input.json \
     --restart "$ckpt" \
     --use-pretrain-script \
     --skip-neighbor-stat \
     "${BRANCH_ARG[@]}" 2>&1 | tee "$LOG"
  rc=${PIPESTATUS[0]}
elif [[ -n "$INIT_FROM" ]]; then
  [[ -f "$INIT_FROM" ]] || { echo "ERROR: --init-from file not found: $INIT_FROM"; exit 2; }
  dp --pytorch train input.json \
     --finetune "$INIT_FROM" \
     --use-pretrain-script \
     --skip-neighbor-stat \
     "${BRANCH_ARG[@]}" 2>&1 | tee "$LOG"
  rc=${PIPESTATUS[0]}
else
  dp --pytorch train input.json \
     --finetune "$(basename "$MODEL")" \
     --use-pretrain-script \
     --skip-neighbor-stat \
     "${BRANCH_ARG[@]}" 2>&1 | tee "$LOG"
  rc=${PIPESTATUS[0]}
fi
set +x

if [[ $rc -ne 0 ]]; then
  echo "[FAIL] dp exited with code $rc" | tee -a "$SESSION"
  echo "[FAIL] tail of $LOG:" | tee -a "$SESSION"
  tail -n 120 "$LOG" | tee -a "$SESSION"
  exit $rc
fi

echo "Done. Checkpoints in: $OUTDIR/ckpt"
echo "[OK] $(ts) completed successfully" | tee -a "$SESSION"
