export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS=1
export DP_INTRA_OP_PARALLELISM_THREADS="${DP_INTRA_OP_PARALLELISM_THREADS:-1}"
export DP_INTER_OP_PARALLELISM_THREADS="${DP_INTER_OP_PARALLELISM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES=0

### export DP_INFER_BATCH_SIZE=8000   # or smaller until it stops. if illegal memory access met
dp --pt train input.json \
   --finetune ./DPA-3.1-3M.pt \
   --model-branch Alex2D \
   --use-pretrain-script

## TEST part
# paths
FT=ckpt/ckpt-01500.pt          # <- or your chosen/best step
BASE=./DPA-3.1-3M.pt           # foundation
OUT1=results_foundation.csv
OUT2=results_finetuned.csv

echo "system,energy_rmse_per_atom,force_rmse" > "$OUT1"
echo "system,energy_rmse_per_atom,force_rmse" > "$OUT2"

for s in data/test/system.*; do
  # Foundation (needs --model-branch Alex2D)
  dp --pt test -m "$BASE" --model-branch Alex2D -s "$s" 2>/dev/null \
  | awk -v sys="$s" '
      /Energy RMSE\/Natoms/ {e=$NF}
      /Force  RMSE/        {f=$(NF-1)}
      END{if(e!="") printf("%s,%s,%s\n",sys,e,f)}' >> "$OUT1"

  # Fine-tuned (no --model-branch needed)
  dp --pt test -m "$FT" -s "$s" 2>/dev/null \
  | awk -v sys="$s" '
      /Energy RMSE\/Natoms/ {e=$NF}
      /Force  RMSE/        {f=$(NF-1)}
      END{if(e!="") printf("%s,%s,%s\n",sys,e,f)}' >> "$OUT2"
done

echo "Wrote $OUT1 and $OUT2"

