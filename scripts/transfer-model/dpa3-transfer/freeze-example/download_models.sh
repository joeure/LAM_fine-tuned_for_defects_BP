export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
branch="MP_traj_v024_alldata_mixu"
# branch="Omat24"
indir="checkpoints"            # where your ckpt-*.pt live
outdir="frozen"                # where to write .pth
prefix="frozen-"        # optional filename prefix

mkdir -p "$outdir"
for ck in "$indir"/*.pt; do
  stem="${ck##*/}"; stem="${stem%.pt}"
  out="$outdir/${prefix}${stem}_${branch}.pth"
  echo "[freeze] $ck -> $out"
  dp --pt freeze -c "$ck" -o "$out" --model-branch "$branch"
done
