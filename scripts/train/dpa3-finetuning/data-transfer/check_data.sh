export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export DP_INTRA_OP_PARALLELISM_THREADS="${DP_INTRA_OP_PARALLELISM_THREADS:-1}"
export DP_INTER_OP_PARALLELISM_THREADS="${DP_INTER_OP_PARALLELISM_THREADS:-1}"

dp --pt show ./DPA-3.1-3M.pt descriptor | sed -n '/branch Alex2D/,+30p'

# env (pair) neighborhood
for s in data/train/system.00000{0..9}; do
  [ -d "$s" ] && dp neighbor-stat -s "$s" -r 6.0
done

# angular neighborhood (optional)
for s in data/train/system.00000{0..9}; do
  [ -d "$s" ] && dp neighbor-stat -s "$s" -r 4.0
done

# Try on ~50 systems to keep it quick; you can loop the rest later
i=0
for s in data/train/system.*; do
  dp --pt test \
     -m ./DPA-3.1-3M.pt \
     --model-branch Alex2D \
     -s "$s"
  i=$((i+1)); [ $i -ge 50 ] && break
done