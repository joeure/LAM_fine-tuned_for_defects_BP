merge_xyz () {  # usage: merge_xyz OUT merged_from1.xyz merged_from2.xyz ...
  local out="$1"; shift
  rm -f "$out"
  for f in "$@"; do
    [ -s "$f" ] || { echo "ERR: missing/empty $f"; exit 2; }
    # append file with guaranteed trailing newline without mutating the source
    sed -e '$a\' "$f" >> "$out"
  done
  echo "Wrote $(wc -l <"$out") lines to $out"
}

# Drop Q1 (keep Q2+Q3+Q4+Q5)
# merge_xyz runs/ablate_q1/train.xyz shards/train_q2.xyz shards/train_q3.xyz shards/train_q4.xyz shards/train_q5.xyz

# Drop Q2
# merge_xyz runs/ablate_q2/train.xyz shards/train_q1.xyz shards/train_q3.xyz shards/train_q4.xyz shards/train_q5.xyz

# ... and so on for q3, q4, q5
# q5 = highest OOD; q1 = lowest.
