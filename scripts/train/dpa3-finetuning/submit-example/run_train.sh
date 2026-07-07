bash ./finetune_dpa3.sh \
  --model DPA-3.1-3M.pt \
  --input run_stretching.json \
  --train-list train_systems.txt \
  --val-list   val_systems.txt \
  --out        runs/dpa3_stretch \
  --branch Omat24 \
  --gpu 0
  # --dry-run
  # inspect, then drop --dry-run