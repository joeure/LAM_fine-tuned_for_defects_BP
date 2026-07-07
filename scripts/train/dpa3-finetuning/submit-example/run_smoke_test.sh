bash ./finetune_dpa3.sh \
  --model DPA-3.1-3M.pt \
  --input smoke_test.json \
  --train-list train_systems.txt \
  --val-list   val_systems.txt \
  --out        runs/dpa3_smoke \
  --branch MP_traj_v024_alldata_mixu \
  --gpu 0
  # --dry-run
  # inspect, then drop --dry-run