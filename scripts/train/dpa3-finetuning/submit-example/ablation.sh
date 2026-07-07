# Ablation A: remove Q1
bash ./finetune_dpa3.sh \
  --model DPA-3.1-3M.pt \
  --input run_noq1.json \
  --train-list train_noQ1_ndef.txt \
  --val-list   val_systems.txt \
  --out        runs/abl_noQ1 \
  --branch     MP_traj_v024_alldata_mixu \
  --gpu 0

# Ablation B: remove Q5
bash ./finetune_dpa3.sh \
  --model DPA-3.1-3M.pt \
  --input run_noq5.json \
  --train-list train_noQ5_ndef.txt \
  --val-list   val_systems.txt \
  --out        runs/abl_noQ5 \
  --branch     MP_traj_v024_alldata_mixu \
  --gpu 0
