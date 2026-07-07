# Ablation seq chain-0: Q1
bash ./finetune_dpa3.sh \
  --model DPA-3.1-3M.pt \
  --input run_Q5.json \
  --train-list train_Q5.txt \
  --val-list   val_systems.txt \
  --out        runs/abl_Q5 \
  --branch     MP_traj_v024_alldata_mixu \
  --gpu 0