bash ./finetune_dpa3.sh \
    --init-from ckpt/ckpt-2120.pt \
    --input run_Q5Q4.json \
    --train-list train_Q1.txt \
    --val-list   val_systems.txt \
    --out        runs/dpa3_Q5Q4Q3Q2Q1 \
    --branch MP_traj_v024_alldata_mixu \
    --gpu 0
