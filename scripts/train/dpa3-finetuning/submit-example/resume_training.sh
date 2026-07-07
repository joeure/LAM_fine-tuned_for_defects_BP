# ./finetune_dpa3.sh --resume auto \
#   --input runs/prev/input.json --train-list runs/prev/train_systems.txt \
#   --val-list runs/prev/val_systems.txt --out runs/prev \
#   --target-steps 100000 --pref-e-limit 1.6 --pref-f-limit 0.4


bash ./finetune_dpa3.sh \
    --resume auto \
    --input continue.json \
    --train-list train_systems.txt \
    --val-list   val_systems.txt \
    --out        runs/dpa3_continue \
    --branch MP_traj_v024_alldata_mixu \
    --gpu 0
