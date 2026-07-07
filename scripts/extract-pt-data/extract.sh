# MPA + OMAT sources together; take 30k at random
python extract_replay_xyz.py \
  --inputs omat/mp_traj_combined_omat.xyz \
  --out64 replay_omat_np_30k_remap.xyz \
  --out32 replay_omat_np_30k_remap_fp32.xyz \
  --n 30000 --seed 42 --method fps --elements "N,P"
