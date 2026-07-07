python export_mace_checkpoints.py \
  --ckpt_glob 'checkpoints/omat_finetune_BP_smoketest_run-42_epoch-*.pt' \
  --keep_last 4 \
  --template_model results/omat_finetune_BP_smoketest_run-42.model \
  --head omat_definet_ft_head \
  --compile \
  --format libtorch

