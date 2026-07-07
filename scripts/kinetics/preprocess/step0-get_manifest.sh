python step0-get_manifest.py \
  --csv1 ../data_center/BP_defects.csv \
  --csv2 ../data_center/definet_defects.csv \
  --dft-dir ../relaxed_structures/BP_1014 \
  --base-parent-dir ../relaxed_structures/process_files_MACE-OMAT/data/high_density_defects/BP_spin_500/test \
  --ft-parent-dir   ../relaxed_structures/process_files_MACE-OMAT-25ep-5swa/data/high_density_defects/BP_spin_500/test \
  --base-dumps-dir  ../relaxed_structures/dumps_MACE-OMAT \
  --ft-dumps-dir    ../relaxed_structures/dumps_MACE-OMAT-25ep-5swa \
  --unrelaxed-cif-dir ../relaxed_structures/CIF \
  --out ../data_center/manifest_test.csv \
  --pristine-dft ../relaxed_structures/BP_1014/P/OUTCAR \
  --pristine-base ../relaxed_structures/process_files_MACE-OMAT/data/high_density_defects/BP_spin_500/reference/results/BP_spin_500_P_ref.data \
  --pristine-ft ../relaxed_structures/process_files_MACE-OMAT-25ep-5swa/data/high_density_defects/BP_spin_500/reference/results/BP_spin_500_P_ref.data \
  --only-test-lam \
  --strict-path-exists
#   --only-test-rows

