export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

DATASET_ROOT="${DEFECT_DATASET_ROOT:?set DEFECT_DATASET_ROOT to the DefiNet-style dataset root}"
rm -f "$DATASET_ROOT/high_density_defects/BP_spin_500/test.csv"
cp ./test.csv "$DATASET_ROOT/high_density_defects/BP_spin_500/"

python ./benchmark_workflow.py --config config.json
