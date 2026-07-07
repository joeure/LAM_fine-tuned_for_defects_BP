export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
# Quiet noisy TF logs (optional)
export TF_CPP_MIN_LOG_LEVEL=2

# Keep TF from grabbing all GPU memory (optional but nice)
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Keep oneDNN on unless you specifically need CPU determinism
# export TF_ENABLE_ONEDNN_OPTS=1   # default is effectively on; set 0 to disable

# DeePMD threading (tune to your CPU):
export DP_INTRA_OP_PARALLELISM_THREADS="${DP_INTRA_OP_PARALLELISM_THREADS:-1}"
export DP_INTER_OP_PARALLELISM_THREADS="${DP_INTER_OP_PARALLELISM_THREADS:-1}"

# Avoid BLAS oversubscription
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

DATASET_ROOT="${DEFECT_DATASET_ROOT:?set DEFECT_DATASET_ROOT to the DefiNet-style dataset root}"
rm -f "$DATASET_ROOT/high_density_defects/BP_spin_500/test.csv"
cp ./test.csv "$DATASET_ROOT/high_density_defects/BP_spin_500/"
python ./benchmark_workflow.py --config config.json
