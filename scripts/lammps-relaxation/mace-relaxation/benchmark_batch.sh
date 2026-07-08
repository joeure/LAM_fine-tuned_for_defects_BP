export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
DATASET_ROOT="${DEFECT_DATASET_ROOT:?set DEFECT_DATASET_ROOT to the DefiNet-style dataset root}"
TEST_SPLIT_CSV="${TEST_SPLIT_CSV:-$REPO_ROOT/data/src/high_density_defects/BP/test.csv}"
rm -f "$DATASET_ROOT/high_density_defects/BP_spin_500/test.csv"
cp "$TEST_SPLIT_CSV" "$DATASET_ROOT/high_density_defects/BP_spin_500/test.csv"

python ./benchmark_workflow.py --config config.json
