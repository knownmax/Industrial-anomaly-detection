#!/usr/bin/env bash
# Convenience wrapper around run_all.py
# Usage: bash scripts/benchmark_all.sh [--categories bottle cable ...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

DATA_ROOT="${DATA_ROOT:-/data/max/kaggle/anomaly_detect/anomaly_ds}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/results}"
CONDA_ENV="${CONDA_ENV:-maxtorch}"

echo "========================================"
echo "  PatchCore MVTec Benchmark"
echo "  Data   : $DATA_ROOT"
echo "  Output : $OUTPUT_DIR"
echo "  Env    : $CONDA_ENV"
echo "========================================"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

python "${PROJECT_DIR}/src/run_all.py" \
    --data_root  "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    "$@"

echo ""
echo "Benchmark complete. Results in: $OUTPUT_DIR"
