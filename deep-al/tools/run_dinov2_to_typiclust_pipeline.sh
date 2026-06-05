#!/usr/bin/env bash
set -euo pipefail

# End-to-end pipeline:
# 1) Train DINOv2-style CIFAR10 embeddings (seed 1..10)
# 2) Run TypiClust-DINOv2 AL (seed 1..10)
# 3) Aggregate and plot comparison with other methods
#
# Usage:
#   bash run_dinov2_to_typiclust_pipeline.sh [dinov2_max_jobs] [typiclust_max_jobs] [gpu_list] [budget] [initial_size]
#
# Example:
#   bash run_dinov2_to_typiclust_pipeline.sh 2 2 "0" 100 0

DINOV2_MAX_JOBS="${1:-2}"
TYPI_MAX_JOBS="${2:-1}"
GPU_LIST="${3:-0}"
BUDGET="${4:-100}"
INITIAL_SIZE="${5:-0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "[1/3] Train DINOv2-style CIFAR10 embeddings (seed 1..10)"
cd "${ROOT_DIR}/scan"
bash run_dinov2_cifar10_seeds_1_10_parallel.sh "${DINOV2_MAX_JOBS}"

echo "[2/3] Run TypiClust-DINOv2 AL (seed 1..10)"
cd "${ROOT_DIR}/deep-al/tools"
bash run_typiclust_dinov2_seeds_1_10_parallel.sh "${BUDGET}" "${INITIAL_SIZE}" "${TYPI_MAX_JOBS}" "${GPU_LIST}"

echo "[3/3] Summarize and plot comparison"
cd "${ROOT_DIR}"
python output/CIFAR10/resnet18/summarize_accuracy_vs_budget.py
python deep-al/tools/plot_cifar10_compare_with_dinov2.py

echo "Pipeline finished."
