#!/usr/bin/env bash
set -euo pipefail

# Run baseline AL methods with the same config and hyperparameters.
# Usage:
#   bash run_baselines.sh [budget] [initial_size]
# Example:
#   bash run_baselines.sh 100 0

BUDGET="${1:-100}"
INITIAL_SIZE="${2:-0}"

CFG="../configs/cifar10/al/RESNET18.yaml"
METHODS=(uncertainty)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

for SEED in $(seq 1 10); do
  for method in "${METHODS[@]}"; do
    exp_name="${method}_seed${SEED}_b${BUDGET}_init${INITIAL_SIZE}"
    echo "============================================================"
    echo "Running method: ${method}"
    echo "Seed: ${SEED}"
    echo "Experiment: ${exp_name}"
    echo "============================================================"

    python train_al.py \
      --cfg "${CFG}" \
      --al "${method}" \
      --exp-name "${exp_name}" \
      --initial_size "${INITIAL_SIZE}" \
      --budget "${BUDGET}" \
      --seed "${SEED}"
  done
done

echo "All baseline runs finished."
