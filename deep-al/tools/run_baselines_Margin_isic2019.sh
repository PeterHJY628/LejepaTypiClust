#!/usr/bin/env bash
set -euo pipefail

# Run RANDOM baseline on ISIC2019 with the same AL setup across seeds.
# Usage:
#   bash run_baselines_random_isic2019.sh [budget] [initial_size]
# Example:
#   bash run_baselines_random_isic2019.sh 100 0

BUDGET="${1:-100}"
INITIAL_SIZE="${2:-0}"

CFG="../configs/isic2019/al/RESNET18.yaml"
METHODS=(margin)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

for SEED in $(seq 1 10); do
  for method in "${METHODS[@]}"; do
    exp_name="isic2019_${method}_seed${SEED}_b${BUDGET}_init${INITIAL_SIZE}"
    echo "============================================================"
    echo "Running method: ${method} (ISIC2019)"
    echo "Seed: ${SEED}"
    echo "Experiment: ${exp_name}"
    echo "Config: ${CFG}"
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

echo "All ISIC2019 random baseline runs finished."
