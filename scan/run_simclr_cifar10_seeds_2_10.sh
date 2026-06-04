#!/usr/bin/env bash
set -euo pipefail

# Run CIFAR-10 SimCLR for seeds 2..10.
# Usage:
#   bash run_simclr_cifar10_seeds_2_10.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CONFIG_ENV="configs/env.yml"
CONFIG_EXP="configs/pretext/simclr_cifar10.yml"
LOG_DIR="logs/simclr_cifar10"

mkdir -p "${LOG_DIR}"

for SEED in $(seq 2 10); do
  LOG_FILE="${LOG_DIR}/seed_${SEED}.log"
  echo "============================================================"
  echo "Running SimCLR CIFAR-10, seed=${SEED}"
  echo "Log: ${LOG_FILE}"
  echo "============================================================"

  python simclr.py \
    --config_env "${CONFIG_ENV}" \
    --config_exp "${CONFIG_EXP}" \
    --seed "${SEED}" 2>&1 | tee "${LOG_FILE}"
done

echo "All SimCLR runs for seeds 2..10 finished."
