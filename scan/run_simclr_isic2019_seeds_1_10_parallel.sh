#!/usr/bin/env bash
set -euo pipefail

# Run SimCLR on ISIC2019 **train split only** for seeds 1..10.
# Uses the dedicated isic2019 dataset entry — no train/test merging.
# Results land in: scan/results/isic2019/pretext/
#
# Usage:
#   bash run_simclr_isic2019_seeds_1_10_parallel.sh [max_parallel_jobs]
# Example (2 seeds in parallel):
#   bash run_simclr_isic2019_seeds_1_10_parallel.sh 2

MAX_JOBS="${1:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SRC_ROOT="$(realpath "${SCRIPT_DIR}/../data/isic2019")"
LOG_DIR="${SCRIPT_DIR}/logs/simclr_isic2019_parallel"
CONFIG_ENV="configs/env.yml"
CONFIG_EXP="configs/pretext/simclr_isic2019.yml"

mkdir -p "${LOG_DIR}"

# Sanity-check that the train split exists
if [ ! -d "${SRC_ROOT}/train" ]; then
  echo "[ERROR] ISIC2019 train split not found at ${SRC_ROOT}/train"
  echo "        Run deep-al/tools/prepare_isic2019_from_kagglehub.py first."
  exit 1
fi

echo "[INFO] Using ISIC2019 train split: ${SRC_ROOT}/train"
echo "[INFO] SimCLR results will be saved to: ${SCRIPT_DIR}/results/isic2019/pretext/"

run_one_seed() {
  local seed="$1"
  local log_file="${LOG_DIR}/seed_${seed}.log"
  echo "[START] seed=${seed}  log=${log_file}"
  python simclr.py \
    --config_env "${CONFIG_ENV}" \
    --config_exp "${CONFIG_EXP}" \
    --seed "${seed}" > "${log_file}" 2>&1
  echo "[DONE ] seed=${seed}"
}

pids=()
for seed in $(seq 1 10); do
  run_one_seed "${seed}" &
  pids+=("$!")
  while [ "$(jobs -pr | wc -l)" -ge "${MAX_JOBS}" ]; do
    sleep 2
  done
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    fail=1
  fi
done

if [ "${fail}" -ne 0 ]; then
  echo "Some seed runs failed. Check logs in ${LOG_DIR}."
  exit 1
fi

echo "All ISIC2019 SimCLR runs (seed 1..10, train-only) finished."
