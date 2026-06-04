#!/usr/bin/env bash
set -euo pipefail

# Parallel run: CIFAR-10 SimCLR for seeds 2..10.
# Usage:
#   bash run_simclr_cifar10_seeds_2_10_parallel.sh [max_parallel_jobs]
# Example:
#   bash run_simclr_cifar10_seeds_2_10_parallel.sh 2

MAX_JOBS="${1:-2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CONFIG_ENV="configs/env.yml"
CONFIG_EXP="configs/pretext/simclr_cifar10.yml"
LOG_DIR="logs/simclr_cifar10_parallel"

mkdir -p "${LOG_DIR}"

run_one_seed() {
  local seed="$1"
  local log_file="${LOG_DIR}/seed_${seed}.log"
  echo "[START] seed=${seed} log=${log_file}"
  python simclr.py \
    --config_env "${CONFIG_ENV}" \
    --config_exp "${CONFIG_EXP}" \
    --seed "${seed}" > "${log_file}" 2>&1
  echo "[DONE ] seed=${seed}"
}

pids=()
for SEED in $(seq 2 10); do
  run_one_seed "${SEED}" &
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

echo "All parallel SimCLR runs for seeds 2..10 finished."
