#!/usr/bin/env bash
set -euo pipefail

# Run DINOv2-style CIFAR10 pretraining for seeds 1..10.
# Usage:
#   bash run_dinov2_cifar10_seeds_1_10_parallel.sh [max_parallel_jobs]
#
# Example:
#   bash run_dinov2_cifar10_seeds_1_10_parallel.sh 2

MAX_JOBS="${1:-2}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LOG_DIR="logs/dinov2_cifar10_parallel"
mkdir -p "${LOG_DIR}"

run_one_seed() {
  local seed="$1"
  local log_file="${LOG_DIR}/seed_${seed}.log"
  echo "[START] seed=${seed} log=${log_file}"
  python dinov2_cifar10.py \
    --seed "${seed}" \
    --epochs 300 \
    --batch_size 512 \
    --num_workers 8 \
    --feature_subdir dinov2 > "${log_file}" 2>&1
  echo "[DONE ] seed=${seed}"
}

pids=()
for SEED in $(seq 1 10); do
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
  echo "Some DINOv2 runs failed. Check logs in ${LOG_DIR}."
  exit 1
fi

echo "All DINOv2 CIFAR10 runs for seeds 1..10 finished."
