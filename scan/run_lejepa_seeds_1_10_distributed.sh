#!/usr/bin/env bash
set -euo pipefail

# Distributed-style launcher for LeJEPA CIFAR10 seeds 1..10.
# It runs multiple seeds in parallel and assigns GPUs in round-robin.
#
# Usage:
#   bash run_lejepa_seeds_1_10_distributed.sh [max_parallel_jobs] [gpu_list] [epochs]
#
# Examples:
#   bash run_lejepa_seeds_1_10_distributed.sh
#   bash run_lejepa_seeds_1_10_distributed.sh 2 "0,1" 500
#
# Args:
#   max_parallel_jobs: max concurrent training jobs (default: 1)
#   gpu_list:          comma-separated GPU ids for round-robin (default: "0")
#   epochs:            training epochs passed to lejepa_cifar10.py (default: 500)

MAX_JOBS="${1:-1}"
GPU_LIST="${2:-0}"
EPOCHS="${3:-500}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LOG_DIR="logs/lejepa_cifar10_distributed"
mkdir -p "${LOG_DIR}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [ "${#GPUS[@]}" -eq 0 ]; then
  echo "Invalid gpu_list: ${GPU_LIST}"
  exit 1
fi

run_one_seed() {
  local seed="$1"
  local gpu="$2"
  local log_file="${LOG_DIR}/seed_${seed}.log"

  echo "[START] seed=${seed} gpu=${gpu} epochs=${EPOCHS} log=${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  python lejepa_cifar10.py --seed "${seed}" --epochs "${EPOCHS}" --feature_subdir "lejepa" > "${log_file}" 2>&1
  echo "[DONE ] seed=${seed} gpu=${gpu}"
}

pids=()
idx=0
for SEED in $(seq 2 10); do
  gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  run_one_seed "${SEED}" "${gpu}" &
  pids+=("$!")
  idx=$((idx + 1))

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
  echo "Some LeJEPA runs failed. Check logs in ${LOG_DIR}."
  exit 1
fi

echo "All LeJEPA runs for seeds 1..10 finished."
