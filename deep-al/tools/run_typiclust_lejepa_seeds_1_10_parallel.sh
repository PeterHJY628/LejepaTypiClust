#!/usr/bin/env bash
set -euo pipefail

# Parallel launcher for TypiClust-LeJEPA on seeds 1..10.
#
# Usage:
#   bash run_typiclust_lejepa_seeds_1_10_parallel.sh [budget] [initial_size] [max_parallel_jobs] [gpu_list]
#
# Example (single GPU, 1 job at a time):
#   bash run_typiclust_lejepa_seeds_1_10_parallel.sh 100 0 1 "0"
#
# Example (2 concurrent jobs on same GPU):
#   bash run_typiclust_lejepa_seeds_1_10_parallel.sh 100 0 2 "0"

BUDGET="${1:-100}"
INITIAL_SIZE="${2:-0}"
MAX_JOBS="${3:-1}"
GPU_LIST="${4:-0}"

CFG="../configs/cifar10/al/RESNET18.yaml"
METHOD="typiclust_lejepa"
LOG_DIR="logs/typiclust_lejepa"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p "${LOG_DIR}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [ "${#GPUS[@]}" -eq 0 ]; then
  echo "Invalid gpu_list: ${GPU_LIST}"
  exit 1
fi

run_one_seed() {
  local seed="$1"
  local gpu="$2"
  local exp_name="${METHOD}_seed${seed}_b${BUDGET}_init${INITIAL_SIZE}"
  local log_file="${LOG_DIR}/${exp_name}.log"

  echo "============================================================"
  echo "[START] method=${METHOD} seed=${seed} gpu=${gpu}"
  echo "exp=${exp_name}"
  echo "log=${log_file}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${gpu}" \
  python train_al.py \
    --cfg "${CFG}" \
    --al "${METHOD}" \
    --exp-name "${exp_name}" \
    --initial_size "${INITIAL_SIZE}" \
    --budget "${BUDGET}" \
    --seed "${seed}" > "${log_file}" 2>&1

  echo "[DONE ] ${exp_name}"
}

pids=()
idx=0
for SEED in $(seq 1 10); do
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
  echo "Some TypiClust-LeJEPA runs failed. Check logs in ${LOG_DIR}."
  exit 1
fi

echo "All TypiClust-LeJEPA runs for seeds 1..10 finished."
