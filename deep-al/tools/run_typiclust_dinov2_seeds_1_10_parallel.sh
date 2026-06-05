#!/usr/bin/env bash
set -euo pipefail

# Parallel launcher for TypiClust-DINOv2 on CIFAR10 seeds 1..10.
#
# Usage:
#   bash run_typiclust_dinov2_seeds_1_10_parallel.sh [budget] [initial_size] [max_parallel_jobs] [gpu_list]
#
# Example:
#   bash run_typiclust_dinov2_seeds_1_10_parallel.sh 100 0 2 "0"

BUDGET="${1:-100}"
INITIAL_SIZE="${2:-0}"
MAX_JOBS="${3:-1}"
GPU_LIST="${4:-0}"

CFG="../configs/cifar10/al/RESNET18.yaml"
METHOD="typiclust_dinov2"
LOG_DIR="logs/typiclust_dinov2"
FEATURE_DIR="../../scan/results/cifar-10/dinov2"

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

  local train_feat="${SCRIPT_DIR}/${FEATURE_DIR}/features_seed${seed}.npy"
  local test_feat="${SCRIPT_DIR}/${FEATURE_DIR}/test_features_seed${seed}.npy"
  if [ ! -f "${train_feat}" ] || [ ! -f "${test_feat}" ]; then
    echo "[SKIP ] seed=${seed}: DINOv2 features not found."
    echo "        expected: ${train_feat}"
    return 0
  fi

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
    --seed "${seed}" \
    --linear_from_features > "${log_file}" 2>&1

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
  echo "Some TypiClust-DINOv2 runs failed. Check logs in ${LOG_DIR}."
  exit 1
fi

echo "All TypiClust-DINOv2 runs for seeds 1..10 finished."
