#!/usr/bin/env bash
set -euo pipefail

# Re-export DINOv2 CIFAR10 features from existing checkpoints (seed 1..10).
# This is fast and avoids retraining.
#
# Usage:
#   bash reextract_dinov2_cifar10_features_seeds_1_10.sh [max_parallel_jobs]

MAX_JOBS="${1:-2}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LOG_DIR="logs/dinov2_cifar10_reextract"
mkdir -p "${LOG_DIR}"

run_one_seed() {
  local seed="$1"
  local log_file="${LOG_DIR}/seed_${seed}.log"
  echo "[START] reextract seed=${seed} log=${log_file}"
  python dinov2_cifar10.py \
    --seed "${seed}" \
    --extract_only \
    --feature_subdir dinov2 > "${log_file}" 2>&1
  echo "[DONE ] reextract seed=${seed}"
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
  echo "Some re-extract runs failed. Check logs in ${LOG_DIR}."
  exit 1
fi

echo "All DINOv2 feature re-extractions finished."
