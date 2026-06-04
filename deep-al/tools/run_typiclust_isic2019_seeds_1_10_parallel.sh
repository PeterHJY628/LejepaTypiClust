#!/usr/bin/env bash
set -euo pipefail

# TypiClust (SimCLR features) on ISIC2019, seeds 1..10, bounded parallelism.
#
# Prerequisites:
#   SimCLR features must exist at:
#   scan/results/imagenet/pretext/features_seed<N>.npy
#   scan/results/imagenet/pretext/test_features_seed<N>.npy
#
# Usage:
#   bash run_typiclust_isic2019_seeds_1_10_parallel.sh \
#       [budget] [initial_size] [max_parallel_jobs] [feature_seed]
#
# Arguments:
#   budget           — labeled samples added per AL round  (default: 100)
#   initial_size     — size of the initial labeled pool    (default: 0)
#   max_parallel_jobs— max concurrent jobs                 (default: 1)
#   feature_seed     — if set, ALL AL seeds share this one .npy embedding
#                      (e.g. "1" means every seed uses features_seed1.npy)
#                      leave empty or omit to use each seed's own .npy
#
# Examples:
#   # each seed uses its own features_seedN.npy
#   bash run_typiclust_isic2019_seeds_1_10_parallel.sh 100 0 1
#
#   # all seeds share features_seed1.npy (only seed1 SimCLR required)
#   bash run_typiclust_isic2019_seeds_1_10_parallel.sh 100 0 1 1

BUDGET="${1:-100}"
INITIAL_SIZE="${2:-0}"
MAX_JOBS="${3:-1}"
FEATURE_SEED="${4:-}"          # empty = each seed uses its own .npy

CFG="../configs/isic2019/al/RESNET18.yaml"
METHOD="typiclust_rp"
LOG_DIR="logs/typiclust_isic2019"
FEATURE_DIR="../../scan/results/imagenet/pretext"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p "${LOG_DIR}"

if [ -n "${FEATURE_SEED}" ]; then
  echo "[INFO] Fixed feature_seed=${FEATURE_SEED}: all AL seeds will use features_seed${FEATURE_SEED}.npy"
else
  echo "[INFO] Each AL seed will use its own features_seedN.npy"
fi

run_one_seed() {
  local seed="$1"

  # Determine which .npy to check/use
  local fseed="${FEATURE_SEED:-${seed}}"
  local train_feat="${SCRIPT_DIR}/${FEATURE_DIR}/features_seed${fseed}.npy"
  local test_feat="${SCRIPT_DIR}/${FEATURE_DIR}/test_features_seed${fseed}.npy"

  if [ ! -f "${train_feat}" ] || [ ! -f "${test_feat}" ]; then
    echo "[SKIP ] seed=${seed}: feature file for fseed=${fseed} not found."
    echo "        expected: ${train_feat}"
    return 0
  fi

  local exp_name="isic2019_${METHOD}_seed${seed}_b${BUDGET}_init${INITIAL_SIZE}"
  # Append feature_seed suffix when a fixed seed is used, to avoid collision
  if [ -n "${FEATURE_SEED}" ]; then
    exp_name="${exp_name}_fseed${FEATURE_SEED}"
  fi
  local log_file="${LOG_DIR}/${exp_name}.log"

  echo "============================================================"
  echo "[START] method=${METHOD}  seed=${seed}  feature_seed=${fseed}"
  echo "        exp=${exp_name}"
  echo "        log=${log_file}"
  echo "============================================================"

  # Build optional --feature_seed argument
  local fseed_arg=""
  if [ -n "${FEATURE_SEED}" ]; then
    fseed_arg="--feature_seed ${FEATURE_SEED}"
  fi

  python train_al.py \
    --cfg "${CFG}" \
    --al "${METHOD}" \
    --exp-name "${exp_name}" \
    --initial_size "${INITIAL_SIZE}" \
    --budget "${BUDGET}" \
    --seed "${seed}" \
    --linear_from_features \
    ${fseed_arg} \
    > "${log_file}" 2>&1

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
  echo "Some runs failed. Check logs in ${LOG_DIR}."
  exit 1
fi

echo "All ISIC2019 TypiClust runs (seed 1..10) finished."
