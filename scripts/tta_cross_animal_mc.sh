#!/bin/bash
# Run TTA on all Monte Carlo cross-animal folds.
# For each fold, adapts held-out sessions (12G + 8J) using both G_model and J_model.
# tta_testing.py automatically intersects their held-out sets to find the 20 shared sessions.
#
# Usage: bash scripts/tta_cross_animal_mc.sh <cross_animal_mc_dir>

set -euo pipefail

export TBFM_DATA_DIR="${TBFM_DATA_DIR:-/mnt/data}"

CROSS_ANIMAL_DIR="${1:-}"
if [ -z "${CROSS_ANIMAL_DIR}" ]; then
    echo "Usage: $0 <cross_animal_mc_dir>"
    exit 1
fi
CROSS_ANIMAL_DIR=$(realpath "${CROSS_ANIMAL_DIR}")

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RETRY="${REPO_ROOT}/scripts/cloud/retry_on_preemption.sh"
SESSIONS_JSON="${CROSS_ANIMAL_DIR}/sessions.json"

if [ ! -f "${SESSIONS_JSON}" ]; then
    echo "ERROR: sessions.json not found in ${CROSS_ANIMAL_DIR}"
    exit 1
fi

NUM_FOLDS=$(python -c "import json; print(len(json.load(open('${SESSIONS_JSON}'))))")

SUPPORT_SIZE="${SUPPORT_SIZE:-1000}"
STANDARD_TTA_FLAGS="--unfreeze-bases --progressive-unfreezing-threshold 0 \
--tta-epochs 7001 --support-sizes ${SUPPORT_SIZE} --max-adapt-sessions 40"

# --auto-gpu-slots: don't hardcode how many workers go on each GPU. Instead,
# tta_testing.py checks real GPU occupancy at startup AND periodically during
# the run, fills whatever slots are actually free (up to MAX_JOBS_PER_GPU),
# and treats any other user's process using more than OTHER_USER_GB_THRESHOLD
# as occupying one of our slots. MAX_JOBS_PER_GPU=1: a careful 1/2/3 sweep
# showed conc=2 is memory-safe (0 failures, 18.3GB/24.6GB peak) but gives
# essentially NO speedup over conc=1 (481s wall vs 464s to drain the same
# job queue -- this job is contention-bound on something other than raw GPU
# compute, so packing more workers per GPU doesn't help throughput here).
# conc=3 collapses (97.5% GPU memory, real OOM failures). So 1/GPU is both
# the fastest-or-tied AND the safest setting -- do not raise this without
# re-measuring wall-clock, not just checking for OOM; a prior run at 8/GPU
# on this same job shape produced cascading OOM failures.
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1}"
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"
OTHER_USER_GB_THRESHOLD="${OTHER_USER_GB_THRESHOLD:-5}"

echo "Cross-animal MC TTA"
echo "  Dir:       ${CROSS_ANIMAL_DIR}"
echo "  NUM_FOLDS: ${NUM_FOLDS}"
echo ""

for i in $(seq 0 $((NUM_FOLDS - 1))); do
    FOLD_DIR="${CROSS_ANIMAL_DIR}/fold${i}"
    G_MODEL="${FOLD_DIR}/G_model"
    J_MODEL="${FOLD_DIR}/J_model"
    OUT_DIR="${FOLD_DIR}/tta_results_${SUPPORT_SIZE}"

    if [ ! -d "${G_MODEL}" ] || [ ! -d "${J_MODEL}" ]; then
        echo "WARNING: fold${i} models not found, skipping"
        continue
    fi

    echo "=========================================="
    echo "TTA fold${i}"
    echo "  G_model: ${G_MODEL}"
    echo "  J_model: ${J_MODEL}"
    echo "  Output:  ${OUT_DIR}"
    echo "=========================================="

    bash "${RETRY}" python -u tta_testing.py \
        --model-paths "G_model:${G_MODEL}/best" "J_model:${J_MODEL}/best" \
        --output-dir "${OUT_DIR}" \
        --use-multi-gpu --gpu-ids ${GPU_CANDIDATES} \
        --auto-gpu-slots --max-jobs-per-gpu "${MAX_JOBS_PER_GPU}" \
        --other-user-gb-threshold "${OTHER_USER_GB_THRESHOLD}" \
        ${STANDARD_TTA_FLAGS} \
        --coadapt-tta \
        2>&1 | tee "${CROSS_ANIMAL_DIR}/tta_fold${i}_support${SUPPORT_SIZE}.log"
done

echo ""
echo "Cross-animal MC TTA complete. Results in: ${CROSS_ANIMAL_DIR}"
