#!/bin/bash
# Run TTA on all Monte Carlo cross-animal folds.
# For each fold, adapts held-out sessions (12G + 8J) using both G_model and J_model.
# tta_testing.py automatically intersects their held-out sets to find the 20 shared sessions.
#
# Usage: bash scripts/tta_cross_animal_mc.sh <cross_animal_mc_dir>

set -euo pipefail

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

STANDARD_TTA_FLAGS="--unfreeze-bases --progressive-unfreezing-threshold 0 \
--tta-epochs 7001 --support-sizes 1000 --max-adapt-sessions 40"

echo "Cross-animal MC TTA"
echo "  Dir:       ${CROSS_ANIMAL_DIR}"
echo "  NUM_FOLDS: ${NUM_FOLDS}"
echo ""

for i in $(seq 0 $((NUM_FOLDS - 1))); do
    FOLD_DIR="${CROSS_ANIMAL_DIR}/fold${i}"
    G_MODEL="${FOLD_DIR}/G_model"
    J_MODEL="${FOLD_DIR}/J_model"
    OUT_DIR="${FOLD_DIR}/tta_results"

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
        --model-paths "G_model:${G_MODEL}" "J_model:${J_MODEL}" \
        --output-dir "${OUT_DIR}" \
        --cuda-device 0 \
        ${STANDARD_TTA_FLAGS} \
        2>&1 | tee "${CROSS_ANIMAL_DIR}/tta_fold${i}.log"
done

echo ""
echo "Cross-animal MC TTA complete. Results in: ${CROSS_ANIMAL_DIR}"
