#!/bin/bash
#SBATCH --job-name=tbfm-tta-maml-zscore
#SBATCH --account=rao
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=44G
#SBATCH --time=02:00:00
#SBATCH --output=logs/tta_%A_%a.log
# NOTE: --time is a placeholder, sized for the largest support (5000) with a
# 1.5x margin over the smoke test's measured duration -- set the real value
# before submitting the full array.
#
# NOTE: --array is intentionally not set here. The task count depends on the
# real held-out-session count from tta_manifest.csv (expected 1600 = 20
# folds x 20 sessions x 4 supports, but verify against the manifest's own
# printed count rather than assuming). Submit with, e.g.:
#   N=$(($(tail -n +2 "$OUT/tta_manifest.csv" | wc -l)))
#   sbatch --array=0-$((N-1))%4 --export=OUT="$OUT" \
#       --dependency=afterok:<training_array_job_id> \
#       scripts/hyak/tta_mccv_maml.sbatch

set -euo pipefail
mkdir -p logs

# Not $(dirname "$0") -- Slurm copies the submitted script into a spool dir,
# so $0 does not point at its location in the repo checkout.
source "${REPO_DIR:-$HOME/py-tbfm}/scripts/hyak/env.sh"

OUT="${OUT:?Set OUT to the output base dir}"
MANIFEST="${OUT}/tta_manifest.csv"

TASK_ID="${SLURM_ARRAY_TASK_ID}"

read -r FOLD SESSION SUPPORT <<< "$(python -c "
import csv
with open('${MANIFEST}') as f:
    for row in csv.DictReader(f):
        if int(row['task_id']) == ${TASK_ID}:
            print(row['fold'], row['session'], row['support'])
            break
")"

if [ -z "${FOLD:-}" ]; then
    echo "ERROR: task_id ${TASK_ID} not found in ${MANIFEST}"
    exit 1
fi

FOLD_IDX="${FOLD#fold}"
FOLD_DIR="${OUT}/fold${FOLD_IDX}"
if [ -f "${FOLD_DIR}/best/tbfm.torch" ]; then
    MODEL_PATH="${FOLD_DIR}/best"
else
    MODEL_PATH="${FOLD_DIR}"
fi
OUT_DIR="${FOLD_DIR}/tta_results_${SUPPORT}"

echo "Task ${TASK_ID}: fold=${FOLD} session=${SESSION} support=${SUPPORT}"
echo "Model: ${MODEL_PATH} -> Output: ${OUT_DIR}"

# --cuda-device 0: inside a Slurm --gpus=1 allocation only one GPU is ever
# visible, always as index 0. tta_testing.py's own default ("1") would try to
# select a second device that doesn't exist here and crash every task.
python tta_testing.py \
    --model-paths "${FOLD}:${MODEL_PATH}" \
    --cuda-device 0 \
    --support-sizes "${SUPPORT}" \
    --adapt-session "${SESSION}" \
    --tta-epochs 7001 \
    --output-dir "${OUT_DIR}" \
    --unfreeze-bases \
    --progressive-unfreezing-threshold 0 \
    --no-plot-display

echo "Task ${TASK_ID}: complete"
