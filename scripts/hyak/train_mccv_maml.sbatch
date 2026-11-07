#!/bin/bash
#SBATCH --job-name=tbfm-train-maml-zscore
#SBATCH --account=rao
#SBATCH --partition=gpu-rtx6k
#SBATCH --array=0-19%4
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=44G
#SBATCH --time=06:00:00
#SBATCH --output=logs/train_%A_%a.log
# NOTE: --time is a placeholder (3.6h estimate x 1.5 margin). Set the real
# value from the smoke test's measured fold0 duration before submitting the
# full array.

set -euo pipefail
mkdir -p logs

source "$(dirname "$0")/env.sh"

OUT="${OUT:?Set OUT to the output base dir, e.g. --export=OUT=/gscratch/scrubbed/danmuir/random_folds_maml_zscore_<ts>}"
SESSIONS_FILE="scripts/random_folds_sessions.json"

FOLD="${SLURM_ARRAY_TASK_ID}"
FOLD_OUT="${OUT}/fold${FOLD}"

SESSIONS=$(python -c "
import json
folds = json.load(open('${SESSIONS_FILE}'))
key = 'fold${FOLD}'
print(','.join(folds[key]))
")

echo "Fold ${FOLD}: training -> ${FOLD_OUT}"
echo "Held-in sessions: ${SESSIONS}"

python tma_standalone.py \
    100 20 0 false 16 5000 true \
    --latent-dim 96 \
    --batch-size-per-session 500 \
    --out-dir "${FOLD_OUT}" \
    --held-in-sessions ${SESSIONS}

echo "Fold ${FOLD}: training complete"
