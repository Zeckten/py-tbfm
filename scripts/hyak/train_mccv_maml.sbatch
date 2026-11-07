#!/bin/bash
#SBATCH --job-name=tbfm-train-maml-zscore
#SBATCH --account=rao
#SBATCH --partition=gpu-rtx6k
#SBATCH --array=0-19%4
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=44G
#SBATCH --time=22:00:00
#SBATCH --output=logs/train_%A_%a.log
# Measured from the smoke test: 300 epochs took ~1270s (4.23s/epoch) on this
# rtx6k. Production uses the default 12001 epochs -> ~14.1h, x1.5 margin.

set -euo pipefail
mkdir -p logs

# Not $(dirname "$0") -- Slurm copies the submitted script into a spool dir,
# so $0 does not point at its location in the repo checkout.
source "${REPO_DIR:-$HOME/py-tbfm}/scripts/hyak/env.sh"

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
