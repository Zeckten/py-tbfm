#!/bin/bash
# Within-session calibration draw sensitivity sweep.
# For a fixed model and fixed set of held-out sessions, draws K different
# random calibration sets of size SUPPORT_SIZE and runs TTA each time.
# Measures variance in test R² across draws to isolate inner-loop sampling
# noise from session-level variability.
#
# Usage:
#   bash scripts/tta_support_sensitivity.sh <fold_dir> <output_base> <session1> [session2 ...]
#
# Env:
#   NUM_DRAWS      default 20
#   SUPPORT_SIZE   default 1000
#   TBFM_DATA_DIR  default /mnt/data
#
# GCP (on VM, inside tmux):
#   BUCKET=py-tbfm-danmuir \
#   FOLD_PATH=random_folds_20260530_213515/fold0 \
#   SESSIONS="MonkeyG_20150914_Session1_S1 MonkeyJ_20160426_Session1_S1" \
#   bash scripts/cloud/cloud_support_sensitivity.sh

set -euo pipefail

export TBFM_DATA_DIR="${TBFM_DATA_DIR:-/mnt/data}"

FOLD_DIR="${1:?Usage: $0 <fold_dir> <output_base> <session1> [session2 ...]}"
OUTPUT_BASE="${2:?Usage: $0 <fold_dir> <output_base> <session1> [session2 ...]}"
shift 2
SESSIONS=("$@")

if [ ${#SESSIONS[@]} -eq 0 ]; then
    echo "ERROR: at least one session ID required"
    exit 1
fi

NUM_DRAWS="${NUM_DRAWS:-20}"
SUPPORT_SIZE="${SUPPORT_SIZE:-1000}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RETRY="${REPO_ROOT}/scripts/cloud/retry_on_preemption.sh"

mkdir -p "${OUTPUT_BASE}"
LOG="${OUTPUT_BASE}/sensitivity.log"

echo "Support sensitivity sweep" | tee "${LOG}"
echo "  Fold:         ${FOLD_DIR}" | tee -a "${LOG}"
echo "  Output:       ${OUTPUT_BASE}" | tee -a "${LOG}"
echo "  Sessions:     ${SESSIONS[*]}" | tee -a "${LOG}"
echo "  NUM_DRAWS:    ${NUM_DRAWS}" | tee -a "${LOG}"
echo "  SUPPORT_SIZE: ${SUPPORT_SIZE}" | tee -a "${LOG}"
echo "" | tee -a "${LOG}"

for k in $(seq 0 $((NUM_DRAWS - 1))); do
    DRAW_DIR="${OUTPUT_BASE}/draw${k}"
    echo "=== Draw ${k}/${NUM_DRAWS} ===" | tee -a "${LOG}"

    bash "${RETRY}" python -u tta_testing.py \
        --model-paths "fold:${FOLD_DIR}/best" \
        --adapt-session "${SESSIONS[@]}" \
        --random-support --support-seed "${k}" \
        --support-sizes "${SUPPORT_SIZE}" \
        --use-multi-gpu --gpu-ids 0 1 2 3 4 5 6 7 \
        --output-dir "${DRAW_DIR}" \
        --unfreeze-bases --progressive-unfreezing-threshold 0 \
        --tta-epochs 7001 \
        2>&1 | tee "${OUTPUT_BASE}/draw${k}.log"

    # Log per-session R² for this draw
    python -c "
import glob, csv, sys
csvs = glob.glob('${DRAW_DIR}/tta_support_*_per_session.csv')
if not csvs:
    print('  draw${k}: no results found')
    sys.exit(0)
rows = list(csv.DictReader(open(csvs[0])))
for r in rows:
    print(f\"  draw${k}: {r['session_id']:<45}  R²={float(r['session_r2']):.4f}\")
" 2>/dev/null | tee -a "${LOG}"

    echo "" | tee -a "${LOG}"
done

echo "Sensitivity sweep complete. Output: ${OUTPUT_BASE}" | tee -a "${LOG}"
