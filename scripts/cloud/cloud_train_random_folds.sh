#!/bin/bash
# Run train_random_folds.sh then tta_random_folds.sh on a GCP VM, with
# incremental GCS rsync throughout so a spot preemption only loses the
# in-progress fold(s).
#
# Usage (on GCP VM, inside tmux):
#   BUCKET=py-tbfm-danmuir bash scripts/cloud/cloud_train_random_folds.sh
#
# To reproduce exact session sets from a prior run, pass the sessions JSON:
#   SESSIONS_FILE=scripts/random_folds_sessions.json \
#   BUCKET=py-tbfm-danmuir bash scripts/cloud/cloud_train_random_folds.sh
#
# Env:
#   BUCKET          required — GCS bucket for incremental + final result sync
#   NUM_GPUS        default 8 — training GPUs (must match VM GPU count)
#   GPU_IDS         default "0 1 2 3 4 5 6 7" — TTA GPU list (space-separated)
#   SESSIONS_FILE   optional — JSON with exact per-fold session lists
#   AUTO_SHUTDOWN   default stop — "stop" halts VM (disk kept), "delete" tears down,
#                   "0" leaves running

set -euo pipefail

BUCKET="${BUCKET:?BUCKET env var required}"
NUM_GPUS="${NUM_GPUS:-8}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-stop}"
SESSIONS_FILE="${SESSIONS_FILE:-}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="random_folds_${TIMESTAMP}"

export TBFM_DATA_DIR="${TBFM_DATA_DIR:-/mnt/data}"
export NUM_GPUS
export OUTPUT_BASE

cd "${REPO_ROOT}"

echo "============================================================"
echo "Random folds training + TTA"
echo "  output:        ${OUTPUT_BASE}"
echo "  train gpus:    ${NUM_GPUS}"
echo "  tta gpu ids:   ${GPU_IDS}"
echo "  sessions file: ${SESSIONS_FILE:-<random seeds>}"
echo "  data dir:      ${TBFM_DATA_DIR}"
echo "  bucket:        gs://${BUCKET}"
echo "============================================================"

# --- Training ---
BUCKET="${BUCKET}" bash "${REPO_ROOT}/scripts/cloud/with_incremental_rsync.sh" \
    "${OUTPUT_BASE}" \
    bash scripts/train_random_folds.sh ${SESSIONS_FILE}

echo ""
echo "Training complete. Syncing models -> gs://${BUCKET}/models/${OUTPUT_BASE}/"
gsutil -m rsync -r "${OUTPUT_BASE}/" "gs://${BUCKET}/models/${OUTPUT_BASE}/"

# --- TTA ---
TTA_OUTPUT="${OUTPUT_BASE}/tta_results_${TIMESTAMP}"
echo ""
echo "============================================================"
echo "Starting TTA (support sizes: 500 1000 2500 5000)"
echo "============================================================"
GPU_IDS="${GPU_IDS}" \
BUCKET="${BUCKET}" bash "${REPO_ROOT}/scripts/cloud/with_incremental_rsync.sh" \
    "${TTA_OUTPUT}" \
    bash scripts/tta_random_folds.sh "${OUTPUT_BASE}"

echo ""
echo "TTA complete. Syncing results -> gs://${BUCKET}/models/${OUTPUT_BASE}/"
gsutil -m rsync -r "${OUTPUT_BASE}/" "gs://${BUCKET}/models/${OUTPUT_BASE}/"

# Auto-shutdown
_vm_meta() { curl -s -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/$1"; }

case "${AUTO_SHUTDOWN}" in
    stop|1)
        echo ""; echo "AUTO_SHUTDOWN: stopping VM in 60s (Ctrl-C to abort)"
        sleep 60
        gcloud compute instances stop "$(hostname)" \
            --zone="$(basename "$(_vm_meta instance/zone)")" \
            --project="$(_vm_meta project/project-id)" \
            --discard-local-ssd=true --quiet
        ;;
    delete)
        echo ""; echo "AUTO_SHUTDOWN: deleting VM in 60s (Ctrl-C to abort)"
        sleep 60
        gcloud compute instances delete "$(hostname)" \
            --zone="$(basename "$(_vm_meta instance/zone)")" \
            --project="$(_vm_meta project/project-id)" --quiet \
            --discard-local-ssd=true
        ;;
    0|"")
        echo "AUTO_SHUTDOWN=0 — leaving VM running" ;;
    *)
        echo "WARNING: unknown AUTO_SHUTDOWN=${AUTO_SHUTDOWN}; leaving VM running" ;;
esac
