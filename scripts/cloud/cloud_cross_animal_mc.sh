#!/bin/bash
# Run train_cross_animal_mc.sh then tta_cross_animal_mc.sh on a GCP VM, with
# incremental GCS rsync throughout.
#
# Usage (on GCP VM, inside tmux):
#   BUCKET=py-tbfm-danmuir bash scripts/cloud/cloud_cross_animal_mc.sh
#
# To reproduce exact session sets from a prior run:
#   SESSIONS_FILE=cross_animal_mc_<ts>/sessions.json \
#   BUCKET=py-tbfm-danmuir bash scripts/cloud/cloud_cross_animal_mc.sh
#
# Env:
#   BUCKET          required
#   NUM_FOLDS       default 4
#   NUM_GPUS        default 8
#   SESSIONS_FILE   optional — JSON from a prior run for reproducibility
#   AUTO_SHUTDOWN   default stop — "stop" halts VM, "delete" tears down, "0" leaves running
#
# Launch VM:
#   bash scripts/cloud/launch_vm.sh cross-animal-mc 8
#   gcloud compute ssh cross-animal-mc --zone=us-east4-c --project=$PROJECT
#   cd /opt/py-tbfm && source .venv/bin/activate
#   tmux new -s run
#   BUCKET=py-tbfm-danmuir bash scripts/cloud/cloud_cross_animal_mc.sh

set -euo pipefail

BUCKET="${BUCKET:?BUCKET env var required}"
NUM_FOLDS="${NUM_FOLDS:-4}"
NUM_GPUS="${NUM_GPUS:-8}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-stop}"
SESSIONS_FILE="${SESSIONS_FILE:-}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="cross_animal_mc_${TIMESTAMP}"

export TBFM_DATA_DIR="${TBFM_DATA_DIR:-/mnt/data}"
export NUM_FOLDS
export NUM_GPUS

cd "${REPO_ROOT}"

echo "============================================================"
echo "Cross-animal MC training + TTA"
echo "  output:        ${OUTPUT_BASE}"
echo "  num_folds:     ${NUM_FOLDS}"
echo "  num_gpus:      ${NUM_GPUS}"
echo "  sessions file: ${SESSIONS_FILE:-<random seeds>}"
echo "  data dir:      ${TBFM_DATA_DIR}"
echo "  bucket:        gs://${BUCKET}"
echo "============================================================"

# --- Training ---
BUCKET="${BUCKET}" bash "${REPO_ROOT}/scripts/cloud/with_incremental_rsync.sh" \
    "${OUTPUT_BASE}" \
    bash scripts/train_cross_animal_mc.sh "${SESSIONS_FILE}" "${OUTPUT_BASE}"

echo ""
echo "Training complete. Syncing -> gs://${BUCKET}/models/${OUTPUT_BASE}/"
gsutil -m rsync -r "${OUTPUT_BASE}/" "gs://${BUCKET}/models/${OUTPUT_BASE}/"

# --- TTA ---
echo ""
echo "============================================================"
echo "Starting TTA (support size: 1000)"
echo "============================================================"
BUCKET="${BUCKET}" bash "${REPO_ROOT}/scripts/cloud/with_incremental_rsync.sh" \
    "${OUTPUT_BASE}" \
    bash scripts/tta_cross_animal_mc.sh "${OUTPUT_BASE}"

echo ""
echo "TTA complete. Syncing -> gs://${BUCKET}/models/${OUTPUT_BASE}/"
gsutil -m rsync -r "${OUTPUT_BASE}/" "gs://${BUCKET}/models/${OUTPUT_BASE}/"

# --- Auto-shutdown ---
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
            --project="$(_vm_meta project/project-id)" \
            --discard-local-ssd=true --quiet
        ;;
    0|"") echo "AUTO_SHUTDOWN=0 — leaving VM running" ;;
    *)    echo "WARNING: unknown AUTO_SHUTDOWN=${AUTO_SHUTDOWN}; leaving VM running" ;;
esac
