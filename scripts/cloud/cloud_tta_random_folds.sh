#!/bin/bash
# Run tta_random_folds.sh on a GCP VM, pulling trained models and any prior
# TTA results from GCS first so interrupted runs resume cleanly.
#
# Usage (on GCP VM, inside tmux):
#   BUCKET=py-tbfm-danmuir FOLDS_DIR=random_folds_20260530_213515 \
#       bash scripts/cloud/cloud_tta_random_folds.sh
#
# Env:
#   BUCKET        required
#   FOLDS_DIR     required — name of the fold output dir (not a full path)
#   GPU_IDS       default "0 1 2 3 4 5 6 7"
#   FOLD_ORDER    default "forward" — set "reverse" for backward sweep
#   AUTO_SHUTDOWN default stop

set -euo pipefail

BUCKET="${BUCKET:?BUCKET env var required}"
FOLDS_DIR="${FOLDS_DIR:?FOLDS_DIR env var required}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
FOLD_ORDER="${FOLD_ORDER:-forward}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-stop}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export TBFM_DATA_DIR="${TBFM_DATA_DIR:-/mnt/data}"

cd "${REPO_ROOT}"

echo "============================================================"
echo "Random folds TTA"
echo "  folds dir:  ${FOLDS_DIR}"
echo "  fold order: ${FOLD_ORDER}"
echo "  gpu ids:    ${GPU_IDS}"
echo "  data dir:   ${TBFM_DATA_DIR}"
echo "  bucket:     gs://${BUCKET}"
echo "============================================================"

# Pull trained fold models from GCS if not already present
MODEL_SRC="gs://${BUCKET}/models/${FOLDS_DIR}/"
if [ ! -d "${FOLDS_DIR}/fold0" ]; then
    echo "Syncing fold models from ${MODEL_SRC}..."
    mkdir -p "${FOLDS_DIR}"
    gsutil -m rsync -r "${MODEL_SRC}" "${FOLDS_DIR}/"
else
    echo "Fold models already present, skipping model sync."
fi

# Pull prior TTA results from the shared models path (covers results from any VM)
PRIOR_TTA=$(gsutil ls "gs://${BUCKET}/models/${FOLDS_DIR}/" 2>/dev/null \
    | grep "tta_results" | sort | tail -1 || true)
if [ -n "${PRIOR_TTA}" ]; then
    TTA_DEST_NAME=$(basename "${PRIOR_TTA%/}")
    echo "Syncing prior TTA results from ${PRIOR_TTA}..."
    mkdir -p "${FOLDS_DIR}/${TTA_DEST_NAME}"
    gsutil -m rsync -r "${PRIOR_TTA}" "${FOLDS_DIR}/${TTA_DEST_NAME}/"
else
    echo "No prior TTA results found, starting fresh."
fi

# Determine resume dir if prior results exist
RESUME_ARG=""
TTA_RESULTS_DIR=$(ls -dt "${FOLDS_DIR}"/tta_results_* 2>/dev/null | head -1 || true)
if [ -n "${TTA_RESULTS_DIR}" ]; then
    echo "Resuming from: ${TTA_RESULTS_DIR}"
    RESUME_ARG="${TTA_RESULTS_DIR}"
fi

# Run TTA with incremental GCS rsync
WATCH_DIR="${TTA_RESULTS_DIR:-${FOLDS_DIR}/tta_results_pending}"
echo ""
echo "Starting TTA..."

GPU_IDS="${GPU_IDS}" FOLD_ORDER="${FOLD_ORDER}" \
BUCKET="${BUCKET}" bash "${REPO_ROOT}/scripts/cloud/with_incremental_rsync.sh" \
    "${WATCH_DIR}" \
    bash scripts/tta_random_folds.sh "${FOLDS_DIR}" ${RESUME_ARG}

# Final sync to models path so everything is in one place
echo ""
echo "Final rsync -> gs://${BUCKET}/models/${FOLDS_DIR}/"
gsutil -m rsync -r "${FOLDS_DIR}/" "gs://${BUCKET}/models/${FOLDS_DIR}/"

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
            --project="$(_vm_meta project/project-id)" \
            --discard-local-ssd=true --quiet
        ;;
    0|"") echo "AUTO_SHUTDOWN=0 — leaving VM running" ;;
    *)    echo "WARNING: unknown AUTO_SHUTDOWN=${AUTO_SHUTDOWN}; leaving VM running" ;;
esac
