#!/bin/bash
# Run tta_support_sensitivity.sh on a GCP VM.
# Downloads a single fold's best/ checkpoint from GCS, runs K-draw sensitivity
# sweep, syncs results back, auto-shuts down.
#
# Usage (on GCP VM, inside tmux):
#   BUCKET=py-tbfm-danmuir \
#   FOLD_PATH=random_folds_20260530_213515/fold0 \
#   SESSIONS="MonkeyG_20150914_Session1_S1 MonkeyJ_20160426_Session1_S1 MonkeyG_20150917_Session2_S1" \
#   bash scripts/cloud/cloud_support_sensitivity.sh
#
# Env:
#   BUCKET        required
#   FOLD_PATH     required — GCS-relative path to fold, e.g. random_folds_.../fold0
#   SESSIONS      required — space-separated held-out session IDs
#   NUM_DRAWS     default 20
#   SUPPORT_SIZE  default 1000
#   AUTO_SHUTDOWN default stop — "stop" halts VM, "delete" tears down, "0" leaves running
#
# Launch VM:
#   bash scripts/cloud/launch_vm.sh support-sensitivity 8
#   gcloud compute ssh support-sensitivity --zone=us-east4-c --project=$PROJECT
#   cd /opt/py-tbfm && source .venv/bin/activate
#   tmux new -s run
#   BUCKET=... FOLD_PATH=... SESSIONS="..." bash scripts/cloud/cloud_support_sensitivity.sh

set -euo pipefail

BUCKET="${BUCKET:?BUCKET env var required}"
FOLD_PATH="${FOLD_PATH:?FOLD_PATH env var required (e.g. random_folds_20260530_213515/fold0)}"
SESSIONS="${SESSIONS:?SESSIONS env var required (space-separated session IDs)}"
NUM_DRAWS="${NUM_DRAWS:-20}"
SUPPORT_SIZE="${SUPPORT_SIZE:-1000}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-stop}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FOLD_NAME=$(basename "${FOLD_PATH}")
RUN_NAME=$(basename "$(dirname "${FOLD_PATH}")")
OUTPUT_BASE="sensitivity_${RUN_NAME}_${FOLD_NAME}_${TIMESTAMP}"
LOCAL_FOLD="${OUTPUT_BASE}/fold_model"

export TBFM_DATA_DIR="${TBFM_DATA_DIR:-/mnt/data}"
export NUM_DRAWS
export SUPPORT_SIZE

cd "${REPO_ROOT}"

echo "============================================================"
echo "Support sensitivity sweep"
echo "  fold:         ${FOLD_PATH}"
echo "  sessions:     ${SESSIONS}"
echo "  num_draws:    ${NUM_DRAWS}"
echo "  support_size: ${SUPPORT_SIZE}"
echo "  output:       ${OUTPUT_BASE}"
echo "  data dir:     ${TBFM_DATA_DIR}"
echo "  bucket:       gs://${BUCKET}"
echo "============================================================"

# Download best/ checkpoint and hisi.torch (held-in sessions list, needed by tta_testing.py)
echo ""
echo "Downloading model checkpoint from gs://${BUCKET}/models/${FOLD_PATH}/ ..."
mkdir -p "${LOCAL_FOLD}/best"
gsutil -m rsync -r "gs://${BUCKET}/models/${FOLD_PATH}/best/" "${LOCAL_FOLD}/best/"
gsutil cp "gs://${BUCKET}/models/${FOLD_PATH}/hisi.torch" "${LOCAL_FOLD}/hisi.torch"
echo "Model ready at: ${LOCAL_FOLD}/"

# Run sensitivity sweep with incremental GCS rsync
echo ""
echo "Starting sensitivity sweep..."
BUCKET="${BUCKET}" bash "${REPO_ROOT}/scripts/cloud/with_incremental_rsync.sh" \
    "${OUTPUT_BASE}" \
    bash scripts/tta_support_sensitivity.sh "${LOCAL_FOLD}" "${OUTPUT_BASE}" ${SESSIONS}

# Final sync
echo ""
echo "Final rsync -> gs://${BUCKET}/models/${OUTPUT_BASE}/"
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
            --project="$(_vm_meta project/project-id)" \
            --discard-local-ssd=true --quiet
        ;;
    0|"") echo "AUTO_SHUTDOWN=0 — leaving VM running" ;;
    *)    echo "WARNING: unknown AUTO_SHUTDOWN=${AUTO_SHUTDOWN}; leaving VM running" ;;
esac
