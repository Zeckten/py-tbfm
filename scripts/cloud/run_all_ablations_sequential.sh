#!/bin/bash
# Run all ablations sequentially on a single multi-GPU VM, dispatching sessions
# across all available GPUs WITHIN each ablation. Designed for g4-standard-384
# (8x RTX PRO 6000) or a2-highgpu-8g (8x A100), but works with fewer GPUs too.
#
# Each ablation gets its own output dir; the wrapper rsyncs to GCS as sessions
# finish. After all ablations succeed, does a final comprehensive sync and
# (optionally) deletes the VM to stop billing.
#
# Usage (on a GCP VM):
#   BUCKET=my-bucket bash scripts/cloud/run_all_ablations_sequential.sh <support_size>
#
# Env:
#   BUCKET           required — GCS bucket for ablation models + result rsync
#   GPU_IDS          default "0 1 2 3 4 5 6 7" — passed to tta_testing.py multi-gpu
#   AUTO_SHUTDOWN    default 1 — delete VM on success. Set to 0 to keep VM alive.
#   ABLATIONS        default all 7 — space-separated list of ablation names
#
# Example:
#   BUCKET=py-tbfm-danmuir bash scripts/cloud/run_all_ablations_sequential.sh 5000

set -euo pipefail

SUPPORT="${1:?usage: $0 <support_size>}"
BUCKET="${BUCKET:?BUCKET env var required}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-1}"
ABLATIONS="${ABLATIONS:-no_ae_recon with_ortho no_rest no_l2 zscore_norm no_tanh no_adapt_ae}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUNNER="${REPO_ROOT}/scripts/cloud/run_ablation_tta.sh"
WRAPPER="${REPO_ROOT}/scripts/cloud/with_incremental_rsync.sh"
RETRY="${REPO_ROOT}/scripts/cloud/retry_on_preemption.sh"
WORK_DIR="${WORK_DIR:-${REPO_ROOT}/ablations_remote}"
LOG_DIR="${WORK_DIR}/sequential_logs"
mkdir -p "${LOG_DIR}"

# These flags are exported into the env for tta_ablations.sh to pick up.
export GPU_FLAGS_OVERRIDE="--use-multi-gpu --gpu-ids ${GPU_IDS}"
export TBFM_DATA_DIR="${TBFM_DATA_DIR:-/mnt/data}"
export BUCKET

cd "${REPO_ROOT}"

echo "============================================================"
echo "Sequential ablation TTA"
echo "  ablations:     ${ABLATIONS}"
echo "  support:       ${SUPPORT}"
echo "  GPU_IDS:       ${GPU_IDS}"
echo "  WORK_DIR:      ${WORK_DIR}"
echo "  TBFM_DATA_DIR: ${TBFM_DATA_DIR}"
echo "  AUTO_SHUTDOWN: ${AUTO_SHUTDOWN}"
echo "============================================================"

declare -a FAILED=()

for abl in ${ABLATIONS}; do
    echo ""
    echo "============================================================"
    echo "Starting ablation: ${abl}  ($(date))"
    echo "============================================================"
    log="${LOG_DIR}/${abl}.log"
    if bash "${RUNNER}" "${abl}" "${SUPPORT}" 2>&1 | tee "${log}"; then
        echo "[OK] ${abl}"
    else
        rc=$?
        echo "[FAIL] ${abl} (exit ${rc})"
        FAILED+=("${abl}")
    fi
done

# Final comprehensive sync. Push everything under WORK_DIR (less the heavy
# session-data subdirs which we never touched on this VM) to GCS.
VM_NAME="$(hostname)"
FINAL_DEST="gs://${BUCKET}/results/${VM_NAME}/"
echo ""
echo "============================================================"
echo "Final rsync ${WORK_DIR} -> ${FINAL_DEST}"
echo "============================================================"
gsutil -m rsync -r \
    -x '.*\.venv/.*|.*__pycache__/.*' \
    "${WORK_DIR}" "${FINAL_DEST}" || true

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All ablations completed successfully."
else
    echo "${#FAILED[@]} ablation(s) failed: ${FAILED[*]}"
    echo "VM kept alive for debugging. Run teardown_vm.sh manually when done."
    exit 1
fi

# Auto-shutdown only on full success.
if [ "${AUTO_SHUTDOWN}" = "1" ]; then
    echo ""
    echo "============================================================"
    echo "AUTO_SHUTDOWN=1: deleting VM ${VM_NAME} in 60s (Ctrl-C to abort)"
    echo "============================================================"
    sleep 60
    # The VM has --scopes=cloud-platform from launch_vm.sh, so it can self-delete.
    # Fetch zone from metadata server (ZONE env var may not be set on the VM).
    ZONE_FULL=$(curl -s -H "Metadata-Flavor: Google" \
        http://metadata.google.internal/computeMetadata/v1/instance/zone || echo "")
    ZONE=$(basename "${ZONE_FULL}")
    PROJECT=$(curl -s -H "Metadata-Flavor: Google" \
        http://metadata.google.internal/computeMetadata/v1/project/project-id || echo "")
    if [ -z "${ZONE}" ] || [ -z "${PROJECT}" ]; then
        echo "ERROR: couldn't resolve zone/project from metadata; skipping self-delete"
        exit 0
    fi
    echo "Self-deleting: gcloud compute instances delete ${VM_NAME} --zone=${ZONE} --project=${PROJECT}"
    gcloud compute instances delete "${VM_NAME}" \
        --zone="${ZONE}" --project="${PROJECT}" --quiet
fi
