#!/bin/bash
# Run TTA for a single ablation on a GCP VM. Pulls the trained ablation model
# from GCS, then invokes scripts/tta_ablations.sh restricted to that ablation.
# Wraps the TTA call in retry_on_preemption.sh so spot evictions get one
# automatic retry before requiring manual relaunch.
#
# Usage (on a GCP VM):
#   BUCKET=my-bucket bash scripts/cloud/run_ablation_tta.sh <ablation> <support_size>
#
# Examples:
#   bash scripts/cloud/run_ablation_tta.sh no_ae_recon 5000
#   bash scripts/cloud/run_ablation_tta.sh with_ortho 5000

set -euo pipefail

ABL="${1:?usage: $0 <ablation> <support_size>}"
SUPPORT="${2:?usage: $0 <ablation> <support_size>}"
BUCKET="${BUCKET:?BUCKET env var required}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RETRY="${REPO_ROOT}/scripts/cloud/retry_on_preemption.sh"
WRAPPER="${REPO_ROOT}/scripts/cloud/with_incremental_rsync.sh"
WORK_DIR="${WORK_DIR:-${REPO_ROOT}/ablations_remote}"

# Pull the trained ablation model from GCS.
mkdir -p "${WORK_DIR}/${ABL}"
echo "Syncing gs://${BUCKET}/ablations/${ABL}/ -> ${WORK_DIR}/${ABL}/"
gsutil -m rsync -r "gs://${BUCKET}/ablations/${ABL}/" "${WORK_DIR}/${ABL}/"

cd "${REPO_ROOT}"
export TBFM_DATA_DIR="${TBFM_DATA_DIR:-/mnt/data}"

# tta_ablations.sh writes per-session adapted models to ${WORK_DIR}/tta_${ABL}_${SUPPORT}/
# as each session finishes. The wrapper rsyncs that dir to GCS every ~2 min so
# spot evictions only cost the in-progress session.
WATCH_DIR="${WORK_DIR}/tta_${ABL}_${SUPPORT}"
echo "Running TTA: ablation=${ABL}, support=${SUPPORT}, work_dir=${WORK_DIR}"

# NOTE: We can't pass ABLATION_NAMES as a bash array env var (arrays don't
# survive being exported — they get stringified to literal "(name)" with parens).
# Use SINGLE_ABLATION as a regular string env var; tta_ablations.sh recognizes it.
SUPPORT_SIZE="${SUPPORT}" \
    SINGLE_ABLATION="${ABL}" \
    bash "${WRAPPER}" "${WATCH_DIR}" \
        "${RETRY}" bash scripts/tta_ablations.sh "${WORK_DIR}"

# Sanity check: verify at least one session-result CSV or adapted_models entry
# exists. tta_ablations.sh prints warnings and exits 0 if it can't find the
# model dir, which silently breaks the sweep — fail loudly here instead.
SESS_DIR="${WATCH_DIR}/adapted_models/${ABL}_support${SUPPORT}_maml"
if [ ! -d "${SESS_DIR}" ] || [ -z "$(ls "${SESS_DIR}" 2>/dev/null)" ]; then
    echo "ERROR: ${ABL} produced no session results (looked in ${SESS_DIR})" >&2
    exit 11
fi
echo "Done. Result: ${WATCH_DIR}/"
