#!/bin/bash
# Run baseline TTA at a given support size N times on the same VM, each rep
# writing to its own output dir, for measuring run-to-run variance of TTA.
#
# Usage (on a GCP VM):
#   BUCKET=my-bucket bash scripts/cloud/run_baseline_reps.sh <support_size> <n_reps>
#
# Example:
#   BUCKET=py-tbfm-danmuir bash scripts/cloud/run_baseline_reps.sh 2500 20
#
# Output: per-rep results land in
#   gs://${BUCKET}/results/incremental/$(hostname)-rep${i}/tta_baseline_${support_size}/
# AUTO_SHUTDOWN=0 for reps 1..N-1, AUTO_SHUTDOWN=stop on rep N, so the VM
# self-stops only after the final rep.

set -euo pipefail

SUPPORT="${1:?usage: $0 <support_size> <n_reps>}"
NREPS="${2:?usage: $0 <support_size> <n_reps>}"
BUCKET="${BUCKET:?BUCKET env var required}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HOST_BASE="$(hostname)"

for i in $(seq 1 "${NREPS}"); do
    echo "============================================================"
    echo "REP ${i}/${NREPS} starting at $(date)"
    echo "============================================================"
    if [ "${i}" -lt "${NREPS}" ]; then
        AS=0
    else
        AS=stop
    fi
    WORK_DIR="${REPO_ROOT}/ablations_remote_rep${i}" \
    VM_NAME="${HOST_BASE}-rep${i}" \
    ABLATIONS="baseline" \
    AUTO_SHUTDOWN="${AS}" \
    BUCKET="${BUCKET}" \
    bash "${REPO_ROOT}/scripts/cloud/run_all_ablations_sequential.sh" "${SUPPORT}"
    echo "REP ${i}/${NREPS} done at $(date)"
done
