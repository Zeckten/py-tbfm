#!/bin/bash
# Wrap a long-running command with periodic rsync of a watch dir to GCS.
#
# Designed for spot-VM TTA experiments where TTA writes per-session results to
# disk as it goes, but the final aggregate CSV is only written at the very end.
# Without this, a preemption at session 17/20 would lose all progress when the
# spot VM is deleted.
#
# Usage:
#   BUCKET=my-bucket bash scripts/cloud/with_incremental_rsync.sh \
#       <watch_dir> <cmd> [args...]
#
# Pushes ${watch_dir} -> gs://${BUCKET}/results/incremental/${VM_NAME}/${watch_dir_basename}/
# every ${RSYNC_INTERVAL:-120}s while <cmd> runs. Final rsync runs on exit
# (success, failure, signal, or preemption).
#
# VM_NAME defaults to `hostname` (the GCP VM name).
#
# Examples:
#   # Wrap an ablation TTA run:
#   bash scripts/cloud/with_incremental_rsync.sh \
#       ablations_remote/tta_no_ae_recon_5000 \
#       bash scripts/tta_ablations.sh ablations_remote
#
#   # Wrap cross-animal TTA:
#   bash scripts/cloud/with_incremental_rsync.sh \
#       cross_animal_20260507/fold_G2J/tta_results \
#       bash scripts/tta_cross_animal.sh cross_animal_20260507

set -euo pipefail

BUCKET="${BUCKET:?BUCKET env var required}"

WATCH_DIR="${1:?usage: $0 <watch_dir> <cmd> [args...]}"
shift
if [ "$#" -eq 0 ]; then
    echo "ERROR: missing command to wrap" >&2
    exit 2
fi

VM_NAME="${VM_NAME:-$(hostname)}"
INTERVAL="${RSYNC_INTERVAL:-120}"
DEST="gs://${BUCKET}/results/incremental/${VM_NAME}/$(basename "${WATCH_DIR}")/"

echo "[wrapper] watch:   ${WATCH_DIR}"
echo "[wrapper] dest:    ${DEST}"
echo "[wrapper] every:   ${INTERVAL}s"
echo "[wrapper] command: $*"

# Background loop. Quiet mode (-q) so the log isn't drowned in per-file lines.
# Errors are tolerated — a transient gsutil failure shouldn't kill the run.
(
    while true; do
        sleep "${INTERVAL}"
        if [ -d "${WATCH_DIR}" ]; then
            gsutil -m -q rsync -r "${WATCH_DIR}/" "${DEST}" 2>/dev/null || true
        fi
    done
) &
RSYNC_PID=$!

# On any exit (incl. signals), stop the loop and do one final non-quiet rsync
# so the user sees what landed.
final_sync() {
    echo "[wrapper] stopping background rsync (pid ${RSYNC_PID})"
    kill "${RSYNC_PID}" 2>/dev/null || true
    if [ -d "${WATCH_DIR}" ]; then
        echo "[wrapper] final rsync ${WATCH_DIR} -> ${DEST}"
        gsutil -m rsync -r "${WATCH_DIR}/" "${DEST}" || true
    else
        echo "[wrapper] watch dir ${WATCH_DIR} never existed; nothing to sync"
    fi
}
trap final_sync EXIT

# Run the wrapped command. Preserve its exit code.
"$@"
