#!/bin/bash
# Run on kalman (NOT Klone) -- pulls finished training/TTA output from Klone's
# scrubbed storage back to /mnt/nsl/danmuir/, incrementally. Safe to run
# repeatedly: rsync only copies what changed, and nothing on the Klone side
# is deleted by this script. This is what protects against scrubbed's 21-day
# purge -- anything copied here is safe even if the Klone copy later ages out.
#
# Usage: ./sync_results_back.sh <run_name>
#   e.g. ./sync_results_back.sh random_folds_maml_zscore_20261106_120000
#
# For the duration of the campaign, run this in a loop inside tmux:
#   tmux new -s maml_zscore_sync
#   while true; do ./sync_results_back.sh <run_name>; sleep 1800; done
set -euo pipefail

RUN_NAME="${1:?Usage: $0 <run_name>}"
SRC_HOST="klone"
SRC="/gscratch/scrubbed/danmuir/${RUN_NAME}/"
DEST="/mnt/nsl/danmuir/${RUN_NAME}/"

mkdir -p "${DEST}"
echo "=== $(date) : syncing ${SRC_HOST}:${SRC} -> ${DEST} ==="
rsync -avP --stats "${SRC_HOST}:${SRC}" "${DEST}"
echo "=== $(date) : sync complete ==="
du -sh "${DEST}"
