#!/bin/bash
# Run on kalman (NOT on Klone -- this is a data-transfer step, which is what
# login nodes are for). Copies only what the loader actually reads for the
# 40 valid sessions: <session>/torchraw/, <session>/embedding_rest/, plus the
# three top-level metadata files. Do NOT rsync the whole
# /var/data/opto-coproc/ tree -- it's 379GB of raw ConditioningBlocks/
# RecordingBlocks/TestingBlocks data the loader never touches.
set -euo pipefail

SRC_BASE="/var/data/opto-coproc"
DEST_HOST="klone"
DEST_BASE="/gscratch/scrubbed/danmuir/opto-coproc"
SESSIONS_FILE="$(dirname "$0")/valid_sessions.txt"

ssh "${DEST_HOST}" "mkdir -p ${DEST_BASE}"

echo "=== Top-level metadata files ==="
rsync -avP \
    "${SRC_BASE}/table_of_experiments.csv" \
    "${SRC_BASE}/bad_channels.pkl" \
    "${SRC_BASE}/electrode_positions.pkl" \
    "${DEST_HOST}:${DEST_BASE}/"

N=$(wc -l < "${SESSIONS_FILE}")
i=0
while read -r SESSION; do
    i=$((i + 1))
    echo "=== [${i}/${N}] ${SESSION} ==="
    ssh "${DEST_HOST}" "mkdir -p ${DEST_BASE}/${SESSION}"
    rsync -avP \
        "${SRC_BASE}/${SESSION}/torchraw" \
        "${SRC_BASE}/${SESSION}/embedding_rest" \
        "${DEST_HOST}:${DEST_BASE}/${SESSION}/"
done < "${SESSIONS_FILE}"

echo "=== Done. Verifying remote size ==="
ssh "${DEST_HOST}" "du -sh ${DEST_BASE}"
