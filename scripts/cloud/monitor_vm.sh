#!/bin/bash
# Monitor a running py-tbfm GCP VM.
#
# Usage:
#   bash scripts/cloud/monitor_vm.sh [vm-name]
#   watch -n 30 bash scripts/cloud/monitor_vm.sh

VM_NAME="${1:-random-folds-train}"
PROJECT="${PROJECT:-nsf-2223495-425310}"
ZONE="${ZONE:-us-central1-f}"
BUCKET="${BUCKET:-py-tbfm-danmuir}"

now() { date '+%Y-%m-%d %H:%M:%S'; }
bar() {
    local done=$1 total=$2 width=30
    local filled=$(( done * width / total ))
    local empty=$(( width - filled ))
    printf '['; printf '%0.s#' $(seq 1 $filled 2>/dev/null); printf '%0.s.' $(seq 1 $empty 2>/dev/null); printf '] %d/%d' "$done" "$total"
}

echo "━━━  ${VM_NAME}  $(now)  ━━━"

# VM status
VM_STATUS=$(gcloud compute instances describe "${VM_NAME}" \
    --project="${PROJECT}" --zone="${ZONE}" \
    --format="value(status)" 2>/dev/null || echo "NOT_FOUND")
echo "VM: ${VM_STATUS}"

if [ "${VM_STATUS}" != "RUNNING" ]; then
    [ "${VM_STATUS}" = "TERMINATED" ] && echo "Stopped. Models at: gs://${BUCKET}/models/"
    [ "${VM_STATUS}" = "NOT_FOUND"  ] && echo "Deleted (completed or preempted)."
    echo ""
    echo "Last GCS activity:"
    gsutil ls -l "gs://${BUCKET}/results/incremental/${VM_NAME}/**" 2>/dev/null \
        | grep -v "^TOTAL" | sort -k2 -r | head -6 || echo "  (none)"
    exit 0
fi

# Single SSH call gathering everything
REMOTE=$(gcloud compute ssh "${VM_NAME}" \
    --project="${PROJECT}" --zone="${ZONE}" \
    --command='
OUTPUT_DIR=$(ls -dt /opt/py-tbfm/random_folds_* 2>/dev/null | head -1 || true)

echo "=META="
echo "${OUTPUT_DIR}"
tmux list-sessions 2>/dev/null | head -3 || echo "(no tmux)"

echo "=TRAIN="
TLOG="${OUTPUT_DIR}/timing_log.txt"
if [ -f "$TLOG" ]; then
    DONE=$(grep -c "COMPLETED" "$TLOG" 2>/dev/null || true)
    FAIL=$(grep -c "FAILED"    "$TLOG" 2>/dev/null || true)
    LAST=$(grep -E "STARTED|COMPLETED|FAILED" "$TLOG" | tail -3)
    echo "done=${DONE} fail=${FAIL}"
    echo "$LAST"
else
    echo "done=0 fail=0"
    echo "(not started)"
fi

echo "=TTA="
TTA_DIR=$(ls -dt "${OUTPUT_DIR}"/tta_results_* 2>/dev/null | head -1 || true)
TTA_LOG="${TTA_DIR}/tta_timing_log.txt"
LIVE_LOG=$(ls /tmp/tta.log 2>/dev/null || true)
if [ -n "$TTA_DIR" ] && [ -f "$TTA_LOG" ]; then
    TDONE=$(grep -c "COMPLETED" "$TTA_LOG" 2>/dev/null || true)
    TFAIL=$(grep -c "FAILED"    "$TTA_LOG" 2>/dev/null || true)
    echo "done=${TDONE} fail=${TFAIL}"
    grep -E "STARTED|COMPLETED|FAILED" "$TTA_LOG" | tail -3
elif [ -n "$LIVE_LOG" ]; then
    # In progress: grab tqdm line + last status
    echo "done=0 fail=0"
    grep "TTA jobs:" "$LIVE_LOG" 2>/dev/null | tail -1 || echo "(initializing)"
    grep "Processing Fold" "$LIVE_LOG" 2>/dev/null | tail -1 || true
else
    echo "done=0 fail=0"
    echo "(not started)"
fi

echo "=GPU="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits 2>/dev/null

echo "=DISK="
df -h /opt/py-tbfm 2>/dev/null | tail -1
' 2>/dev/null) || { echo "SSH failed"; exit 0; }

sec() { echo "${REMOTE}" | awk "/^=${1}=/{p=1;next} /^=[A-Z_]/{p=0} p"; }

# ── Meta ──────────────────────────────────────────────────────────────────────
META=$(sec "META")
OUT_DIR=$(echo "$META" | head -1)
TMUX_LINE=$(echo "$META" | tail -n +2)
echo "Dir:  ${OUT_DIR:-none}"
echo "tmux: ${TMUX_LINE}"

# ── Training ──────────────────────────────────────────────────────────────────
TRAIN=$(sec "TRAIN")
T_STATS=$(echo "$TRAIN" | head -1)
T_DONE=$(echo "$T_STATS" | grep -o 'done=[0-9]*' | cut -d= -f2)
T_FAIL=$(echo "$T_STATS" | grep -o 'fail=[0-9]*' | cut -d= -f2)
T_DONE=${T_DONE:-0}; T_FAIL=${T_FAIL:-0}

echo ""
echo "── Training ─────────────────────────────────────"
printf "  "; bar "$T_DONE" 20; echo ""
echo "$TRAIN" | tail -n +2 | sed 's/^/  /'
[ "$T_FAIL" -gt 0 ] && echo "  ⚠ ${T_FAIL} failed"

# ── TTA ───────────────────────────────────────────────────────────────────────
TTA=$(sec "TTA")
TTA_STATS=$(echo "$TTA" | head -1)
TTA_DONE=$(echo "$TTA_STATS" | grep -o 'done=[0-9]*' | cut -d= -f2)
TTA_FAIL=$(echo "$TTA_STATS" | grep -o 'fail=[0-9]*' | cut -d= -f2)
TTA_DONE=${TTA_DONE:-0}; TTA_FAIL=${TTA_FAIL:-0}

echo ""
echo "── TTA ──────────────────────────────────────────"
printf "  "; bar "$TTA_DONE" 20; echo ""
echo "$TTA" | tail -n +2 | sed 's/^/  /'
[ "$TTA_FAIL" -gt 0 ] && echo "  ⚠ ${TTA_FAIL} failed"

# ── GPUs ──────────────────────────────────────────────────────────────────────
echo ""
echo "── GPUs ─────────────────────────────────────────"
sec "GPU" | awk -F', ' '{
    util=$2; mem_used=$3; mem_tot=$4
    bar=""
    filled=int(util/5)
    for(i=0;i<filled;i++) bar=bar"█"
    for(i=filled;i<20;i++) bar=bar"░"
    printf "  GPU%s  %s %3d%%  %5dMB/%dMB\n", $1, bar, util, mem_used, mem_tot
}'

# ── Disk ──────────────────────────────────────────────────────────────────────
echo ""
echo "── Disk ─────────────────────────────────────────"
sec "DISK" | sed 's/^/  /'

echo ""
