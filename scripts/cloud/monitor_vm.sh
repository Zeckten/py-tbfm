#!/bin/bash
# Monitor one or more running py-tbfm GCP VMs.
#
# Usage:
#   bash scripts/cloud/monitor_vm.sh                    # auto-detect all running random-folds VMs
#   bash scripts/cloud/monitor_vm.sh vm1 vm2            # specific VMs
#   watch -n 60 bash scripts/cloud/monitor_vm.sh

PROJECT="${PROJECT:-nsf-2223495-425310}"
ZONE="${ZONE:-us-central1-f}"
BUCKET="${BUCKET:-py-tbfm-danmuir}"

bar() {
    local done=$1 total=$2 width=${3:-30}
    local filled=$(( total > 0 ? done * width / total : 0 ))
    local empty=$(( width - filled ))
    printf '['; printf '%0.s#' $(seq 1 $filled 2>/dev/null); printf '%0.s.' $(seq 1 $empty 2>/dev/null); printf '] %d/%d' "$done" "$total"
}

# Collect VM list
if [ $# -gt 0 ]; then
    VMS=("$@")
else
    mapfile -t VMS < <(gcloud compute instances list \
        --project="${PROJECT}" \
        --filter="status=RUNNING AND name~random-folds" \
        --format="value(name)" 2>/dev/null)
    [ ${#VMS[@]} -eq 0 ] && VMS=("random-folds-train")
fi

echo "━━━  $(date '+%Y-%m-%d %H:%M:%S')  (${#VMS[@]} VM(s))  ━━━"
echo ""

REMOTE_CMD='
OUTPUT_DIR=$(ls -dt /opt/py-tbfm/random_folds_* 2>/dev/null | head -1 || true)

echo "=META="
echo "${OUTPUT_DIR}"
tmux list-sessions 2>/dev/null | head -3 || echo "(no tmux)"

echo "=TRAIN="
TLOG="${OUTPUT_DIR}/timing_log.txt"
if [ -f "$TLOG" ]; then
    DONE=$(grep -c "COMPLETED" "$TLOG" 2>/dev/null || true); DONE=${DONE:-0}
    FAIL=$(grep -c "FAILED"    "$TLOG" 2>/dev/null || true); FAIL=${FAIL:-0}
    printf "done=%s fail=%s\n" "$DONE" "$FAIL"
    grep -E "STARTED|COMPLETED|FAILED" "$TLOG" | tail -2
else
    printf "done=0 fail=0\n"
    echo "(not started)"
fi

echo "=TTA="
TTA_DIR=$(ls -dt "${OUTPUT_DIR}"/tta_results_* 2>/dev/null | head -1 || true)
TTA_LOG="${TTA_DIR}/tta_timing_log.txt"
if [ -n "$TTA_DIR" ]; then
    TDONE=$(find "${TTA_DIR}" -maxdepth 2 -name "tta_support_*.json" 2>/dev/null | \
            xargs -I{} dirname {} | sort -u | wc -l); TDONE=${TDONE:-0}
    TFAIL=$(grep -c "FAILED" "$TTA_LOG" 2>/dev/null || true); TFAIL=${TFAIL:-0}
    printf "done=%s fail=%s\n" "$TDONE" "$TFAIL"
    grep -E "STARTED|COMPLETED|FAILED" "$TTA_LOG" 2>/dev/null | tail -2 || true
else
    printf "done=0 fail=0\n"
    echo "(not started)"
fi

echo "=TTADETAIL="
python3 -c "
import os, time, math, re
from pathlib import Path
from collections import defaultdict

output_dir = \"${OUTPUT_DIR}\"
tta_dirs = sorted(Path(output_dir).glob(\"tta_results_*\"))
if not tta_dirs:
    print(\"no_tta\")
else:
    base = tta_dirs[-1]
    active = None
    tlog = base / \"tta_timing_log.txt\"
    if tlog.exists():
        for line in reversed(tlog.read_text().splitlines()):
            m = re.search(r\"Fold (\\d+): STARTED\", line)
            if m:
                active = base / f\"fold{m.group(1)}\"
                break
    if active is None or not active.exists():
        fold_dirs = sorted(base.glob(\"fold*\"), key=lambda p: int(p.name.replace(\"fold\",\"\")))
        for fd in reversed(fold_dirs):
            if (fd / \"adapted_models\").exists():
                active = fd
                break
    if active is None:
        print(\"initializing\")
    else:
        fold_num = active.name
        am = active / \"adapted_models\"
        now = time.time()
        by_sup = defaultdict(list)
        for sup_dir in sorted(am.glob(\"*_support*\")):
            try:
                sup = int(sup_dir.name.split(\"support\")[1].split(\"_\")[0])
            except:
                continue
            for sess in sup_dir.iterdir():
                mf = sess / \"metadata.torch\"
                if mf.exists():
                    by_sup[sup].append(mf.stat().st_mtime)
        if not by_sup:
            print(f\"{fold_num} initializing\")
        else:
            all_times = sorted(t for ts in by_sup.values() for t in ts)
            elapsed = now - all_times[0]
            print(f\"{fold_num} elapsed={elapsed:.0f}s\")
            for sup in sorted(by_sup):
                times = sorted(by_sup[sup])
                n = len(times)
                age = now - times[-1]
                if n >= 8:
                    per_job = (times[-1] - times[-8]) / math.ceil(8/8)
                elif n >= 2:
                    per_job = (times[-1] - times[0]) / max(1, math.ceil(n/8))
                else:
                    per_job = None
                remaining = 20 - n
                if per_job and remaining > 0:
                    eta_str = f\"ETA ~{math.ceil(remaining/8)*per_job/60:.0f}m\"
                elif remaining == 0:
                    eta_str = \"done\"
                else:
                    eta_str = \"\"
                print(f\"  sup={sup:5d}: {n:2d}/20  last={age/60:.1f}m ago  {eta_str}\")
            total_done = sum(len(v) for v in by_sup.values())
            print(f\"  total_jobs={total_done}/80\")
" 2>/dev/null || echo "no_tta"

echo "=GPU="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits 2>/dev/null

echo "=DISK="
df -h /opt/py-tbfm 2>/dev/null | tail -1
'

sec() { echo "$1" | awk "/^=${2}=/{p=1;next} /^=[A-Z_]/{p=0} p"; }

show_vm() {
    local VM_NAME="$1"
    echo "┌─  ${VM_NAME}  ─────────────────────────────────────────────"

    local VM_STATUS
    VM_STATUS=$(gcloud compute instances describe "${VM_NAME}" \
        --project="${PROJECT}" --zone="${ZONE}" \
        --format="value(status)" 2>/dev/null || echo "NOT_FOUND")

    if [ "${VM_STATUS}" != "RUNNING" ]; then
        echo "│  VM: ${VM_STATUS}"
        if [ "${VM_STATUS}" = "TERMINATED" ]; then
            echo "│  Stopped. Check gs://${BUCKET}/models/ for results."
        else
            echo "│  Not found (deleted or wrong zone)."
            echo "│  Last GCS activity:"
            gsutil ls -l "gs://${BUCKET}/results/incremental/${VM_NAME}/**" 2>/dev/null \
                | grep -v "^TOTAL" | sort -k2 -r | head -4 | sed 's/^/│    /' || echo "│    (none)"
        fi
        echo "└────────────────────────────────────────────────────────"
        echo ""
        return
    fi

    local REMOTE
    REMOTE=$(gcloud compute ssh "${VM_NAME}" \
        --project="${PROJECT}" --zone="${ZONE}" \
        --command="${REMOTE_CMD}" 2>/dev/null) || { echo "│  SSH failed"; echo "└────────────────────────────────────────────────────────"; echo ""; return; }

    # Meta
    local META OUT_DIR TMUX_LINE
    META=$(sec "$REMOTE" "META")
    OUT_DIR=$(echo "$META" | head -1)
    TMUX_LINE=$(echo "$META" | tail -n +2)
    echo "│  Dir:  ${OUT_DIR:-none}"
    echo "│  tmux: ${TMUX_LINE}"

    # Training
    local TRAIN T_DONE T_FAIL TRAIN_BODY
    TRAIN=$(sec "$REMOTE" "TRAIN")
    T_DONE=$(echo "$TRAIN" | head -1 | grep -o 'done=[0-9]*' | cut -d= -f2); T_DONE=${T_DONE:-0}
    T_FAIL=$(echo "$TRAIN" | head -1 | grep -o 'fail=[0-9]*' | cut -d= -f2); T_FAIL=${T_FAIL:-0}
    TRAIN_BODY=$(echo "$TRAIN" | tail -n +2)
    if [ "$TRAIN_BODY" != "(not started)" ]; then
        echo "│"
        echo "│  Training: $(bar $T_DONE 20)$( [ "$T_FAIL" -gt 0 ] && echo "  ⚠ ${T_FAIL} failed" )"
        echo "$TRAIN_BODY" | sed 's/^/│    /'
    fi

    # TTA fold-level
    local TTA TTA_DONE TTA_FAIL
    TTA=$(sec "$REMOTE" "TTA")
    TTA_DONE=$(echo "$TTA" | head -1 | grep -o 'done=[0-9]*' | cut -d= -f2); TTA_DONE=${TTA_DONE:-0}
    TTA_FAIL=$(echo "$TTA" | head -1 | grep -o 'fail=[0-9]*' | cut -d= -f2); TTA_FAIL=${TTA_FAIL:-0}
    echo "│"
    echo "│  TTA folds: $(bar $TTA_DONE 20)$( [ "$TTA_FAIL" -gt 0 ] && echo "  ⚠ ${TTA_FAIL} failed" )"
    echo "$TTA" | tail -n +2 | sed 's/^/│    /'

    # TTA detail
    local DETAIL FOLD_LINE FOLD_NAME ELAPSED ELAPSED_MIN
    DETAIL=$(sec "$REMOTE" "TTADETAIL")
    if [ -n "$DETAIL" ] && [ "$DETAIL" != "no_tta" ] && [ "$DETAIL" != "initializing" ]; then
        FOLD_LINE=$(echo "$DETAIL" | head -1)
        FOLD_NAME=$(echo "$FOLD_LINE" | awk '{print $1}')
        ELAPSED=$(echo "$FOLD_LINE" | grep -o 'elapsed=[0-9]*' | cut -d= -f2)
        ELAPSED_MIN=$(( ${ELAPSED:-0} / 60 ))
        echo "│"
        echo "│  Active: ${FOLD_NAME}  (${ELAPSED_MIN}m elapsed)"
        echo "$DETAIL" | tail -n +2 | sed 's/^/│    /'
    fi

    # GPUs
    echo "│"
    echo "│  GPUs:"
    sec "$REMOTE" "GPU" | awk -F', ' '{
        util=$2; mem_used=$3; mem_tot=$4
        filled=int(util/5); bar=""
        for(i=0;i<filled;i++) bar=bar"█"
        for(i=filled;i<20;i++) bar=bar"░"
        printf "│    GPU%s  %s %3d%%  %5dMB/%dMB\n", $1, bar, util, mem_used, mem_tot
    }'

    # Disk
    echo "│"
    printf "│  Disk: "; sec "$REMOTE" "DISK"
    echo "└────────────────────────────────────────────────────────"
    echo ""
}

for VM in "${VMS[@]}"; do
    show_vm "$VM"
done
