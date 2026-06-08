#!/bin/bash
# Monitor a running support-sensitivity GCP VM.
# Shows draw progress, latest per-session R² values, and GPU utilization.
#
# Usage:
#   bash scripts/cloud/monitor_support_sensitivity.sh [vm-name]
#   watch -n 30 bash scripts/cloud/monitor_support_sensitivity.sh

PROJECT="${PROJECT:-nsf-2223495-425310}"
ZONE="${ZONE:-us-east4-c}"
VM_NAME="${1:-support-sensitivity}"

REMOTE_CMD='
OUT_DIR=$(ls -dt /opt/py-tbfm/sensitivity_* 2>/dev/null | head -1 || true)

echo "=META="
echo "${OUT_DIR:-none}"
tmux list-sessions 2>/dev/null | head -3 || echo "(no tmux)"

echo "=DRAWS="
if [ -z "${OUT_DIR}" ] || [ ! -d "${OUT_DIR}" ]; then
    echo "none"
else
    python3 -c "
import os, csv, glob
from pathlib import Path

out = Path(\"${OUT_DIR}\")

# Count total draws expected
log = out / \"sensitivity.log\"
num_draws = 20
if log.exists():
    for line in log.read_text().splitlines():
        if line.strip().startswith(\"NUM_DRAWS:\"):
            try: num_draws = int(line.split(\":\")[1].strip())
            except: pass

# Count completed draws (have a per_session CSV)
draw_dirs = sorted(out.glob(\"draw*/\"))
done_draws = [d for d in draw_dirs if list(d.glob(\"tta_support_*_per_session.csv\"))]
in_progress = [d for d in draw_dirs if d not in done_draws]

print(f\"done={len(done_draws)} total={num_draws}\")

# Per-session R² across completed draws
session_r2s = {}
for d in done_draws:
    csvs = list(d.glob(\"tta_support_*_per_session.csv\"))
    if not csvs:
        continue
    draw_idx = int(d.name.replace(\"draw\", \"\"))
    for row in csv.DictReader(open(csvs[0])):
        sid = row[\"session_id\"]
        r2 = float(row[\"session_r2\"])
        session_r2s.setdefault(sid, []).append((draw_idx, r2))

import statistics
for sid, vals in sorted(session_r2s.items()):
    r2s = [v[1] for v in vals]
    mean = statistics.mean(r2s)
    var = statistics.variance(r2s) if len(r2s) > 1 else 0.0
    n = len(r2s)
    print(f\"session|{sid}|n={n}|mean={mean:.4f}|var={var:.5f}\")

# Show current draw log tail if in progress
if in_progress:
    cur = sorted(in_progress)[0]
    cur_log = out / f\"{cur.name}.log\"
    if cur_log.exists():
        lines = cur_log.read_text().splitlines()
        for l in reversed(lines):
            if l.strip() and \"\\r\" not in l:
                print(f\"current|{cur.name}: {l.strip()[-70:]}\")
                break
" 2>/dev/null || echo "error"
fi

echo "=GPU="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader,nounits 2>/dev/null

echo "=LOG="
tail -6 /tmp/sensitivity.log 2>/dev/null || echo "(no log yet)"
'

echo "━━━  support-sensitivity  $(date '+%H:%M:%S')  ━━━"
echo ""

VM_STATUS=$(gcloud compute instances describe "${VM_NAME}" \
    --project="${PROJECT}" --zone="${ZONE}" \
    --format="value(status)" 2>/dev/null || echo "NOT_FOUND")

if [ "${VM_STATUS}" != "RUNNING" ]; then
    echo "  VM status: ${VM_STATUS}"
    [ "${VM_STATUS}" = "TERMINATED" ] && echo "  Experiment finished (VM stopped)."
    exit 0
fi

REMOTE=$(gcloud compute ssh "${VM_NAME}" \
    --project="${PROJECT}" --zone="${ZONE}" \
    --command="${REMOTE_CMD}" \
    --ssh-flag="-o ConnectTimeout=10" 2>/dev/null) || {
    echo "  SSH failed — VM may still be starting up"
    exit 0
}

sec() { echo "$REMOTE" | awk "/^=${1}=/{p=1;next} /^=[A-Z_]/{p=0} p"; }

# Meta
META=$(sec META)
OUT_DIR=$(echo "$META" | head -1)
TMUX=$(echo "$META" | tail -n +2)
echo "  Output: ${OUT_DIR}"
echo "  tmux:   ${TMUX}"
echo ""

# Draw progress
DRAWS=$(sec DRAWS)
if [ "${DRAWS}" = "none" ] || [ "${DRAWS}" = "error" ] || [ -z "${DRAWS}" ]; then
    echo "  (not started)"
else
    SUMMARY=$(echo "$DRAWS" | grep "^done=")
    DONE=$(echo "$SUMMARY" | sed 's/done=\([0-9]*\).*/\1/')
    TOTAL=$(echo "$SUMMARY" | sed 's/.*total=\([0-9]*\)/\1/')
    echo "  Draws: ${DONE:-?}/${TOTAL:-?}"
    echo ""

    echo "  Per-session R² (mean ± var across completed draws):"
    echo "$DRAWS" | grep "^session|" | while IFS='|' read -r _ sid n mean var; do
        printf "    %-45s  %s  mean=%-7s  var=%s\n" "${sid}" "${n}" "${mean#mean=}" "${var#var=}"
    done

    CURRENT=$(echo "$DRAWS" | grep "^current|")
    if [ -n "${CURRENT}" ]; then
        echo ""
        echo "  In progress: $(echo "$CURRENT" | sed 's/^current|//')"
    fi
fi
echo ""

# GPU utilization
echo "  GPUs:"
sec GPU | awk -F', ' '{
    idx=$1; util=$2; mem_used=$3; mem_tot=$4; temp=$5
    filled=int(util/5); bar=""
    for(i=0;i<filled;i++) bar=bar"█"
    for(i=filled;i<20;i++) bar=bar"░"
    printf "    GPU%s  %s %3d%%  mem=%5d/%dMB  %s°C\n", idx, bar, util, mem_used, mem_tot, temp
}'
echo ""

# Recent log
echo "  Recent log:"
sec LOG | sed 's/^/    /'
