#!/bin/bash
# Monitor a running cross-animal-mc GCP VM.
# Shows GPU utilization and per-fold training/TTA progress.
#
# Usage:
#   bash scripts/cloud/monitor_cross_animal_mc.sh [vm-name]
#   watch -n 30 bash scripts/cloud/monitor_cross_animal_mc.sh

PROJECT="${PROJECT:-nsf-2223495-425310}"
ZONE="${ZONE:-us-east4-c}"
VM_NAME="${1:-cross-animal-mc}"

bar() {
    local done=$1 total=$2 width=${3:-20}
    local filled=$(( total > 0 ? done * width / total : 0 ))
    local empty=$(( width - filled ))
    printf '['; printf '%0.s#' $(seq 1 $filled 2>/dev/null); printf '%0.s-' $(seq 1 $empty 2>/dev/null); printf '] %d/%d' "$done" "$total"
}

REMOTE_CMD='
OUT_DIR=$(ls -dt /opt/py-tbfm/cross_animal_mc_* 2>/dev/null | head -1 || true)

echo "=META="
echo "${OUT_DIR:-none}"
tmux list-sessions 2>/dev/null | head -5 || echo "(no tmux)"

echo "=FOLDS="
if [ -z "${OUT_DIR}" ] || [ ! -d "${OUT_DIR}" ]; then
    echo "none"
else
    python3 -c "
import os, json, time
from pathlib import Path

out = Path(\"${OUT_DIR}\")
sessions_json = out / \"sessions.json\"
timing_log = out / \"timing_log.txt\"

folds = {}
if sessions_json.exists():
    folds = json.loads(sessions_json.read_text())

timing = {}
if timing_log.exists():
    for line in timing_log.read_text().splitlines():
        for fold_key in folds or [f\"fold{i}\" for i in range(10)]:
            if line.startswith(f\"{fold_key}:\"):
                timing[fold_key] = line.split(\":\", 1)[1].strip()

num_folds = max(len(folds), len(list(out.glob(\"fold*\"))))

for i in range(num_folds):
    key = f\"fold{i}\"
    fold_dir = out / key
    g_dir = fold_dir / \"G_model\"
    j_dir = fold_dir / \"J_model\"
    tta_dir = fold_dir / \"tta_results\"

    def model_status(d):
        if not d.exists():
            return \"waiting\"
        if (d / \"hisi.torch\").exists():
            mtime = (d / \"hisi.torch\").stat().st_mtime
            age = int(time.time() - mtime)
            return f\"done({age//60}m ago)\"
        log = d.parent / f\"{d.name}.log\"
        if log.exists():
            lines = log.read_text().splitlines()
            # Look for most recent epoch line: ---- <epoch> <trloss> <teloss> <tr2> <te_r2>
            for l in reversed(lines):
                if l.startswith(\"----\"):
                    parts = l.split()
                    if len(parts) >= 6:
                        try:
                            ep = int(parts[1])
                            tr2 = float(parts[4])
                            te_r2 = float(parts[5])
                            return f\"ep={ep}  train_r2={tr2:.3f}  test_r2={te_r2:.3f}\"
                        except ValueError:
                            pass
            # Fall back to last non-empty line
            for l in reversed(lines):
                if l.strip():
                    return f\"starting: {l.strip()[-55:]}\"
            return \"running\"
        return \"starting\"

    def tta_status(d):
        if not d.exists():
            return \"waiting\"
        # Check sweep logs for per-job completion lines
        sweep_logs = sorted(d.glob(\"tta_sweep_*.log\"))
        if sweep_logs:
            text = sweep_logs[-1].read_text()
            lines = text.splitlines()
            total = 0
            for l in lines:
                if l.startswith(\"Total TTA jobs:\"):
                    try: total = int(l.split(\":\")[1].strip())
                    except: pass
            complete_lines = [l for l in lines if \"] Complete |\" in l]
            done = len(complete_lines)
            if total > 0 and done >= total:
                # summarise final r2s
                r2s = []
                for l in complete_lines:
                    if \"R²=\" in l:
                        try: r2s.append(float(l.split(\"R²=\")[1].split()[0]))
                        except: pass
                mean_r2 = sum(r2s)/len(r2s) if r2s else 0
                return f\"done  {done}/{total} jobs  mean_r2={mean_r2:.3f}\"
            if done > 0:
                last = complete_lines[-1]
                # extract model, r2
                try:
                    model = last.split(\"Model=\")[1].split(\"|\")[0].strip()
                    r2 = last.split(\"R²=\")[1].split()[0]
                    last_str = f\"{model} R²={r2}\"
                except:
                    last_str = last[-40:]
                return f\"{done}/{total} jobs  last: {last_str}\"
            if total > 0:
                return f\"0/{total} jobs  loading...\"
        # Fallback: recent line from fold log
        logs = list(d.parent.glob(f\"tta_fold{i}.log\"))
        if logs:
            lines = logs[0].read_text().splitlines()
            for l in reversed(lines):
                if l.strip() and \"\\r\" not in l:
                    return f\"starting: {l.strip()[-55:]}\"
        return \"running\"

    g_stat = model_status(g_dir)
    j_stat = model_status(j_dir)
    t_stat = tta_status(tta_dir)
    timing_str = timing.get(key, \"\")
    print(f\"{key}|G:{g_stat}|J:{j_stat}|TTA:{t_stat}|{timing_str}\")
" 2>/dev/null || echo "error"
fi

echo "=GPU="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader,nounits 2>/dev/null

echo "=LOG="
tail -8 /tmp/ca_mc.log 2>/dev/null || echo "(no log yet)"
'

echo "━━━  cross-animal-mc  $(date '+%H:%M:%S')  ━━━"
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

# Per-fold training + TTA status
echo "  Folds:"
FOLDS=$(sec FOLDS)
if [ "${FOLDS}" = "none" ] || [ "${FOLDS}" = "error" ] || [ -z "${FOLDS}" ]; then
    echo "    (not started)"
else
    while IFS='|' read -r fold g_stat j_stat tta_stat timing; do
        [ -z "$fold" ] && continue
        printf "  %-7s  G:%-35s  J:%-35s  TTA:%s\n" \
            "${fold}" "${g_stat#G:}" "${j_stat#J:}" "${tta_stat#TTA:}"
        [ -n "${timing}" ] && printf "           %s\n" "${timing}"
    done <<< "$FOLDS"
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
