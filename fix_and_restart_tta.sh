#!/bin/bash
# Run this inside tmux on the g4 VM (random-folds-tta)
# cd /opt/py-tbfm && bash fix_and_restart_tta.sh

set -euo pipefail
source .venv/bin/activate
TTA_DIR="random_folds_20260530_213515/tta_results_20260601_022544"

echo "=== Step 1: Kill any remaining GPU workers ==="
sudo kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ') 2>/dev/null || true
echo "done"

echo "=== Step 2: Reconstruct fold 2 results from metadata.torch ==="
python3 - <<'PYEOF'
import torch, json, math
from pathlib import Path

for fold_n in [0, 1, 2]:
    BASE = Path(f"random_folds_20260530_213515/tta_results_20260601_022544/fold{fold_n}")
    am = BASE / "adapted_models"
    if not am.exists():
        continue
    # Skip if a real JSON already exists
    if list(BASE.glob("tta_support_[0-9]*.json")):
        print(f"fold{fold_n}: real JSON exists, skipping")
        continue
    if (BASE / "tta_support_reconstructed.json").exists():
        print(f"fold{fold_n}: reconstructed JSON exists, skipping")
        continue
    runs = []
    for sup_dir in sorted(am.glob("*_support*")):
        sup = int(sup_dir.name.split("support")[1].split("_")[0])
        per_session, finals = {}, []
        for s in sorted(sup_dir.iterdir()):
            m = s / "metadata.torch"
            if m.exists():
                d = torch.load(m, weights_only=False)
                per_session.update(d.get("per_session_r2s", {}))
                finals.append(d.get("final_r2", float("nan")))
        if per_session:
            v = [x for x in finals if not math.isnan(x)]
            mean_r2 = sum(v)/len(v) if v else float("nan")
            runs.append({"model": f"fold{fold_n}", "strategy": "inner_outer",
                         "support_size": sup, "r2": mean_r2, "per_session_r2s": per_session})
            print(f"  fold{fold_n} sup={sup}: n={len(per_session)} mean_final_r2={mean_r2:.4f}")
    if runs:
        out = BASE / "tta_support_reconstructed.json"
        with open(out, "w") as f:
            json.dump({"runs": runs}, f)
        print(f"fold{fold_n}: written {out}")
PYEOF

echo "=== Step 3: Restart TTA for remaining folds in tmux ==="
BUCKET=py-tbfm-danmuir
FOLDS_DIR="random_folds_20260530_213515"
RESUME_DIR="${TTA_DIR}"

tmux kill-session -t run 2>/dev/null || true
tmux new-session -d -s run -x 220 -y 50 \
    "TBFM_DATA_DIR=/mnt/data GPU_IDS='0 1 2 3 4 5 6 7' BUCKET=${BUCKET} \
    bash scripts/cloud/with_incremental_rsync.sh \
        '${RESUME_DIR}' \
        bash scripts/tta_random_folds.sh '${FOLDS_DIR}' '${RESUME_DIR}' \
    2>&1 | tee -a /tmp/tta.log; \
    gsutil -m rsync -r '${RESUME_DIR}/' \
        'gs://${BUCKET}/models/${FOLDS_DIR}/tta_results_20260601_022544/' && \
    ZONE=\$(basename \$(curl -sf -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/zone)) && \
    PROJECT=\$(curl -sf -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/project/project-id) && \
    echo 'Stopping VM...' && \
    gcloud compute instances stop \$(hostname) --zone=\${ZONE} --project=\${PROJECT} --discard-local-ssd=true --quiet"

echo "Started tmux session 'run'. Attach with: tmux attach -t run"
