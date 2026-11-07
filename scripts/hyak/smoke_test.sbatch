#!/bin/bash
#SBATCH --job-name=tbfm-smoke-test
#SBATCH --account=rao
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=44G
#SBATCH --time=0:45:00
#SBATCH --output=%x_%j.log

# One-off validation run: short fold0 training + one TTA task at support=1000.
# Writes to its own directory, never the real campaign's $OUT (tma_standalone.py
# wipes --out-dir on start).
set -euo pipefail

source "${REPO_DIR:-$HOME/py-tbfm}/scripts/hyak/env.sh"

SMOKE_OUT="/gscratch/scrubbed/danmuir/smoke_test_maml_zscore"
SESSIONS_FILE="scripts/random_folds_sessions.json"

SESSIONS=$(python -c "
import json
folds = json.load(open('${SESSIONS_FILE}'))
print(','.join(folds['fold0']))
")

echo "=== Smoke training: fold0, 300 epochs -> ${SMOKE_OUT} ==="
python tma_standalone.py \
    100 20 0 false 16 5000 true \
    --latent-dim 96 \
    --batch-size-per-session 500 \
    --out-dir "${SMOKE_OUT}" \
    --held-in-sessions ${SESSIONS} \
    --epochs 300

echo ""
echo "=== Check 1: normalizer in hyperparameters.torch ==="
python -c "
import torch
hp = torch.load('${SMOKE_OUT}/hyperparameters.torch', weights_only=False)
print('normalizer:', hp['normalizer'])
assert 'ScalerZscore' in hp['normalizer'], f\"expected ScalerZscore, got {hp['normalizer']}\"
print('OK: z-score confirmed')
"

echo ""
echo "=== Check 2: held-in sessions match fold0 ==="
python -c "
import json, torch
expected = set(json.load(open('${SESSIONS_FILE}'))['fold0'])
actual = set(torch.load('${SMOKE_OUT}/hisi.torch', weights_only=False))
print('match:', expected == actual)
if expected != actual:
    print('expected-actual:', expected - actual)
    print('actual-expected:', actual - expected)
"

echo ""
echo "=== Check 3: norms.torch has z-score (mean/std) keys, not IQR (a/b) ==="
python -c "
import torch
norms = torch.load('${SMOKE_OUT}/best/norms.torch', weights_only=False)
print(type(norms))
print(list(norms.keys()) if hasattr(norms, 'keys') else norms)
"

echo ""
echo "=== Smoke TTA: fold0, support=1000, one session ==="
TTA_OUT="${SMOKE_OUT}/tta_smoke_1000"
python tta_testing.py \
    --model-paths "fold0:${SMOKE_OUT}/best" \
    --cuda-device 0 \
    --support-sizes 1000 \
    --adapt-session MonkeyJ_20160702_Session4_S1 \
    --tta-epochs 7001 \
    --output-dir "${TTA_OUT}" \
    --unfreeze-bases \
    --progressive-unfreezing-threshold 0 \
    --no-plot-display

echo ""
echo "=== Check 4: strategy_key is inner_outer ==="
python -c "
import torch, glob
paths = glob.glob('${TTA_OUT}/adapted_models/*/*/metadata.torch')
print('found', len(paths), 'metadata.torch file(s)')
for p in paths:
    m = torch.load(p, weights_only=False)
    print(p)
    print('  strategy_key:', m.get('strategy_key'))
    print('  support_size:', m.get('support_size'))
    print('  final_r2:', m.get('final_r2'))
"

echo ""
echo "=== Check 5: GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "=== Smoke test complete ==="
