#!/bin/bash
#SBATCH --job-name=tbfm-smoke-tta
#SBATCH --account=rao
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=44G
#SBATCH --time=2:30:00
#SBATCH --output=%x_%j.log

# TTA-only half of the smoke test, run standalone against the already-trained
# fold0 checkpoint from smoke_test.sbatch's first (successful) run -- avoids
# re-paying for training after the first attempt timed out during TTA.
set -euo pipefail

source "${REPO_DIR:-$HOME/py-tbfm}/scripts/hyak/env.sh"

SMOKE_OUT="/gscratch/scrubbed/danmuir/smoke_test_maml_zscore"
TTA_OUT="${SMOKE_OUT}/tta_smoke_1000"
rm -rf "${TTA_OUT}"

echo "=== Smoke TTA: fold0, support=1000, one session ==="
time python tta_testing.py \
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
echo "=== Check: strategy_key is inner_outer ==="
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
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

echo ""
echo "=== Smoke TTA complete ==="
