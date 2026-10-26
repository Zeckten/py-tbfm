#!/usr/bin/env bash
set -euo pipefail

export TBFM_DATA_DIR=/var/data/opto-coproc

time .venv/bin/python tta_testing.py \
    --use-multi-gpu --gpu-ids 0 \
    --support-sizes 1000 \
    --model-paths 'fold0:/mnt/nsl/danmuir/random_folds_20260530_213515/fold0/model' \
    --adapt-session MonkeyG_20150914_Session1_S1 \
    --allow-held-in \
    --unfreeze-bases \
    --progressive-unfreezing-threshold 0 \
    --max-adapt-sessions 1 \
    --tta-epochs 7001 \
    --output-dir /tmp/tta_single_timing \
    --no-plot-display
