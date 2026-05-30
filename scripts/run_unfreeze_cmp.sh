#!/bin/bash
set -e

SESSIONS="MonkeyG_20150914_Session1_S1 MonkeyG_20150914_Session3_S1 MonkeyG_20150915_Session5_S1 MonkeyG_20150917_Session1_M1 MonkeyG_20150917_Session2_S1 MonkeyG_20150917_Session3_S1 MonkeyG_20150918_Session1_M1 MonkeyG_20150918_Session1_S1 MonkeyG_20150921_Session3_S1 MonkeyG_20150922_Session1_S1 MonkeyG_20150925_Session1_S1 MonkeyG_20150925_Session2_S1 MonkeyJ_20160426_Session1_S1 MonkeyJ_20160426_Session2_S1 MonkeyJ_20160429_Session1_S1 MonkeyJ_20160502_Session1_S1 MonkeyJ_20160627_Session1_S1 MonkeyJ_20160630_Session3_S1 MonkeyJ_20160702_Session2_S1 MonkeyJ_20160702_Session4_S1"
COMMON="--model-paths baseline:baseline --support-sizes 1000 --adapt-session $SESSIONS --no-plot-display --use-multi-gpu --gpu-ids 0 1"
OUTDIR=/home/danmuir/GitHub/py-tbfm/unfreeze_cmp

echo "=== Run 1: no unfreezing ==="
python tta_testing.py $COMMON --output-dir $OUTDIR/off_20

echo "=== Run 2: all unfrozen ==="
python tta_testing.py $COMMON --unfreeze-basis-weights --unfreeze-bases --output-dir $OUTDIR/on_20
