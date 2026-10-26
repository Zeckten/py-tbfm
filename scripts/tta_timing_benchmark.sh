#!/usr/bin/env bash
# TTA timing benchmark: 4 reps × {1k, 2.5k} × all 40 sessions using fold0's model.
# Uses --allow-held-in to include fold0's training sessions.
# Results land in tta_timing_benchmark/rep<N>/; timing is in per_session CSV.

set -euo pipefail

export TBFM_DATA_DIR="/var/data/opto-coproc"

MODEL_PATH="/mnt/nsl/danmuir/random_folds_20260530_213515/fold0/model"
OUT_BASE="tta_timing_benchmark_gpu1"
REPS=1
SUPPORT_SIZES=(1000 2500)

ALL_SESSIONS=(
    MonkeyG_20150914_Session1_S1
    MonkeyG_20150914_Session3_S1
    MonkeyG_20150915_Session2_S1
    MonkeyG_20150915_Session3_S1
    MonkeyG_20150915_Session4_S1
    MonkeyG_20150915_Session5_S1
    MonkeyG_20150916_Session4_S1
    MonkeyG_20150917_Session1_M1
    MonkeyG_20150917_Session1_S1
    MonkeyG_20150917_Session2_M1
    MonkeyG_20150917_Session2_S1
    MonkeyG_20150917_Session3_M1
    MonkeyG_20150917_Session3_S1
    MonkeyG_20150918_Session1_M1
    MonkeyG_20150918_Session1_S1
    MonkeyG_20150921_Session3_S1
    MonkeyG_20150921_Session5_S1
    MonkeyG_20150922_Session1_S1
    MonkeyG_20150922_Session2_S1
    MonkeyG_20150922_Session3_S1
    MonkeyG_20150925_Session1_S1
    MonkeyG_20150925_Session2_S1
    MonkeyJ_20160426_Session1_S1
    MonkeyJ_20160426_Session2_S1
    MonkeyJ_20160426_Session3_S1
    MonkeyJ_20160428_Session2_S1
    MonkeyJ_20160428_Session3_S1
    MonkeyJ_20160429_Session1_S1
    MonkeyJ_20160429_Session3_S1
    MonkeyJ_20160502_Session1_S1
    MonkeyJ_20160624_Session3_S1
    MonkeyJ_20160624_Session4_S1
    MonkeyJ_20160625_Session4_S1
    MonkeyJ_20160625_Session5_S1
    MonkeyJ_20160627_Session1_S1
    MonkeyJ_20160627_Session2_S1
    MonkeyJ_20160630_Session1_S1
    MonkeyJ_20160630_Session3_S1
    MonkeyJ_20160702_Session2_S1
    MonkeyJ_20160702_Session4_S1
)

mkdir -p "${OUT_BASE}"

for rep in $(seq 1 ${REPS}); do
    echo "=== Rep ${rep}/${REPS} ==="
    OUT_DIR="${OUT_BASE}/rep${rep}"
    mkdir -p "${OUT_DIR}"

    .venv/bin/python tta_testing.py \
        --use-multi-gpu --gpu-ids 1 \
        --support-sizes "${SUPPORT_SIZES[@]}" \
        --model-paths "fold0:${MODEL_PATH}" \
        --adapt-session "${ALL_SESSIONS[@]}" \
        --allow-held-in \
        --unfreeze-bases \
        --progressive-unfreezing-threshold 0 \
        --max-adapt-sessions 40 \
        --tta-epochs 7001 \
        --output-dir "${OUT_DIR}" \
        --no-plot-display

    echo "Rep ${rep} done -> ${OUT_DIR}"
done

echo "All reps complete. Results in ${OUT_BASE}/"
