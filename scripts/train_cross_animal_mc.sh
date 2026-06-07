#!/bin/bash
# Monte Carlo cross-animal generalization: sample 10 sessions from each monkey,
# train a G-model and J-model per fold, repeat for NUM_FOLDS folds.
#
# With 8 GPUs: PAIRS_PER_BATCH=4 folds run in parallel (G on even GPU, J on odd).
# Batches automatically if NUM_FOLDS > PAIRS_PER_BATCH.
#
# Usage: bash scripts/train_cross_animal_mc.sh [sessions_json] [output_dir]
#
# Env:
#   NUM_FOLDS     default 4
#   NUM_GPUS      default 8
#
# GCP (on VM, inside tmux):
#   BUCKET=py-tbfm-danmuir bash scripts/cloud/cloud_cross_animal_mc.sh

set -euo pipefail

SESSIONS_FILE="${1:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="${2:-cross_animal_mc_${TIMESTAMP}}"
NUM_FOLDS="${NUM_FOLDS:-4}"
NUM_GPUS="${NUM_GPUS:-8}"
PAIRS_PER_BATCH=$((NUM_GPUS / 2))

NUM_BASES=100
LATENT_DIM=96
BASIS_RESIDUAL_RANK=16
TRAIN_SIZE=5000
BATCH_SIZE=500

mkdir -p "${OUTPUT_BASE}"
SESSIONS_JSON="${OUTPUT_BASE}/sessions.json"
TIMING_LOG="${OUTPUT_BASE}/timing_log.txt"

echo "Cross-animal MC training" | tee "${TIMING_LOG}"
echo "Started:         ${TIMESTAMP}" | tee -a "${TIMING_LOG}"
echo "Output:          ${OUTPUT_BASE}" | tee -a "${TIMING_LOG}"
echo "NUM_FOLDS:       ${NUM_FOLDS}" | tee -a "${TIMING_LOG}"
echo "NUM_GPUS:        ${NUM_GPUS} (${PAIRS_PER_BATCH} folds/batch)" | tee -a "${TIMING_LOG}"
echo "" | tee -a "${TIMING_LOG}"

MONKEY_G_POOL="MonkeyG_20150914_Session1_S1 MonkeyG_20150914_Session3_S1 \
MonkeyG_20150915_Session2_S1 MonkeyG_20150915_Session3_S1 MonkeyG_20150915_Session4_S1 \
MonkeyG_20150915_Session5_S1 MonkeyG_20150916_Session4_S1 MonkeyG_20150917_Session1_M1 \
MonkeyG_20150917_Session1_S1 MonkeyG_20150917_Session2_M1 MonkeyG_20150917_Session2_S1 \
MonkeyG_20150917_Session3_M1 MonkeyG_20150917_Session3_S1 MonkeyG_20150918_Session1_M1 \
MonkeyG_20150918_Session1_S1 MonkeyG_20150921_Session3_S1 MonkeyG_20150921_Session5_S1 \
MonkeyG_20150922_Session1_S1 MonkeyG_20150922_Session2_S1 MonkeyG_20150922_Session3_S1 \
MonkeyG_20150925_Session1_S1 MonkeyG_20150925_Session2_S1"

MONKEY_J_POOL="MonkeyJ_20160426_Session1_S1 MonkeyJ_20160426_Session2_S1 \
MonkeyJ_20160426_Session3_S1 MonkeyJ_20160428_Session2_S1 MonkeyJ_20160428_Session3_S1 \
MonkeyJ_20160429_Session1_S1 MonkeyJ_20160429_Session3_S1 MonkeyJ_20160502_Session1_S1 \
MonkeyJ_20160624_Session3_S1 MonkeyJ_20160624_Session4_S1 MonkeyJ_20160625_Session4_S1 \
MonkeyJ_20160625_Session5_S1 MonkeyJ_20160627_Session1_S1 MonkeyJ_20160627_Session2_S1 \
MonkeyJ_20160630_Session1_S1 MonkeyJ_20160630_Session3_S1 MonkeyJ_20160702_Session2_S1 \
MonkeyJ_20160702_Session4_S1"

# -----------------------------------------------------------------------
# Generate or load sessions JSON
# -----------------------------------------------------------------------
if [ -n "${SESSIONS_FILE}" ]; then
    echo "Using sessions from: ${SESSIONS_FILE}"
    cp "${SESSIONS_FILE}" "${SESSIONS_JSON}"
    NUM_FOLDS=$(python -c "import json; print(len(json.load(open('${SESSIONS_JSON}'))))")
    echo "NUM_FOLDS from sessions file: ${NUM_FOLDS}"
else
    python -c "
import random, json
G_pool = '${MONKEY_G_POOL}'.split()
J_pool = '${MONKEY_J_POOL}'.split()
folds = {}
for i in range(${NUM_FOLDS}):
    rng = random.Random(i)
    folds[f'fold{i}'] = {
        'G_sessions': rng.sample(G_pool, 10),
        'J_sessions': rng.sample(J_pool, 10),
    }
with open('${SESSIONS_JSON}', 'w') as f:
    json.dump(folds, f, indent=2)
print('Sessions saved to: ${SESSIONS_JSON}')
for fold, s in folds.items():
    g = s['G_sessions']
    j = s['J_sessions']
    print(f'  {fold}: {len(g)}G + {len(j)}J held-in, {22-len(g)}G + {18-len(j)}J held-out')
"
fi
echo ""

# -----------------------------------------------------------------------
# Train: batch through folds PAIRS_PER_BATCH at a time
# -----------------------------------------------------------------------
fold=0
while [ $fold -lt $NUM_FOLDS ]; do
    batch_end=$((fold + PAIRS_PER_BATCH - 1))
    [ $batch_end -ge $NUM_FOLDS ] && batch_end=$((NUM_FOLDS - 1))

    echo "=== Batch: folds ${fold}..${batch_end} ==="

    PIDS_G=()
    PIDS_J=()
    BATCH_FOLDS=()
    START_TIMES=()

    for j in $(seq $fold $batch_end); do
        local_j=$((j - fold))
        GPU_G=$((local_j * 2))
        GPU_J=$((local_j * 2 + 1))
        FOLD_DIR="${OUTPUT_BASE}/fold${j}"
        mkdir -p "${FOLD_DIR}"

        G_SESSIONS=$(python -c "
import json
print(','.join(json.load(open('${SESSIONS_JSON}'))['fold${j}']['G_sessions']))
")
        J_SESSIONS=$(python -c "
import json
print(','.join(json.load(open('${SESSIONS_JSON}'))['fold${j}']['J_sessions']))
")

        START_TS=$(date +%Y%m%d_%H%M%S)
        echo "  fold${j}: G→GPU${GPU_G}, J→GPU${GPU_J} (${START_TS})"
        echo "fold${j}: STARTED ${START_TS} (G→GPU${GPU_G} J→GPU${GPU_J})" >> "${TIMING_LOG}"

        python -u tma_standalone.py \
            ${NUM_BASES} 10 ${GPU_G} false ${BASIS_RESIDUAL_RANK} ${TRAIN_SIZE} true \
            --latent-dim ${LATENT_DIM} \
            --batch-size-per-session ${BATCH_SIZE} \
            --out-dir "${FOLD_DIR}/G_model" \
            --held-in-sessions "${G_SESSIONS}" \
            > "${FOLD_DIR}/G_model.log" 2>&1 &
        PIDS_G+=($!)

        python -u tma_standalone.py \
            ${NUM_BASES} 10 ${GPU_J} false ${BASIS_RESIDUAL_RANK} ${TRAIN_SIZE} true \
            --latent-dim ${LATENT_DIM} \
            --batch-size-per-session ${BATCH_SIZE} \
            --out-dir "${FOLD_DIR}/J_model" \
            --held-in-sessions "${J_SESSIONS}" \
            > "${FOLD_DIR}/J_model.log" 2>&1 &
        PIDS_J+=($!)

        BATCH_FOLDS+=($j)
        START_TIMES+=($(date +%s))
    done

    echo "  Waiting for batch..."
    BATCH_FAILED=0
    for idx in "${!BATCH_FOLDS[@]}"; do
        j=${BATCH_FOLDS[$idx]}
        wait "${PIDS_G[$idx]}" || {
            echo "ERROR: fold${j}/G_model failed — see ${OUTPUT_BASE}/fold${j}/G_model.log"
            BATCH_FAILED=1
        }
        wait "${PIDS_J[$idx]}" || {
            echo "ERROR: fold${j}/J_model failed — see ${OUTPUT_BASE}/fold${j}/J_model.log"
            BATCH_FAILED=1
        }
        DURATION=$(( $(date +%s) - ${START_TIMES[$idx]} ))
        END_TS=$(date +%Y%m%d_%H%M%S)
        if [ $BATCH_FAILED -eq 0 ]; then
            echo "  fold${j} complete (${DURATION}s)"
            echo "fold${j}: COMPLETED ${END_TS} (${DURATION}s)" >> "${TIMING_LOG}"
        else
            echo "fold${j}: FAILED ${END_TS}" >> "${TIMING_LOG}"
        fi
    done

    [ $BATCH_FAILED -ne 0 ] && exit 1

    fold=$((batch_end + 1))
done

echo ""
echo "All ${NUM_FOLDS} folds complete. Output: ${OUTPUT_BASE}"
echo "Completed: $(date +%Y%m%d_%H%M%S)" >> "${TIMING_LOG}"
