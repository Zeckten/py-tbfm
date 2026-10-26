#!/bin/bash
# Run TTA evaluation (co-adaptation strategy) on all held-out sessions for each fold
# from a coadapt random-folds training run.
#
# Usage: $0 <path_to_coadapt_random_folds_results> [existing_tta_output_dir]
# Example: $0 random_folds_coadapt_20260201_120000
# Example (resume): $0 random_folds_coadapt_20260201_120000 random_folds_coadapt_20260201_120000/tta_results_500_20260201_130000

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_coadapt_random_folds_results> [existing_tta_output_dir]"
    echo "Example: $0 random_folds_coadapt_20260201_120000"
    echo "Example (resume): $0 random_folds_coadapt_20260201_120000 random_folds_coadapt_20260201_120000/tta_results_500_20260201_130000"
    exit 1
fi

FOLDS_DIR="$1"
RESUME_DIR="$2"

if [ ! -d "$FOLDS_DIR" ]; then
    echo "ERROR: Directory not found: $FOLDS_DIR"
    exit 1
fi

echo "========================================="
echo "TTA Cross-Validation on Coadapt Random Folds"
echo "========================================="
echo "Folds directory: $FOLDS_DIR"
echo ""

# TTA parameters
SUPPORT_SIZE=${SUPPORT_SIZE:-1000}
TTA_EPOCHS=7001
NUM_FOLDS=20
# --auto-gpu-slots: GPU_CANDIDATES lists which physical GPUs are eligible
# (not repeated -- the fill level per GPU is MAX_JOBS_PER_GPU, checked live
# against real GPU occupancy at startup and periodically during the run, not
# hardcoded). Any other user's process using more than OTHER_USER_GB_THRESHOLD
# counts as occupying one of our slots. MAX_JOBS_PER_GPU=1: a careful 1/2/3
# sweep against this exact job (real fold0 checkpoint, support=1000) showed
# conc=2 is memory-safe (0/20 failed, 18.4GB/24.6GB peak) but gives
# essentially no speedup over conc=1 (268s wall vs 259s to drain the same
# 20-job queue), while conc=3 collapses (18/20 jobs failed -- a mix of
# "CUDA error: out of memory" and "CUDA error: invalid argument"; the sweep's
# OOM-detection grep originally only matched the literal "CUDA out of
# memory" string and missed the invalid-argument failures entirely --
# always grep "Error processing job" broadly, not a specific CUDA message,
# when auditing a sweep's failure count). So 1/GPU is the fastest-or-tied
# AND safest setting here -- re-sweep with wall-clock comparisons, not just
# an OOM check, before raising it. See run_queue.sh for why the old flat
# "8/GPU" default was wrong. Failed-but-partially-complete folds can be
# backfilled afterward by harvesting
# adapted_models/<fold>_support<N>_coadapt/<session>/metadata.torch's
# 'final_r2' field directly (NOT 'per_session_r2s', which is ~2x too high).
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1}"
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"
OTHER_USER_GB_THRESHOLD="${OTHER_USER_GB_THRESHOLD:-5}"

# Create or reuse TTA output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ -n "$RESUME_DIR" ]; then
    if [ ! -d "$RESUME_DIR" ]; then
        echo "ERROR: Resume directory not found: $RESUME_DIR"
        exit 1
    fi
    TTA_OUTPUT_DIR="$RESUME_DIR"
    echo "Resuming previous run from: ${TTA_OUTPUT_DIR}"
else
    TTA_OUTPUT_DIR="${FOLDS_DIR}/tta_results_${SUPPORT_SIZE}_${TIMESTAMP}"
    mkdir -p ${TTA_OUTPUT_DIR}
fi

# Create logs
TIMING_LOG="${TTA_OUTPUT_DIR}/tta_timing_log.txt"
RESULTS_SUMMARY="${TTA_OUTPUT_DIR}/tta_summary.csv"

echo "TTA Coadapt Cross-Validation Log" > ${TIMING_LOG}
echo "Started at: ${TIMESTAMP}" >> ${TIMING_LOG}
echo "========================================" >> ${TIMING_LOG}
echo "" >> ${TIMING_LOG}

echo "fold,session_id,strategy,support_size,r2,start_time,end_time,duration_sec,status" > ${RESULTS_SUMMARY}


# Count available folds
AVAILABLE_FOLDS=0
for i in $(seq 0 $((NUM_FOLDS - 1))); do
    FOLD_DIR="${FOLDS_DIR}/fold${i}"
    if [ -d "$FOLD_DIR" ]; then
        AVAILABLE_FOLDS=$((AVAILABLE_FOLDS + 1))
    fi
done

echo "Found ${AVAILABLE_FOLDS} folds to process"
echo "Support size: ${SUPPORT_SIZE}"
echo "TTA epochs: ${TTA_EPOCHS}"
echo "Output directory: ${TTA_OUTPUT_DIR}"
echo ""

# Process each fold
for i in $(seq 0 $((NUM_FOLDS - 1))); do
    FOLD_DIR="${FOLDS_DIR}/fold${i}"

    if [ ! -d "$FOLD_DIR" ]; then
        echo "WARNING: Fold ${i} directory not found, skipping..."
        continue
    fi

    # Check if model files exist (monolithic or split format, matching
    # scripts/tta_random_folds.sh's check)
    if [ ! -f "${FOLD_DIR}/model_nf_1.torch" ] && [ ! -f "${FOLD_DIR}/model.torch" ] \
            && [ ! -f "${FOLD_DIR}/model/tbfm.torch" ] && [ ! -f "${FOLD_DIR}/tbfm.torch" ]; then
        echo "WARNING: No model file found in fold ${i}, skipping..."
        continue
    fi

    # Check if hisi file exists
    if [ ! -f "${FOLD_DIR}/hisi_nf_1.torch" ] && [ ! -f "${FOLD_DIR}/hisi.torch" ]; then
        echo "WARNING: No hisi file found in fold ${i}, skipping..."
        continue
    fi

    # Check if this fold was already completed
    FOLD_TTA_DIR="${TTA_OUTPUT_DIR}/fold${i}"
    if [ -d "$FOLD_TTA_DIR" ] && ls ${FOLD_TTA_DIR}/tta_support_*.json 1>/dev/null 2>&1; then
        echo "Fold ${i} already completed (results found in ${FOLD_TTA_DIR}), skipping..."
        continue
    fi

    echo "========================================="
    echo "Processing Fold ${i}"
    echo "========================================="

    FOLD_START=$(date +%s)
    FOLD_START_TS=$(date +%Y%m%d_%H%M%S)

    echo "Fold ${i}: STARTED at ${FOLD_START_TS}" >> ${TIMING_LOG}

    mkdir -p ${FOLD_TTA_DIR}

    # Prefer the best-validation-R2 checkpoint (current-code training writes
    # this via model_save_path=best/) over the final-training-epoch model in
    # the fold root, matching scripts/tta_random_folds.sh's same check.
    if [ -f "${FOLD_DIR}/best/tbfm.torch" ]; then
        MODEL_PATH="${FOLD_DIR}/best"
    else
        MODEL_PATH="${FOLD_DIR}"
    fi

    # Run TTA for this fold using co-adaptation strategy
    python tta_testing.py \
        --model-paths fold${i}:${MODEL_PATH} \
        --use-multi-gpu \
        --gpu-ids ${GPU_CANDIDATES} \
        --auto-gpu-slots --max-jobs-per-gpu "${MAX_JOBS_PER_GPU}" \
        --other-user-gb-threshold "${OTHER_USER_GB_THRESHOLD}" \
        --support-sizes ${SUPPORT_SIZE} \
        --max-adapt-sessions 20 \
        --tta-epochs ${TTA_EPOCHS} \
        --output-dir ${FOLD_TTA_DIR} \
        --unfreeze-bases \
        --progressive-unfreezing-threshold 0 \
        --coadapt-tta

    EXIT_CODE=$?
    FOLD_END=$(date +%s)
    FOLD_DURATION=$((FOLD_END - FOLD_START))
    FOLD_END_TS=$(date +%Y%m%d_%H%M%S)

    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "ERROR: TTA failed for fold ${i}"
        echo "Fold ${i}: FAILED at ${FOLD_END_TS} (duration: ${FOLD_DURATION}s)" >> ${TIMING_LOG}
        echo "${i},ALL,ALL,${SUPPORT_SIZE},NA,${FOLD_START_TS},${FOLD_END_TS},${FOLD_DURATION},FAILED" >> ${RESULTS_SUMMARY}
        echo "Continuing with remaining folds..."
        echo ""
        continue
    fi

    echo "Fold ${i}: COMPLETED at ${FOLD_END_TS} (duration: ${FOLD_DURATION}s)" >> ${TIMING_LOG}

    # Extract per-session results
    RESULTS_JSON=$(ls -t ${FOLD_TTA_DIR}/tta_support_*.json 2>/dev/null | head -n 1)

    if [ -f "$RESULTS_JSON" ]; then
        echo "  Extracting results from: $(basename $RESULTS_JSON)"

        python -c "
import json
import sys

try:
    with open('${RESULTS_JSON}', 'r') as f:
        results = json.load(f)

    for run in results.get('runs', []):
        strategy = run.get('strategy', 'unknown')
        support_size = run.get('support_size', 0)
        per_session_r2s = run.get('per_session_r2s', {})

        if per_session_r2s:
            for session_id, r2 in per_session_r2s.items():
                print(f'${i},{session_id},{strategy},{support_size},{r2},${FOLD_START_TS},${FOLD_END_TS},${FOLD_DURATION},SUCCESS')
        else:
            overall_r2 = run.get('r2', 'NA')
            print(f'${i},OVERALL,{strategy},{support_size},{overall_r2},${FOLD_START_TS},${FOLD_END_TS},${FOLD_DURATION},SUCCESS')
except Exception as e:
    print(f'ERROR: Failed to parse results: {e}', file=sys.stderr)
    sys.exit(1)
" >> ${RESULTS_SUMMARY}

        if [ $? -ne 0 ]; then
            echo "  WARNING: Failed to extract per-session results"
        fi
    else
        echo "  WARNING: No results JSON file found"
        echo "${i},ALL,ALL,${SUPPORT_SIZE},NA,${FOLD_START_TS},${FOLD_END_TS},${FOLD_DURATION},NO_RESULTS" >> ${RESULTS_SUMMARY}
    fi

    python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()" 2>/dev/null || true
    sleep 2

    echo "Fold ${i} complete (${FOLD_DURATION}s)"
    echo ""
done

# Generate summary statistics
COMPLETION_TS=$(date +%Y%m%d_%H%M%S)
echo "" >> ${TIMING_LOG}
echo "========================================" >> ${TIMING_LOG}
echo "All folds completed: ${COMPLETION_TS}" >> ${TIMING_LOG}
echo "========================================" >> ${TIMING_LOG}

SUMMARY_STATS=$(python -c "
import pandas as pd
import numpy as np

try:
    df = pd.read_csv('${RESULTS_SUMMARY}')
    success_df = df[df['status'] == 'SUCCESS']

    if len(success_df) > 0:
        unique_folds = success_df['fold'].nunique()
        total_sessions = len(success_df)

        strategies = success_df['strategy'].unique()

        print(f'Processed {unique_folds} folds with {total_sessions} total session results')
        print('')
        print('Per-strategy statistics:')

        for strategy in strategies:
            strategy_df = success_df[success_df['strategy'] == strategy]
            r2_vals = pd.to_numeric(strategy_df['r2'], errors='coerce')

            if not r2_vals.isna().all():
                mean_r2 = r2_vals.mean()
                std_r2 = r2_vals.std()
                min_r2 = r2_vals.min()
                max_r2 = r2_vals.max()
                count = len(r2_vals.dropna())

                print(f'  {strategy}:')
                print(f'    Sessions: {count}')
                print(f'    Mean R²: {mean_r2:.4f} ± {std_r2:.4f}')
                print(f'    Range: [{min_r2:.4f}, {max_r2:.4f}]')
                print('')

        print('Per-fold average R²:')
        fold_avg = success_df.groupby('fold')['r2'].apply(lambda x: pd.to_numeric(x, errors='coerce').mean())
        for fold, avg_r2 in fold_avg.items():
            print(f'  Fold {fold}: {avg_r2:.4f}')

        print('')
        print(f'Overall mean R²: {pd.to_numeric(success_df[\"r2\"], errors=\"coerce\").mean():.4f}')
    else:
        print('No successful results found')
except Exception as e:
    print(f'Error computing summary: {e}')
" 2>/dev/null)

echo "========================================="
echo "TTA Coadapt Cross-Validation Complete!"
echo "========================================="
echo "Results directory: ${TTA_OUTPUT_DIR}"
echo ""
echo "$SUMMARY_STATS"
echo ""
echo "Output files:"
echo "  - ${TIMING_LOG}: Detailed timing log"
echo "  - ${RESULTS_SUMMARY}: Per-session results CSV"
echo ""
echo "Per-fold results saved to:"
for i in $(seq 0 $((NUM_FOLDS - 1))); do
    FOLD_TTA_DIR="${TTA_OUTPUT_DIR}/fold${i}"
    if [ -d "$FOLD_TTA_DIR" ]; then
        echo "  - ${FOLD_TTA_DIR}/"
    fi
done
echo ""

echo "" >> ${TIMING_LOG}
echo "========================================" >> ${TIMING_LOG}
echo "Summary Statistics" >> ${TIMING_LOG}
echo "========================================" >> ${TIMING_LOG}
echo "$SUMMARY_STATS" >> ${TIMING_LOG}
echo "" >> ${TIMING_LOG}
echo "========================================" >> ${TIMING_LOG}
echo "Completed at: ${COMPLETION_TS}" >> ${TIMING_LOG}
echo "========================================" >> ${TIMING_LOG}

echo "Full timing log:"
cat ${TIMING_LOG}
