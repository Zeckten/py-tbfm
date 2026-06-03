#!/bin/bash
# Run TTA evaluation on all held-out sessions for each fold from random folds training
# This is the second half of cross-validation: testing on held-out sessions

set -e  # Exit on error

# Check if fold results directory is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_random_folds_results> [existing_tta_output_dir]"
    echo "Example: $0 random_folds_20260108_123456"
    echo "Example (resume): $0 random_folds_20260108_123456 random_folds_20260108_123456/tta_results_2500_20260110_153000"
    exit 1
fi

FOLDS_DIR="$1"
RESUME_DIR="$2"

if [ ! -d "$FOLDS_DIR" ]; then
    echo "ERROR: Directory not found: $FOLDS_DIR"
    exit 1
fi

echo "========================================="
echo "TTA Cross-Validation on Random Folds"
echo "========================================="
echo "Folds directory: $FOLDS_DIR"
echo ""

# TTA parameters
SUPPORT_SIZES="${SUPPORT_SIZES:-500 1000 2500 5000}"
GPU_IDS="${GPU_IDS:-0 1}"
TTA_EPOCHS=7001
NUM_FOLDS=20
FOLD_ORDER="${FOLD_ORDER:-forward}"

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
    TTA_OUTPUT_DIR="${FOLDS_DIR}/tta_results_${TIMESTAMP}"
    mkdir -p ${TTA_OUTPUT_DIR}
fi

# Create logs
TIMING_LOG="${TTA_OUTPUT_DIR}/tta_timing_log.txt"
RESULTS_SUMMARY="${TTA_OUTPUT_DIR}/tta_summary.csv"

# Append on resume; create fresh otherwise
if [ -n "$RESUME_DIR" ] && [ -f "${TIMING_LOG}" ]; then
    echo "" >> ${TIMING_LOG}
    echo "--- Resumed at: ${TIMESTAMP} ---" >> ${TIMING_LOG}
else
    echo "TTA Cross-Validation Log" > ${TIMING_LOG}
    echo "Started at: ${TIMESTAMP}" >> ${TIMING_LOG}
    echo "========================================" >> ${TIMING_LOG}
    echo "" >> ${TIMING_LOG}
fi

# Append summary rows on resume
if [ -n "$RESUME_DIR" ] && [ -f "${RESULTS_SUMMARY}" ]; then
    true  # keep existing rows
else
    echo "fold,session_id,strategy,support_size,r2,start_time,end_time,duration_sec,status" > ${RESULTS_SUMMARY}
fi


# Count available folds
AVAILABLE_FOLDS=0
for i in $(seq 0 $((NUM_FOLDS - 1))); do
    FOLD_DIR="${FOLDS_DIR}/fold${i}"
    if [ -d "$FOLD_DIR" ]; then
        AVAILABLE_FOLDS=$((AVAILABLE_FOLDS + 1))
    fi
done

echo "Found ${AVAILABLE_FOLDS} folds to process"
echo "Support sizes: ${SUPPORT_SIZES}"
echo "TTA epochs: ${TTA_EPOCHS}"
echo "Output directory: ${TTA_OUTPUT_DIR}"
echo ""

# Process each fold
if [ "$FOLD_ORDER" = "reverse" ]; then
    FOLD_SEQ=$(seq $((NUM_FOLDS - 1)) -1 0)
else
    FOLD_SEQ=$(seq 0 $((NUM_FOLDS - 1)))
fi
for i in $FOLD_SEQ; do
    FOLD_DIR="${FOLDS_DIR}/fold${i}"

    if [ ! -d "$FOLD_DIR" ]; then
        echo "WARNING: Fold ${i} directory not found, skipping..."
        continue
    fi

    # Check if model files exist (monolithic or split format)
    if [ ! -f "${FOLD_DIR}/model_nf_1.torch" ] && [ ! -f "${FOLD_DIR}/model.torch" ] \
            && [ ! -f "${FOLD_DIR}/model/tbfm.torch" ]; then
        echo "WARNING: No model file found in fold ${i}, skipping..."
        continue
    fi

    # Check if hisi file exists
    if [ ! -f "${FOLD_DIR}/hisi_nf_1.torch" ] && [ ! -f "${FOLD_DIR}/hisi.torch" ]; then
        echo "WARNING: No hisi file found in fold ${i}, skipping..."
        continue
    fi

    # Check if this fold was already completed — local or in any VM's GCS bucket
    FOLD_TTA_DIR="${TTA_OUTPUT_DIR}/fold${i}"
    _n_sup=$(echo "${SUPPORT_SIZES}" | wc -w)
    _expected_jobs=$(( 20 * _n_sup ))
    _jobs_done=$(find "${FOLD_TTA_DIR}/adapted_models" -name "metadata.torch" 2>/dev/null | wc -l)
    _gcs_json=0
    _gcs_jobs=0
    if [ -n "${BUCKET:-}" ]; then
        _tta_name=$(basename ${TTA_OUTPUT_DIR})
        _gcs_json=$(gsutil ls "gs://${BUCKET}/results/incremental/*/${_tta_name}/fold${i}/tta_support_*.json" 2>/dev/null | wc -l)
        _gcs_jobs=$(gsutil ls "gs://${BUCKET}/results/incremental/*/${_tta_name}/fold${i}/adapted_models/**/metadata.torch" 2>/dev/null | wc -l)
    fi
    if { ls ${FOLD_TTA_DIR}/tta_support_*.json 1>/dev/null 2>&1 \
        || [ "$_jobs_done" -ge "$_expected_jobs" ] \
        || [ "$_gcs_json" -gt 0 ] \
        || [ "$_gcs_jobs" -ge "$_expected_jobs" ]; }; then
        echo "Fold ${i} already completed (local_jobs=${_jobs_done} gcs_jobs=${_gcs_jobs} gcs_json=${_gcs_json}), skipping..."
        continue
    fi

    echo "========================================="
    echo "Processing Fold ${i}"
    echo "========================================="

    FOLD_START=$(date +%s)
    FOLD_START_TS=$(date +%Y%m%d_%H%M%S)

    echo "Fold ${i}: STARTED at ${FOLD_START_TS}" >> ${TIMING_LOG}

    mkdir -p ${FOLD_TTA_DIR}

    # For split-format saves (twostage branch), point at best/ subdir so
    # tta_testing.py finds tbfm.torch; it searches parent for hisi.torch.
    if [ -f "${FOLD_DIR}/best/tbfm.torch" ]; then
        MODEL_PATH="${FOLD_DIR}/best"
    else
        MODEL_PATH="${FOLD_DIR}"
    fi

    # Run TTA for this fold on all held-out sessions
    # Use `&&/||` guard so set -e doesn't trigger before EXIT_CODE is captured
    python tta_testing.py \
        --model-paths fold${i}:${MODEL_PATH} \
        --use-multi-gpu \
        --gpu-ids ${GPU_IDS} \
        --support-sizes ${SUPPORT_SIZES} \
        --max-adapt-sessions 20 \
        --tta-epochs ${TTA_EPOCHS} \
        --output-dir ${FOLD_TTA_DIR} \
        --unfreeze-bases \
        --progressive-unfreezing-threshold 0 \
        --no-plot-display \
        && EXIT_CODE=0 || EXIT_CODE=$?
    FOLD_END=$(date +%s)
    FOLD_DURATION=$((FOLD_END - FOLD_START))
    FOLD_END_TS=$(date +%Y%m%d_%H%M%S)

    # If no results JSON exists but adapted_models do, reconstruct from metadata.torch.
    # Covers both non-zero exit and silent failures (exit 0 but no JSON written).
    if [ -d "${FOLD_TTA_DIR}/adapted_models" ] && ! ls ${FOLD_TTA_DIR}/tta_support_*.json 1>/dev/null 2>&1; then
        echo "No results JSON for fold ${i} — reconstructing from adapted_models"
        ${PYTHON_BIN:-python} -c "
import torch, json, math
from pathlib import Path
BASE = Path('${FOLD_TTA_DIR}')
am = BASE / 'adapted_models'
runs = []
for sd in sorted(am.glob('*_support*')):
    sup = int(sd.name.split('support')[1].split('_')[0])
    ps, finals = {}, []
    for s in sorted(sd.iterdir()):
        m = s / 'metadata.torch'
        if m.exists():
            d = torch.load(m, weights_only=False)
            final_r2 = d.get('final_r2', float('nan'))
            for sid in d.get('adapt_session_ids', []):
                ps[sid] = float(final_r2)
            finals.append(final_r2)
    if ps:
        v = [x for x in finals if not math.isnan(float(x))]
        runs.append({'model': 'fold${i}', 'strategy': 'inner_outer', 'support_size': sup,
                     'r2': float(sum(v)/len(v)) if v else None, 'per_session_r2s': ps})
if runs:
    out = BASE / 'tta_support_reconstructed.json'
    json.dump({'runs': runs}, open(out,'w'))
    print('Reconstructed:', out)
" 2>/dev/null || true
    fi

    if [ ${EXIT_CODE} -ne 0 ] && ! ls ${FOLD_TTA_DIR}/tta_support_*.json 1>/dev/null 2>&1; then
        echo "ERROR: TTA failed for fold ${i}"
        echo "Fold ${i}: FAILED at ${FOLD_END_TS} (duration: ${FOLD_DURATION}s)" >> ${TIMING_LOG}

        # Log failure for all sessions in this fold
        echo "${i},ALL,ALL,ALL,NA,${FOLD_START_TS},${FOLD_END_TS},${FOLD_DURATION},FAILED" >> ${RESULTS_SUMMARY}

        # Don't exit - continue with other folds
        echo "Continuing with remaining folds..."
        echo ""
        continue
    fi

    echo "Fold ${i}: COMPLETED at ${FOLD_END_TS} (duration: ${FOLD_DURATION}s)" >> ${TIMING_LOG}

    # Extract per-session results from the TTA output
    # Look for the most recent results JSON file
    RESULTS_JSON=$(ls -t ${FOLD_TTA_DIR}/tta_support_*.json 2>/dev/null | head -n 1)

    if [ -f "$RESULTS_JSON" ]; then
        echo "  Extracting results from: $(basename $RESULTS_JSON)"

        # Parse JSON and extract per-session R² scores
        ${PYTHON_BIN:-python} -c "
import json
import sys

try:
    with open('${RESULTS_JSON}', 'r') as f:
        results = json.load(f)

    # Extract per-session results
    for run in results.get('runs', []):
        strategy = run.get('strategy', 'unknown')
        support_size = run.get('support_size', 0)
        per_session_r2s = run.get('per_session_r2s', {})

        if per_session_r2s:
            for session_id, r2 in per_session_r2s.items():
                print(f'${i},{session_id},{strategy},{support_size},{r2},${FOLD_START_TS},${FOLD_END_TS},${FOLD_DURATION},SUCCESS')
        else:
            # No per-session data, record overall R²
            overall_r2 = run.get('r2', 'NA')
            print(f'${i},OVERALL,{strategy},{support_size},{overall_r2},${FOLD_START_TS},${FOLD_END_TS},${FOLD_DURATION},SUCCESS')
except Exception as e:
    print(f'ERROR: Failed to parse results: {e}', file=sys.stderr)
    sys.exit(1)
" >> ${RESULTS_SUMMARY} || echo "  WARNING: Failed to extract per-session results"
    else
        echo "  WARNING: No results JSON file found"
        echo "${i},ALL,ALL,ALL,NA,${FOLD_START_TS},${FOLD_END_TS},${FOLD_DURATION},NO_RESULTS" >> ${RESULTS_SUMMARY}
    fi

    # Force GPU memory cleanup between folds
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

# Compute summary statistics
SUMMARY_STATS=$(python -c "
import pandas as pd
import numpy as np

try:
    df = pd.read_csv('${RESULTS_SUMMARY}')

    # Filter successful runs
    success_df = df[df['status'] == 'SUCCESS']

    if len(success_df) > 0:
        # Get unique folds
        unique_folds = success_df['fold'].nunique()
        total_sessions = len(success_df)

        # Compute R² statistics per strategy
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

        # Compute per-fold average R²
        print('Per-fold average R² (across all sessions and strategies):')
        fold_avg = success_df.groupby('fold')['r2'].apply(lambda x: pd.to_numeric(x, errors='coerce').mean())
        for fold, avg_r2 in fold_avg.items():
            print(f'  Fold {fold}: {avg_r2:.4f}')

        print('')
        print(f'Overall mean R² across all sessions and strategies: {pd.to_numeric(success_df[\"r2\"], errors=\"coerce\").mean():.4f}')
    else:
        print('No successful results found')
except Exception as e:
    print(f'Error computing summary: {e}')
" 2>/dev/null)

echo "========================================="
echo "TTA Cross-Validation Complete!"
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

# Write summary to timing log
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
