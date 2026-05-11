#!/bin/bash
# Run all 7 ablations + no_adapt_ae in parallel on a single multi-GPU VM.
# Each ablation runs on its own GPU, controlled via CUDA_VISIBLE_DEVICES.
# Designed for an a2-highgpu-8g (8x A100) VM — one GPU per ablation, one idle.
#
# Usage (on a GCP VM with at least 7 GPUs):
#   BUCKET=my-bucket bash scripts/cloud/run_all_ablations_parallel.sh <support_size>
#
# Examples:
#   bash scripts/cloud/run_all_ablations_parallel.sh 5000
#
# Each ablation gets its own background process that:
#   1. Pulls its trained model from GCS
#   2. Runs scripts/tta_ablations.sh restricted to that one ablation
#   3. The wrapper rsyncs per-session results to GCS every 2 min
#
# When all background jobs finish, the script exits.

set -euo pipefail

SUPPORT="${1:?usage: $0 <support_size>}"
BUCKET="${BUCKET:?BUCKET env var required}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUNNER="${REPO_ROOT}/scripts/cloud/run_ablation_tta.sh"
WORK_DIR="${WORK_DIR:-${REPO_ROOT}/ablations_remote}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/ablations_remote/parallel_logs}"
mkdir -p "${LOG_DIR}"

# Each ablation gets one GPU. Map via CUDA_VISIBLE_DEVICES so the inner
# tta_testing.py only sees its assigned device.
declare -A ABLATION_GPU=(
    ["no_ae_recon"]=0
    ["with_ortho"]=1
    ["no_rest"]=2
    ["no_l2"]=3
    ["zscore_norm"]=4
    ["no_tanh"]=5
    ["no_adapt_ae"]=6
)

declare -A PIDS

echo "Launching ${#ABLATION_GPU[@]} ablations in parallel at support=${SUPPORT}"
echo "Logs: ${LOG_DIR}/"
echo ""

# Helper invoked per ablation. Defined as a function so it gets called via
# `bash -c "func name args"` with positional args — that way each background
# job's view of $ABL and $GPU is its own argv, not a shared loop variable
# that other iterations might mutate before the subshell starts evaluating.
launch_one() {
    local abl="$1"
    local gpu="$2"
    local support="$3"
    export GPU_FLAGS_OVERRIDE="--cuda-device ${gpu}"
    export TBFM_DATA_DIR="${TBFM_DATA_DIR:-/mnt/data}"
    if [ "${abl}" = "no_adapt_ae" ]; then
        # Reuses baseline model with --no-adapt-ae at TTA time.
        mkdir -p "${WORK_DIR}/baseline"
        gsutil -m -q rsync -r "gs://${BUCKET}/ablations/baseline/" "${WORK_DIR}/baseline/"
        cd "${REPO_ROOT}"
        local watch="${WORK_DIR}/tta_no_adapt_ae_${support}"
        mkdir -p "${watch}"
        bash "${REPO_ROOT}/scripts/cloud/with_incremental_rsync.sh" "${watch}" \
            bash "${REPO_ROOT}/scripts/cloud/retry_on_preemption.sh" \
            python -u tta_testing.py \
                --model-paths "no_adapt_ae:${WORK_DIR}/baseline" \
                --output-dir "${watch}" \
                --support-sizes "${support}" \
                --no-adapt-ae \
                --unfreeze-bases --progressive-unfreezing-threshold 0 \
                --max-adapt-sessions 20 --tta-epochs 7001 \
                --cuda-device "${gpu}"
    else
        bash "${RUNNER}" "${abl}" "${support}"
    fi
}
export -f launch_one
export REPO_ROOT WORK_DIR RUNNER BUCKET

for abl in "${!ABLATION_GPU[@]}"; do
    gpu="${ABLATION_GPU[$abl]}"
    log="${LOG_DIR}/${abl}.log"
    echo "  [${abl}] GPU=${gpu}  log=${log}"

    # Pass abl/gpu/support as positional args so each background job has its own
    # argv. Avoids shared-variable races we've hit with naked subshells.
    bash -c 'launch_one "$@"' _ "${abl}" "${gpu}" "${SUPPORT}" > "${log}" 2>&1 &

    PIDS[$abl]=$!
    sleep 1   # stagger launches slightly to avoid simultaneous gsutil contention
done

echo ""
echo "All ablations launched. PIDs:"
for abl in "${!PIDS[@]}"; do
    echo "  ${abl}: ${PIDS[$abl]}"
done

echo ""
echo "Waiting for all ablations to finish..."
FAIL=0
for abl in "${!PIDS[@]}"; do
    pid="${PIDS[$abl]}"
    if wait "${pid}"; then
        echo "[OK]   ${abl} (pid ${pid})"
    else
        rc=$?
        echo "[FAIL] ${abl} (pid ${pid}, exit ${rc})"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
if [ ${FAIL} -eq 0 ]; then
    echo "All ${#PIDS[@]} ablations completed successfully."
else
    echo "${FAIL}/${#PIDS[@]} ablations failed. See per-ablation logs in ${LOG_DIR}/"
fi
exit ${FAIL}
