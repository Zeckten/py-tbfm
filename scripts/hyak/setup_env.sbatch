#!/bin/bash
#SBATCH --job-name=tbfm-setup-env
#SBATCH --account=rao
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=%x_%j.log

# One-time setup: shallow-clone py-tbfm at a pinned commit and build the
# conda environment in /gscratch/rao. Re-run is safe (wipes and recreates
# both the clone and the env).
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/py-tbfm}"
REPO_URL="https://github.com/Zeckten/py-tbfm.git"
BRANCH="twostage-experiments"
# Pin to the commit pushed for this sweep. If the branch has moved since,
# abort rather than silently run against unreviewed code.
EXPECTED_COMMIT="5b3e1524c00936f82034a1289e8858be359b7a13"
ENV_PREFIX="${ENV_PREFIX:-/gscratch/rao/danmuir/envs/tbfm}"

echo "=== Cloning ${REPO_URL}@${BRANCH} (shallow -- full history is ~39GB and would blow the home quota) ==="
rm -rf "${REPO_DIR}"
git clone --depth 1 --branch "${BRANCH}" "${REPO_URL}" "${REPO_DIR}"
cd "${REPO_DIR}"
ACTUAL_COMMIT=$(git rev-parse HEAD)
echo "Cloned commit: ${ACTUAL_COMMIT}"
if [ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]; then
    echo "ERROR: expected commit ${EXPECTED_COMMIT}, got ${ACTUAL_COMMIT}."
    echo "The branch has moved since this script was written -- stop and confirm before proceeding."
    exit 1
fi

echo "=== Building conda env at ${ENV_PREFIX} ==="
module load conda
mkdir -p "$(dirname "${ENV_PREFIX}")"
rm -rf "${ENV_PREFIX}"
conda create --yes --prefix "${ENV_PREFIX}" python=3.10
conda activate "${ENV_PREFIX}"

pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .

echo "=== Verifying CUDA / GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "
import torch
print('cuda available:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0))
print('device mem (GB):', torch.cuda.get_device_properties(0).total_memory / 1e9)
print('arch list:', torch.cuda.get_arch_list())
import tbfm
print('tbfm importable, version:', getattr(tbfm, '__version__', 'n/a'))
"

echo "=== Done. Env ready at ${ENV_PREFIX}, repo at ${REPO_DIR}@${ACTUAL_COMMIT} ==="
