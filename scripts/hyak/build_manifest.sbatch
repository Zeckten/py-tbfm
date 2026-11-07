#!/bin/bash
#SBATCH --job-name=tbfm-build-manifest
#SBATCH --account=rao
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:15:00
#SBATCH --output=%x_%j.log

# Builds $OUT/tta_manifest.csv. Needs the conda env (for `import tbfm`) and
# the data transfer done (dataset.load_meta reads table_of_experiments.csv),
# but NOT trained models -- held-in sessions come from random_folds_sessions.json
# directly, so this can run as soon as setup_env + rsync_data are done.
set -euo pipefail

# Not $(dirname "$0") -- Slurm copies the submitted script into a spool dir,
# so $0 does not point at its location in the repo checkout.
source "${REPO_DIR:-$HOME/py-tbfm}/scripts/hyak/env.sh"

OUT="${OUT:?Set OUT to the output base dir, e.g. --export=OUT=/gscratch/scrubbed/danmuir/random_folds_maml_zscore_<ts>}"
mkdir -p "${OUT}"

python scripts/hyak/build_tta_manifest.py --out "${OUT}/tta_manifest.csv"
