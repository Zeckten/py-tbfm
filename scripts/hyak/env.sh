#!/bin/bash
# Sourced by every sbatch script in this directory. Never run standalone on a
# login node -- module load only works on compute nodes on Klone.
set -euo pipefail

export TBFM_DATA_DIR="/gscratch/scrubbed/danmuir/opto-coproc/"

REPO_DIR="${REPO_DIR:-$HOME/py-tbfm}"
ENV_PREFIX="${ENV_PREFIX:-/gscratch/rao/danmuir/envs/tbfm}"

module load conda
conda activate "${ENV_PREFIX}"
cd "${REPO_DIR}"
