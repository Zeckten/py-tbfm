# MAML MCCV z-score sweep on Hyak (Klone), rao slice

Full plan: see the project's internal plan notes for context. This directory
holds everything needed to reproduce the random-folds MAML MCCV with the
current z-score normalizer, on the `rao` lab's owned `gpu-rtx6k` slice only.

**Every script here hard-codes `--account=rao --partition=gpu-rtx6k`. Never
run anything in this directory against `ckpt`, `cse`, or `stf`.**

## Layout

- `valid_sessions.txt` — the 40 valid session IDs (same list embedded in
  `../train_random_folds.sh`), one per line. Used to build the minimal rsync
  of input data.
- `env.sh` — sourced by every sbatch script. Sets `TBFM_DATA_DIR`, activates
  the conda env, `cd`s to the repo checkout.
- `setup_env.sbatch` — one-time: shallow-clones this repo at a pinned commit
  and builds the conda environment in `/gscratch/rao`.
- `rsync_data.sh` — run from kalman: copies only the 40 sessions'
  `torchraw/`+`embedding_rest/` (~31GB) plus the three top-level metadata
  files to Klone's scrubbed storage. **Do not `rsync -a` the whole
  `/var/data/opto-coproc/` tree** — it's 379GB of raw block data the loader
  never reads.
- `build_tta_manifest.py` / `build_manifest.sbatch` — computes the real
  held-out-session list per fold (via the same `dataset.load_meta` +
  `hisi.torch` logic `tta_testing.py` uses) and writes
  `$OUT/tta_manifest.csv`, one row per (fold, session, support) triple. The
  TTA array indexes into this file rather than hard-assuming 20 held-out
  sessions/fold.
- `train_mccv_maml.sbatch` — `--array=0-19`, one fold per task.
- `tta_mccv_maml.sbatch` — one task per manifest row, single-session TTA via
  `--adapt-session`.
- `sync_results_back.sh` — run from kalman: rsyncs a finished fold's output
  back to `/mnt/nsl/danmuir/`.

## Why a shallow clone

`git clone` of this repo pulls its full history, which is currently ~39GB
packed (old binary files committed and later removed) — that alone would
blow past home's 10GB quota and eat most of `/gscratch/rao`'s free space.
`setup_env.sbatch` always clones with `--depth 1`, which only fetches the
objects needed to materialize the tip commit (~8MB tree). It also verifies
the cloned commit matches the pinned one and aborts if the branch moved
since this was written — never silently run against unreviewed code.
