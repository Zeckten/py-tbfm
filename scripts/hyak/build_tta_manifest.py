#!/usr/bin/env python3
"""
Build the TTA task manifest for the MAML MCCV z-score sweep on Hyak.

Computes held-out sessions per fold the way tta_testing.py's TTA sweep
actually resolves them: all_session_ids - held_in_sessions, where
all_session_ids comes from dataset.load_meta(TBFM_DATA_DIR) -- but
dataset.load_meta() returns every session passing its own, broader
"Number of Lasers during Conditioning == 2" filter (confirmed live: more
than the 40 sessions used for folds), not just the 40 valid/preprocessed
ones. tta_testing.py's own select_adapt_sessions() handles this by only
keeping candidates that actually have a cached embedding_rest/er.torch file
and taking the first --max-adapt-sessions of those in sorted order -- so we
reproduce that same filter here, against scripts/hyak/valid_sessions.txt
(the 40 sessions actually transferred to Klone, all of which have
embedding_rest present; nothing outside that list does). Since held-in
sessions are always drawn from this same 40-session pool, valid-pool
held-out sessions are always exactly 20/fold, matching the auto-selected set
the existing MAML-ref and coadapt z-score sweeps on kalman produced -- this
is what makes per-session pairing against those runs meaningful.

Reads held-in sessions directly from random_folds_sessions.json instead of
reading back hisi.torch from a trained model -- so this can run before
training finishes, right after the data transfer + env setup are done.

Writes one CSV row per (fold, session, support_size) triple. The TTA sbatch
array indexes into this file by row number (SLURM_ARRAY_TASK_ID), so the
array size is exactly len(manifest), not a hard-coded guess.

Sanity check: aborts if any fold's filtered held-out count isn't exactly 20,
since that would mean a fold's held-in sessions aren't a clean subset of the
40-session valid pool -- which should never happen given random_folds_sessions.json
was verified against the MAML-ref folds, but is cheap to check again here.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from tbfm import dataset

SUPPORT_SIZES = [500, 1000, 2500, 5000]
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions-file", default=str(HERE.parent / "random_folds_sessions.json"))
    parser.add_argument("--valid-sessions-file", default=str(HERE / "valid_sessions.txt"))
    parser.add_argument("--out", required=True, help="Path to write tta_manifest.csv")
    args = parser.parse_args()

    valid_sessions = set(Path(args.valid_sessions_file).read_text().split())
    folds = json.load(open(args.sessions_file))

    data_dir = os.environ["TBFM_DATA_DIR"]
    meta = dataset.load_meta(data_dir)
    all_session_ids = sorted(m.session_id for m in meta)

    rows = []
    for fold_name in sorted(folds, key=lambda k: int(k.replace("fold", ""))):
        held_in = set(folds[fold_name])
        held_out_all = sorted(set(all_session_ids) - held_in)
        held_out = [s for s in held_out_all if s in valid_sessions]
        excluded = [s for s in held_out_all if s not in valid_sessions]

        if excluded:
            print(
                f"{fold_name}: excluding {len(excluded)} held-out session(s) with no "
                f"transferred data (matches select_adapt_sessions' er.torch filter): "
                f"{excluded[:3]}{'...' if len(excluded) > 3 else ''}"
            )

        if len(held_out) != 20:
            sys.exit(
                f"ERROR: {fold_name} has {len(held_out)} valid-pool held-out sessions, "
                f"expected exactly 20. held_in sessions for this fold are not a clean "
                f"subset of the 40-session valid pool -- stop and check "
                f"random_folds_sessions.json before proceeding."
            )

        for session in held_out:
            for support in SUPPORT_SIZES:
                rows.append({"fold": fold_name, "session": session, "support": support})

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "fold", "session", "support"])
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({"task_id": i, **row})

    n_folds = len(folds)
    n_sessions_per_fold = len(rows) // (n_folds * len(SUPPORT_SIZES)) if n_folds else 0
    print(f"Wrote {len(rows)} rows -> {args.out}")
    print(f"  Folds: {n_folds}")
    print(f"  Held-out sessions/fold: {n_sessions_per_fold} (expected 20)")
    print(f"  Support sizes: {SUPPORT_SIZES}")
    print(f"  Array range for sbatch: --array=0-{len(rows) - 1}")


if __name__ == "__main__":
    main()
