#!/usr/bin/env python3
"""
Build the TTA task manifest for the MAML MCCV z-score sweep on Hyak.

Computes held-out sessions per fold exactly the way tta_testing.py's
load_held_in_sessions() does (all_session_ids - held_in_sessions, where
all_session_ids comes from dataset.load_meta(TBFM_DATA_DIR)), but reads
held-in sessions directly from random_folds_sessions.json instead of reading
back hisi.torch from a trained model -- so this can run before training
finishes, right after the data transfer + env setup are done.

Writes one CSV row per (fold, session, support_size) triple. The TTA sbatch
array indexes into this file by row number (SLURM_ARRAY_TASK_ID), so the
array size is exactly len(manifest), not a hard-coded guess.

Safety check: every computed held-out session must be one of the 40 sessions
actually transferred to Klone (scripts/hyak/valid_sessions.txt). If
dataset.load_meta() ever returns a broader session universe than that,
abort rather than let a TTA task later fail (or worse, silently load the
wrong data) for a session we never copied.
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
        held_out = sorted(set(all_session_ids) - held_in)

        bad = [s for s in held_out if s not in valid_sessions]
        if bad:
            sys.exit(
                f"ERROR: {fold_name} held-out sessions include {len(bad)} session(s) "
                f"not in {args.valid_sessions_file} (data for these was never "
                f"transferred to Klone): {bad[:5]}{'...' if len(bad) > 5 else ''}"
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
