#!/usr/bin/env python3
"""
Package cross-animal joint-TTA results into the same shape as the existing
MAML cross-animal CSVs (cross_animal_mc_sessions.csv / cross_animal_mc_results.csv).

The run this packages (cross_animal_mc_joint_20261017_091918) was evaluated
with the old `--coadapt-tta` flag, which at the time dispatched to
``test_time_adaptation_joint`` (the alternating embedding/AE strategy) rather
than the paper's actual "coadapt" method (single shared forward/backward
pass, now restored as ``test_time_adaptation_coadapt``). Output is
named/labeled "joint" accordingly, even though the pretraining run directory
itself is still (correctly) named "coadapt" -- the pretraining method and the
TTA strategy are independent axes, and only the TTA strategy was mislabeled.

Reads:  cross_animal_mc_coadapt_*/fold*/tta_results/adapted_models/*/*/metadata.torch
            (per-session TTA R², read directly from metadata.torch rather than
            the run's per_session.csv -- matches this project's standing lesson,
            see memory: "Timing log isn't source of truth" / verify from the
            ground-truth per-session result files, not a summary.)
Writes: cross_animal_mc_joint_sessions.csv  (session-level, mirrors cross_animal_mc_sessions.csv)
        cross_animal_mc_joint_results.csv   (aggregated by fold/model/held_out_animal,
                                               mirrors cross_animal_mc_results.csv)

Note: unlike the MAML sessions CSV, per-session train_r2 and adapt_time_s are
not recorded in metadata.torch for the joint TTA strategy, so those columns
are left blank here rather than fabricated.
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import torch

CROSS_ANIMAL_DIR = Path("/mnt/nsl/danmuir/cross_animal_mc_joint_20261017_091918")
SESSIONS_OUT = Path("cross_animal_mc_joint_sessions.csv")
RESULTS_OUT = Path("cross_animal_mc_joint_results.csv")
EXPECTED_STRATEGY = "joint"


def _animal_from_session_id(session_id: str) -> str:
    m = re.match(r"(Monkey[A-Za-z])_", session_id)
    if not m:
        raise ValueError(f"Could not parse animal from session_id: {session_id}")
    return m.group(1).replace("Monkey", "")


def _load_sessions(cross_animal_dir: Path) -> pd.DataFrame:
    rows = []
    for metadata_path in cross_animal_dir.glob("fold*/tta_results/adapted_models/*/*/metadata.torch"):
        fold_name = metadata_path.relative_to(cross_animal_dir).parts[0]
        m = torch.load(metadata_path, map_location="cpu")
        strategy_key = m.get("strategy_key")
        if strategy_key != EXPECTED_STRATEGY:
            sys.exit(
                f"ERROR: {metadata_path} has strategy_key={strategy_key!r}, "
                f"expected {EXPECTED_STRATEGY!r}. This script packages joint-TTA "
                f"results specifically -- mixing strategies into one CSV would "
                f"silently repeat the exact mislabeling this file was renamed to fix."
            )
        session_id = metadata_path.parent.name
        model = m["model_key"]
        rows.append(
            {
                "fold": fold_name,
                "model": model,
                "held_out_animal": _animal_from_session_id(session_id),
                "session_id": session_id,
                "test_r2": m["final_r2"],
                "train_r2": None,
                "adapt_time_s": None,
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Package cross-animal coadapt TTA data")
    parser.add_argument("--cross-animal-dir", default=str(CROSS_ANIMAL_DIR))
    parser.add_argument("--sessions-output", default=str(SESSIONS_OUT))
    parser.add_argument("--results-output", default=str(RESULTS_OUT))
    args = parser.parse_args()

    cross_animal_dir = Path(args.cross_animal_dir)
    sessions = _load_sessions(cross_animal_dir)
    if sessions.empty:
        raise SystemExit(f"ERROR: no completed sessions found under {cross_animal_dir}")

    sessions = sessions.sort_values(["fold", "model", "held_out_animal", "session_id"]).reset_index(drop=True)
    sessions.to_csv(args.sessions_output, index=False)
    print(f"Wrote {len(sessions)} rows -> {args.sessions_output}")
    print(f"  Folds:  {sorted(sessions['fold'].unique())}")
    print(f"  Models: {sorted(sessions['model'].unique())}")

    results = (
        sessions.groupby(["fold", "model", "held_out_animal"])
        .agg(
            n_sessions=("session_id", "count"),
            test_r2_mean=("test_r2", "mean"),
            test_r2_std=("test_r2", "std"),
            train_r2_mean=("train_r2", "mean"),
            train_r2_std=("train_r2", "std"),
        )
        .reset_index()
        .sort_values(["fold", "model", "held_out_animal"])
    )
    results.to_csv(args.results_output, index=False)
    print(f"Wrote {len(results)} rows -> {args.results_output}")


if __name__ == "__main__":
    main()
