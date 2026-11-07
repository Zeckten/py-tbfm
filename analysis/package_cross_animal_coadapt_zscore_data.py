#!/usr/bin/env python3
"""
Package real-coadapt + z-score cross-animal TTA results into the same shape
as the existing cross-animal CSVs (cross_animal_mc_sessions.csv /
cross_animal_mc_results.csv, cross_animal_mc_joint_sessions.csv /
cross_animal_mc_joint_results.csv).

Unlike package_cross_animal_coadapt_data.py (which packages
cross_animal_mc_joint_20261017_091918 -- trained with the old IQR
normalizer and TTA'd before the Oct 26 coadapt restoration, so its TTA
strategy_key is actually "joint", not real coadapt), this packages a run
trained AND evaluated after the restoration: base models were trained with
conf/normalizers/default.yaml set to ScalerZscore (the current default), and
`--coadapt-tta` dispatches to the paper's actual "coadapt" method
(test_time_adaptation_coadapt, single shared forward/backward pass).

Reads:  cross_animal_mc_coadapt_zscore_*/fold*/tta_results*/adapted_models/*/*/metadata.torch
Writes: cross_animal_mc_coadapt_zscore_sessions.csv
        cross_animal_mc_coadapt_zscore_results.csv
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import torch

CROSS_ANIMAL_DIR = Path("/mnt/nsl/danmuir/cross_animal_mc_coadapt_zscore_20261104_205434")
SESSIONS_OUT = Path("cross_animal_mc_coadapt_zscore_sessions.csv")
RESULTS_OUT = Path("cross_animal_mc_coadapt_zscore_results.csv")
EXPECTED_STRATEGY = "coadapt"


def _animal_from_session_id(session_id: str) -> str:
    m = re.match(r"(Monkey[A-Za-z])_", session_id)
    if not m:
        raise ValueError(f"Could not parse animal from session_id: {session_id}")
    return m.group(1).replace("Monkey", "")


def _load_sessions(cross_animal_dir: Path) -> pd.DataFrame:
    rows = []
    for metadata_path in cross_animal_dir.glob("fold*/tta_results*/adapted_models/*/*/metadata.torch"):
        fold_name = metadata_path.relative_to(cross_animal_dir).parts[0]
        m = torch.load(metadata_path, map_location="cpu")
        strategy_key = m.get("strategy_key")
        if strategy_key != EXPECTED_STRATEGY:
            sys.exit(
                f"ERROR: {metadata_path} has strategy_key={strategy_key!r}, "
                f"expected {EXPECTED_STRATEGY!r}. This script packages real-coadapt "
                f"results specifically -- mixing strategies into one CSV would "
                f"produce the same kind of mislabeling package_cross_animal_coadapt_data.py "
                f"documents from the pre-restoration run."
            )
        session_id = metadata_path.parent.name
        model = m["model_key"]
        rows.append(
            {
                "fold": fold_name,
                "model": model,
                "held_out_animal": _animal_from_session_id(session_id),
                "session_id": session_id,
                "support_samples": m["support_size"],
                "test_r2": m["final_r2"],
                "train_r2": None,
                "adapt_time_s": None,
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Package real-coadapt + z-score cross-animal TTA data")
    parser.add_argument("--cross-animal-dir", default=str(CROSS_ANIMAL_DIR))
    parser.add_argument("--sessions-output", default=str(SESSIONS_OUT))
    parser.add_argument("--results-output", default=str(RESULTS_OUT))
    args = parser.parse_args()

    cross_animal_dir = Path(args.cross_animal_dir)
    sessions = _load_sessions(cross_animal_dir)
    if sessions.empty:
        raise SystemExit(f"ERROR: no completed sessions found under {cross_animal_dir}")

    sessions = sessions.sort_values(
        ["fold", "model", "held_out_animal", "session_id"]
    ).reset_index(drop=True)
    sessions.to_csv(args.sessions_output, index=False)
    print(f"Wrote {len(sessions)} rows -> {args.sessions_output}")
    print(f"  Folds:  {sorted(sessions['fold'].unique())}")
    print(f"  Models: {sorted(sessions['model'].unique())}")
    print(f"  Support sizes: {sorted(sessions['support_samples'].unique())}")

    results = (
        sessions.groupby(["fold", "model", "held_out_animal", "support_samples"])
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
