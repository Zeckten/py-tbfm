#!/usr/bin/env python3
"""
Package real-coadapt + z-score TTA results into a portable CSV.

Unlike package_coadapt_data.py (which packages the old, pre-restoration run
that was mislabeled "joint"), this packages a run trained AND evaluated after
the Oct 26 restoration: `--coadapt-tta` dispatches to the paper's actual
"coadapt" method (`test_time_adaptation_coadapt`, single shared
forward/backward pass per outer step), and the base models were trained with
`conf/normalizers/default.yaml` set to `ScalerZscore` (the new default,
c^rest left untouched).

Reads:  random_folds_coadapt_20261031_160052/tta_results_1000_*/fold*/adapted_models/*/*/metadata.torch
            (per-session TTA R², read directly rather than from tta_summary.csv --
            see _load_from_metadata for why)
        tbfm_cv_data_twostage.csv  (vanilla R² baseline)
Writes: tbfm_cv_data_coadapt_zscore.csv

The vanilla R² baseline is shared with MAML — it is the session-level performance
of a vanilla model and does not depend on the adaptation strategy.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

COADAPT_DIR  = Path("/mnt/nsl/danmuir/random_folds_coadapt_20261031_160052")
MAML_CSV     = Path("tbfm_cv_data_twostage.csv")
OUTPUT_CSV   = Path("tbfm_cv_data_coadapt_zscore.csv")
SUPPORT_SIZES = [500, 1000, 2500, 5000]
EXPECTED_STRATEGY = "coadapt"


def _latest_tta_dir(coadapt_dir: Path, support_size: int) -> Path | None:
    # tta_random_folds_coadapt.sh names each run's output
    # tta_results_<support>_<timestamp> -- timestamps aren't known ahead of
    # time (especially for a run still in progress across multiple support
    # sizes launched hours/days apart), so glob for them. Pick by data
    # volume (most completed sessions on disk), not just the latest
    # timestamp -- an aborted/restarted run can leave a near-empty dir with
    # a LATER timestamp than the real one (e.g. a run killed seconds after
    # launch, then properly resumed under its original directory), which
    # would otherwise silently win a plain "sort by name, take last"
    # comparison.
    candidates = list(coadapt_dir.glob(f"tta_results_{support_size}_*"))
    if not candidates:
        return None

    def _session_count(d: Path) -> int:
        return sum(1 for _ in d.glob(f"fold*/adapted_models/*_support{support_size}_*/*/metadata.torch"))

    candidates.sort(key=lambda d: (_session_count(d), d.name))
    return candidates[-1]


def _load_from_metadata(tta_dir: Path, support_size: int) -> pd.DataFrame:
    """
    Read every completed session's result directly from its metadata.torch,
    rather than tta_summary.csv. tta_summary.csv is only appended to by the
    SPECIFIC tta_testing.py invocation that computed each row -- a session
    finished in an earlier interrupted attempt and then skipped (not
    recomputed) on resume is silently missing from the summary, even though
    its real result is saved on disk right next to it. Matches this
    project's standing lesson (see memory: "Timing log isn't source of
    truth") -- check per-session result files, not
    tta_summary.csv/timing_log.txt, for true completeness.
    """
    rows = []
    pattern = f"fold*/adapted_models/*_support{support_size}_*/*/metadata.torch"
    for metadata_path in tta_dir.glob(pattern):
        fold_name = metadata_path.relative_to(tta_dir).parts[0]
        session_id = metadata_path.parent.name
        m = torch.load(metadata_path, map_location="cpu")
        strategy_key = m.get("strategy_key")
        if strategy_key != EXPECTED_STRATEGY:
            sys.exit(
                f"ERROR: {metadata_path} has strategy_key={strategy_key!r}, "
                f"expected {EXPECTED_STRATEGY!r}. This script packages real-coadapt "
                f"results specifically -- mixing strategies into one CSV would "
                f"produce the same kind of mislabeling package_coadapt_data.py "
                f"documents from the pre-restoration run."
            )
        rows.append(
            {
                "fold": fold_name,
                "session_id": session_id,
                "support_size": m["support_size"],
                "r2": m["final_r2"],
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Package real-coadapt + z-score TTA data")
    parser.add_argument("--coadapt-dir", default=str(COADAPT_DIR),
                        help="Directory containing the coadapt folds + tta_results_* subdirs")
    parser.add_argument("--maml-csv", default=str(MAML_CSV),
                        help="Path to MAML tbfm_cv_data.csv for vanilla R² baseline")
    parser.add_argument("--output", default=str(OUTPUT_CSV),
                        help="Output CSV path")
    args = parser.parse_args()

    maml_path = Path(args.maml_csv)
    out_path  = Path(args.output)
    coadapt_dir = Path(args.coadapt_dir)

    TTA_DIRS = {}
    for support_size in SUPPORT_SIZES:
        tta_dir = _latest_tta_dir(coadapt_dir, support_size)
        if tta_dir is None:
            print(f"  WARNING: no tta_results_{support_size}_* dir found under {coadapt_dir}")
            continue
        TTA_DIRS[support_size] = tta_dir

    # ── Vanilla R² lookup from MAML CSV ────────────────────────────────────────
    if not maml_path.exists():
        sys.exit(f"ERROR: MAML CSV not found: {maml_path}")
    maml = pd.read_csv(maml_path)

    # Session metadata (monkey/area/num_channels) is fold- AND support-size-
    # invariant -- one value per session is correct here.
    session_meta = (
        maml.groupby("session_id")
        .agg(
            monkey=("monkey", "first"),
            area=("area", "first"),
            num_channels=("num_channels", "first"),
        )
        .reset_index()
    )

    # Vanilla R² is NOT support-size-invariant -- it's the performance of a
    # vanilla model trained on that support set, so it genuinely rises with
    # support size. A session_id-only lookup would silently collapse this to
    # whichever support size happened to appear first per session. Must be
    # keyed on (session_id, support_samples), matching what each coadapt row
    # actually used.
    vanilla_lookup = (
        maml.groupby(["session_id", "support_samples"])
        .agg(
            vanilla_train_r2=("vanilla_train_r2", "first"),
            vanilla_test_r2=("vanilla_test_r2", "first"),
        )
        .reset_index()
    )

    # ── Read coadapt TTA results directly from per-session metadata.torch ──────
    frames = []
    for support_size, tta_dir in TTA_DIRS.items():
        df = _load_from_metadata(tta_dir, support_size)
        if df.empty:
            print(f"  WARNING: no completed sessions found under {tta_dir}")
            continue
        frames.append(df)
        print(f"  support={support_size}: {len(df)} sessions from {tta_dir}")

    if not frames:
        sys.exit("ERROR: no completed TTA sessions found")

    tta = pd.concat(frames, ignore_index=True)
    tta = tta.rename(columns={"r2": "tta_r2", "support_size": "support_samples"})

    # ── Join session metadata, then join vanilla baseline per support size ─────
    merged = tta.merge(session_meta, on="session_id", how="left")
    merged = merged.merge(vanilla_lookup, on=["session_id", "support_samples"], how="left")

    n_missing = merged["vanilla_test_r2"].isna().sum()
    if n_missing:
        missing_pairs = merged.loc[
            merged["vanilla_test_r2"].isna(), ["session_id", "support_samples"]
        ].drop_duplicates()
        print(f"  WARNING: {n_missing} rows missing vanilla R² for (session, support_samples) pairs:")
        print(missing_pairs.to_string(index=False))

    merged["improvement_vs_vanilla"] = merged["tta_r2"] - merged["vanilla_test_r2"]

    # ── Select and order columns ────────────────────────────────────────────────
    out = merged[[
        "session_id", "monkey", "area", "num_channels",
        "fold", "support_samples",
        "tta_r2", "vanilla_train_r2", "vanilla_test_r2",
        "improvement_vs_vanilla",
    ]].sort_values(["support_samples", "session_id", "fold"]).reset_index(drop=True)

    out.to_csv(out_path, index=False)

    print(f"\nWrote {len(out)} rows → {out_path}")
    print(f"  Sessions:      {out['session_id'].nunique()}")
    print(f"  Support sizes: {sorted(out['support_samples'].unique())}")
    print(f"  Folds:         {out['fold'].nunique()}")
    print(f"  Monkeys:       {sorted(out['monkey'].unique())}")
    print(f"  Areas:         {sorted(out['area'].unique())}")


if __name__ == "__main__":
    main()
