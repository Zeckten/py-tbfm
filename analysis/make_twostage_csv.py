#!/usr/bin/env python3
"""
Build tbfm_cv_data_twostage.csv from the May-30 TTA sweep.

Reads:  /mnt/nsl/danmuir/random_folds_20260530_213515/tta_results_20260601_022544/**
        tbfm_cv_data.csv  (for per-support vanilla baselines + monkey/area/num_channels)
Writes: tbfm_cv_data_twostage.csv
"""

import re
from pathlib import Path

import pandas as pd
import torch

TTA_BASE = Path("/mnt/nsl/danmuir/random_folds_20260530_213515/tta_results_20260601_022544")
REF_CSV = Path("tbfm_cv_data.csv")
OUTPUT_CSV = Path("tbfm_cv_data_twostage.csv")


def load_vanilla_ref() -> pd.DataFrame:
    """Load per-support vanilla baselines from tbfm_cv_data.csv, deduped by (session, support)."""
    df = pd.read_csv(REF_CSV)
    return (
        df.groupby(["session_id", "support_samples"])
        .agg(
            monkey=("monkey", "first"),
            area=("area", "first"),
            num_channels=("num_channels", "first"),
            vanilla_train_r2=("vanilla_train_r2", "first"),
            vanilla_test_r2=("vanilla_test_r2", "first"),
        )
        .reset_index()
    )


def collect_rows(tta_base: Path) -> pd.DataFrame:
    rows = []
    for meta_path in sorted(tta_base.rglob("metadata.torch")):
        # fold7/adapted_models/fold7_support500_inner_outer/<session>/metadata.torch
        parts = meta_path.parts
        fold_dir = None
        session_id = None
        for i, p in enumerate(parts):
            if re.match(r"^fold\d+$", p):
                fold_dir = p
            if i > 0 and parts[i - 1] != "adapted_models" and re.match(
                r"^Monkey[GJ]_\d+_Session\d+_[SM]\d+$", p
            ):
                session_id = p

        if fold_dir is None or session_id is None:
            print(f"  SKIP (parse failed): {meta_path}")
            continue

        fold_num = int(fold_dir.replace("fold", ""))

        try:
            m = torch.load(meta_path, weights_only=False)
        except Exception as e:
            print(f"  SKIP (load error): {meta_path}: {e}")
            continue

        final_r2 = m.get("final_r2")
        support_size = m.get("support_size")
        if final_r2 is None or support_size is None:
            print(f"  SKIP (missing keys): {meta_path}")
            continue

        rows.append(
            {
                "session_id": session_id,
                "fold": fold_num,
                "support_samples": int(support_size),
                "tta_r2": float(final_r2),
            }
        )

    return pd.DataFrame(rows)


def main():
    print(f"Loading vanilla reference from {REF_CSV} …")
    ref = load_vanilla_ref()
    print(f"  {len(ref)} rows  ({ref['session_id'].nunique()} sessions × {ref['support_samples'].nunique()} support sizes)")

    print(f"\nScanning {TTA_BASE} …")
    df = collect_rows(TTA_BASE)
    print(f"  {len(df)} metadata entries found")
    print(
        f"  folds: {sorted(df['fold'].unique())}  "
        f"  support sizes: {sorted(df['support_samples'].unique())}"
    )

    # Join vanilla reference on (session_id, support_samples)
    df = df.merge(ref, on=["session_id", "support_samples"], how="left")

    missing = df["vanilla_test_r2"].isna().sum()
    if missing:
        unmatched = df[df["vanilla_test_r2"].isna()][["session_id", "support_samples"]].drop_duplicates()
        print(f"\n  WARNING: {missing} rows have no vanilla reference")
        print(unmatched.to_string(index=False))

    df["improvement_vs_vanilla"] = df["tta_r2"] - df["vanilla_test_r2"]

    # Reorder columns to match tbfm_cv_data.csv
    df = df[
        [
            "session_id",
            "monkey",
            "area",
            "num_channels",
            "fold",
            "support_samples",
            "tta_r2",
            "vanilla_train_r2",
            "vanilla_test_r2",
            "improvement_vs_vanilla",
        ]
    ].sort_values(["support_samples", "session_id", "fold"]).reset_index(drop=True)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows → {OUTPUT_CSV}")
    print(f"  Sessions:      {df['session_id'].nunique()}")
    print(f"  Support sizes: {sorted(df['support_samples'].unique())}")
    print(f"  Folds:         {df['fold'].nunique()}")

    # Quick sanity: mean tta_r2 per support
    print("\n  Mean final_r2 per support size:")
    for ss, g in df.groupby("support_samples"):
        print(f"    {ss:5d}: {g['tta_r2'].mean():.4f}  (n={len(g)})")


if __name__ == "__main__":
    main()
