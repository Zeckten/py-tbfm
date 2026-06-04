#!/usr/bin/env python3
"""
Portable TTA Cross-Validation Analysis
=======================================
Reads a single CSV (tbfm_cv_data.csv) and produces:

  Fig 1  –  Learning curve with 95 % CI and 95 % PI
  Fig 2  –  Violin plot of R² distributions (TTA vs Vanilla) + improvement
  Fig 3  –  Violin of cross-fold R² deviations within sessions

  P-values (per support size):
    • Sign test          H0: P(TTA > Vanilla) = 0.5
    • Wilcoxon (paired)  H0: mean improvement = 0  (one-sided)
    • Brown-Forsythe     H0: var(TTA) = var(Vanilla)

Usage:
    python portable_analysis.py
    python portable_analysis.py --data path/to/tbfm_cv_data.csv --output-dir figs/

Dependencies: numpy, pandas, matplotlib, seaborn, scipy
"""

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ── Style ─────────────────────────────────────────────────────────────────────
mpl.rcParams["text.usetex"] = False
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 16,
        "axes.titlesize": 13,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 16,
        "figure.dpi": 600,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "mathtext.fontset": "dejavuserif",
    }
)


TTA_COLOR = "#2196F3"  # blue
VAN_COLOR = "#FF5722"  # orange-red
DIFF_COLOR = "#43A047"  # green


# ── I/O helpers ───────────────────────────────────────────────────────────────
def load_data(path: Path) -> pd.DataFrame:
    """Load and validate the packaged CSV."""
    df = pd.read_csv(path)
    required = {
        "session_id",
        "fold",
        "support_samples",
        "tta_r2",
        "vanilla_test_r2",
        "vanilla_train_r2",
        "improvement_vs_vanilla",
    }
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: CSV is missing columns: {missing}")
    return df


# ── Aggregation ───────────────────────────────────────────────────────────────
def compute_session_means(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average TTA R² across folds for each (session, support_size).
    Vanilla R² is fold-invariant so we just take the first value.
    """
    g = df.groupby(["session_id", "support_samples"])
    out = g.agg(
        tta_mean=("tta_r2", "mean"),
        tta_std=("tta_r2", "std"),
        n_folds=("tta_r2", "count"),
        vanilla_r2=("vanilla_test_r2", "first"),
        monkey=("monkey", "first"),
        area=("area", "first"),
    ).reset_index()
    out["improvement"] = out["tta_mean"] - out["vanilla_r2"]
    return out


def compute_lc_stats(smeans: pd.DataFrame) -> pd.DataFrame:
    """
    For each support size compute mean, 95 % CI, and 95 % PI across sessions.

    CI: x̄ ± t * SE         (uncertainty about the mean)
    PI: x̄ ± t * SD√(1+1/n) (range for a new session)
    """
    rows = []
    for ss, g in smeans.groupby("support_samples"):
        tta = g["tta_mean"].values
        van = g["vanilla_r2"].values
        n = len(tta)
        tc = stats.t.ppf(0.975, df=n - 1)

        def _stats(x):
            mu = x.mean()
            se = x.std(ddof=1) / np.sqrt(n)
            sd = x.std(ddof=1)
            return (
                mu,
                mu - tc * se,
                mu + tc * se,
                mu - tc * sd * np.sqrt(1 + 1 / n),
                mu + tc * sd * np.sqrt(1 + 1 / n),
            )

        t_mu, t_ci_lo, t_ci_hi, t_pi_lo, t_pi_hi = _stats(tta)
        v_mu, v_ci_lo, v_ci_hi, v_pi_lo, v_pi_hi = _stats(van)

        rows.append(
            {
                "support_samples": ss,
                "n": n,
                "tta_mean": t_mu,
                "tta_ci_lo": t_ci_lo,
                "tta_ci_hi": t_ci_hi,
                "tta_pi_lo": t_pi_lo,
                "tta_pi_hi": t_pi_hi,
                "van_mean": v_mu,
                "van_ci_lo": v_ci_lo,
                "van_ci_hi": v_ci_hi,
                "van_pi_lo": v_pi_lo,
                "van_pi_hi": v_pi_hi,
            }
        )
    return pd.DataFrame(rows).sort_values("support_samples").reset_index(drop=True)


# ── Figure 1: Learning curve ──────────────────────────────────────────────────
def fig1_learning_curve(lc: pd.DataFrame, out: Path) -> None:
    fontsize_main = 16

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(lc))
    labels = lc["support_samples"].astype(str).tolist()

    # TTA: PI as shaded region, CI as error bars
    ax.fill_between(
        x,
        lc["tta_pi_lo"],
        lc["tta_pi_hi"],
        color=TTA_COLOR,
        alpha=0.15,
        label="MAML 95% PI",
    )
    ax.errorbar(
        x,
        lc["tta_mean"],
        yerr=[lc["tta_mean"] - lc["tta_ci_lo"], lc["tta_ci_hi"] - lc["tta_mean"]],
        fmt="o-",
        color=TTA_COLOR,
        lw=2,
        ms=7,
        capsize=5,
        capthick=1.5,
        elinewidth=1.5,
        label="MAML mean ± 95% CI",
    )

    # Vanilla: PI as shaded region, CI as error bars
    ax.fill_between(
        x,
        lc["van_pi_lo"],
        lc["van_pi_hi"],
        color=VAN_COLOR,
        alpha=0.15,
        label="Vanilla 95% PI",
    )
    ax.errorbar(
        x,
        lc["van_mean"],
        yerr=[lc["van_mean"] - lc["van_ci_lo"], lc["van_ci_hi"] - lc["van_mean"]],
        fmt="s--",
        color=VAN_COLOR,
        lw=2,
        ms=7,
        capsize=5,
        capthick=1.5,
        elinewidth=1.5,
        label="Vanilla mean ± 95% CI",
    )

    fontsize_ticks = fontsize_main - 2
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=fontsize_ticks)
    ax.tick_params(axis="y", labelsize=fontsize_ticks)
    ax.set_xlabel("Calibration set size", fontsize=fontsize_main)
    ax.set_ylabel("Test set $R^2$", fontsize=fontsize_main)
    # ax.set_title("Fig 1  ·  Learning curve with CI and PI")
    ax.legend(loc="lower right", framealpha=0.75, fontsize=fontsize_main - 1)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    ax.set_ylim(-0.51, 1.05)
    ax.set_autoscaley_on(False)
    ax.margins(y=0)

    path = out / "scaling_curve.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 2: Violin plot ─────────────────────────────────────────────────────
def fig2_violin_compare(
    smeans: pd.DataFrame, out: Path, smeans_coadapt: pd.DataFrame = None
) -> None:
    # --- reshape to long form ---
    van_df = smeans[["session_id", "support_samples", "vanilla_r2"]].copy()
    van_df = van_df.rename(columns={"vanilla_r2": "r2"})
    van_df["method"] = "Vanilla"

    tta_df = smeans[["session_id", "support_samples", "tta_mean"]].copy()
    tta_df = tta_df.rename(columns={"tta_mean": "r2"})
    tta_df["method"] = "MAML"

    long = pd.concat([van_df, tta_df], ignore_index=True)

    # Add coadapt data if provided
    if smeans_coadapt is not None:
        coadapt_df = smeans_coadapt[
            ["session_id", "support_samples", "tta_mean"]
        ].copy()
        coadapt_df = coadapt_df.rename(columns={"tta_mean": "r2"})
        coadapt_df["method"] = "Coadapt"
        long = pd.concat([long, coadapt_df], ignore_index=True)

    long["support_samples"] = long["support_samples"].astype(str)

    support_order = [str(s) for s in sorted(smeans["support_samples"].unique())]

    fig, ax = plt.subplots(figsize=(6, 5))

    palette = {"Vanilla": VAN_COLOR, "MAML": TTA_COLOR}
    if smeans_coadapt is not None:
        palette["Coadapt"] = "#9C27B0"  # purple
        hue_order = ["Vanilla", "Coadapt", "MAML"]
    else:
        hue_order = ["Vanilla", "MAML"]

    sns.violinplot(
        data=long,
        x="support_samples",
        y="r2",
        hue="method",
        hue_order=hue_order,
        split=False if smeans_coadapt is not None else True,
        inner="quart",
        palette=palette,
        order=support_order,
        ax=ax,
        linewidth=1.2,
    )
    # ax.set_title("R² distributions: MAML vs Vanilla")
    ax.set_xlabel("Calibration set size")
    ax.set_ylabel("Test set $R^2$")
    ax.legend(title=None)
    ax.grid(True, axis="y", alpha=0.35)

    fig.tight_layout()
    path = out / "violin_compare.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def fig2_violin_improvement(smeans: pd.DataFrame, out: Path) -> None:
    impr = smeans[["support_samples", "improvement"]].copy()
    impr["support_samples"] = impr["support_samples"].astype(str)

    support_order = [str(s) for s in sorted(smeans["support_samples"].unique())]

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.violinplot(
        data=impr,
        x="support_samples",
        y="improvement",
        color=DIFF_COLOR,
        inner="quart",
        order=support_order,
        ax=ax,
        linewidth=1.2,
    )
    ax.axhline(0, color="k", lw=1.5)
    # ax.set_title("Improvement: MAML − Vanilla")
    ax.set_xlabel("Calibration set size")
    ax.set_ylabel("Test set $\Delta R^2$")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.35)

    fig.tight_layout()
    path = out / "violin_improvement.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 3: Within-session cross-fold variability ───────────────────────────
def fig3_fold_variability(df: pd.DataFrame, out: Path) -> None:
    """
    For each (session, support_size) compute each fold's deviation from the
    session's mean R².  The violin shows how much fold assignment shifts R²
    within a session.
    """
    rows = []
    for (sid, ss), g in df.groupby(["session_id", "support_samples"]):
        r2s = g["tta_r2"].values
        if len(r2s) < 2:
            continue
        mu = r2s.mean()
        for r2 in r2s:
            rows.append(
                {
                    "session_id": sid,
                    "support_samples": str(ss),
                    "deviation": r2 - mu,
                }
            )
    dev = pd.DataFrame(rows)
    support_order = [str(s) for s in sorted(df["support_samples"].unique())]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.violinplot(
        data=dev,
        x="support_samples",
        y="deviation",
        color=TTA_COLOR,
        inner="quart",
        ax=ax,
        order=support_order,
        linewidth=1.2,
    )
    ax.axhline(0, color="k", lw=1.5, linestyle="--")
    ax.set_xlabel("Calibration set size")
    ax.set_ylabel("Test set $R^2$\ndeviation from session mean")
    # ax.set_title("Fig 3  ·  Cross-fold variability within sessions")
    ax.grid(True, axis="y", alpha=0.35)

    fig.tight_layout()
    path = out / "fold_variability.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── P-value table ─────────────────────────────────────────────────────────────
def compute_pvalues(smeans: pd.DataFrame) -> pd.DataFrame:
    """
    Per support size:
      1. Sign test (binomial)   H0: P(MAML > Vanilla) = 0.5
      2. Wilcoxon signed-rank   H0: median improvement = 0  (one-sided, >)
      3. Brown-Forsythe         H0: var(MAML) = var(Vanilla)
    """
    rows = []
    for ss, g in smeans.groupby("support_samples"):
        tta = g["tta_mean"].values
        van = g["vanilla_r2"].values
        diff = tta - van
        n = len(diff)

        # 1. Sign test
        n_wins = int((diff > 0).sum())
        win_rate = n_wins / n
        binom_p = stats.binomtest(n_wins, n, p=0.5, alternative="greater").pvalue

        # 2. Wilcoxon signed-rank (one-sided: improvement > 0)
        if np.all(diff == 0):
            w_stat, w_p = np.nan, np.nan
        else:
            w_res = stats.wilcoxon(diff, alternative="greater")
            w_stat, w_p = w_res.statistic, w_res.pvalue

        # 3. Brown-Forsythe (levene with center='median')
        bf_stat, bf_p = stats.levene(tta, van, center="median")

        rows.append(
            {
                "support_samples": ss,
                "n_sessions": n,
                "win_rate": win_rate,
                "n_wins": n_wins,
                "win_rate_p": binom_p,
                "wilcoxon_W": w_stat,
                "wilcoxon_p": w_p,
                "BF_stat": bf_stat,
                "BF_p": bf_p,
            }
        )
    return pd.DataFrame(rows)


def print_pvalues(pv: pd.DataFrame) -> None:
    rule = "=" * 68
    print()
    print(rule)
    print("  P-VALUE SUMMARY")
    print(rule)
    hdr = (
        f"{'Support':>8}  {'n':>4}  "
        f"{'Win%':>6}  {'Binom p':>10}  "
        f"{'Wilcox p':>10}  {'BF p':>10}"
    )
    print(hdr)
    print("-" * 68)
    for _, row in pv.iterrows():
        print(
            f"{int(row['support_samples']):>8}  "
            f"{int(row['n_sessions']):>4}  "
            f"{row['win_rate']:>5.1%}  "
            f"{row['win_rate_p']:>10.3g}  "
            f"{row['wilcoxon_p']:>10.3g}  "
            f"{row['BF_p']:>10.3g}"
        )
    print(rule)
    print("  Win%     = fraction of sessions where MAML R² > Vanilla R²")
    print("  Binom p  = sign test H0: P(win)=50%  (one-sided, >)")
    print("  Wilcox p = Wilcoxon signed-rank H0: improvement=0  (one-sided, >)")
    print("  BF p     = Brown-Forsythe H0: var(MAML)=var(Vanilla)")
    print(rule)


def compute_maml_vs_coadapt_pvalues(
    smeans: pd.DataFrame, smeans_coadapt: pd.DataFrame
) -> pd.DataFrame:
    """
    Per support size, test differences between MAML and Coadapt using Wilcoxon signed-rank.
    H0: median(MAML - Coadapt) = 0  (paired test)
    """
    rows = []
    for ss, g_maml in smeans.groupby("support_samples"):
        g_coadapt = smeans_coadapt[smeans_coadapt["support_samples"] == ss]

        # Get MAML and Coadapt R² values (matched by session)
        maml_vals = g_maml.set_index("session_id")["tta_mean"]
        coadapt_vals = g_coadapt.set_index("session_id")["tta_mean"]

        # Find common sessions
        common_sessions = maml_vals.index.intersection(coadapt_vals.index)
        if len(common_sessions) < 2:
            continue

        maml = maml_vals.loc[common_sessions].values
        coadapt = coadapt_vals.loc[common_sessions].values
        diff = maml - coadapt
        n = len(diff)

        # Wilcoxon signed-rank test (two-sided)
        if np.all(diff == 0):
            w_stat, w_p = np.nan, np.nan
        else:
            w_res = stats.wilcoxon(diff, alternative="two-sided")
            w_stat, w_p = w_res.statistic, w_res.pvalue

        # Brown-Forsythe test (H0: var(MAML) = var(Coadapt))
        bf_stat, bf_p = stats.levene(maml, coadapt, center="median")

        rows.append(
            {
                "support_samples": ss,
                "n_sessions": n,
                "wilcoxon_W": w_stat,
                "wilcoxon_p": w_p,
                "BF_stat": bf_stat,
                "BF_p": bf_p,
            }
        )
    return pd.DataFrame(rows)


def print_maml_vs_coadapt_pvalues(pv: pd.DataFrame) -> None:
    rule = "=" * 82
    print()
    print(rule)
    print("  MAML vs COADAPT COMPARISON")
    print(rule)
    hdr = f"{'Support':>8}  {'n':>4}  {'Wilcox W':>12}  {'Wilcox p':>10}  {'BF p':>10}"
    print(hdr)
    print("-" * 82)
    for _, row in pv.iterrows():
        print(
            f"{int(row['support_samples']):>8}  "
            f"{int(row['n_sessions']):>4}  "
            f"{row['wilcoxon_W']:>12.1f}  "
            f"{row['wilcoxon_p']:>10.3g}  "
            f"{row['BF_p']:>10.3g}"
        )
    print(rule)
    print("  Wilcox p = Wilcoxon signed-rank H0: MAML = Coadapt  (two-sided)")
    print("  BF p     = Brown-Forsythe H0: var(MAML) = var(Coadapt)")
    print(rule)


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Portable TTA analysis: 3 figures + p-values"
    )
    parser.add_argument(
        "--data",
        default="tbfm_cv_data.csv",
        help="Path to packaged CSV  (default: tbfm_cv_data.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="figures",
        help="Directory for output PNGs  (default: figures/)",
    )
    parser.add_argument(
        "--coadapt-data",
        default="tbfm_cv_data_coadapt.csv",
        help="Path to coadapt CSV (same format as --data) for MAML vs Coadapt comparison",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {data_path} …")
    df = load_data(data_path)
    print(
        f"  {len(df)} rows  |  "
        f"{df['session_id'].nunique()} sessions  |  "
        f"{df['fold'].nunique()} folds  |  "
        f"support sizes: {sorted(df['support_samples'].unique())}"
    )

    smeans = compute_session_means(df)
    lc = compute_lc_stats(smeans)

    if args.coadapt_data:
        coadapt_path = Path(args.coadapt_data)
        print(f"\nLoading coadapt data from {coadapt_path} …")
        df_coadapt = load_data(coadapt_path)
        smeans_coadapt = compute_session_means(df_coadapt)
        lc_coadapt = compute_lc_stats(smeans_coadapt)
        print(
            f"  {len(df_coadapt)} rows  |  "
            f"{df_coadapt['session_id'].nunique()} sessions  |  "
            f"{df_coadapt['fold'].nunique()} folds  |  "
            f"support sizes: {sorted(df_coadapt['support_samples'].unique())}"
        )
    else:
        smeans_coadapt = None
        lc_coadapt = None

    # Print means and variances by method and support sample size
    print("\n" + "=" * 70)
    print("Means and Variances by Method and Support Sample Size")
    print("=" * 70)
    for ss in sorted(smeans["support_samples"].unique()):
        ss_data = smeans[smeans["support_samples"] == ss]
        print(f"\nSupport size: {ss}")
        print(
            f"  MAML R²     — mean: {ss_data['tta_mean'].mean():.4f}, var: {ss_data['tta_mean'].var():.6f}"
        )
        print(
            f"  Vanilla R²  — mean: {ss_data['vanilla_r2'].mean():.4f}, var: {ss_data['vanilla_r2'].var():.6f}"
        )
        if smeans_coadapt is not None:
            coadapt_data = smeans_coadapt[smeans_coadapt["support_samples"] == ss]
            print(
                f"  Coadapt R²  — mean: {coadapt_data['tta_mean'].mean():.4f}, var: {coadapt_data['tta_mean'].var():.6f}"
            )

    # Count sessions with R² below thresholds
    print("\n" + "=" * 70)
    print("Count of Sessions with R² below Threshold")
    print("=" * 70)

    support_sizes = sorted(smeans["support_samples"].unique())

    # Threshold 0.05
    print("\nR² < 0.05:")
    header = f"{'Support Size':<15} {'MAML Count':<15} {'Vanilla Count':<15}"
    if smeans_coadapt is not None:
        header += f" {'Coadapt Count':<15}"
    print(header)
    print("-" * (45 if smeans_coadapt is None else 60))
    for ss in support_sizes:
        ss_data = smeans[smeans["support_samples"] == ss]
        tta_below = (ss_data["tta_mean"] < 0.05).sum()
        van_below = (ss_data["vanilla_r2"] < 0.05).sum()
        line = f"{ss:<15} {tta_below:<15} {van_below:<15}"
        if smeans_coadapt is not None:
            coadapt_data = smeans_coadapt[smeans_coadapt["support_samples"] == ss]
            coadapt_below = (coadapt_data["tta_mean"] < 0.05).sum()
            line += f" {coadapt_below:<15}"
        print(line)

    # Threshold 0.15
    print("\nR² < 0.15:")
    header = f"{'Support Size':<15} {'MAML Count':<15} {'Vanilla Count':<15}"
    if smeans_coadapt is not None:
        header += f" {'Coadapt Count':<15}"
    print(header)
    print("-" * (45 if smeans_coadapt is None else 60))
    for ss in support_sizes:
        ss_data = smeans[smeans["support_samples"] == ss]
        tta_below = (ss_data["tta_mean"] < 0.15).sum()
        van_below = (ss_data["vanilla_r2"] < 0.15).sum()
        line = f"{ss:<15} {tta_below:<15} {van_below:<15}"
        if smeans_coadapt is not None:
            coadapt_data = smeans_coadapt[smeans_coadapt["support_samples"] == ss]
            coadapt_below = (coadapt_data["tta_mean"] < 0.15).sum()
            line += f" {coadapt_below:<15}"
        print(line)

    print("\nGenerating figures …")
    fig1_learning_curve(lc, out_dir)
    fig2_violin_compare(smeans, out_dir, smeans_coadapt)
    fig2_violin_improvement(smeans, out_dir)
    fig3_fold_variability(df, out_dir)

    pv = compute_pvalues(smeans)
    print_pvalues(pv)

    if smeans_coadapt is not None:
        pv_maml_vs_coadapt = compute_maml_vs_coadapt_pvalues(smeans, smeans_coadapt)
        print_maml_vs_coadapt_pvalues(pv_maml_vs_coadapt)

    pv_path = out_dir / "pvalues.csv"
    pv.to_csv(pv_path, index=False)
    print(f"\n  P-value table saved to {pv_path}")


if __name__ == "__main__":
    main()
