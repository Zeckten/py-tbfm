#!/usr/bin/env python3
"""
Portable TTA Cross-Validation Analysis
=======================================
Reads the MAML CSV (--data) and the coadapt CSV (--coadapt-data) and
regenerates every figure, statistic and number the paper cites.

Statistical unit: one value per (session, support size).  For MAML and coadapt
this is the mean of tta_r2 over the folds in which the session was held out;
vanilla is a single fold-invariant value per session.  n = 40 sessions.

Figures (written to --output-dir):
  Fig 1  –  Learning curve with 95 % CI and 95 % PI (scaling_curve.png);
            PI-free version with Coadapt added (scaling_curve_ci.png)
  Fig 2  –  Horizontal box plots with data points: R² by method + improvement
  Fig 3  –  Violin of cross-fold R² deviations within sessions

Legacy p-value tables (pvalues.csv and the console tables): MAML vs Vanilla and
MAML vs Coadapt, per support size.

Paired comparisons (reference -> new), configured in PAIRS:
  A  Vanilla -> MAML     B  Vanilla -> Coadapt     C  Coadapt -> MAML
  • Wilcoxon signed-rank (sidedness per pair), sign test (pair A), and the
    number of sessions where new > reference
  • Brown-Forsythe, one-sided, H1: var(new) < var(reference), computed
    directly as a pooled t-test on |x - median(x)| (the t statistic is
    reported; the two-sided version is checked against scipy's levene)
  • Exact McNemar test on failures (R² below 0.05 and 0.15)
  • CVaR (mean R² of the worst 25 % of sessions) with a paired bootstrap CI
  • Holm correction across the four support sizes within each (test, pair)
  • Fold-noise sanity check and a "Numbers cited in paper" block

Additional outputs: stats_report.md, stats_long.csv, supplement_table.tex.

Data integrity is asserted first (identical sessions in both CSVs, identical
vanilla R² in both CSVs); the script stops with a clear error otherwise.

Usage:
    python portable_analysis.py
    python portable_analysis.py --data a.csv --coadapt-data b.csv --output-dir figs/ --seed 0

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
COADAPT_COLOR = "#9C27B0"  # purple

# ── Test configuration ────────────────────────────────────────────────────────
METHOD_LABELS = {"vanilla": "Vanilla", "maml": "MAML", "coadapt": "Coadapt"}
ALT_TEXT = {"greater": "one-sided, >", "less": "one-sided, <", "two-sided": "two-sided"}

# (reference, new): "new" is the method expected to do better.  Sidedness of
# every test lives here.  Wilcoxon / sign / McNemar "greater" means new > ref
# (McNemar: more reference-only failures than new-only failures); BF "less"
# means var(new) < var(ref).
PAIRS = {
    "A": {
        "ref": "vanilla",
        "new": "maml",
        "wilcoxon_alt": "greater",
        "sign_alt": "greater",
        "bf_alt": "less",
        "mcnemar_alt": "greater",
    },
    "B": {
        "ref": "vanilla",
        "new": "coadapt",
        "wilcoxon_alt": "greater",
        "sign_alt": None,
        "bf_alt": "less",
        "mcnemar_alt": "greater",
    },
    "C": {
        "ref": "coadapt",
        "new": "maml",
        "wilcoxon_alt": "two-sided",
        "sign_alt": None,
        "bf_alt": "less",
        "mcnemar_alt": "greater",
    },
}
THRESHOLDS = (0.05, 0.15)
CVAR_Q = 0.25
N_BOOT = 10_000
MIN_DISCORDANT = 5  # McNemar p is NaN below this many discordant sessions
FOLD_NOISE_TOL = 0.20  # flag if single-fold SD differs from session-mean SD by more
VANILLA_TOL = 1e-9
SESSION_KEY = ["session_id", "support_samples"]


def brown_forsythe(new, ref, alternative):
    """
    Brown-Forsythe test as a pooled t-test on absolute deviations from the
    median.  H1 'less' means var(new) < var(ref).  Returns (t, p).  The groups
    are paired but BF treats them as independent (conservative).
    """
    d_new = np.abs(new - np.median(new))
    d_ref = np.abs(ref - np.median(ref))
    res = stats.ttest_ind(d_new, d_ref, equal_var=True, alternative=alternative)
    return float(res.statistic), float(res.pvalue)


def bf_matches_levene(new, ref) -> float:
    """Absolute gap between the two-sided BF p above and scipy's levene p."""
    _, p2 = brown_forsythe(new, ref, "two-sided")
    return abs(p2 - stats.levene(new, ref, center="median").pvalue)


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
def fig1_learning_curve(
    lc: pd.DataFrame,
    out: Path,
    lc_coadapt: pd.DataFrame = None,
    show_pi: bool = True,
) -> None:
    """
    Learning curve of mean test R² with 95 % CIs.

    show_pi:     shade the 95 % PI bands (saved as scaling_curve.png);
                 otherwise saved as scaling_curve_ci.png.
    lc_coadapt:  if given, adds a Coadapt curve (from compute_lc_stats).
    """
    fontsize_main = 16

    fig, ax = plt.subplots(figsize=(6, 3.2))
    x = np.arange(len(lc))
    labels = lc["support_samples"].astype(str).tolist()

    curves = [
        ("Vanilla", "van", lc, VAN_COLOR, "s--"),
    ]
    if lc_coadapt is not None:
        curves.append(("Coadapt", "tta", lc_coadapt, COADAPT_COLOR, "^-."))

    curves.append(("MAML", "tta", lc, TTA_COLOR, "o-"))

    for name, prefix, data, color, fmt in curves:
        if show_pi:
            ax.fill_between(
                x,
                data[f"{prefix}_pi_lo"],
                data[f"{prefix}_pi_hi"],
                color=color,
                alpha=0.15,
                label=f"{name} 95% PI",
            )
        ax.errorbar(
            x,
            data[f"{prefix}_mean"],
            yerr=[
                data[f"{prefix}_mean"] - data[f"{prefix}_ci_lo"],
                data[f"{prefix}_ci_hi"] - data[f"{prefix}_mean"],
            ],
            fmt=fmt,
            color=color,
            lw=2,
            ms=7,
            capsize=5,
            capthick=1.5,
            elinewidth=1.5,
            label=f"{name}",
        )

    fontsize_ticks = fontsize_main - 2
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=fontsize_ticks)
    ax.tick_params(axis="y", labelsize=fontsize_ticks)
    ax.set_xlabel("Calibration set size", fontsize=fontsize_main)
    ax.set_ylabel(r"Test set $R^2$ ($95\%$ CI)", fontsize=fontsize_main)
    # ax.set_title("Fig 1  ·  Learning curve with CI and PI")
    ax.legend(loc="lower right", framealpha=0.75, fontsize=fontsize_main - 1)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    #ax.set_ylim(-0.51, 1.05)
    ax.set_ylim(-0.2, 0.6)
    ax.set_autoscaley_on(False)
    ax.margins(y=0)

    path = out / ("scaling_curve.png" if show_pi else "scaling_curve_ci.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 2: Horizontal box plots with data points ───────────────────────────
def _box_with_points(
    ax, data, x, order, hue=None, hue_order=None, palette=None, color=None
):
    """Horizontal box plot (one box per support size) with every point overlaid."""
    color_kw = {"palette": palette} if palette else {"color": color}
    common = dict(
        data=data,
        x=x,
        y="support_samples",
        orient="h",
        order=order,
        hue=hue,
        hue_order=hue_order,
        ax=ax,
    )
    sns.boxplot(
        **common,
        **color_kw,
        fill=False,
        linewidth=1.5,
        showfliers=False,  # all points are drawn by stripplot below
        whis=np.inf,
    )
    sns.stripplot(
        **common,
        **color_kw,
        dodge=hue is not None,
        jitter=0.15,
        size=5,
        alpha=0.7,
        linewidth=0.3,
        edgecolor="0.25",
        legend=False,
    )

    # if hue is not None:
    #    w = 0.8  # seaborn's default group width
    #    offs = -w / 2 + (np.arange(len(hue_order)) + 0.5) * w / len(hue_order)
    #    med = data.groupby([hue, "support_samples"])[x].median()
    #    for i, h in enumerate(hue_order):
    #        xs = [med[(h, s)] for s in order]
    #        ys = [k + offs[i] for k in range(len(order))]
    #        ax.plot(xs, ys, color=palette[h], lw=1, alpha=0.7, zorder=1)


def fig2_box_compare(
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

    _box_with_points(
        ax,
        long,
        x="r2",
        order=support_order,
        hue="method",
        hue_order=hue_order,
        palette=palette,
    )
    # ax.set_title("R² distributions: MAML vs Vanilla")
    ax.axvline(0.05, color="0.7", linestyle="--", linewidth=1, zorder=0)
    ax.set_xlabel("Test set $R^2$")
    ax.set_ylabel("Calibration set size")
    ax.legend(title=None)
    ax.grid(True, axis="x", alpha=0.35)

    fig.tight_layout()
    path = out / "boxwhisker_compare.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def fig2_box_improvement(smeans: pd.DataFrame, out: Path) -> None:
    impr = smeans[["support_samples", "improvement"]].copy()
    impr["support_samples"] = impr["support_samples"].astype(str)

    support_order = [str(s) for s in sorted(smeans["support_samples"].unique())]

    fig, ax = plt.subplots(figsize=(6, 5))

    _box_with_points(ax, impr, x="improvement", order=support_order, color=DIFF_COLOR)
    ax.axvline(0, color="k", lw=1.5)
    # ax.set_title("Improvement: MAML − Vanilla")
    ax.set_xlabel(r"Test set $\Delta R^2$")
    ax.set_ylabel("Calibration set size")
    ax.grid(True, axis="x", alpha=0.35)

    fig.tight_layout()
    path = out / "boxwhisker_improvement.png"
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
      3. Brown-Forsythe         H0: var(MAML) = var(Vanilla)  (one-sided, <)
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
        binom_p = stats.binomtest(
            n_wins, n, p=0.5, alternative=PAIRS["A"]["sign_alt"]
        ).pvalue

        # 2. Wilcoxon signed-rank (alternative from PAIRS["A"]: improvement > 0)
        if np.all(diff == 0):
            w_stat, w_p = np.nan, np.nan
        else:
            w_res = stats.wilcoxon(diff, alternative=PAIRS["A"]["wilcoxon_alt"])
            w_stat, w_p = w_res.statistic, w_res.pvalue

        # 3. Brown-Forsythe, one-sided: var(MAML) < var(Vanilla); reports t
        bf_stat, bf_p = brown_forsythe(tta, van, PAIRS["A"]["bf_alt"])

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
    a = PAIRS["A"]
    print(f"  Binom p  = sign test H0: P(win)=50%  ({ALT_TEXT[a['sign_alt']]})")
    print(
        "  Wilcox p = Wilcoxon signed-rank H0: improvement=0  "
        f"({ALT_TEXT[a['wilcoxon_alt']]})"
    )
    print(
        "  BF p     = Brown-Forsythe H0: var(MAML)=var(Vanilla)  "
        f"({ALT_TEXT[a['bf_alt']]}); BF_stat in pvalues.csv is the t statistic"
    )
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

        # Wilcoxon signed-rank test (alternative from PAIRS["C"])
        if np.all(diff == 0):
            w_stat, w_p = np.nan, np.nan
        else:
            w_res = stats.wilcoxon(diff, alternative=PAIRS["C"]["wilcoxon_alt"])
            w_stat, w_p = w_res.statistic, w_res.pvalue

        # Brown-Forsythe test (H0: var(MAML) = var(Coadapt)); reports t
        bf_stat, bf_p = brown_forsythe(maml, coadapt, PAIRS["C"]["bf_alt"])

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
    c = PAIRS["C"]
    print(
        "  Wilcox p = Wilcoxon signed-rank H0: MAML = Coadapt  "
        f"({ALT_TEXT[c['wilcoxon_alt']]})"
    )
    print(
        "  BF p     = Brown-Forsythe H0: var(MAML) = var(Coadapt)  "
        f"({ALT_TEXT[c['bf_alt']]}: var(MAML) < var(Coadapt))"
    )
    print(rule)


# ── Formatting helpers ────────────────────────────────────────────────────────
def fmt_p(p) -> str:
    return "n/a" if pd.isna(p) else f"{p:.3g}"


def fmt_p_tex(p) -> str:
    if pd.isna(p):
        return "---"
    if p < 1e-3:
        mant, exp = f"{p:.1e}".split("e")
        return rf"${float(mant):.1f}\times10^{{{int(exp)}}}$"
    return f"{p:.3f}"


def md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def pair_label(cfg) -> str:
    return f"{METHOD_LABELS[cfg['ref']]} → {METHOD_LABELS[cfg['new']]}"


def holm_adjust(p) -> np.ndarray:
    """Holm step-down adjusted p-values; NaNs are skipped and stay NaN."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.where(~np.isnan(p))[0]
    m = len(valid)
    running = 0.0
    for rank, i in enumerate(valid[np.argsort(p[valid])]):
        running = max(running, (m - rank) * p[i])
        out[i] = min(1.0, running)
    return out


def cvar(x, q: float = CVAR_Q) -> float:
    """Mean of the worst q-fraction of values."""
    k = max(1, int(np.ceil(q * len(x))))
    return float(np.sort(x)[:k].mean())


# ── Step 1: data integrity ────────────────────────────────────────────────────
def check_data_integrity(df: pd.DataFrame, df_coadapt) -> pd.DataFrame:
    """
    Stop with a clear error if the MAML and coadapt CSVs are not comparable.
    Returns the per-method fold-count table (folds held out per session).
    """
    frames = {"maml": df}
    if df_coadapt is not None:
        frames["coadapt"] = df_coadapt
        sa, sb = set(df["session_id"]), set(df_coadapt["session_id"])
        if sa != sb:
            sys.exit(
                "ERROR: session_id sets differ between the MAML and coadapt CSVs.\n"
                f"  only in MAML: {sorted(sa - sb)}\n  only in coadapt: {sorted(sb - sa)}"
            )
        va = df.groupby(SESSION_KEY)["vanilla_test_r2"].first()
        vb = df_coadapt.groupby(SESSION_KEY)["vanilla_test_r2"].first()
        if not va.index.equals(vb.index):
            sys.exit(
                "ERROR: (session, support size) combinations differ between the CSVs: "
                f"{len(va.index.symmetric_difference(vb.index))} unmatched."
            )
        gap = (va - vb).abs()
        bad = gap[gap > VANILLA_TOL]
        if len(bad):
            comp = pd.DataFrame({"maml_csv": va, "coadapt_csv": vb, "abs_diff": gap})
            print(
                f"\nVanilla test R² differs between the CSVs for {len(bad)} of "
                f"{len(gap)} (session, support size) values:"
            )
            print(bad.groupby(level="support_samples").size().rename("n_differ").to_string())
            print("\nLargest differences:")
            print(comp.loc[bad.index].sort_values("abs_diff", ascending=False).head(10).to_string())
            sys.exit(
                "ERROR: vanilla_test_r2 must be identical across the MAML and coadapt "
                "CSVs (the paired comparisons need one shared vanilla baseline). Stopping."
            )

    rows = []
    for method, d in frames.items():
        spread = d.groupby(SESSION_KEY)["vanilla_test_r2"].agg(lambda s: s.max() - s.min())
        if spread.max() > VANILLA_TOL:
            sys.exit(f"ERROR: vanilla_test_r2 is not fold-invariant in the {method} CSV.")
        counts = d.groupby(SESSION_KEY).size().unstack("support_samples")
        for ss in counts.columns:
            c = counts[ss]
            rows.append(
                {
                    "method": method,
                    "support_samples": ss,
                    "n_sessions": len(c),
                    "folds_min": int(c.min()),
                    "folds_median": float(c.median()),
                    "folds_max": int(c.max()),
                }
            )
    return pd.DataFrame(rows)


# ── Step 2: statistics ────────────────────────────────────────────────────────
def build_wide(smeans: pd.DataFrame, smeans_coadapt) -> dict:
    """Per support size: DataFrame indexed by session with one column per method."""
    wide = {}
    for ss, g in smeans.groupby("support_samples"):
        g = g.set_index("session_id").sort_index()
        w = pd.DataFrame({"vanilla": g["vanilla_r2"], "maml": g["tta_mean"]})
        if smeans_coadapt is not None:
            gc = smeans_coadapt[smeans_coadapt["support_samples"] == ss]
            w["coadapt"] = gc.set_index("session_id")["tta_mean"].reindex(w.index)
        if w.isna().any().any():
            sys.exit(f"ERROR: missing per-session values at support size {ss}.")
        wide[ss] = w
    return wide


def compute_stats(wide: dict, seed: int):
    """
    All paired tests for every pair in PAIRS at every support size.
    Returns (long DataFrame with Holm p, CVaR-per-method DataFrame, max |BF - levene|).
    """
    rng = np.random.default_rng(seed)
    rows, cvar_rows, bf_gap = [], [], 0.0
    for ss in sorted(wide):
        w = wide[ss]
        n = len(w)
        # One resample of sessions per support size, shared by all methods and pairs
        idx = rng.integers(0, n, size=(N_BOOT, n))
        k = max(1, int(np.ceil(CVAR_Q * n)))
        for m in METHOD_LABELS:
            if m in w:
                cvar_rows.append(
                    {"support_samples": ss, "method": m, "cvar": cvar(w[m].to_numpy())}
                )
        for pair, cfg in PAIRS.items():
            if cfg["ref"] not in w or cfg["new"] not in w:
                continue
            ref, new = w[cfg["ref"]].to_numpy(), w[cfg["new"]].to_numpy()
            diff = new - ref
            n_gt = int((diff > 0).sum())
            base = {
                "support_samples": ss,
                "pair": pair,
                "reference": cfg["ref"],
                "new": cfg["new"],
                "n": n,
            }

            if cfg["sign_alt"]:
                rows.append(
                    {
                        **base,
                        "test": "sign",
                        "alternative": cfg["sign_alt"],
                        "statistic_name": "n_new_gt_ref",
                        "statistic": n_gt,
                        "p_raw": stats.binomtest(
                            n_gt, n, 0.5, alternative=cfg["sign_alt"]
                        ).pvalue,
                        "n_new_gt_ref": n_gt,
                    }
                )

            if np.all(diff == 0):
                w_stat, w_p = np.nan, np.nan
            else:
                res = stats.wilcoxon(diff, alternative=cfg["wilcoxon_alt"])
                w_stat, w_p = float(res.statistic), float(res.pvalue)
            rows.append(
                {
                    **base,
                    "test": "wilcoxon",
                    "alternative": cfg["wilcoxon_alt"],
                    "statistic_name": "W",
                    "statistic": w_stat,
                    "p_raw": w_p,
                    "n_new_gt_ref": n_gt,
                }
            )

            t, p = brown_forsythe(new, ref, cfg["bf_alt"])
            bf_gap = max(bf_gap, bf_matches_levene(new, ref))
            rows.append(
                {
                    **base,
                    "test": "brown_forsythe",
                    "alternative": cfg["bf_alt"],
                    "statistic_name": "t",
                    "statistic": t,
                    "p_raw": p,
                    "var_ref": float(ref.var(ddof=1)),
                    "var_new": float(new.var(ddof=1)),
                }
            )

            for thr in THRESHOLDS:
                fail_ref, fail_new = ref < thr, new < thr
                b = int((fail_ref & ~fail_new).sum())
                c = int((fail_new & ~fail_ref).sum())
                flag = ""
                if b + c < MIN_DISCORDANT:
                    p, flag = np.nan, "too few discordant sessions"
                else:
                    p = stats.binomtest(
                        b, b + c, 0.5, alternative=cfg["mcnemar_alt"]
                    ).pvalue
                rows.append(
                    {
                        **base,
                        "test": f"mcnemar_{thr:g}",
                        "alternative": cfg["mcnemar_alt"],
                        "statistic_name": "b",
                        "statistic": b,
                        "p_raw": p,
                        "n_fail_ref": int(fail_ref.sum()),
                        "n_fail_new": int(fail_new.sum()),
                        "b": b,
                        "c": c,
                        "flag": flag,
                    }
                )

            def boot_cvar(x):
                return np.sort(x[idx], axis=1)[:, :k].mean(axis=1)

            d_boot = boot_cvar(new) - boot_cvar(ref)
            lo, hi = np.percentile(d_boot, [2.5, 97.5])
            rows.append(
                {
                    **base,
                    "test": "cvar_diff",
                    "alternative": "n/a",
                    "statistic_name": "cvar_new - cvar_ref",
                    "statistic": cvar(new) - cvar(ref),
                    "ci_lo": float(lo),
                    "ci_hi": float(hi),
                    "cvar_ref": cvar(ref),
                    "cvar_new": cvar(new),
                }
            )

    long = pd.DataFrame(rows)
    long["p_holm"] = np.nan
    for _, g in long.groupby(["test", "pair"]):
        g = g.sort_values("support_samples")
        long.loc[g.index, "p_holm"] = holm_adjust(g["p_raw"].to_numpy())
    test_order = ["sign", "wilcoxon", "brown_forsythe"] + [
        f"mcnemar_{t:g}" for t in THRESHOLDS
    ] + ["cvar_diff"]
    long["test"] = pd.Categorical(long["test"], categories=test_order, ordered=True)
    long = long.sort_values(["support_samples", "pair", "test"]).reset_index(drop=True)
    long["test"] = long["test"].astype(str)
    cols = [
        "support_samples", "pair", "reference", "new", "test", "alternative", "n",
        "statistic_name", "statistic", "p_raw", "p_holm", "n_new_gt_ref",
        "n_fail_ref", "n_fail_new", "b", "c", "flag",
        "ci_lo", "ci_hi", "cvar_ref", "cvar_new", "var_ref", "var_new",
    ]
    return long[cols], pd.DataFrame(cvar_rows), bf_gap


# ── Step 3: fold-noise sanity check ───────────────────────────────────────────
def compute_fold_noise(df, smeans, method: str) -> pd.DataFrame:
    """
    Within-session SD across folds vs. between-session SD, for the session means
    and for a single fold only (first fold available).
    """
    first = (
        df.sort_values("fold")
        .groupby(SESSION_KEY)["tta_r2"]
        .first()
        .rename("first_fold")
        .reset_index()
    )
    sm = smeans.merge(first, on=SESSION_KEY)
    rows = []
    for ss, g in sm.groupby("support_samples"):
        sd_mean = float(g["tta_mean"].std(ddof=1))
        sd_single = float(g["first_fold"].std(ddof=1))
        rel = sd_single / sd_mean - 1
        rows.append(
            {
                "method": method,
                "support_samples": ss,
                "within_sd_median": float(np.nanmedian(g["tta_std"])),
                "between_sd_session_mean": sd_mean,
                "between_sd_single_fold": sd_single,
                "rel_diff": rel,
                "flag": abs(rel) > FOLD_NOISE_TOL,
            }
        )
    return pd.DataFrame(rows)


# ── Step 4: numbers cited in the paper ────────────────────────────────────────
def compute_cited_numbers(wide: dict) -> dict:
    sizes = sorted(wide)
    methods = [m for m in METHOD_LABELS if m in wide[sizes[0]]]
    summary = []
    for ss in sizes:
        w = wide[ss]
        n = len(w)
        tc = stats.t.ppf(0.975, df=n - 1)
        for m in methods:
            x = w[m].to_numpy()
            row = {
                "support_samples": ss,
                "method": m,
                "mean": float(x.mean()),
                "ci_half": float(tc * x.std(ddof=1) / np.sqrt(n)),
            }
            for thr in THRESHOLDS:
                row[f"below_{thr:g}"] = int((x < thr).sum())
            summary.append(row)
    summary = pd.DataFrame(summary)

    beats = [
        {
            "support_samples": ss,
            "n": len(wide[ss]),
            "vanilla_beats_maml": int((wide[ss]["vanilla"] > wide[ss]["maml"]).sum()),
        }
        for ss in sizes
    ]

    van_means = {ss: wide[ss]["vanilla"].mean() for ss in sizes}
    savings = []
    for ss in sizes:
        maml_mean = wide[ss]["maml"].mean()
        match = next((s for s in sizes if van_means[s] >= maml_mean), None)
        savings.append(
            {
                "maml_size": ss,
                "maml_mean": float(maml_mean),
                "vanilla_size": match,
                "vanilla_mean": None if match is None else float(van_means[match]),
                "savings": None if match is None else 1 - ss / match,
            }
        )
    return {
        "summary": summary,
        "beats": pd.DataFrame(beats),
        "savings": pd.DataFrame(savings),
        "methods": methods,
    }


# ── Report rendering ──────────────────────────────────────────────────────────
def render_integrity(fold_counts: pd.DataFrame, coadapt_checked: bool) -> str:
    out = ["## 1. Data integrity", ""]
    if coadapt_checked:
        out += [
            "- Session IDs are identical in the MAML and coadapt CSVs.",
            "- `vanilla_test_r2` is fold-invariant within each CSV and identical "
            "across the two CSVs (tolerance 1e-9).",
        ]
    else:
        out += ["- No coadapt CSV supplied: cross-CSV checks skipped."]
    out += ["", "Folds in which each session was held out (across sessions):", ""]
    out.append(
        md_table(
            ["Method", "Support", "Sessions", "Min", "Median", "Max"],
            [
                [
                    METHOD_LABELS[r.method], r.support_samples, r.n_sessions,
                    r.folds_min, f"{r.folds_median:g}", r.folds_max,
                ]
                for r in fold_counts.itertuples()
            ],
        )
    )
    return "\n".join(out)


def _tests(long, test):
    return long[long["test"] == test]


def render_tests(long: pd.DataFrame, cvar_df: pd.DataFrame, bf_gap: float) -> str:
    out = ["## 2. Paired comparisons", ""]
    out.append("Pairs (reference → new) and sidedness:")
    out.append("")
    out.append(
        md_table(
            ["Pair", "Comparison", "Wilcoxon", "Sign", "Brown-Forsythe", "McNemar"],
            [
                [
                    p, pair_label(c), ALT_TEXT[c["wilcoxon_alt"]],
                    ALT_TEXT[c["sign_alt"]] if c["sign_alt"] else "n/a",
                    ALT_TEXT[c["bf_alt"]] + " (var(new) < var(ref))",
                    ALT_TEXT[c["mcnemar_alt"]] + " (ref-only failures > new-only)",
                ]
                for p, c in PAIRS.items()
                if p in set(long["pair"])
            ],
        )
    )
    out += [
        "",
        "p-values are Holm-adjusted across the four support sizes within each "
        "(test, pair) family (McNemar: (threshold, pair)); NaN p-values are "
        "skipped when counting the family size.",
        "",
        "### 2a. Brown-Forsythe (H1: var(new) < var(ref))",
        "",
        "Computed as a pooled t-test on |x − median(x)|; the t statistic is "
        "reported. The groups are paired but BF treats them as independent "
        "(conservative). Sanity check: the two-sided version equals "
        f"`scipy.stats.levene(center='median')` (max |Δp| = {bf_gap:.2e}).",
        "",
    ]
    out.append(
        md_table(
            ["Support", "Pair", "var(ref)", "var(new)", "t", "p", "p (Holm)"],
            [
                [
                    r.support_samples, r.pair, f"{r.var_ref:.4f}", f"{r.var_new:.4f}",
                    f"{r.statistic:.2f}", fmt_p(r.p_raw), fmt_p(r.p_holm),
                ]
                for r in _tests(long, "brown_forsythe").itertuples()
            ],
        )
    )
    out += ["", "### 2b. Wilcoxon signed-rank and sign test", ""]
    sign = _tests(long, "sign").set_index(["support_samples", "pair"])
    rows = []
    for r in _tests(long, "wilcoxon").itertuples():
        s = sign.loc[(r.support_samples, r.pair)] if (r.support_samples, r.pair) in sign.index else None
        rows.append(
            [
                r.support_samples, r.pair, f"{int(r.n_new_gt_ref)}/{int(r.n)}",
                f"{r.statistic:.1f}", fmt_p(r.p_raw), fmt_p(r.p_holm),
                "n/a" if s is None else fmt_p(s["p_raw"]),
                "n/a" if s is None else fmt_p(s["p_holm"]),
            ]
        )
    out.append(
        md_table(
            ["Support", "Pair", "new > ref", "W", "Wilcoxon p", "p (Holm)", "Sign p", "p (Holm)"],
            rows,
        )
    )
    out += ["", "### 2c. Exact McNemar test on failures", ""]
    out.append(
        "b = only the reference fails; c = only the new method fails; "
        f"p is NaN when b + c < {MIN_DISCORDANT}."
    )
    out.append("")
    rows = []
    for thr in THRESHOLDS:
        for r in _tests(long, f"mcnemar_{thr:g}").itertuples():
            rows.append(
                [
                    r.support_samples, r.pair, f"{thr:g}", int(r.n_fail_ref), int(r.n_fail_new),
                    int(r.b), int(r.c), fmt_p(r.p_raw), fmt_p(r.p_holm), r.flag,
                ]
            )
    out.append(
        md_table(
            ["Support", "Pair", "Thr", "fail ref", "fail new", "b", "c", "p", "p (Holm)", "Note"],
            sorted(rows, key=lambda r: (r[2], r[0], r[1])),
        )
    )
    out += ["", f"### 2d. CVaR (mean R² of the worst {CVAR_Q:.0%} of sessions)", ""]
    pivot = cvar_df.pivot(index="support_samples", columns="method", values="cvar")
    cols = [m for m in METHOD_LABELS if m in pivot]
    out.append(
        md_table(
            ["Support"] + [METHOD_LABELS[m] for m in cols],
            [[ss] + [f"{pivot.loc[ss, m]:.3f}" for m in cols] for ss in pivot.index],
        )
    )
    out += [
        "",
        f"Difference (new − reference) with a 95% paired bootstrap CI "
        f"({N_BOOT:,} resamples of sessions, both methods resampled jointly):",
        "",
    ]
    out.append(
        md_table(
            ["Support", "Pair", "CVaR ref", "CVaR new", "Diff", "95% CI"],
            [
                [
                    r.support_samples, r.pair, f"{r.cvar_ref:.3f}", f"{r.cvar_new:.3f}",
                    f"{r.statistic:+.3f}", f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]",
                ]
                for r in _tests(long, "cvar_diff").itertuples()
            ],
        )
    )
    return "\n".join(out)


def render_fold_noise(fn: pd.DataFrame, wide: dict) -> str:
    out = [
        "## 3. Fold-noise sanity check",
        "",
        "Within = median across sessions of the within-session SD of `tta_r2` "
        "across folds. Between = SD across sessions of the session means, and "
        "of a single fold only (first fold available). Flag if the two "
        f"between-session SDs differ by more than {FOLD_NOISE_TOL:.0%}.",
        "",
    ]
    rows = []
    for r in fn.itertuples():
        rows.append(
            [
                METHOD_LABELS[r.method], r.support_samples,
                f"{r.within_sd_median:.4f}", f"{r.between_sd_session_mean:.4f}",
                f"{r.between_sd_single_fold:.4f}", f"{r.rel_diff:+.1%}",
                "**FLAG**" if r.flag else "ok",
            ]
        )
    for ss in sorted(wide):
        rows.append(
            [
                "Vanilla (single value)", ss, "n/a",
                f"{wide[ss]['vanilla'].std(ddof=1):.4f}", "n/a", "n/a", "n/a",
            ]
        )
    out.append(
        md_table(
            ["Method", "Support", "Within SD (median)", "Between SD (session mean)",
             "Between SD (single fold)", "Single vs mean", "Flag"],
            rows,
        )
    )
    return "\n".join(out)


def render_cited(c: dict) -> str:
    s, methods = c["summary"], c["methods"]
    out = ["## 4. Numbers cited in paper", ""]
    out += ["### Mean R² ± 95% CI half-width (t-based, n sessions)", ""]
    rows, ratios = [], {}
    for ss, g in s.groupby("support_samples"):
        g = g.set_index("method")
        row = [ss] + [f"{g.loc[m, 'mean']:.3f} ± {g.loc[m, 'ci_half']:.3f}" for m in methods]
        ratio = g.loc["vanilla", "ci_half"] / g.loc["maml", "ci_half"]
        rows.append(row + [f"{ratio:.2f}"])
    out.append(
        md_table(
            ["Support"] + [METHOD_LABELS[m] for m in methods] + ["Vanilla/MAML CI-width ratio"],
            rows,
        )
    )
    for thr in THRESHOLDS:
        out += ["", f"### Sessions with R² < {thr:g}", ""]
        piv = s.pivot(index="support_samples", columns="method", values=f"below_{thr:g}")
        out.append(
            md_table(
                ["Support"] + [METHOD_LABELS[m] for m in methods],
                [[ss] + [int(piv.loc[ss, m]) for m in methods] for ss in piv.index],
            )
        )
    out += ["", "### Sessions where vanilla beats MAML", ""]
    out.append(
        md_table(
            ["Support", "Vanilla > MAML"],
            [[r.support_samples, f"{r.vanilla_beats_maml} of {r.n}"] for r in c["beats"].itertuples()],
        )
    )
    out += [
        "",
        "### Calibration savings",
        "",
        "Smallest vanilla calibration size whose mean R² is at least the MAML "
        "mean at the given size (restricted to the sizes in the data). 'none' "
        "means vanilla never reaches it within the largest size, so the savings "
        "are at least 1 − size / largest size (shown as '>').",
        "",
    ]
    largest = max(c["savings"]["maml_size"])
    out.append(
        md_table(
            ["MAML size", "MAML mean R²", "Vanilla size needed", "Vanilla mean R²", "Savings"],
            [
                [
                    r.maml_size, f"{r.maml_mean:.3f}",
                    "none" if pd.isna(r.vanilla_size) else int(r.vanilla_size),
                    "n/a" if pd.isna(r.vanilla_mean) else f"{r.vanilla_mean:.3f}",
                    f">{1 - r.maml_size / largest:.0%}"
                    if pd.isna(r.savings)
                    else f"{r.savings:.0%}",
                ]
                for r in c["savings"].itertuples()
            ],
        )
    )
    return "\n".join(out)


def render_surprises(long, fn, fold_counts, bf_gap) -> str:
    notes = []
    m = long[long["test"].str.startswith("mcnemar") & (long["flag"] != "")]
    if len(m):
        notes.append(
            f"{len(m)} McNemar tests have fewer than {MIN_DISCORDANT} discordant "
            "sessions (p reported as NaN): "
            + ", ".join(
                f"{r.test.split('_')[1]}/{r.pair}/{r.support_samples}"
                for r in m.itertuples()
            )
            + "."
        )
    flagged = fn[fn["flag"]]
    if len(flagged):
        notes.append(
            "Fold-noise flag: single-fold between-session SD differs from the "
            "session-mean SD by more than "
            f"{FOLD_NOISE_TOL:.0%} for "
            + ", ".join(
                f"{METHOD_LABELS[r.method]}@{r.support_samples} ({r.rel_diff:+.0%})"
                for r in flagged.itertuples()
            )
            + "."
        )
    else:
        notes.append("Fold-noise check: no flags; single-fold and session-mean between-session SDs agree.")
    fc = fold_counts.pivot(index="support_samples", columns="method", values="folds_median")
    mm = fold_counts.pivot(index="support_samples", columns="method", values="folds_min")
    if "coadapt" in fc:
        diff_sizes = [
            ss for ss in fc.index
            if fc.loc[ss, "maml"] != fc.loc[ss, "coadapt"] or mm.loc[ss, "maml"] != mm.loc[ss, "coadapt"]
        ]
        if diff_sizes:
            notes.append(
                "Fold counts differ between MAML and coadapt at support size(s) "
                f"{diff_sizes}; see Section 1."
            )
    sig = long[long["p_raw"].notna()]
    lost = sig[(sig["p_raw"] < 0.05) & (sig["p_holm"] >= 0.05)]
    if len(lost):
        notes.append(
            "Significant before but not after Holm correction: "
            + ", ".join(f"{r.test}/{r.pair}/{r.support_samples}" for r in lost.itertuples())
            + "."
        )
    notes.append(f"BF direct t-test vs scipy levene: max |Δp| = {bf_gap:.2e} (two-sided).")
    return "## Summary of anything surprising\n\n" + "\n".join(f"- {n}" for n in notes)


def write_supplement_table(long: pd.DataFrame, path: Path) -> None:
    lines = [
        "% Generated by portable_analysis.py -- do not edit by hand.",
        r"\begin{tabular}{rlcrcrcrr}",
        r"\hline",
        r"Set size & Comparison & Fail $<0.05$ (ref/new) & McNemar $p$ & "
        r"Fail $<0.15$ (ref/new) & McNemar $p$ & CVaR diff [95\% CI] & "
        r"Wilcoxon $p$ & BF $p$ \\",
        r"\hline",
    ]
    L = long.set_index(["support_samples", "pair", "test"])
    pairs = [p for p in PAIRS if p in set(long["pair"])]
    for i, ss in enumerate(sorted(long["support_samples"].unique())):
        for j, pair in enumerate(pairs):
            cfg = PAIRS[pair]
            m1, m2 = L.loc[(ss, pair, "mcnemar_0.05")], L.loc[(ss, pair, "mcnemar_0.15")]
            cv = L.loc[(ss, pair, "cvar_diff")]
            lines.append(
                " & ".join(
                    [
                        str(ss) if j == 0 else "",
                        f"{METHOD_LABELS[cfg['ref']]} $\\to$ {METHOD_LABELS[cfg['new']]}",
                        f"{int(m1['n_fail_ref'])}/{int(m1['n_fail_new'])}",
                        fmt_p_tex(m1["p_holm"]),
                        f"{int(m2['n_fail_ref'])}/{int(m2['n_fail_new'])}",
                        fmt_p_tex(m2["p_holm"]),
                        f"${cv['statistic']:+.3f}$ [${cv['ci_lo']:+.3f}$, ${cv['ci_hi']:+.3f}$]",
                        fmt_p_tex(L.loc[(ss, pair, "wilcoxon"), "p_holm"]),
                        fmt_p_tex(L.loc[(ss, pair, "brown_forsythe"), "p_holm"]),
                    ]
                )
                + r" \\"
            )
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    ALT_TEX = {"greater": "one-sided ($>$)", "less": "one-sided ($<$)", "two-sided": "two-sided"}
    sided = "; ".join(
        f"{METHOD_LABELS[PAIRS[p]['ref']]}$\\to${METHOD_LABELS[PAIRS[p]['new']]}: "
        f"Wilcoxon {ALT_TEX[PAIRS[p]['wilcoxon_alt']]}, "
        f"Brown--Forsythe {ALT_TEX[PAIRS[p]['bf_alt']]}, "
        f"McNemar {ALT_TEX[PAIRS[p]['mcnemar_alt']]}"
        for p in pairs
    )
    lines.append(
        r"\par\smallskip{\footnotesize Sidedness (new = method after the arrow; "
        + sided
        + r"). Brown--Forsythe is a pooled $t$-test on absolute deviations from "
        r"the median; one-sided means H$_1$: var(new) $<$ var(reference). McNemar is an "
        r"exact binomial test on sessions with $R^2$ below the threshold; "
        r"`---' marks fewer than "
        f"{MIN_DISCORDANT} discordant sessions. CVaR is the mean $R^2$ of the worst "
        f"{CVAR_Q * 100:g}\\% of sessions, difference = new $-$ reference with a paired "
        r"bootstrap CI. All $p$-values are Holm-adjusted across the four "
        r"calibration set sizes within each test and comparison.}"
    )
    path.write_text("\n".join(lines) + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Portable TTA analysis: figures, p-values and the stats report"
    )
    parser.add_argument(
        "--data",
        default="tbfm_cv_data_twostage.csv",
        help="Path to the MAML CSV  (default: tbfm_cv_data_twostage.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="figures",
        help="Directory for all outputs  (default: figures/)",
    )
    parser.add_argument(
        "--coadapt-data",
        default="tbfm_cv_data_joint.csv",
        help="Path to coadapt/joint-comparison CSV (same format as --data). "
        "Default points at tbfm_cv_data_joint.csv -- the current data is from "
        "the joint TTA strategy, not the paper's actual coadapt method; see "
        "tbfm_cv_data_coadapt_20260225.csv for the (older, differently-matched) "
        "real coadapt run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the paired bootstrap  (default: 0)",
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
        df_coadapt = None
        smeans_coadapt = None
        lc_coadapt = None

    print("\nChecking data integrity …")
    fold_counts = check_data_integrity(df, df_coadapt)
    print("  OK")

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
    fig1_learning_curve(lc, out_dir, show_pi=True)
    fig1_learning_curve(lc, out_dir, lc_coadapt=lc_coadapt, show_pi=False)
    fig2_box_compare(smeans, out_dir, smeans_coadapt)
    fig2_box_improvement(smeans, out_dir)
    fig3_fold_variability(df, out_dir)

    pv = compute_pvalues(smeans)
    print_pvalues(pv)

    if smeans_coadapt is not None:
        pv_maml_vs_coadapt = compute_maml_vs_coadapt_pvalues(smeans, smeans_coadapt)
        print_maml_vs_coadapt_pvalues(pv_maml_vs_coadapt)

    pv_path = out_dir / "pvalues.csv"
    pv.to_csv(pv_path, index=False)
    print(f"\n  P-value table saved to {pv_path}")

    # ── New statistics, report and supplement table ───────────────────────────
    print("\nComputing paired statistics …")
    wide = build_wide(smeans, smeans_coadapt)
    long, cvar_df, bf_gap = compute_stats(wide, args.seed)
    fold_noise = [compute_fold_noise(df, smeans, "maml")]
    if df_coadapt is not None:
        fold_noise.append(compute_fold_noise(df_coadapt, smeans_coadapt, "coadapt"))
    fold_noise = pd.concat(fold_noise, ignore_index=True)
    cited = compute_cited_numbers(wide)

    n_sessions = len(next(iter(wide.values())))
    header = "\n".join(
        [
            "# Statistics report",
            "",
            f"- MAML data: `{data_path}`",
            f"- Coadapt data: `{args.coadapt_data or 'none'}`",
            f"- Seed: {args.seed} (paired bootstrap, {N_BOOT:,} resamples)",
            f"- Sessions: n = {n_sessions}",
            "- Statistical unit: one value per (session, support size). MAML and "
            "coadapt use the mean of `tta_r2` over the folds in which the session "
            "was held out; vanilla is a single fold-invariant value per session.",
        ]
    )
    report = "\n\n".join(
        [
            header,
            render_integrity(fold_counts, df_coadapt is not None),
            render_tests(long, cvar_df, bf_gap),
            render_fold_noise(fold_noise, wide),
            render_cited(cited),
            render_surprises(long, fold_noise, fold_counts, bf_gap),
        ]
    )
    print("\n" + report)

    (out_dir / "stats_report.md").write_text(report + "\n")
    long.to_csv(out_dir / "stats_long.csv", index=False)
    write_supplement_table(long, out_dir / "supplement_table.tex")
    print(
        f"\n  Saved {out_dir / 'stats_report.md'}, {out_dir / 'stats_long.csv'}, "
        f"{out_dir / 'supplement_table.tex'}"
    )


if __name__ == "__main__":
    main()
