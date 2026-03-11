#!/usr/bin/env python3
"""
10_interest_rate_exposure.py — Interest rate sensitivity vs AI exposure.

Addresses Brynjolfsson et al. (2025, Nov) Figure A22 for the Swedish context.
Their finding: AI exposure and interest rate exposure are NEGATIVELY correlated
in the US (Zens et al. 2020 measure), meaning AI-exposed occupations are less
rate-sensitive. We test whether this holds in Sweden.

Key insight: The 7-month window between the Riksbank rate hike (April 2022) and
ChatGPT launch (December 2022) provides a clean "revealed preference" measure of
each occupation's monetary policy sensitivity, uncontaminated by AI effects. Any
posting decline in this window is macro-driven by definition.

Interpretation:
  - Negative rate_sensitivity = postings declined = occupation IS rate-sensitive
  - Zero / positive = postings held up = occupation is rate-INsensitive
  - Positive correlation (DAIOE vs rate_sensitivity) would mean AI-exposed
    occupations are LESS rate-sensitive (same direction as US finding)
  - Negative correlation would mean AI-exposed occupations are MORE rate-
    sensitive, supporting the macro-confound explanation

Outputs:
  - figA_rate_sensitivity_scatter.png  (scatter: rate sensitivity × DAIOE)
  - figA_rate_sensitivity_boxplot.png  (box plot by exposure quartile)
  - rate_sensitivity_correlation.csv   (correlation statistics)
  - rate_sensitivity_by_quartile.csv   (quartile-level summary)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    PROCESSED, FIGDIR, TABDIR,
    RIKSBANKEN_HIKE, CHATGPT_LAUNCH,
    DARK_BLUE, ORANGE, TEAL, GRAY, LIGHT_GRAY,
    Q_COLORS, set_rcparams,
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

set_rcparams()


def compute_rate_sensitivity(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Compute each occupation's revealed monetary policy sensitivity.

    Economic logic:
      - Pre-hike period: Jan-Mar 2022 (Riksbank still at 0%)
      - Post-hike, pre-ChatGPT period: May-Nov 2022
      - The log change in mean postings between these windows captures
        how much each occupation responded to the rate hike cycle.
      - This is free from AI contamination (ChatGPT launched Nov 30 2022).
      - Negative values = postings declined = occupation is rate-sensitive.

    Returns one row per SSYK4 with rate_sensitivity (log change)
    and posting volume for weighting.
    """
    df = merged.copy()
    df["date"] = pd.to_datetime(df["year_month"] + "-01")

    # Define windows
    pre_start = pd.Timestamp("2022-01-01")
    pre_end = pd.Timestamp("2022-03-01")    # Jan-Mar 2022
    post_start = pd.Timestamp("2022-05-01")  # Skip April (transition month)
    post_end = pd.Timestamp("2022-11-01")    # May-Nov 2022

    pre = df[(df["date"] >= pre_start) & (df["date"] <= pre_end)]
    post = df[(df["date"] >= post_start) & (df["date"] <= post_end)]

    # Mean postings per occupation in each window
    pre_mean = (
        pre.groupby("ssyk4")["n_ads"]
        .mean()
        .rename("pre_mean")
    )
    post_mean = (
        post.groupby("ssyk4")["n_ads"]
        .mean()
        .rename("post_mean")
    )

    # Total posting volume (for weighting and filtering)
    total_ads = (
        df.groupby("ssyk4")["n_ads"]
        .sum()
        .rename("total_ads")
    )

    # Combine
    occ = pd.concat([pre_mean, post_mean, total_ads], axis=1).dropna()

    # Log change: negative = decline after rate hike = rate-sensitive
    occ["rate_sensitivity"] = np.log(occ["post_mean"] + 1) - np.log(occ["pre_mean"] + 1)

    # Filter: require at least 3 ads/month in pre-period for stable estimates
    occ = occ[occ["pre_mean"] >= 3].copy()

    print(f"  Occupations with valid rate sensitivity: {len(occ)}")
    print(f"  Rate sensitivity range: {occ['rate_sensitivity'].min():.3f} "
          f"to {occ['rate_sensitivity'].max():.3f}")
    print(f"  Mean: {occ['rate_sensitivity'].mean():.3f}, "
          f"Median: {occ['rate_sensitivity'].median():.3f}")

    return occ.reset_index()


def merge_with_daioe(occ: pd.DataFrame, daioe: pd.DataFrame) -> pd.DataFrame:
    """Merge rate sensitivity with DAIOE exposure data."""
    occ["ssyk4"] = occ["ssyk4"].astype(str).str.zfill(4)
    daioe["ssyk4"] = daioe["ssyk4"].astype(str).str.zfill(4)

    merged = occ.merge(daioe, on="ssyk4", how="inner")
    print(f"  Matched occupations: {len(merged)}")
    return merged


def weighted_ols(x, y, w):
    """WLS regression: returns slope, intercept, r, p, se."""
    # Weighted means
    xm = np.average(x, weights=w)
    ym = np.average(y, weights=w)
    # Weighted covariance and variance
    dx = x - xm
    dy = y - ym
    cov_xy = np.sum(w * dx * dy) / np.sum(w)
    var_x = np.sum(w * dx**2) / np.sum(w)
    var_y = np.sum(w * dy**2) / np.sum(w)
    slope = cov_xy / var_x
    intercept = ym - slope * xm
    r = cov_xy / np.sqrt(var_x * var_y)
    # Weighted residuals for SE
    resid = y - (intercept + slope * x)
    n = len(x)
    mse = np.sum(w * resid**2) / np.sum(w) * n / (n - 2)
    se = np.sqrt(mse / (np.sum(w * dx**2)))
    t_stat = slope / se
    p = 2 * stats.t.sf(np.abs(t_stat), df=n - 2)
    return slope, intercept, r, p, se


def plot_scatter(df: pd.DataFrame):
    """
    Scatter plot: DAIOE genAI exposure (x) vs revealed rate sensitivity (y).

    Uses posting-volume-weighted regression line to avoid small-occupation noise.
    Analogous to Brynjolfsson et al. (2025) Figure A22, but using
    Swedish revealed-preference measure instead of Zens et al. (2020).
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Colour by quartile
    colors = {
        "Q1 (lowest)": LIGHT_GRAY,
        "Q2": "#DCE6F2",
        "Q3": TEAL,
        "Q4 (highest)": ORANGE,
    }

    for q in ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]:
        mask = df["exposure_quartile"] == q
        ax.scatter(
            df.loc[mask, "pctl_rank_genai"],
            df.loc[mask, "rate_sensitivity"],
            c=colors[q],
            s=np.sqrt(df.loc[mask, "total_ads"]) * 0.5,
            alpha=0.6,
            edgecolors="white",
            linewidth=0.3,
            label=q,
            zorder=3,
        )

    # Unweighted OLS
    slope_uw, intercept_uw, r_uw, p_uw, se_uw = stats.linregress(
        df["pctl_rank_genai"], df["rate_sensitivity"]
    )[:5]

    # Weighted OLS (by posting volume) — primary line
    x = df["pctl_rank_genai"].values
    y = df["rate_sensitivity"].values
    w = df["total_ads"].values
    slope_w, intercept_w, r_w, p_w, se_w = weighted_ols(x, y, w)

    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, intercept_w + slope_w * x_line, color=DARK_BLUE,
            linewidth=2, linestyle="-", zorder=4, label="WLS fit")
    ax.plot(x_line, intercept_uw + slope_uw * x_line, color=GRAY,
            linewidth=1.2, linestyle="--", zorder=4, alpha=0.6, label="OLS fit")

    # Annotation box
    ax.text(
        0.03, 0.97,
        f"WLS (posting-weighted):\n"
        f"  r = {r_w:.3f}, p = {p_w:.3f}, slope = {slope_w:.4f}\n"
        f"OLS (unweighted):\n"
        f"  r = {r_uw:.3f}, p = {p_uw:.3f}\n"
        f"N = {len(df)} occupations",
        transform=ax.transAxes, fontsize=9,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )

    # Interpretation note at bottom
    ax.text(
        0.97, 0.03,
        "< rate-sensitive            rate-insensitive >",
        transform=ax.transAxes, fontsize=8, color=GRAY,
        va="bottom", ha="right", style="italic",
    )

    ax.axhline(0, color=GRAY, linewidth=0.5, linestyle=":")
    ax.set_xlabel("DAIOE genAI exposure (percentile rank)")
    ax.set_ylabel("Posting change after Riksbank rate hike\n"
                  "(log change, May–Nov vs Jan–Mar 2022)")
    ax.set_title("AI exposure vs monetary policy sensitivity in Sweden")
    ax.legend(title="AI exposure quartile", loc="lower left", fontsize=8,
              title_fontsize=9)

    plt.tight_layout()
    out = FIGDIR / "figA_rate_sensitivity_scatter.png"
    fig.savefig(out, dpi=300)
    plt.close()
    print(f"  Saved: {out.name}")

    return slope_uw, intercept_uw, r_uw, p_uw, se_uw, r_w, p_w


def plot_boxplot(df: pd.DataFrame):
    """
    Box plot: rate sensitivity by AI exposure quartile.

    Clearer visual than scatter for showing the distribution of monetary
    policy sensitivity within each AI exposure group.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    quartiles = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
    data = [df.loc[df["exposure_quartile"] == q, "rate_sensitivity"].values
            for q in quartiles]

    colors_list = [LIGHT_GRAY, "#DCE6F2", TEAL, ORANGE]

    bp = ax.boxplot(
        data,
        tick_labels=["Q1\n(lowest)", "Q2", "Q3", "Q4\n(highest)"],
        patch_artist=True,
        widths=0.6,
        medianprops=dict(color=DARK_BLUE, linewidth=2),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )
    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add means as diamonds
    means = [np.mean(d) for d in data]
    ax.scatter(range(1, 5), means, marker="D", color=DARK_BLUE,
               s=50, zorder=5, label="Mean")

    ax.axhline(0, color=GRAY, linewidth=0.5, linestyle=":")
    ax.set_xlabel("DAIOE genAI exposure quartile")
    ax.set_ylabel("Posting change after Riksbank rate hike\n"
                  "(log change, May–Nov vs Jan–Mar 2022)")
    ax.set_title("Monetary policy sensitivity by AI exposure quartile")
    ax.legend(loc="upper left", fontsize=9)

    # Interpretation arrows
    ax.annotate("< rate-sensitive", xy=(0.02, 0.02), xycoords="axes fraction",
                fontsize=8, color=GRAY, style="italic", va="bottom")
    ax.annotate("rate-insensitive >", xy=(0.98, 0.98), xycoords="axes fraction",
                fontsize=8, color=GRAY, style="italic", va="top", ha="right")

    # Kruskal-Wallis test
    h_stat, kw_p = stats.kruskal(*data)
    ax.text(
        0.97, 0.05,
        f"Kruskal-Wallis: H = {h_stat:.1f}, p = {kw_p:.3f}",
        transform=ax.transAxes, fontsize=9,
        va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    out = FIGDIR / "figA_rate_sensitivity_boxplot.png"
    fig.savefig(out, dpi=300)
    plt.close()
    print(f"  Saved: {out.name}")


def save_statistics(df, slope_uw, intercept_uw, r_uw, p_uw, se_uw, r_w, p_w):
    """Save correlation statistics to CSV for reference."""
    quartile_stats = (
        df.groupby("exposure_quartile")["rate_sensitivity"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )

    stats_dict = {
        "measure": ["unweighted_r", "unweighted_p", "unweighted_slope",
                     "unweighted_slope_se", "weighted_r", "weighted_p",
                     "n_occupations"],
        "value": [r_uw, p_uw, slope_uw, se_uw, r_w, p_w, len(df)],
    }
    stats_df = pd.DataFrame(stats_dict)

    out1 = TABDIR / "rate_sensitivity_correlation.csv"
    stats_df.to_csv(out1, index=False)

    out2 = TABDIR / "rate_sensitivity_by_quartile.csv"
    quartile_stats.to_csv(out2, index=False)

    print(f"  Saved: {out1.name}, {out2.name}")
    print("\n  Quartile means (negative = more rate-sensitive):")
    for _, row in quartile_stats.iterrows():
        print(f"    {row['exposure_quartile']}: "
              f"mean={row['mean']:.4f}, median={row['median']:.4f}, "
              f"n={row['count']:.0f}")


def main():
    print("=" * 70)
    print("STEP 10: Interest rate sensitivity vs AI exposure")
    print("=" * 70)

    # Load data
    merged = pd.read_csv(PROCESSED / "postings_daioe_merged.csv")
    daioe = pd.read_csv(PROCESSED / "daioe_quartiles.csv")

    print(f"Input: {len(merged):,} occupation×month rows")

    # Compute rate sensitivity
    print("\nComputing revealed rate sensitivity...")
    occ = compute_rate_sensitivity(merged)

    # Merge with DAIOE
    print("\nMerging with DAIOE...")
    analysis = merge_with_daioe(occ, daioe)

    # Scatter plot
    print("\nPlotting scatter (with weighted + unweighted fit lines)...")
    slope_uw, intercept_uw, r_uw, p_uw, se_uw, r_w, p_w = plot_scatter(analysis)

    # Interpret
    print(f"\n  KEY RESULTS:")
    print(f"  Unweighted: r = {r_uw:.3f} (p = {p_uw:.3f})")
    print(f"  Weighted:   r = {r_w:.3f} (p = {p_w:.3f})")
    print()
    if abs(r_uw) < 0.1 and p_uw > 0.05:
        print("  → No significant correlation between AI exposure and")
        print("    monetary policy sensitivity. The Riksbank rate hike")
        print("    affected postings broadly across all exposure levels.")
        print("  → This means the posting decline cannot be attributed to")
        print("    either AI or differential interest rate sensitivity —")
        print("    it was a broad-based macro shock.")
    elif r_uw > 0:
        print("  → Positive r: AI-exposed occupations are slightly LESS")
        print("    rate-sensitive (same direction as US, Brynjolfsson A22).")
    else:
        print("  → Negative r: AI-exposed occupations are MORE rate-sensitive.")
        print("    This supports the macro-confound explanation.")

    # Box plot
    print("\nPlotting box plot...")
    plot_boxplot(analysis)

    # Save statistics
    print("\nSaving statistics...")
    save_statistics(analysis, slope_uw, intercept_uw, r_uw, p_uw, se_uw, r_w, p_w)

    print("\nDone.")


if __name__ == "__main__":
    main()
