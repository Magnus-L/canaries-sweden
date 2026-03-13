#!/usr/bin/env python3
"""
31_cumulative_pretrend_bias.py -- Cumulative pre-trend bias calculation.

NOT a MONA script. Reads event study CSV output locally.

PURPOSE:
  Pre-trend tests in the event study formally reject (p=0.000) for most
  age groups because the sample is so large that even tiny pre-period
  coefficients are statistically significant. But statistical significance
  is not economic significance. The question is: if we linearly
  extrapolate the pre-treatment trend into the post-treatment window,
  how much of the observed post-treatment effect could it explain?

  This gives a concrete, interpretable number: "X% of the endpoint
  effect could be attributed to continuation of the pre-trend."

APPROACH:
  1. Read the event study coefficients (ref = 2022H1).
  2. For each age group, fit OLS through the pre-treatment coefficients:
       coef_t = alpha + beta * t   (t = 0, 1, ..., 5 for 2019H1-2021H2)
  3. beta = per-half-year pre-trend slope.
  4. Cumulative bias = beta * N_post, where N_post = number of post-
     treatment half-years.
  5. Compare to the actual endpoint coefficient (2025H1).
  6. Bias share = cumulative_bias / endpoint_coef.

  A small bias share (e.g., <20%) means the pre-trend, even if
  statistically significant, explains little of the post-treatment
  divergence. The formal test is overpowered, not informative.

OUTPUT:
  - pretrend_bias_table.csv
  - pretrend_bias_summary.txt
  - Printed table to stdout
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys


# ======================================================================
#   CONFIGURATION
# ======================================================================

PROJECT = Path(__file__).resolve().parent.parent

# Try multiple locations for the event study CSV
CANDIDATE_PATHS = [
    PROJECT / "data" / "output" / "corrected_es_all_ref2022H1.csv",
    PROJECT / "output_18" / "corrected_es_all_ref2022H1.csv",
]

REF_PERIOD = "2022H1"

# All periods in order (for assigning numeric time index)
ALL_PERIODS = [
    "2019H1", "2019H2",
    "2020H1", "2020H2",
    "2021H1", "2021H2",
    "2022H1",
    "2022H2",
    "2023H1", "2023H2",
    "2024H1", "2024H2",
    "2025H1",
]

# Pre-treatment: periods before the reference (2022H1)
PRE_PERIODS = [p for p in ALL_PERIODS if p < REF_PERIOD]
# Post-treatment: periods after the reference
POST_PERIODS = [p for p in ALL_PERIODS if p > REF_PERIOD]
# Endpoint: last post-treatment period
ENDPOINT = POST_PERIODS[-1]  # "2025H1"


# ======================================================================
#   FIND INPUT FILE
# ======================================================================

def find_input_csv():
    """
    Locate the event study CSV. Checks standard locations first,
    then falls back to a command-line argument if provided.
    """
    # Check command-line argument first (highest priority)
    if len(sys.argv) > 1:
        cli_path = Path(sys.argv[1])
        if cli_path.exists():
            return cli_path
        else:
            print(f"  WARNING: Command-line path not found: {cli_path}")

    # Check standard locations
    for p in CANDIDATE_PATHS:
        if p.exists():
            return p

    # Nothing found
    print("ERROR: Could not find corrected_es_all_ref2022H1.csv")
    print("  Searched:")
    for p in CANDIDATE_PATHS:
        print(f"    {p}")
    print("  You can also pass the path as a command-line argument:")
    print("    python 31_cumulative_pretrend_bias.py /path/to/file.csv")
    sys.exit(1)


# ======================================================================
#   MAIN ANALYSIS
# ======================================================================

def compute_pretrend_bias(es_df):
    """
    For each age group, fit a linear trend through pre-treatment
    event study coefficients and compute cumulative bias.

    The linear trend is: coef = alpha + beta * t
    where t is a numeric index (0 for first pre-period, 1 for second, etc.)

    Cumulative bias = beta * N_post_periods, i.e. how far the trend
    would carry if extrapolated through all post-treatment half-years.

    Returns a DataFrame with one row per age group.
    """
    results = []

    age_groups = es_df["age_group"].unique()

    for ag in sorted(age_groups):
        sub = es_df[es_df["age_group"] == ag].copy()

        # Get pre-treatment coefficients
        pre = sub[sub["period"].isin(PRE_PERIODS)].sort_values("period")
        if len(pre) < 2:
            print(f"  {ag}: Too few pre-treatment periods ({len(pre)}), skipping")
            continue

        # Assign numeric time index to pre-treatment periods
        # t = 0 for first pre-period, 1 for second, etc.
        pre = pre.copy()
        pre["t"] = range(len(pre))

        # Fit linear trend: coef = alpha + beta * t
        # Using numpy polyfit (degree 1) for simplicity
        coeffs = np.polyfit(pre["t"].values, pre["coef"].values, deg=1)
        beta = coeffs[0]   # slope (per half-year)
        alpha = coeffs[1]  # intercept

        # R-squared of the linear fit
        predicted = alpha + beta * pre["t"].values
        ss_res = np.sum((pre["coef"].values - predicted) ** 2)
        ss_tot = np.sum((pre["coef"].values - pre["coef"].mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # Number of post-treatment periods
        n_post = len(POST_PERIODS)

        # Cumulative bias: extrapolate the pre-trend across all post periods
        # From the reference period (end of pre-treatment), the trend
        # continues for n_post additional half-years.
        cumulative_bias = beta * n_post

        # Actual endpoint coefficient
        endpoint_row = sub[sub["period"] == ENDPOINT]
        if endpoint_row.empty:
            print(f"  {ag}: No endpoint coefficient found for {ENDPOINT}")
            continue
        endpoint_coef = endpoint_row["coef"].values[0]
        endpoint_se = endpoint_row["se"].values[0]

        # Bias share: what fraction of the endpoint effect could
        # the pre-trend explain?
        if endpoint_coef != 0:
            bias_share = cumulative_bias / endpoint_coef
        else:
            bias_share = np.nan

        results.append({
            "age_group": ag,
            "n_pre_periods": len(pre),
            "pretrend_slope": beta,
            "pretrend_r2": r_squared,
            "n_post_periods": n_post,
            "cumulative_bias": cumulative_bias,
            "endpoint_coef": endpoint_coef,
            "endpoint_se": endpoint_se,
            "bias_share": bias_share,
        })

    return pd.DataFrame(results)


def format_table(bias_df):
    """Format the bias table for readable stdout output."""
    lines = []
    lines.append("")
    lines.append("=" * 90)
    lines.append("CUMULATIVE PRE-TREND BIAS ANALYSIS")
    lines.append("=" * 90)
    lines.append("")
    lines.append(
        f"{'Age Group':>10s} | {'Pre-trend':>10s} | {'R2':>6s} | "
        f"{'x periods':>9s} | {'= Cum.bias':>10s} | "
        f"{'Endpoint':>10s} | {'Bias/Endpt':>10s}"
    )
    lines.append("-" * 90)

    for _, row in bias_df.iterrows():
        bias_pct = f"{row['bias_share']:.0%}" if not np.isnan(row['bias_share']) else "n/a"
        lines.append(
            f"{row['age_group']:>10s} | "
            f"{row['pretrend_slope']:>+10.4f} | "
            f"{row['pretrend_r2']:>6.3f} | "
            f"x {int(row['n_post_periods']):>6d} | "
            f"{row['cumulative_bias']:>+10.4f} | "
            f"{row['endpoint_coef']:>+10.4f} | "
            f"{bias_pct:>10s}"
        )

    lines.append("-" * 90)
    return "\n".join(lines)


def write_summary(bias_df, input_path):
    """Write narrative summary of the pre-trend bias analysis."""
    output_dir = PROJECT / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "=" * 60,
        "CUMULATIVE PRE-TREND BIAS ANALYSIS",
        "Script: 31_cumulative_pretrend_bias.py",
        f"Input: {input_path}",
        f"Reference period: {REF_PERIOD}",
        f"Pre-treatment periods: {', '.join(PRE_PERIODS)} ({len(PRE_PERIODS)} periods)",
        f"Post-treatment periods: {', '.join(POST_PERIODS)} ({len(POST_PERIODS)} periods)",
        f"Endpoint: {ENDPOINT}",
        "=" * 60,
        "",
        "METHOD:",
        "  For each age group, fit OLS through pre-treatment event study",
        "  coefficients: coef_t = alpha + beta * t. Then extrapolate:",
        "  cumulative_bias = beta * N_post_periods. Compare to the actual",
        "  endpoint coefficient to get the 'bias share': what fraction of",
        "  the observed effect could be explained by trend continuation.",
        "",
        "RESULTS:",
    ]

    # Add the formatted table
    table = format_table(bias_df)
    lines.append(table)
    lines.append("")

    # Interpretation for key age groups
    lines.append("INTERPRETATION:")
    for _, row in bias_df.iterrows():
        ag = row["age_group"]
        bs = row["bias_share"]
        slope = row["pretrend_slope"]
        endpoint = row["endpoint_coef"]

        if np.isnan(bs):
            lines.append(f"  {ag}: Cannot compute bias share (endpoint = 0).")
            continue

        # Direction check: do pre-trend and endpoint go in the same direction?
        same_direction = (slope < 0 and endpoint < 0) or (slope > 0 and endpoint > 0)

        if not same_direction:
            lines.append(
                f"  {ag}: Pre-trend slope ({slope:+.4f}) goes in OPPOSITE direction "
                f"to endpoint ({endpoint:+.4f}). Pre-trend biases AGAINST the finding. "
                f"The true effect may be larger than estimated."
            )
        elif abs(bs) < 0.20:
            lines.append(
                f"  {ag}: Bias share = {bs:.0%}. Pre-trend explains less than 20% of "
                f"the endpoint. Overpowered pre-trend test is not economically "
                f"informative."
            )
        elif abs(bs) < 0.50:
            lines.append(
                f"  {ag}: Bias share = {bs:.0%}. Pre-trend explains a moderate share "
                f"of the endpoint. Some caution warranted, but the post-treatment "
                f"effect substantially exceeds the trend."
            )
        else:
            lines.append(
                f"  {ag}: Bias share = {bs:.0%}. Pre-trend could explain a large "
                f"share of the endpoint. The post-treatment effect may partly "
                f"reflect trend continuation rather than a treatment effect."
            )

    lines.extend([
        "",
        "NOTE ON R-SQUARED:",
        "  R2 of the linear fit through pre-treatment coefficients indicates",
        "  how well a simple linear trend describes the pre-period pattern.",
        "  Low R2 means the pre-period coefficients are noisy / non-linear,",
        "  making linear extrapolation less meaningful. High R2 means the",
        "  pre-trend is well-characterised and the extrapolation is informative.",
        "",
        "NOTE ON DIRECTION:",
        "  When the pre-trend slope goes in the opposite direction to the",
        "  post-treatment effect, the pre-trend works AGAINST our finding.",
        "  In this case, the formal pre-trend test failure (p<0.05) actually",
        "  makes our results more conservative, not less credible.",
    ])

    summary_text = "\n".join(lines)

    # Save
    (output_dir / "pretrend_bias_summary.txt").write_text(summary_text)

    return summary_text


# ======================================================================
#   MAIN
# ======================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("31_cumulative_pretrend_bias.py")
    print("Cumulative pre-trend bias calculation")
    print("(local script -- no MONA access needed)")
    print("=" * 70)

    # Find input file
    input_path = find_input_csv()
    print(f"\n  Input: {input_path}")

    # Read event study coefficients
    es_df = pd.read_csv(input_path)
    print(f"  Loaded {len(es_df)} rows, {es_df['age_group'].nunique()} age groups")
    print(f"  Periods: {sorted(es_df['period'].unique())}")

    # Compute pre-trend bias
    bias_df = compute_pretrend_bias(es_df)

    # Print formatted table
    table_str = format_table(bias_df)
    print(table_str)

    # Save CSV
    output_dir = PROJECT / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    bias_df.to_csv(output_dir / "pretrend_bias_table.csv", index=False)
    print(f"\n  Saved: {output_dir / 'pretrend_bias_table.csv'}")

    # Write summary
    summary = write_summary(bias_df, input_path)
    print(f"  Saved: {output_dir / 'pretrend_bias_summary.txt'}")

    # Print full summary
    print("\n" + summary)
