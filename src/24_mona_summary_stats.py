#!/usr/bin/env python3
"""
24_mona_summary_stats.py -- Employment summary statistics for appendix Table A2.

======================================================================
  THIS SCRIPT IS DESIGNED TO RUN IN SCB's MONA SECURE ENVIRONMENT
  Do NOT run outside MONA -- the data is not available externally.
======================================================================

PURPOSE:
  Computes descriptive statistics for the balanced employer-level panels
  used in the employment DiD and event study regressions (scripts 15, 18).
  These statistics document the panel structure for the reader and help
  diagnose potential zero-cell bias.

WHAT IT COMPUTES:
  1. N employers (unique employer_id across all age groups)
  2. N employer×quartile cells (unique fe_emp_q combinations)
  3. Mean and median cell size (n_emp) by age group
  4. Share of zero-employment cells by age_group × quartile × half-year
  5. How the zero-cell share evolves over time (the key diagnostic)

  The zero-cell share diagnostic is important because our balanced-panel
  approach zero-fills missing cells. If zero-cells increase over time in
  Q4 young-worker cells specifically, that is consistent with AI-driven
  displacement at the extensive margin (entire cohorts disappearing from
  employer×quartile cells).

DATA PIPELINE:
  Identical to script 18 (18_mona_eventstudy_corrected.py):
  - pyodbc → SQL Server on MONA (year-by-year UNION ALL)
  - Cascading SSYK lookup for years >= 2023
  - DAIOE quartile merge
  - MIN_EMPLOYER_SIZE = 5 filter
  - Balanced panel with zero-filling per age group

OUTPUT FILES (to output_24/):
  - employment_summary_stats.csv  -- main summary table
  - zero_cell_shares.csv          -- zero-cell share by age_group × quartile × half_year
  - zero_cell_shares_by_age.png   -- figure: zero-cell share over time, one line per age group
  - summary_stats.txt             -- human-readable report

ESTIMATED RUNTIME:
  ~15-30 min (data loading dominates; no regression estimation).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import time
warnings.filterwarnings("ignore")


# ======================================================================
#   CONFIGURATION
# ======================================================================

# --- MONA SQL connection (same as scripts 14, 15, 18) ---
import pyodbc
conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=monasql.micro.intra;"
    "DATABASE=P1207;"
    "Trusted_Connection=yes;"
)

# --- DAIOE quartile file ---
DAIOE_PATH = r"\\micro.intra\Projekt\P1207$\P1207_Gem\Lydia P1207\daioe_quartiles.csv"

# --- Output directory ---
OUTPUT_DIR = Path("output_24")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Age group definitions (same as scripts 15, 18) ---
AGE_GROUPS = {
    "22-25": (22, 25),
    "26-30": (26, 30),
    "31-34": (31, 34),
    "35-40": (35, 40),
    "41-49": (41, 49),
    "50+":   (50, 69),
}

# --- Minimum employer size ---
MIN_EMPLOYER_SIZE = 5

# --- Colours (consistent with other project figures) ---
ORANGE = "#E8873A"
TEAL = "#2E7D6F"
GRAY = "#8C8C8C"


# ======================================================================
#   STEP 1: LOAD DATA (copied from script 18 for self-containedness)
# ======================================================================

def pull_year(year, conn):
    """
    Pull one year of AGI data with cascading SSYK lookup (2023 -> 2022 -> 2021).
    Same approach as scripts 14, 15, and 18 for data consistency.

    For years >= 2023, SSYK is looked up via COALESCE across Individ_2023,
    Individ_2022, Individ_2021 because LISA-based SSYK codes end at 2023.
    For years < 2023, a single Individ_{year} join suffices.

    Year 2025 uses _prel suffix (preliminary data) and only months 1-6.
    All other years use _def suffix (definitive) and months 1-12.
    """
    print(f"  Processing year {year}...")

    individ_year = min(year, 2023)

    if year < 2025:
        suffix = "_def"
        max_month = 12
    else:
        suffix = "_prel"
        max_month = 6

    monthly_queries = []
    for month in range(1, max_month + 1):
        ym = f"{year}{month:02d}"

        if individ_year >= 2023:
            monthly_queries.append(f"""
                SELECT
                    agi.P1207_LOPNR_PEORGNR AS employer_id,
                    agi.PERIOD AS period,
                    COALESCE(ind23.Ssyk4_2012_J16,
                             ind22.Ssyk4_2012_J16,
                             ind21.Ssyk4_2012_J16) AS ssyk4,
                    COALESCE(ind23.FodelseAr,
                             ind22.FodelseAr,
                             ind21.FodelseAr) AS birth_year,
                    agi.P1207_LOPNR_PERSONNR AS person_id
                FROM dbo.Arb_AGIIndivid{ym}{suffix} agi
                LEFT JOIN dbo.Individ_2023 ind23
                    ON agi.P1207_LOPNR_PERSONNR = ind23.P1207_LopNr_PersonNr
                LEFT JOIN dbo.Individ_2022 ind22
                    ON agi.P1207_LOPNR_PERSONNR = ind22.P1207_LopNr_PersonNr
                LEFT JOIN dbo.Individ_2021 ind21
                    ON agi.P1207_LOPNR_PERSONNR = ind21.P1207_LopNr_PersonNr
            """)
        else:
            monthly_queries.append(f"""
                SELECT
                    agi.P1207_LOPNR_PEORGNR AS employer_id,
                    agi.PERIOD AS period,
                    ind.Ssyk4_2012_J16 AS ssyk4,
                    ind.FodelseAr AS birth_year,
                    agi.P1207_LOPNR_PERSONNR AS person_id
                FROM dbo.Arb_AGIIndivid{ym}{suffix} agi
                LEFT JOIN dbo.Individ_{individ_year} ind
                    ON agi.P1207_LOPNR_PERSONNR = ind.P1207_LopNr_PersonNr
            """)

    union_query = "\nUNION ALL\n".join(monthly_queries)

    query = f"""
    WITH base AS (
        {union_query}
    ),
    age_calc AS (
        SELECT
            employer_id,
            period,
            RIGHT('0000'+CAST(ssyk4 AS VARCHAR(4)),4) AS ssyk4,
            person_id,
            CAST(LEFT(period,4) AS INT) - birth_year AS age
        FROM base
        WHERE birth_year IS NOT NULL
    )
    SELECT
        employer_id,
        LEFT(period,4) + '-' + SUBSTRING(period,5,2) AS year_month,
        ssyk4,
        CASE
            WHEN age BETWEEN 22 AND 25 THEN '22-25'
            WHEN age BETWEEN 26 AND 30 THEN '26-30'
            WHEN age BETWEEN 31 AND 34 THEN '31-34'
            WHEN age BETWEEN 35 AND 40 THEN '35-40'
            WHEN age BETWEEN 41 AND 49 THEN '41-49'
            WHEN age BETWEEN 50 AND 69 THEN '50+'
            ELSE NULL
        END AS age_group,
        COUNT(DISTINCT person_id) AS n_emp
    FROM age_calc
    WHERE age BETWEEN 22 AND 69
    GROUP BY
        employer_id,
        period,
        ssyk4,
        CASE
            WHEN age BETWEEN 22 AND 25 THEN '22-25'
            WHEN age BETWEEN 26 AND 30 THEN '26-30'
            WHEN age BETWEEN 31 AND 34 THEN '31-34'
            WHEN age BETWEEN 35 AND 40 THEN '35-40'
            WHEN age BETWEEN 41 AND 49 THEN '41-49'
            WHEN age BETWEEN 50 AND 69 THEN '50+'
            ELSE NULL
        END
    """
    return pd.read_sql(query, conn)


def load_and_prepare():
    """
    Load AGI data year by year (same pipeline as scripts 14, 15, 18),
    merge DAIOE quartiles, assign age groups,
    aggregate to employer x quartile x age_group x month cells.

    Returns the aggregated DataFrame with columns:
        employer_id, exposure_quartile, age_group, year_month, n_emp
    """
    print("=" * 70)
    print("STEP 1: Loading and preparing data")
    print("=" * 70)

    # Pull year by year (consistent with scripts 14, 15, 18)
    frames = []
    t0 = time.time()
    for year in range(2019, 2026):
        frames.append(pull_year(year, conn))
    df = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {len(df):,} records in {time.time()-t0:.0f}s")

    # Drop rows without SSYK or age group
    df = df[df["ssyk4"].notna() & (df["ssyk4"] != "None")].copy()
    df = df[df["age_group"].notna()].copy()

    # SSYK4 as zero-padded string
    df["ssyk4"] = df["ssyk4"].astype(str).str.zfill(4)

    # Merge DAIOE quartiles
    print("  Merging DAIOE quartiles...")
    daioe = pd.read_csv(DAIOE_PATH)
    daioe["ssyk4"] = daioe["ssyk4"].astype(str).str.zfill(4)
    if daioe["exposure_quartile"].dtype == object:
        q_map = {"Q1 (lowest)": 1, "Q2": 2, "Q3": 3, "Q4 (highest)": 4}
        daioe["exposure_quartile"] = daioe["exposure_quartile"].map(q_map)
    df = df.merge(daioe[["ssyk4", "exposure_quartile"]], on="ssyk4", how="inner")
    print(f"  After DAIOE merge: {len(df):,} records")

    # Rename for consistency with downstream code
    df = df.rename(columns={"n_emp": "person_count"})

    # Filter small employers (total person-count across all records < threshold)
    emp_size = df.groupby("employer_id")["person_count"].sum()
    large_emp = emp_size[emp_size >= MIN_EMPLOYER_SIZE].index
    df = df[df["employer_id"].isin(large_emp)].copy()
    print(f"  Employers with >={MIN_EMPLOYER_SIZE} workers: "
          f"{df['employer_id'].nunique():,}")

    # Aggregate to employer x quartile x age_group x month
    # (already aggregated in SQL by ssyk4, but re-aggregate to collapse
    #  across ssyk4 codes within each quartile)
    print("  Aggregating to employer x quartile x age_group x month...")
    agg = (
        df.groupby(["employer_id", "exposure_quartile", "age_group", "year_month"])
        ["person_count"]
        .sum()
        .reset_index()
        .rename(columns={"person_count": "n_emp"})
    )
    print(f"  Panel cells: {len(agg):,}")
    print(f"  Period: {agg['year_month'].min()} to {agg['year_month'].max()}")

    return agg


def balance_panel_for_age(agg, age_label):
    """
    Build balanced panel for one age group: every (employer, quartile)
    combination the employer is ever observed in x all months.
    Missing cells filled with n_emp = 0.

    Applies identification restriction: employers must have workers in
    both Q4 and at least one of Q1-Q3 (needed for within-employer
    high-vs-low comparison).

    WHY ZERO-FILLING MATTERS:
      Without zero-filling, firms that shed ALL workers of an age group
      in a quartile disappear from the data. This drops the strongest
      treatment cases and biases the DiD toward zero. The zero-fill
      approach captures the extensive margin (complete cohort exit).

    Returns a DataFrame with columns:
        employer_id, exposure_quartile, year_month, n_emp, age_group
    """
    sub = agg[agg["age_group"] == age_label].copy()
    all_months = sorted(agg["year_month"].unique())

    # Unique employer-quartile pairs for this age group
    emp_q = (
        sub.groupby(["employer_id", "exposure_quartile"])
        .size()
        .reset_index()[["employer_id", "exposure_quartile"]]
    )

    # Identification restriction: employers spanning Q4 and Q1-Q3
    q4_emps = set(emp_q.loc[emp_q["exposure_quartile"] == 4, "employer_id"])
    low_emps = set(emp_q.loc[emp_q["exposure_quartile"] < 4, "employer_id"])
    valid_emps = q4_emps & low_emps
    emp_q = emp_q[emp_q["employer_id"].isin(valid_emps)]

    # Cross join: employer-quartile pairs x all months
    months_df = pd.DataFrame({"year_month": all_months})
    emp_q["_k"] = 1
    months_df["_k"] = 1
    balanced = emp_q.merge(months_df, on="_k").drop(columns="_k")

    # Left join actual employment counts
    balanced = balanced.merge(
        sub[["employer_id", "exposure_quartile", "year_month", "n_emp"]],
        on=["employer_id", "exposure_quartile", "year_month"],
        how="left",
    )
    balanced["n_emp"] = balanced["n_emp"].fillna(0).astype(int)
    balanced["age_group"] = age_label

    n_zeros = (balanced["n_emp"] == 0).sum()
    pct_zero = 100 * n_zeros / len(balanced) if len(balanced) > 0 else 0
    print(f"  Balanced panel ({age_label}): {len(balanced):,} cells "
          f"({n_zeros:,} zeros = {pct_zero:.1f}%)")
    print(f"  Employers with both Q4 and Q1-Q3: {len(valid_emps):,}")

    return balanced


# ======================================================================
#   STEP 1b: HALF-YEAR HELPER (same as script 18)
# ======================================================================

def assign_halfyear(ym):
    """Map year-month string to half-year label, e.g. '2022-03' -> '2022H1'."""
    year = ym[:4]
    month = int(ym[5:7])
    half = "H1" if month <= 6 else "H2"
    return f"{year}{half}"


# ======================================================================
#   STEP 2: COMPUTE SUMMARY STATISTICS
# ======================================================================

def compute_summary_stats(agg):
    """
    Compute panel summary statistics across all age groups.

    Returns:
        summary_df: DataFrame with one row per age group, columns for
                    N employers, N emp×quartile cells, mean/median n_emp,
                    total cells, zero cells, zero share.
        zero_shares: DataFrame with zero-cell share by age_group ×
                     quartile × half_year.
        balanced_all: Concatenated balanced panels (for downstream use).
    """
    print("\n" + "=" * 70)
    print("STEP 2: Computing summary statistics")
    print("=" * 70)

    summary_rows = []
    zero_share_rows = []
    balanced_parts = []

    for age_label in AGE_GROUPS:
        print(f"\n--- {age_label} ---")

        # Build balanced panel (same as regression scripts)
        balanced = balance_panel_for_age(agg, age_label)
        balanced_parts.append(balanced)

        # Construct the employer x quartile identifier
        balanced["fe_emp_q"] = (
            balanced["employer_id"].astype(str) + "_" +
            balanced["exposure_quartile"].astype(str)
        )

        # --- Panel-level summary ---
        n_employers = balanced["employer_id"].nunique()
        n_emp_q_cells = balanced["fe_emp_q"].nunique()
        total_cells = len(balanced)
        n_zeros = (balanced["n_emp"] == 0).sum()
        zero_share = n_zeros / total_cells if total_cells > 0 else 0

        # Cell size statistics (including zeros, since that is the
        # balanced panel the regressions actually use)
        mean_n_emp = balanced["n_emp"].mean()
        median_n_emp = balanced["n_emp"].median()
        p25_n_emp = balanced["n_emp"].quantile(0.25)
        p75_n_emp = balanced["n_emp"].quantile(0.75)

        # Cell size statistics excluding zeros (for comparison)
        positive = balanced.loc[balanced["n_emp"] > 0, "n_emp"]
        mean_n_emp_pos = positive.mean() if len(positive) > 0 else np.nan
        median_n_emp_pos = positive.median() if len(positive) > 0 else np.nan

        summary_rows.append({
            "age_group": age_label,
            "n_employers": n_employers,
            "n_emp_q_cells": n_emp_q_cells,
            "total_panel_cells": total_cells,
            "n_zero_cells": n_zeros,
            "zero_cell_share": round(zero_share, 4),
            "mean_n_emp": round(mean_n_emp, 2),
            "median_n_emp": round(median_n_emp, 1),
            "p25_n_emp": round(p25_n_emp, 1),
            "p75_n_emp": round(p75_n_emp, 1),
            "mean_n_emp_positive": round(mean_n_emp_pos, 2),
            "median_n_emp_positive": round(median_n_emp_pos, 1),
        })

        # --- Zero-cell shares by quartile x half-year ---
        # This is the key diagnostic: if zero-cells in Q4 young workers
        # increase sharply post-ChatGPT, it confirms the extensive margin story
        balanced["halfyear"] = balanced["year_month"].apply(assign_halfyear)

        for q in sorted(balanced["exposure_quartile"].unique()):
            for hy in sorted(balanced["halfyear"].unique()):
                mask = (
                    (balanced["exposure_quartile"] == q) &
                    (balanced["halfyear"] == hy)
                )
                sub_hy = balanced[mask]
                n_total = len(sub_hy)
                n_zero = (sub_hy["n_emp"] == 0).sum()
                share = n_zero / n_total if n_total > 0 else np.nan

                zero_share_rows.append({
                    "age_group": age_label,
                    "exposure_quartile": q,
                    "halfyear": hy,
                    "n_cells": n_total,
                    "n_zero_cells": n_zero,
                    "zero_cell_share": round(share, 4),
                })

    summary_df = pd.DataFrame(summary_rows)
    zero_shares = pd.DataFrame(zero_share_rows)
    balanced_all = pd.concat(balanced_parts, ignore_index=True)

    return summary_df, zero_shares, balanced_all


# ======================================================================
#   STEP 3: FIGURES
# ======================================================================

def plot_zero_cell_shares(zero_shares):
    """
    Plot zero-cell share over time, one line per age group.
    Averaged across quartiles (overall diagnostic).
    Also a Q4-vs-rest comparison panel.
    """
    print("\n" + "=" * 70)
    print("STEP 3: Plotting zero-cell share diagnostics")
    print("=" * 70)

    # --- Figure A: Overall zero-cell share by age group ---
    # Average across quartiles within each age group x half-year
    avg_by_age = (
        zero_shares.groupby(["age_group", "halfyear"])
        ["zero_cell_share"]
        .mean()
        .reset_index()
    )

    # Sort half-year periods chronologically
    all_halfyears = sorted(avg_by_age["halfyear"].unique())
    x_map = {hy: i for i, hy in enumerate(all_halfyears)}

    fig, ax = plt.subplots(figsize=(10, 5))

    # Colour cycle for age groups
    colours = ["#E8873A", "#2E7D6F", "#5B8DBE", "#D4A037", "#8C8C8C", "#C44E52"]

    for idx, age_label in enumerate(AGE_GROUPS):
        sub = avg_by_age[avg_by_age["age_group"] == age_label].copy()
        sub["x"] = sub["halfyear"].map(x_map)
        sub = sub.sort_values("x")
        ax.plot(sub["x"], sub["zero_cell_share"], "o-",
                color=colours[idx % len(colours)],
                label=age_label, linewidth=1.5, markersize=4)

    ax.set_xticks(range(len(all_halfyears)))
    ax.set_xticklabels(all_halfyears, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Share of zero-employment cells", fontsize=11)
    ax.set_title("Zero-cell share in balanced panels over time", fontsize=13)
    ax.legend(fontsize=9, title="Age group", loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "zero_cell_shares_by_age.png", dpi=150)
    plt.close(fig)
    print("  Saved: zero_cell_shares_by_age.png")

    # --- Figure B: Q4 vs Q1-Q3 comparison (2x3 panel by age group) ---
    age_list = list(AGE_GROUPS.keys())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)

    for idx, age_label in enumerate(age_list):
        ax = axes[idx // 3, idx % 3]
        sub = zero_shares[zero_shares["age_group"] == age_label].copy()
        sub["x"] = sub["halfyear"].map(x_map)

        # Q4 (high AI exposure)
        q4 = sub[sub["exposure_quartile"] == 4].sort_values("x")
        # Q1-Q3 average (low AI exposure)
        low = (
            sub[sub["exposure_quartile"] < 4]
            .groupby("halfyear")
            .agg({"zero_cell_share": "mean", "x": "first"})
            .reset_index()
            .sort_values("x")
        )

        ax.plot(q4["x"], q4["zero_cell_share"], "o-",
                color=ORANGE, linewidth=1.5, markersize=4, label="Q4 (high AI)")
        ax.plot(low["x"], low["zero_cell_share"], "s-",
                color=TEAL, linewidth=1.5, markersize=4, label="Q1-Q3 (low AI)")

        ax.set_xticks(range(len(all_halfyears)))
        ax.set_xticklabels(all_halfyears, rotation=45, ha="right", fontsize=7)
        ax.set_title(age_label, fontsize=11)
        ax.grid(axis="y", alpha=0.3)

        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Zero-cell share: Q4 (high AI exposure) vs Q1-Q3",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "zero_cell_shares_q4_vs_rest.png", dpi=150)
    plt.close(fig)
    print("  Saved: zero_cell_shares_q4_vs_rest.png")


# ======================================================================
#   STEP 4: HUMAN-READABLE REPORT
# ======================================================================

def write_report(summary_df, zero_shares):
    """Write a plain-text summary report for quick inspection."""
    lines = []
    lines.append("=" * 70)
    lines.append("EMPLOYMENT SUMMARY STATISTICS — CANARIES-SWEDEN")
    lines.append("Script: 24_mona_summary_stats.py")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 70)

    lines.append("\n--- PANEL STRUCTURE BY AGE GROUP ---\n")
    lines.append(summary_df.to_string(index=False))

    lines.append("\n\n--- INTERPRETATION ---")
    lines.append("N employers: unique employer_id with both Q4 and Q1-Q3 workers")
    lines.append("N emp×q cells: unique employer × quartile combinations in panel")
    lines.append("total_panel_cells: emp×q cells × 78 months (Jan 2019 – Jun 2025)")
    lines.append("zero_cell_share: fraction of cells where n_emp = 0 (zero-filled)")
    lines.append("mean/median_n_emp: cell size including zeros")
    lines.append("mean/median_n_emp_positive: cell size excluding zeros")

    # Key comparison: Q4 vs Q1-Q3 zero-cell evolution for youngest group
    lines.append("\n\n--- ZERO-CELL SHARE DIAGNOSTIC (22-25, Q4 vs Q1-Q3) ---\n")
    young_q4 = zero_shares[
        (zero_shares["age_group"] == "22-25") &
        (zero_shares["exposure_quartile"] == 4)
    ][["halfyear", "zero_cell_share"]].rename(
        columns={"zero_cell_share": "Q4_zero_share"}
    )
    young_low = (
        zero_shares[
            (zero_shares["age_group"] == "22-25") &
            (zero_shares["exposure_quartile"] < 4)
        ]
        .groupby("halfyear")["zero_cell_share"]
        .mean()
        .reset_index()
        .rename(columns={"zero_cell_share": "Q1-Q3_zero_share"})
    )
    comparison = young_q4.merge(young_low, on="halfyear")
    comparison["Q4_minus_Q1Q3"] = (
        comparison["Q4_zero_share"] - comparison["Q1-Q3_zero_share"]
    )
    lines.append(comparison.to_string(index=False))

    lines.append("\n\nIf Q4_minus_Q1Q3 increases post-2022H2, it means young workers")
    lines.append("in high-AI-exposure occupations are disappearing from employer")
    lines.append("panels at a higher rate than workers in low-AI occupations.")
    lines.append("This is the extensive-margin channel: employers stop hiring")
    lines.append("young workers into AI-exposed roles entirely.")

    # Same comparison for 50+ (should show opposite or flat pattern)
    lines.append("\n\n--- ZERO-CELL SHARE DIAGNOSTIC (50+, Q4 vs Q1-Q3) ---\n")
    old_q4 = zero_shares[
        (zero_shares["age_group"] == "50+") &
        (zero_shares["exposure_quartile"] == 4)
    ][["halfyear", "zero_cell_share"]].rename(
        columns={"zero_cell_share": "Q4_zero_share"}
    )
    old_low = (
        zero_shares[
            (zero_shares["age_group"] == "50+") &
            (zero_shares["exposure_quartile"] < 4)
        ]
        .groupby("halfyear")["zero_cell_share"]
        .mean()
        .reset_index()
        .rename(columns={"zero_cell_share": "Q1-Q3_zero_share"})
    )
    comparison_old = old_q4.merge(old_low, on="halfyear")
    comparison_old["Q4_minus_Q1Q3"] = (
        comparison_old["Q4_zero_share"] - comparison_old["Q1-Q3_zero_share"]
    )
    lines.append(comparison_old.to_string(index=False))

    report = "\n".join(lines)
    (OUTPUT_DIR / "summary_stats.txt").write_text(report, encoding="utf-8")
    print(f"\n  Saved: summary_stats.txt")
    return report


# ======================================================================
#   MAIN
# ======================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("24_mona_summary_stats.py")
    print("Employment summary statistics for appendix Table A2")
    print("=" * 70)

    t_start = time.time()

    # Step 1: Load data (same pipeline as scripts 15, 18)
    agg = load_and_prepare()

    # Step 2: Compute summary statistics with balanced panels
    summary_df, zero_shares, balanced_all = compute_summary_stats(agg)

    # Save CSVs
    summary_df.to_csv(OUTPUT_DIR / "employment_summary_stats.csv", index=False)
    print(f"\n  Saved: employment_summary_stats.csv")

    zero_shares.to_csv(OUTPUT_DIR / "zero_cell_shares.csv", index=False)
    print(f"  Saved: zero_cell_shares.csv")

    # Step 3: Figures
    plot_zero_cell_shares(zero_shares)

    # Step 4: Human-readable report
    report = write_report(summary_df, zero_shares)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"DONE. Total runtime: {elapsed/60:.1f} minutes")
    print(f"All output in: {OUTPUT_DIR}/")
    print(f"{'=' * 70}")

    # Print report to console as well
    print(f"\n{report}")
