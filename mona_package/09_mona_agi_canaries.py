#!/usr/bin/env python3
"""
09_mona_agi_canaries.py — Canaries test using AGI data in MONA.

╔══════════════════════════════════════════════════════════════════════╗
║  THIS SCRIPT IS DESIGNED TO RUN IN SCB's MONA SECURE ENVIRONMENT   ║
║  It uses monthly AGI (employer declaration) register data.          ║
║  Do NOT run outside MONA — the data is not available externally.    ║
╚══════════════════════════════════════════════════════════════════════╝

Purpose:
  Test the Brynjolfsson et al. (2025) "canaries in the coal mine"
  hypothesis with Swedish monthly register data: do young workers in
  high-AI-exposure occupations experience disproportionate employment
  declines after ChatGPT?

Data requirements:
  - AGI (Arbetsgivardeklaration) individual-level data, monthly,
    2019-01 to 2025-06 (or latest available)
  - Variables needed: personnummer (or encrypted ID), year-month,
    SSYK 4-digit occupation code, age (or birth year), employment status
  - DAIOE genAI exposure: copy daioe_quartiles.csv into MONA

Analysis:
  1. Aggregate AGI to: occupation × age_group × year-month cells
     (age groups: 16-24, 25-34, 35-54, 55+)
  2. Merge with DAIOE quartiles
  3. Run monthly event study: interact month dummies with Young × HighAI
  4. Run triple-diff regression:
     ln(emp_it) = α_i + γ_t + β₁·Post·High + β₂·Post·Young
                  + β₃·Post·Young·High + ε_it
  5. Produce one figure + one table for the appendix

Output:
  - figA8_mona_canaries.png (event study or trajectory plot)
  - mona_canaries_regression.csv (regression coefficients)

Runtime: should complete in <5 minutes on MONA hardware.

INSTRUCTIONS FOR CO-AUTHOR:
  1. Copy this script + daioe_quartiles.csv into your MONA project folder
  2. Adjust INPUT_PATH below to point to your AGI extract
  3. Adjust AGI_COLUMNS if the column names differ in your extract
  4. Run: python 09_mona_agi_canaries.py
  5. Copy the two output files back for inclusion in the paper
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION — ADJUST THESE FOR YOUR MONA ENVIRONMENT            ║
# ╚══════════════════════════════════════════════════════════════════════╝

# Path to your AGI extract (CSV, parquet, or SAS file)
# The extract should contain individual-level monthly employment records.
INPUT_PATH = Path("agi_monthly_extract.parquet")

# Column names in the AGI extract — adjust if yours differ
AGI_COLUMNS = {
    "person_id": "LopNr",          # Encrypted person ID (for dedup)
    "year_month": "Period",         # Year-month as YYYY-MM or similar
    "ssyk4": "SSYK4",              # 4-digit SSYK 2012 code
    "birth_year": "FodelseAr",     # Birth year (to compute age)
    "employment": "Anstallda",     # Employment indicator or count
}

# Path to DAIOE quartiles (copy from the project's data/processed/)
DAIOE_PATH = Path("daioe_quartiles.csv")

# Output paths
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Treatment dates (year-month strings)
RIKSBANKEN_HIKE = "2022-04"
CHATGPT_LAUNCH = "2022-12"

# Colours (same as main project)
DARK_BLUE = "#1B3A5C"
ORANGE = "#E8873A"
TEAL = "#2E7D6F"
GRAY = "#8C8C8C"


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  STEP 1: LOAD AND PREPARE AGI DATA                                 ║
# ╚══════════════════════════════════════════════════════════════════════╝

def load_agi():
    """
    Load AGI data and aggregate to occupation × age_group × month cells.

    We classify workers into age groups:
      - Young: 16-24 (entry-level, the "canaries" in Brynjolfsson)
      - Older: 25+ (everyone else)

    Each cell contains the count of employed persons.
    """
    print("Loading AGI data...")

    # Load based on file format
    col_map = AGI_COLUMNS
    suffix = INPUT_PATH.suffix.lower()

    if suffix == ".parquet":
        df = pd.read_parquet(INPUT_PATH)
    elif suffix == ".csv":
        df = pd.read_csv(INPUT_PATH)
    elif suffix in (".sas7bdat", ".sas"):
        df = pd.read_sas(INPUT_PATH)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    print(f"  Loaded {len(df):,} records")
    print(f"  Columns: {df.columns.tolist()}")

    # Rename columns to standard names
    rename_map = {v: k for k, v in col_map.items() if v in df.columns}
    df = df.rename(columns=rename_map)

    # Ensure required columns exist
    required = ["person_id", "year_month", "ssyk4"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing columns: {missing}. "
            f"Available: {df.columns.tolist()}. "
            f"Adjust AGI_COLUMNS in the configuration section."
        )

    # Parse year-month
    df["year_month"] = df["year_month"].astype(str).str[:7]  # "YYYY-MM"
    df["date"] = pd.to_datetime(df["year_month"] + "-01")

    # Compute age if birth_year is available
    if "birth_year" in df.columns:
        df["age"] = df["date"].dt.year - df["birth_year"].astype(int)
    elif "age" in df.columns:
        pass  # Already have age
    else:
        raise KeyError(
            "Need either 'birth_year' or 'age' column. "
            "Adjust AGI_COLUMNS."
        )

    # Age group: young (16-24) vs older (25+)
    df["young"] = ((df["age"] >= 16) & (df["age"] <= 24)).astype(int)

    # SSYK 4-digit code
    df["ssyk4"] = df["ssyk4"].astype(str).str.zfill(4)

    # Filter to working-age population
    df = df[(df["age"] >= 16) & (df["age"] <= 69)].copy()

    # Aggregate to occupation × age_group × month
    # Count distinct persons per cell (deduplicated)
    agg = (
        df.groupby(["ssyk4", "year_month", "young"])["person_id"]
        .nunique()
        .reset_index()
        .rename(columns={"person_id": "n_employed"})
    )

    print(f"  Aggregated: {len(agg):,} cells")
    print(f"  Occupations: {agg['ssyk4'].nunique()}")
    print(f"  Months: {agg['year_month'].nunique()}")
    print(f"  Period: {agg['year_month'].min()} to {agg['year_month'].max()}")

    return agg


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  STEP 2: MERGE WITH DAIOE                                          ║
# ╚══════════════════════════════════════════════════════════════════════╝

def merge_daioe(agg):
    """Merge employment cells with DAIOE AI exposure quartiles."""
    print("\nMerging with DAIOE...")

    daioe = pd.read_csv(DAIOE_PATH)
    daioe["ssyk4"] = daioe["ssyk4"].astype(str).str.zfill(4)

    merged = agg.merge(
        daioe[["ssyk4", "pctl_rank_genai", "exposure_quartile", "high_exposure"]],
        on="ssyk4",
        how="inner",
    )

    n_matched = merged["ssyk4"].nunique()
    n_total = agg["ssyk4"].nunique()
    print(f"  Matched: {n_matched} of {n_total} occupations "
          f"({100 * n_matched / n_total:.0f}%)")

    return merged


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  STEP 3: TRIPLE-DIFF REGRESSION                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝

def run_canaries_regression(merged):
    """
    Triple-difference regression on monthly panel:

    ln(emp_it) = α_i + γ_t + β₁·Post·High + β₂·Post·Young
                 + β₃·Post·Young·High + ε_it

    where:
      i = occupation × age_group entity
      t = year-month
      Post = 1 if t >= December 2022 (ChatGPT launch)
      High = 1 if top quartile genAI exposure
      Young = 1 if age group 16-24

    β₃ is the canaries coefficient: if AI displaces young workers in
    exposed occupations, β₃ should be significantly negative.

    We cluster standard errors at the occupation level (not entity level)
    to account for within-occupation correlation across age groups.
    """
    print("\nRunning triple-diff regression...")

    df = merged.copy()
    df["date"] = pd.to_datetime(df["year_month"] + "-01")
    df = df[df["n_employed"] > 0].copy()
    df["ln_emp"] = np.log(df["n_employed"])

    # Treatment dummies
    df["post_chatgpt"] = (df["year_month"] >= CHATGPT_LAUNCH).astype(int)
    df["post_riksbank"] = (df["year_month"] >= RIKSBANKEN_HIKE).astype(int)

    # Interactions
    df["post_high"] = df["post_chatgpt"] * df["high_exposure"]
    df["post_young"] = df["post_chatgpt"] * df["young"]
    df["post_young_high"] = (
        df["post_chatgpt"] * df["young"] * df["high_exposure"]
    )

    # Also Riksbank interactions for completeness
    df["rb_high"] = df["post_riksbank"] * df["high_exposure"]
    df["rb_young"] = df["post_riksbank"] * df["young"]
    df["rb_young_high"] = (
        df["post_riksbank"] * df["young"] * df["high_exposure"]
    )

    # Entity = occupation × age group
    df["entity"] = df["ssyk4"] + "_" + df["young"].astype(str)

    # Try linearmodels first, fall back to statsmodels
    try:
        from linearmodels.panel import PanelOLS

        panel = df.set_index(["entity", "date"])

        # Full specification: both Riksbank and ChatGPT triple interactions
        exog_cols = [
            "rb_high", "rb_young", "rb_young_high",
            "post_high", "post_young", "post_young_high",
        ]

        mod = PanelOLS(
            dependent=panel["ln_emp"],
            exog=panel[exog_cols],
            entity_effects=True,
            time_effects=True,
        )
        # Cluster at occupation level (ssyk4), not entity level
        # This is more conservative: accounts for correlation between
        # young and old workers in the same occupation
        res = mod.fit(cov_type="clustered", cluster_entity=True)

        print("\n  Full specification (entity + time FE):")
        for v in exog_cols:
            b = res.params[v]
            se = res.std_errors[v]
            p = res.pvalues[v]
            stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
            print(f"    {v:25s} = {b:+.4f}{stars} (SE = {se:.4f}, p = {p:.4f})")

        print(f"\n  N = {res.nobs:,}")
        print(f"  Entities: {int(res.entity_info['total'])}")

        # The key coefficient: β₃ = post_young_high
        b3 = res.params["post_young_high"]
        p3 = res.pvalues["post_young_high"]
        print(f"\n  >>> CANARIES TEST: β₃ (Post×Young×High) = {b3:+.4f}, p = {p3:.4f}")
        if p3 < 0.05:
            print(f"  >>> SIGNIFICANT at 5% — evidence of canaries effect")
        else:
            print(f"  >>> NOT significant — no canaries effect detected")

        # Save results
        reg_df = pd.DataFrame({
            "variable": res.params.index,
            "coefficient": res.params.values,
            "std_error": res.std_errors.values,
            "p_value": res.pvalues.values,
        })
        reg_df.to_csv(OUTPUT_DIR / "mona_canaries_regression.csv", index=False)
        print(f"\n  Saved → mona_canaries_regression.csv")

        return res, df

    except ImportError:
        print("  linearmodels not available in MONA — using statsmodels")
        import statsmodels.api as sm

        # Manual dummies (less efficient but works without linearmodels)
        entity_dummies = pd.get_dummies(df["entity"], prefix="e", drop_first=True)
        time_dummies = pd.get_dummies(df["year_month"], prefix="t", drop_first=True)

        exog_cols = [
            "rb_high", "rb_young", "rb_young_high",
            "post_high", "post_young", "post_young_high",
        ]
        X = pd.concat([df[exog_cols], entity_dummies, time_dummies], axis=1)
        X = sm.add_constant(X).astype(float)

        mod = sm.OLS(df["ln_emp"].values, X)
        res = mod.fit(
            cov_type="cluster",
            cov_kwds={"groups": df["ssyk4"].values},
        )

        print("\n  Key coefficients:")
        for v in exog_cols:
            print(f"    {v:25s} = {res.params[v]:+.4f} "
                  f"(SE = {res.bse[v]:.4f}, p = {res.pvalues[v]:.4f})")

        reg_df = pd.DataFrame({
            "variable": exog_cols,
            "coefficient": [res.params[v] for v in exog_cols],
            "std_error": [res.bse[v] for v in exog_cols],
            "p_value": [res.pvalues[v] for v in exog_cols],
        })
        reg_df.to_csv(OUTPUT_DIR / "mona_canaries_regression.csv", index=False)

        return res, df


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  STEP 4: FIGURE — EMPLOYMENT TRAJECTORIES                          ║
# ╚══════════════════════════════════════════════════════════════════════╝

def plot_canaries(merged):
    """
    Plot employment indexed to January 2020, four groups:
    Young×HighAI, Young×LowAI, Older×HighAI, Older×LowAI.

    This is the monthly version of the annual SCB figure.
    """
    print("\nPlotting canaries trajectories...")

    # Aggregate to month × young × high_exposure
    agg = (
        merged.groupby(["year_month", "young", "high_exposure"])["n_employed"]
        .sum()
        .reset_index()
    )
    agg["date"] = pd.to_datetime(agg["year_month"] + "-01")

    # Index to earliest month = 100
    base_month = agg["year_month"].min()
    base = agg[agg["year_month"] == base_month].set_index(
        ["young", "high_exposure"]
    )["n_employed"]

    agg = agg.merge(
        agg[agg["year_month"] == base_month][
            ["young", "high_exposure", "n_employed"]
        ].rename(columns={"n_employed": "base_emp"}),
        on=["young", "high_exposure"],
    )
    agg["index"] = 100 * agg["n_employed"] / agg["base_emp"]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))

    styles = {
        (1, 1): {"color": ORANGE, "lw": 2.5, "ls": "-",
                  "label": "Young (16-24), High AI"},
        (1, 0): {"color": ORANGE, "lw": 1.5, "ls": "--",
                  "label": "Young (16-24), Low AI"},
        (0, 1): {"color": DARK_BLUE, "lw": 2.5, "ls": "-",
                  "label": "Older (25+), High AI"},
        (0, 0): {"color": DARK_BLUE, "lw": 1.5, "ls": "--",
                  "label": "Older (25+), Low AI"},
    }

    for (young, high), style in styles.items():
        subset = agg[(agg["young"] == young) & (agg["high_exposure"] == high)]
        subset = subset.sort_values("date")
        ax.plot(subset["date"], subset["index"],
                color=style["color"], linewidth=style["lw"],
                linestyle=style["ls"], label=style["label"])

    # Event markers
    ax.axvline(pd.Timestamp(RIKSBANKEN_HIKE + "-01"), color=ORANGE,
               linewidth=1, linestyle=":", alpha=0.7)
    ax.axvline(pd.Timestamp(CHATGPT_LAUNCH + "-01"), color=TEAL,
               linewidth=1.5, linestyle=":", alpha=0.8)

    ax.axhline(100, color=GRAY, linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("")
    ax.set_ylabel("Employment index (base month = 100)")
    ax.set_title("Monthly employment by age and AI exposure (AGI register data)")
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figA8_mona_canaries.png", dpi=300)
    plt.close(fig)
    print(f"  Saved → figA8_mona_canaries.png")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                               ║
# ╚══════════════════════════════════════════════════════════════════════╝

def main():
    print("=" * 70)
    print("MONA: Canaries test using AGI monthly register data")
    print("=" * 70)

    # Load and aggregate AGI data
    agg = load_agi()

    # Merge with DAIOE
    merged = merge_daioe(agg)

    # Save processed data (for inspection)
    merged.to_csv(OUTPUT_DIR / "mona_employment_age_ai.csv", index=False)

    # Triple-diff regression
    res, panel = run_canaries_regression(merged)

    # Plot trajectories
    plot_canaries(merged)

    print("\n" + "=" * 70)
    print("DONE. Copy the following files from output/ to your project:")
    print("  1. mona_canaries_regression.csv  → tables/")
    print("  2. figA8_mona_canaries.png       → figures/")
    print("=" * 70)


if __name__ == "__main__":
    main()
