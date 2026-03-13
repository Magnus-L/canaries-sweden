#!/usr/bin/env python3
"""
30_mona_entrant_composition.py -- Entrant composition analysis by DAIOE quartile.

======================================================================
  THIS SCRIPT IS DESIGNED TO RUN IN SCB's MONA SECURE ENVIRONMENT
  Do NOT run outside MONA -- the data is not available externally.
======================================================================

PURPOSE:
  The paper currently reports that ~13% of young workers are in Q4
  occupations. But this static share does not rule out a composition
  channel: if young entrants increasingly avoid Q4 occupations after
  ChatGPT, the "canaries" finding might partly reflect sorting rather
  than within-occupation displacement.

  This script tests whether the Q4 share among labour market entrants
  (people appearing in AGI for the first time) changes differentially
  after ChatGPT launch. A stable Q4 share across years would support
  the displacement interpretation. A declining Q4 share would suggest
  that some of the employment composition shift is driven by entrants
  sorting away from high-AI occupations.

APPROACH:
  1. For each year 2019-2025, pull person-level records (person_id,
     year, ssyk4, age) from AGI.
  2. Identify "entrants" = individuals whose first appearance in any
     AGI table falls in that year (min observed year per person_id).
  3. Merge SSYK4 to DAIOE quartiles.
  4. For young entrants (22-25 and 26-30), compute the share entering
     each DAIOE quartile, by year.
  5. Report the Q4 share time series and test for a structural break.

OUTPUT (to output_30/):
  - entrant_q4_share_by_year.csv
  - entrant_quartile_shares.csv
  - entrant_composition.png
  - entrant_summary.txt
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

import pyodbc
conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=monasql.micro.intra;"
    "DATABASE=P1207;"
    "Trusted_Connection=yes;"
)

DAIOE_PATH = r"\\micro.intra\Projekt\P1207$\P1207_Gem\Lydia P1207\daioe_quartiles.csv"

OUTPUT_DIR = Path("output_30")
OUTPUT_DIR.mkdir(exist_ok=True)

# Treatment dates (for figure annotations)
RIKSBANK_YEAR = 2022   # April 2022
CHATGPT_YEAR = 2022    # December 2022 (shows up in 2023 data)

# Age groups of interest
YOUNG_GROUPS = ["22-25", "26-30"]

# Colours (consistent with project palette)
ORANGE = "#E8873A"
TEAL = "#2E7D6F"
GRAY = "#8C8C8C"
DARK_BLUE = "#1B3A5C"
LIGHT_GRAY = "#C8C8C8"


# ======================================================================
#   STEP 1: PULL PERSON-LEVEL DATA (year, person_id, ssyk4, age)
# ======================================================================

def pull_person_year(year, conn):
    """
    Pull person-level data for one year from AGI.

    Returns one row per person per year with their SSYK4 code and age.
    Uses DISTINCT on person_id to get one record per person (taking
    the modal/first SSYK4 via SQL aggregation).

    Same cascading SSYK lookup as script 18: for years >= 2023,
    COALESCE across Individ_2023, Individ_2022, Individ_2021.
    """
    print(f"  Pulling year {year}...")
    t0 = time.time()

    individ_year = min(year, 2023)

    if year < 2025:
        suffix = "_def"
        max_month = 12
    else:
        suffix = "_prel"
        max_month = 6

    # Build UNION ALL across all months for this year.
    # We only need one observation per person per year, so we take
    # DISTINCT person_id with their SSYK4 from the earliest month
    # they appear in that year (MIN(period) to break ties).
    monthly_queries = []
    for month in range(1, max_month + 1):
        ym = f"{year}{month:02d}"

        if individ_year >= 2023:
            monthly_queries.append(f"""
                SELECT
                    agi.P1207_LOPNR_PERSONNR AS person_id,
                    agi.PERIOD AS period,
                    COALESCE(ind23.Ssyk4_2012_J16,
                             ind22.Ssyk4_2012_J16,
                             ind21.Ssyk4_2012_J16) AS ssyk4,
                    COALESCE(ind23.FodelseAr,
                             ind22.FodelseAr,
                             ind21.FodelseAr) AS birth_year
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
                    agi.P1207_LOPNR_PERSONNR AS person_id,
                    agi.PERIOD AS period,
                    ind.Ssyk4_2012_J16 AS ssyk4,
                    ind.FodelseAr AS birth_year
                FROM dbo.Arb_AGIIndivid{ym}{suffix} agi
                LEFT JOIN dbo.Individ_{individ_year} ind
                    ON agi.P1207_LOPNR_PERSONNR = ind.P1207_LopNr_PersonNr
            """)

    union_query = "\nUNION ALL\n".join(monthly_queries)

    # Aggregate to one row per person per year:
    # - Take the SSYK4 from their earliest month (MIN(period))
    # - Compute age from birth_year
    # We use a CTE with ROW_NUMBER to pick the first month's record.
    query = f"""
    WITH all_months AS (
        {union_query}
    ),
    ranked AS (
        SELECT
            person_id,
            RIGHT('0000'+CAST(ssyk4 AS VARCHAR(4)),4) AS ssyk4,
            CAST(LEFT(CAST(period AS VARCHAR(6)),4) AS INT) - birth_year AS age,
            ROW_NUMBER() OVER (
                PARTITION BY person_id
                ORDER BY period ASC
            ) AS rn
        FROM all_months
        WHERE birth_year IS NOT NULL
          AND ssyk4 IS NOT NULL
    )
    SELECT
        person_id,
        ssyk4,
        age
    FROM ranked
    WHERE rn = 1
    """

    df = pd.read_sql(query, conn)
    df["obs_year"] = year

    elapsed = time.time() - t0
    print(f"    {len(df):,} unique persons in {elapsed:.0f}s")

    return df


def load_all_persons():
    """
    Pull person-level data for all years 2019-2025.
    Returns a DataFrame with columns: person_id, ssyk4, age, obs_year.
    One row per person per year.
    """
    print("=" * 70)
    print("STEP 1: Loading person-level data from AGI (2019-2025)")
    print("=" * 70)

    frames = []
    t0 = time.time()
    for year in range(2019, 2026):
        frames.append(pull_person_year(year, conn))
    df = pd.concat(frames, ignore_index=True)
    print(f"\n  Total records: {len(df):,} in {time.time()-t0:.0f}s")

    return df


# ======================================================================
#   STEP 2: IDENTIFY ENTRANTS (first year in AGI)
# ======================================================================

def identify_entrants(df):
    """
    For each person, find their first observed year in AGI.
    An "entrant" in year Y is someone whose min(obs_year) == Y.

    Note: people first observed in 2019 may have been working before
    our data window. The 2019 cohort is therefore a mix of genuine
    entrants and incumbents. Results for 2019 should be interpreted
    with caution; the meaningful comparison is 2020-2025 trends.

    Returns the input DataFrame filtered to entrant-year observations,
    with an added 'age_group' column.
    """
    print("\n" + "=" * 70)
    print("STEP 2: Identifying entrants (first AGI appearance)")
    print("=" * 70)

    # Find first year per person
    first_year = df.groupby("person_id")["obs_year"].min().reset_index()
    first_year.columns = ["person_id", "first_year"]

    # Merge back: keep only the row where obs_year == first_year
    # (i.e., the entrant observation)
    df = df.merge(first_year, on="person_id", how="inner")
    entrants = df[df["obs_year"] == df["first_year"]].copy()

    # Assign age groups
    entrants["age_group"] = pd.cut(
        entrants["age"],
        bins=[21, 25, 30, 34, 40, 49, 69],
        labels=["22-25", "26-30", "31-34", "35-40", "41-49", "50+"],
        right=True,
    )
    entrants = entrants[entrants["age_group"].notna()].copy()

    for yr in sorted(entrants["obs_year"].unique()):
        n = len(entrants[entrants["obs_year"] == yr])
        print(f"  {yr}: {n:,} entrants (age 22-69)")

    return entrants


# ======================================================================
#   STEP 3: MERGE DAIOE QUARTILES AND COMPUTE SHARES
# ======================================================================

def compute_quartile_shares(entrants):
    """
    Merge SSYK4 -> DAIOE quartile, then compute the share of entrants
    in each quartile by year and age group.

    Returns two DataFrames:
      - shares_full: year x age_group x quartile shares
      - q4_summary: year x age_group with Q4 share only
    """
    print("\n" + "=" * 70)
    print("STEP 3: Computing quartile shares among entrants")
    print("=" * 70)

    # Load DAIOE quartiles
    daioe = pd.read_csv(DAIOE_PATH)
    daioe["ssyk4"] = daioe["ssyk4"].astype(str).str.zfill(4)
    if daioe["exposure_quartile"].dtype == object:
        q_map = {"Q1 (lowest)": 1, "Q2": 2, "Q3": 3, "Q4 (highest)": 4}
        daioe["exposure_quartile"] = daioe["exposure_quartile"].map(q_map)

    # Clean SSYK4 in entrants
    entrants["ssyk4"] = entrants["ssyk4"].astype(str).str.zfill(4)

    # Merge
    merged = entrants.merge(
        daioe[["ssyk4", "exposure_quartile"]], on="ssyk4", how="inner"
    )
    n_before = len(entrants)
    n_after = len(merged)
    print(f"  DAIOE merge: {n_after:,} / {n_before:,} entrants matched "
          f"({100*n_after/n_before:.1f}%)")

    # Count entrants by year x age_group x quartile
    counts = (
        merged.groupby(["obs_year", "age_group", "exposure_quartile"])
        .size()
        .reset_index(name="n_entrants")
    )

    # Total entrants by year x age_group (denominator)
    totals = (
        counts.groupby(["obs_year", "age_group"])["n_entrants"]
        .sum()
        .reset_index(name="n_total")
    )

    # Shares
    shares = counts.merge(totals, on=["obs_year", "age_group"])
    shares["share"] = shares["n_entrants"] / shares["n_total"]

    # Q4 summary
    q4 = shares[shares["exposure_quartile"] == 4][
        ["obs_year", "age_group", "n_entrants", "n_total", "share"]
    ].copy()
    q4 = q4.rename(columns={"share": "q4_share", "n_entrants": "n_q4"})

    # Print summary for young groups
    for ag in YOUNG_GROUPS:
        sub = q4[q4["age_group"] == ag].sort_values("obs_year")
        print(f"\n  Q4 share among {ag} entrants:")
        for _, row in sub.iterrows():
            print(f"    {int(row['obs_year'])}: {row['q4_share']:.3f} "
                  f"({int(row['n_q4']):,} / {int(row['n_total']):,})")

    return shares, q4


# ======================================================================
#   STEP 4: FIGURE
# ======================================================================

def plot_entrant_composition(q4, shares):
    """
    Plot Q4 share among young entrants over time, with vertical lines
    at Riksbank rate hike (Apr 2022) and ChatGPT launch (Dec 2022).
    """
    print("\n" + "=" * 70)
    print("STEP 4: Plotting entrant composition figure")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for idx, ag in enumerate(YOUNG_GROUPS):
        ax = axes[idx]
        sub = q4[q4["age_group"] == ag].sort_values("obs_year")

        if sub.empty:
            ax.set_title(f"Age {ag}")
            continue

        years = sub["obs_year"].values
        q4_share = sub["q4_share"].values

        ax.plot(years, q4_share, "o-", color=ORANGE, linewidth=2,
                markersize=7, label="Q4 share")

        # Vertical lines for treatment dates
        # Riksbank: between 2022 and 2023 on the x-axis
        ax.axvline(2022.3, color=GRAY, linestyle="--", linewidth=1,
                   label="Riksbank rate hike")
        # ChatGPT: end of 2022
        ax.axvline(2022.9, color=TEAL, linestyle="--", linewidth=1,
                   label="ChatGPT launch")

        ax.set_xlabel("Year", fontsize=12)
        if idx == 0:
            ax.set_ylabel("Q4 share among entrants", fontsize=12)
        ax.set_title(f"Age {ag}", fontsize=13)
        ax.set_xticks(range(2019, 2026))
        ax.set_xticklabels([str(y) for y in range(2019, 2026)],
                           rotation=45, ha="right", fontsize=10)
        ax.legend(fontsize=9, loc="best")

        # Add percentage labels on points
        for yr, sh in zip(years, q4_share):
            ax.annotate(f"{sh:.1%}", (yr, sh),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=9, color=ORANGE)

    fig.suptitle(
        "Share of labour market entrants in Q4 (high AI exposure) occupations",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "entrant_composition.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved entrant_composition.png")

    # --- Also plot full quartile distribution for 22-25 ---
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sub_all = shares[shares["age_group"] == "22-25"].copy()
    q_colors = {1: LIGHT_GRAY, 2: "#DCE6F2", 3: TEAL, 4: ORANGE}

    for q in [1, 2, 3, 4]:
        qdata = sub_all[sub_all["exposure_quartile"] == q].sort_values("obs_year")
        ax2.plot(qdata["obs_year"], qdata["share"], "o-",
                 color=q_colors[q], linewidth=2, markersize=6,
                 label=f"Q{q}")

    ax2.axvline(2022.3, color=GRAY, linestyle="--", linewidth=1,
                label="Riksbank")
    ax2.axvline(2022.9, color=DARK_BLUE, linestyle="--", linewidth=1,
                label="ChatGPT")
    ax2.set_xlabel("Year", fontsize=12)
    ax2.set_ylabel("Share of entrants", fontsize=12)
    ax2.set_title("Quartile distribution of 22-25 entrants", fontsize=13)
    ax2.set_xticks(range(2019, 2026))
    ax2.legend(fontsize=10, ncol=3)
    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / "entrant_quartile_dist_22_25.png", dpi=150)
    plt.close(fig2)
    print("  Saved entrant_quartile_dist_22_25.png")


# ======================================================================
#   STEP 5: SUMMARY AND INTERPRETATION
# ======================================================================

def write_summary(q4, shares):
    """Write narrative summary of the entrant composition analysis."""
    print("\n" + "=" * 70)
    print("STEP 5: Writing summary")
    print("=" * 70)

    lines = [
        "=" * 60,
        "ENTRANT COMPOSITION ANALYSIS",
        "Script: 30_mona_entrant_composition.py",
        "=" * 60,
        "",
        "QUESTION: Does the Q4 share among young entrants change",
        "differentially after ChatGPT, suggesting a composition/sorting",
        "channel rather than (or in addition to) within-occupation",
        "displacement?",
        "",
    ]

    for ag in YOUNG_GROUPS:
        sub = q4[q4["age_group"] == ag].sort_values("obs_year")
        if sub.empty:
            continue

        lines.append(f"--- Age group: {ag} ---")
        lines.append(f"{'Year':>6s}  {'Q4 share':>10s}  {'N(Q4)':>8s}  {'N(total)':>10s}")

        for _, row in sub.iterrows():
            lines.append(
                f"{int(row['obs_year']):>6d}  {row['q4_share']:>10.4f}  "
                f"{int(row['n_q4']):>8,d}  {int(row['n_total']):>10,d}"
            )

        # Pre/post comparison
        pre = sub[sub["obs_year"] <= 2022]["q4_share"]
        post = sub[sub["obs_year"] > 2022]["q4_share"]
        if len(pre) > 0 and len(post) > 0:
            pre_mean = pre.mean()
            post_mean = post.mean()
            diff = post_mean - pre_mean
            lines.append(f"  Pre-ChatGPT mean (2019-2022): {pre_mean:.4f}")
            lines.append(f"  Post-ChatGPT mean (2023-2025): {post_mean:.4f}")
            lines.append(f"  Difference: {diff:+.4f}")
            if abs(diff) < 0.01:
                lines.append("  --> STABLE: Q4 share essentially unchanged.")
                lines.append("      Supports displacement, not sorting.")
            elif diff < -0.01:
                lines.append("  --> DECLINING: Young entrants sorting away from Q4.")
                lines.append("      Composition channel may contribute to canaries result.")
            else:
                lines.append("  --> RISING: Young entrants moving toward Q4.")
                lines.append("      No sorting-away; displacement result is conservative.")
        lines.append("")

    lines.extend([
        "INTERPRETATION:",
        "  If Q4 share is stable across years, the canaries finding in the",
        "  main employment analysis reflects within-occupation displacement",
        "  (fewer young workers hired into Q4 roles), not a change in which",
        "  occupations young workers sort into.",
        "",
        "  If Q4 share declines post-ChatGPT, both channels may operate:",
        "  young workers are simultaneously displaced from Q4 occupations",
        "  (main result) AND sorting away from them (this result).",
        "",
        "  CAVEAT: 2019 'entrants' include incumbents who entered before",
        "  our data window. The 2019 cohort overstates entrants. Focus",
        "  the interpretation on 2020-2025 trends.",
    ])

    summary_text = "\n".join(lines)

    (OUTPUT_DIR / "entrant_summary.txt").write_text(summary_text)
    print(summary_text)

    return summary_text


# ======================================================================
#   MAIN
# ======================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("30_mona_entrant_composition.py")
    print("Entrant composition analysis: Q4 share over time")
    print("=" * 70)

    t_start = time.time()

    # Step 1: Pull person-level data
    all_persons = load_all_persons()

    # Step 2: Identify entrants
    entrants = identify_entrants(all_persons)

    # Step 3: Compute quartile shares
    shares, q4 = compute_quartile_shares(entrants)

    # Step 4: Plot
    plot_entrant_composition(q4, shares)

    # Step 5: Save CSVs and summary
    shares.to_csv(OUTPUT_DIR / "entrant_quartile_shares.csv", index=False)
    print(f"  Saved entrant_quartile_shares.csv")

    q4.to_csv(OUTPUT_DIR / "entrant_q4_share_by_year.csv", index=False)
    print(f"  Saved entrant_q4_share_by_year.csv")

    write_summary(q4, shares)

    elapsed = time.time() - t_start
    print(f"\n  Total runtime: {elapsed/60:.1f} minutes")
    print(f"  All output in: {OUTPUT_DIR}/")
