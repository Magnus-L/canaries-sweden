#!/usr/bin/env python3
"""
17_mona_pctchange_figure.py — Brynjolfsson-style percentage change figure.

======================================================================
  THIS SCRIPT RUNS IN SCB's MONA SECURE ENVIRONMENT.
  Uses monthly AGI register data via the same SQL connection as script 15.
  Do NOT run outside MONA — the data is not available externally.
======================================================================

Purpose:
  Produce a monthly employment figure showing PERCENTAGE CHANGE from
  a base month (Oct 2022), rather than the indexed level (100 = base).
  This is the format used in Brynjolfsson et al. (2025) and their
  associated blog post / Stanford working paper, where the headline
  figure shows e.g. "-15.7%" for young workers in top-AI occupations.

  Our existing canaries figures (scripts 09, 15) show employment
  indexed to 100. That format is standard in economics but less
  immediately legible for a policy audience. The percentage-change
  format puts the magnitude front and centre.

Figures produced:
  1. figA_pctchange_headline.png
     Single line: 22-25-year-olds in Q4 (top AI exposure) occupations.
     Percentage deviation from Oct 2022 baseline.
     Annotated with the final-month value (e.g., "-12.3%").

  2. figA_pctchange_comparison.png
     Four lines comparing:
       - Young (22-25), High AI (Q4)     [orange solid — the "canaries"]
       - Young (22-25), Low AI (Q1-Q3)   [orange dashed]
       - Older (26+), High AI (Q4)       [dark blue solid]
       - Older (26+), Low AI (Q1-Q3)     [dark blue dashed]
     All as percentage deviation from Oct 2022.
     This figure makes the divergence between groups visually stark.

  3. pctchange_data.csv
     Underlying monthly data for all four groups, so we can also
     re-plot or tweak the figure locally without MONA access.

Data pipeline:
  - Uses same SQL connection as script 15 (pyodbc → MONA SQL Server)
  - Pulls AGI employer declarations, aggregates to age_group × month
  - Merges DAIOE genAI exposure quartiles
  - Computes percentage change from base month

INSTRUCTIONS FOR LYDIA:
  1. Copy this script into your MONA project folder (same location as
     script 14 and 15)
  2. Run: python 17_mona_pctchange_figure.py
  3. Export the three output files from output_17/
  4. Share the CSV — Magnus can re-plot locally if needed

Runtime: ~5 minutes (same data pull as script 15, lighter computation).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


# ======================================================================
#   CONFIGURATION
# ======================================================================

# --- MONA connection (same as scripts 14-15) ---
import pyodbc
conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=monasql.micro.intra;"
    "DATABASE=P1207;"
    "Trusted_Connection=yes;"
)

# --- DAIOE quartile file (same as scripts 14-15) ---
DAIOE_PATH = r"\\micro.intra\Projekt\P1207$\P1207_Gem\Lydia P1207\daioe_quartiles.csv"

# --- Output directory ---
OUTPUT_DIR = Path("output_17")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Base month for percentage change ---
# Oct 2022: just before ChatGPT launch, after Riksbank hike has begun.
# Same base as the existing canaries figures for consistency.
BASE_MONTH = "2022-10"

# --- Event dates ---
RIKSBANK_YM = "2022-04"
CHATGPT_YM = "2022-12"

# --- Colours (same as the rest of the paper) ---
ORANGE = "#E8873A"
TEAL = "#2E7D6F"
GRAY = "#8C8C8C"
DARK_BLUE = "#1B3A5C"
LIGHT_GRAY = "#D0D0D0"
MILESTONE_COLOR = "#7B2D8E"

# --- GenAI capability milestones (same as script 09) ---
GENAI_MILESTONES = [
    (2023, 3, "GPT-4"),
    (2023, 11, "Enterprise tools"),
    (2024, 5, "GPT-4o"),
    (2024, 9, "o1"),
    (2025, 1, "DeepSeek R1"),
]

# --- Age group definitions ---
# We produce the headline for 22-25 (direct Brynjolfsson comparison)
# and the four-group comparison using young (22-25) vs older (26+).
AGE_GROUPS = {
    "22-25": (22, 25),
    "26-30": (26, 30),
    "31-34": (31, 34),
    "35-40": (35, 40),
    "41-49": (41, 49),
    "50+":   (50, 69),
}

# --- Minimum employer size (same as scripts 14-15) ---
MIN_EMPLOYER_SIZE = 5

# --- Moving average window ---
MA_WINDOW = 3  # 3-month centred moving average


# ======================================================================
#   STEP 1: PULL DATA (same SQL structure as script 15)
# ======================================================================

def pull_year(year, conn):
    """
    Pull one year of AGI data, aggregated to
    employer x year_month x ssyk4 x age_group cells.

    Same as script 15 but without gender (not needed here).
    """
    print(f"  Pulling year {year}...")

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
            # Cascading SSYK lookup: try Individ_2023, then 2022, then 2021.
            # LISA-based Individ tables end at 2023. Workers who entered the
            # labour market after the latest Individ vintage will be missing
            # from Individ_2023 but may appear in an earlier vintage.
            # COALESCE picks the most recent available match.
            monthly_queries.append(f"""
                SELECT
                    agi.P1207_LOPNR_PEORGNR AS employer_id,
                    agi.PERIOD AS period,
                    COALESCE(ind23.Ssyk4_2012_J16, ind22.Ssyk4_2012_J16, ind21.Ssyk4_2012_J16) AS ssyk4,
                    COALESCE(ind23.FodelseAr, ind22.FodelseAr, ind21.FodelseAr) AS birth_year,
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


def load_all_data():
    """
    Pull all years 2019-2025, merge DAIOE quartiles, apply employer
    size filter. Returns panel at age_group × AI_binary × month level.
    """
    print("=" * 70)
    print("STEP 1: Pulling data from MONA (2019-2025)")
    print("=" * 70)

    years = list(range(2019, 2026))
    all_years = []
    for y in years:
        df_year = pull_year(y, conn)
        all_years.append(df_year)
        print(f"    {y}: {len(df_year):,} rows")

    raw = pd.concat(all_years, ignore_index=True)
    print(f"\n  Total rows from SQL: {len(raw):,}")

    # Drop rows with NULL age group (outside 22-69)
    raw = raw[raw["age_group"].notna()].copy()

    # --- Merge DAIOE quartiles ---
    print("  Merging DAIOE quartiles...")
    daioe = pd.read_csv(DAIOE_PATH)
    daioe["ssyk4"] = daioe["ssyk4"].astype(str).str.zfill(4)
    daioe["exposure_quartile"] = (
        daioe["exposure_quartile"]
        .str.strip()
        .str.extract(r"Q(\d)")
        .astype(int)
    )
    raw["ssyk4"] = raw["ssyk4"].astype(str).str.zfill(4)
    raw = raw.merge(daioe[["ssyk4", "exposure_quartile"]], on="ssyk4", how="inner")
    print(f"  After DAIOE merge: {len(raw):,} rows")

    # --- Employer size filter ---
    emp_size = raw.groupby("employer_id")["n_emp"].sum()
    large_emp = emp_size[emp_size >= MIN_EMPLOYER_SIZE].index
    n_before = raw["employer_id"].nunique()
    raw = raw[raw["employer_id"].isin(large_emp)].copy()
    print(f"  Employers: {n_before:,} -> {raw['employer_id'].nunique():,} "
          f"(>={MIN_EMPLOYER_SIZE} total person-months)")

    # --- Create binary variables ---
    # AI exposure: Q4 (top quartile) vs Q1-Q3
    raw["high_ai"] = (raw["exposure_quartile"] == 4).astype(int)

    # Age: young (22-25) vs older (26+)
    raw["young"] = (raw["age_group"] == "22-25").astype(int)

    print(f"\n  Period: {raw['year_month'].min()} to {raw['year_month'].max()}")
    print(f"  Months: {raw['year_month'].nunique()}")

    return raw


# ======================================================================
#   STEP 2: AGGREGATE AND COMPUTE PERCENTAGE CHANGE
# ======================================================================

def compute_pctchange(raw):
    """
    Aggregate employment to four groups (young/older × high/low AI)
    by month, then compute percentage change from BASE_MONTH.

    Returns a DataFrame with columns:
      year_month, date, group, n_emp, n_emp_ma, pct_change

    Economic intuition: The percentage change shows *how much*
    employment in each group has moved relative to the pre-ChatGPT
    baseline. A -15% value means 15% fewer workers in that group
    compared to October 2022.
    """
    print("\n" + "=" * 70)
    print("STEP 2: Computing percentage change from base month")
    print("=" * 70)

    # Aggregate to group × month
    # Four groups: Young×High, Young×Low, Older×High, Older×Low
    agg = (
        raw.groupby(["year_month", "young", "high_ai"])["n_emp"]
        .sum()
        .reset_index()
    )
    agg["date"] = pd.to_datetime(agg["year_month"] + "-01")
    agg = agg.sort_values(["young", "high_ai", "date"])

    # Group labels for readability
    label_map = {
        (1, 1): "Young (22–25), High AI",
        (1, 0): "Young (22–25), Low AI",
        (0, 1): "Older (26+), High AI",
        (0, 0): "Older (26+), Low AI",
    }
    agg["group"] = agg.apply(
        lambda r: label_map[(r["young"], r["high_ai"])], axis=1
    )

    # 3-month centred moving average (smooths monthly noise)
    agg["n_emp_ma"] = (
        agg.groupby("group")["n_emp"]
        .transform(lambda x: x.rolling(MA_WINDOW, min_periods=1, center=True).mean())
    )

    # Percentage change from base month
    # pct_change = (current / base - 1) × 100
    base = agg[agg["year_month"] == BASE_MONTH][["group", "n_emp_ma"]].copy()
    base = base.rename(columns={"n_emp_ma": "base_val"})
    agg = agg.merge(base, on="group", how="left")
    agg["pct_change"] = (agg["n_emp_ma"] / agg["base_val"] - 1) * 100

    # Print summary for each group at the last available month
    last_month = agg["year_month"].max()
    print(f"\n  Base month: {BASE_MONTH}")
    print(f"  Latest month: {last_month}")
    print(f"\n  Percentage change from {BASE_MONTH} to {last_month}:")
    for grp in label_map.values():
        val = agg[(agg["group"] == grp) & (agg["year_month"] == last_month)]
        if not val.empty:
            pct = val["pct_change"].values[0]
            n = val["n_emp"].values[0]
            print(f"    {grp:35s}: {pct:+.1f}%  (N = {n:,.0f})")

    return agg


# ======================================================================
#   STEP 3: FIGURES
# ======================================================================

def add_event_lines(ax, y_ref=0):
    """
    Draw vertical annotation lines for events and GenAI milestones.
    Same style as the rest of the paper.

    y_ref: the horizontal reference line value (0 for pct change,
           100 for index).
    """
    # Riksbank rate hike (macro tightening)
    ax.axvline(pd.Timestamp(RIKSBANK_YM + "-01"),
               color=ORANGE, linewidth=1, linestyle=":", alpha=0.7,
               label="_nolegend_")

    # ChatGPT launch
    ax.axvline(pd.Timestamp(CHATGPT_YM + "-01"),
               color=TEAL, linewidth=1.5, linestyle=":", alpha=0.9,
               label="_nolegend_")

    # GenAI capability milestones
    for year, month, label in GENAI_MILESTONES:
        date = pd.Timestamp(f"{year}-{month:02d}-01")
        ax.axvline(date, color=MILESTONE_COLOR, linewidth=0.9,
                   linestyle="--", alpha=0.6, label="_nolegend_")
        ax.text(date, ax.get_ylim()[1], label,
                ha="center", va="bottom", fontsize=6,
                color=MILESTONE_COLOR, fontweight="bold", alpha=0.8,
                rotation=45)

    # Horizontal reference at y_ref (0% change line)
    ax.axhline(y_ref, color=GRAY, linewidth=0.8, linestyle="--", alpha=0.5)


def plot_headline(agg):
    """
    Figure 1: Single-line headline figure.

    Shows only the "canaries" group: 22-25-year-olds in Q4 (top AI
    exposure) occupations. Percentage deviation from Oct 2022 = 0%.

    Annotated with the final-month value prominently displayed,
    matching the Brynjolfsson blog post style.
    """
    print("\n  Plotting headline figure...")

    grp = "Young (22–25), High AI"
    sub = agg[agg["group"] == grp].sort_values("date").copy()

    if sub.empty:
        print("    WARNING: no data for headline group — skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    # Main line
    ax.plot(sub["date"], sub["pct_change"],
            color=ORANGE, linewidth=2.5, zorder=5)

    # Fill below zero (shading the decline region)
    ax.fill_between(sub["date"], sub["pct_change"], 0,
                    where=(sub["pct_change"] < 0),
                    alpha=0.15, color=ORANGE, zorder=3)

    # Annotate the final value prominently
    last = sub.iloc[-1]
    pct_val = last["pct_change"]
    ax.annotate(
        f"{pct_val:+.1f}%",
        xy=(last["date"], pct_val),
        xytext=(15, -20),
        textcoords="offset points",
        fontsize=16, fontweight="bold", color=ORANGE,
        arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5),
        zorder=10,
    )

    # Event lines (0% is the reference)
    add_event_lines(ax, y_ref=0)

    # Labels for the two vertical event lines
    ylims = ax.get_ylim()
    mid_y = (ylims[0] + ylims[1]) / 2
    ax.text(pd.Timestamp(RIKSBANK_YM + "-01"), mid_y,
            " Riksbank\n rate hike", fontsize=7, color=ORANGE,
            alpha=0.8, va="center")
    ax.text(pd.Timestamp(CHATGPT_YM + "-01"), mid_y,
            " ChatGPT\n launch", fontsize=7, color=TEAL,
            alpha=0.8, va="center")

    ax.set_ylabel("Employment change from Oct 2022 (%)", fontsize=11)
    ax.set_title(
        "Monthly employment of 22–25-year-olds in top-quartile AI-exposed occupations\n"
        "(Swedish AGI register data, 3-month moving average)",
        fontsize=11, fontweight="bold",
    )
    ax.tick_params(axis="x", rotation=30)

    # Y-axis: percentage format
    from matplotlib.ticker import FormatStrFormatter
    ax.yaxis.set_major_formatter(FormatStrFormatter('%+.0f%%'))

    fig.tight_layout()
    out = OUTPUT_DIR / "figA_pctchange_headline.png"
    fig.savefig(out, dpi=300)
    plt.close()
    print(f"    Saved -> {out.name}")


def plot_comparison(agg):
    """
    Figure 2: Four-group comparison figure.

    Shows percentage change from Oct 2022 for all four groups:
      - Young (22-25), High AI (Q4)     [orange solid — the canaries]
      - Young (22-25), Low AI (Q1-Q3)   [orange dashed]
      - Older (26+), High AI (Q4)       [dark blue solid]
      - Older (26+), Low AI (Q1-Q3)     [dark blue dashed]

    This is the key analytical figure: the divergence between the
    orange solid line and the other three lines IS the canaries effect.
    """
    print("\n  Plotting comparison figure...")

    styles = {
        "Young (22–25), High AI":  {"color": ORANGE, "ls": "-", "lw": 2.5},
        "Young (22–25), Low AI":   {"color": ORANGE, "ls": "--", "lw": 1.5},
        "Older (26+), High AI":    {"color": DARK_BLUE, "ls": "-", "lw": 2.5},
        "Older (26+), Low AI":     {"color": DARK_BLUE, "ls": "--", "lw": 1.5},
    }

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for grp, style in styles.items():
        sub = agg[agg["group"] == grp].sort_values("date")
        if sub.empty:
            continue
        ax.plot(sub["date"], sub["pct_change"], label=grp, **style)

        # Annotate final value at the end of each line
        last = sub.iloc[-1]
        ax.annotate(
            f"{last['pct_change']:+.1f}%",
            xy=(last["date"], last["pct_change"]),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=8, fontweight="bold",
            color=style["color"], va="center",
        )

    # Event lines (0% reference)
    add_event_lines(ax, y_ref=0)

    ax.set_ylabel("Employment change from Oct 2022 (%)", fontsize=11)
    ax.set_title(
        "Monthly employment by age and AI exposure — percentage change\n"
        "(Swedish AGI register data, 3-month moving average)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.tick_params(axis="x", rotation=30)

    # Y-axis: percentage format
    from matplotlib.ticker import FormatStrFormatter
    ax.yaxis.set_major_formatter(FormatStrFormatter('%+.0f%%'))

    fig.tight_layout()
    out = OUTPUT_DIR / "figA_pctchange_comparison.png"
    fig.savefig(out, dpi=300)
    plt.close()
    print(f"    Saved -> {out.name}")


# ======================================================================
#   STEP 4: EXPORT DATA
# ======================================================================

def export_csv(agg):
    """
    Save the underlying monthly data so Magnus can re-plot locally
    without needing MONA access.

    Columns exported:
      year_month, group, n_emp, n_emp_ma, pct_change
    """
    out = OUTPUT_DIR / "pctchange_data.csv"
    export = agg[["year_month", "group", "n_emp", "n_emp_ma", "pct_change"]].copy()
    export = export.sort_values(["group", "year_month"])
    export.to_csv(out, index=False)
    print(f"\n  Saved data -> {out.name}")
    print(f"  ({len(export)} rows, {export['group'].nunique()} groups, "
          f"{export['year_month'].nunique()} months)")


# ======================================================================
#   MAIN
# ======================================================================

def main():
    print("=" * 70)
    print("SCRIPT 17: Brynjolfsson-style percentage change figure")
    print("=" * 70)

    # Step 1: Pull and prepare data
    raw = load_all_data()

    # Step 2: Aggregate and compute percentage change
    agg = compute_pctchange(raw)

    # Step 3: Figures
    plot_headline(agg)
    plot_comparison(agg)

    # Step 4: Export data for local re-plotting
    export_csv(agg)

    print("\n" + "=" * 70)
    print("DONE. Export these files from MONA:")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("  1. figA_pctchange_headline.png    — single-line Brynjolfsson style")
    print("  2. figA_pctchange_comparison.png  — four-group comparison")
    print("  3. pctchange_data.csv             — raw data for local re-plotting")
    print("=" * 70)


if __name__ == "__main__":
    main()
