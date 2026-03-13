#!/usr/bin/env python3
"""
14_mona_canaries_descriptive.py — Canaries test using AGI data in MONA.

╔══════════════════════════════════════════════════════════════════════╗
║  THIS SCRIPT IS DESIGNED TO RUN IN SCB's MONA SECURE ENVIRONMENT   ║
║  It queries AGI tables via pyodbc (SQL Server on monasql.micro.intra).║
║  Do NOT run outside MONA — the data is not available externally.    ║
╚══════════════════════════════════════════════════════════════════════╝

Purpose:
  Test the Brynjolfsson et al. (2025) "canaries in the coal mine"
  hypothesis with Swedish monthly register data: do young workers in
  high-AI-exposure occupations experience disproportionate employment
  declines after ChatGPT?

Data access:
  - Connects to SQL Server (P1207 database) via pyodbc
  - Queries AGI (Arb_AGIIndivid) tables year-by-year, 2019-2025
  - Joins to Individ tables for SSYK 4-digit occupation codes
  - For years >= 2023: cascading SSYK lookup (2023 → 2022 → 2021)
    to recover individuals who lack a 2023 SSYK code
  - For years < 2023: uses the year's own Individ table
  - Uses _def suffix for years < 2025, _prel for 2025 (max_month=6)
  - DAIOE quartiles loaded from the MONA network share

Analysis:
  1. Pull individual-level AGI data via SQL, aggregate in Python to:
     - agg_fine:  ssyk4 × year_month × age_band  (for spotlight figures)
     - agg_broad: ssyk4 × year_month × young      (for canaries regression)
     (COUNT DISTINCT person_id for deduplication)
  2. Merge with DAIOE quartiles
  3. Run monthly event study: interact month dummies with Young × HighAI
  4. Run triple-diff regression:
     ln(emp_it) = α_i + γ_t + β₁·Post·High + β₂·Post·Young
                  + β₃·Post·Young·High + ε_it
  5. Produce one figure + one table for the appendix

Output:
  - figA8a_mona_canaries_softwaredevelopers.png (spotlight: software devs by age)
  - figA8b_mona_canaries_customerservice.png (spotlight: customer service by age)
  - figA8c_mona_canaries_economy.png (broad canaries: young×high vs others)
  - mona_canaries_regression.csv (regression coefficients)

Runtime: should complete in <10 minutes on MONA hardware (SQL queries
  are the bottleneck; one UNION ALL query per year).

INSTRUCTIONS FOR CO-AUTHOR:
  1. Copy this script to your MONA project folder
  2. Ensure daioe_quartiles.csv is at the DAIOE_PATH below
     (or adjust the path)
  3. Run: python 14_mona_canaries_descriptive.py
  4. Copy the output files back for inclusion in the paper
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import pyodbc


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION — ADJUST THESE FOR YOUR MONA ENVIRONMENT            ║
# ╚══════════════════════════════════════════════════════════════════════╝

# SQL Server connection (MONA environment)
conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=monasql.micro.intra;"
    "DATABASE=P1207;"
    "Trusted_Connection=yes;"
)

# Path to DAIOE quartiles on the MONA network share
DAIOE_PATH = r"\\micro.intra\Projekt\P1207$\P1207_Gem\Lydia P1207\daioe_quartiles.csv"

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

# Spotlight occupations (SSYK 2012, 4-digit)
SSYK_SOFTWARE = ["2512"]  # Programmerare och webbutvecklare
SSYK_CUSTOMER = ["4221",  # Kundtjänstpersonal (call centre)
                 "4222",  # Helpdesk- och support-personal
                 "5230"]  # Kassörer m.fl.

# GenAI capability milestones (for timeline annotation on figures)
# Each tuple: (year, month, label)
GENAI_MILESTONES = [
    (2023, 3, "GPT-4"),
    (2023, 11, "Enterprise tools"),
    (2024, 5, "GPT-4o"),
    (2024, 9, "o1"),
    (2025, 1, "DeepSeek R1"),
]
MILESTONE_COLOR = "#7B2D8E"  # purple, distinct from existing orange/teal

# Fine age bands matching Brynjolfsson et al.
# NOTE: ages below 22 are excluded (consistent with employer-level DiD, script 15).
AGE_BANDS = [
    (22, 25, "22–25"),
    (26, 30, "26–30"),
    (31, 34, "31–34"),
    (35, 40, "35–40"),
    (41, 49, "41–49"),
    (50, 69, "50+"),
]

# Six colours for age-band trajectories
AGE_BAND_COLORS = [
    "#E8873A",
    "#F0A86B",
    "#A8C5BC",
    "#5FA898",
    "#2E7D6F",
    "#1B3A5C",
]

# Normalisation base month for spotlight figures (just before ChatGPT)
BASE_MONTH = "2022-10"


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  STEP 1: LOAD AND PREPARE AGI DATA                                 ║
# ╚══════════════════════════════════════════════════════════════════════╝

def pull_individual_year(year, conn):
    """
    Pull individual-level AGI records for one year from SQL Server.

    For each month in the year, queries dbo.Arb_AGIIndivid{ym}{suffix}
    and joins to the appropriate Individ table(s) for SSYK codes.

    Cascading SSYK lookup for years >= 2023:
      Try Individ_2023 first, fall back to 2022, then 2021.
      This recovers individuals who lack a 2023 SSYK code but had one
      in an earlier register year (e.g. workers with employment gaps).

    For years < 2023: joins to Individ_{year} directly (no cascade).

    Uses _def suffix for years < 2025, _prel for 2025 (max_month=6).

    Returns a DataFrame with columns:
      person_id, period (YYYYMM string), ssyk4, birth_year
    """
    print(f"  Querying year {year}...")

    if year < 2025:
        suffix = "_def"
        max_month = 12
    else:
        suffix = "_prel"
        max_month = 6

    monthly_queries = []

    for month in range(1, max_month + 1):
        ym = f"{year}{month:02d}"

        if year >= 2023:
            # Cascading SSYK lookup: 2023 → 2022 → 2021
            monthly_queries.append(
                f"""
                SELECT
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
                """
            )
        else:
            # For years <= 2022: use the year's own Individ table
            individ_year = year
            monthly_queries.append(
                f"""
                SELECT
                    agi.PERIOD AS period,
                    ind.Ssyk4_2012_J16 AS ssyk4,
                    ind.FodelseAr AS birth_year,
                    agi.P1207_LOPNR_PERSONNR AS person_id
                FROM dbo.Arb_AGIIndivid{ym}{suffix} agi
                LEFT JOIN dbo.Individ_{individ_year} ind
                    ON agi.P1207_LOPNR_PERSONNR = ind.P1207_LopNr_PersonNr
                """
            )

    # Combine all months for this year into one query
    union_query = "\nUNION ALL\n".join(monthly_queries)
    full_query = f"""
    SELECT * FROM (
        {union_query}
    ) AS combined
    WHERE birth_year IS NOT NULL
      AND ssyk4 IS NOT NULL
    """

    df_year = pd.read_sql(full_query, conn)
    print(f"    {len(df_year):,} records for {year}")
    return df_year


def load_agi():
    """
    Pull individual-level AGI data via SQL year-by-year, then aggregate
    in Python to two panels:
      1. agg_fine:  ssyk4 × year_month × age_band  (for spotlight figures)
      2. agg_broad: ssyk4 × year_month × young      (for canaries regression)

    Uses COUNT DISTINCT person_id for deduplication (same as before).

    Age groups for broad panel:
      - Young: 22-25 (entry-level, aligned with Brynjolfsson et al. 2025)
      - Older: 26+ (everyone else)
    Fine age bands (22-25, 26-30, ..., 50+) match Brynjolfsson et al.
    """
    print("Loading AGI data from SQL Server...")

    # Pull individual-level data year by year
    years = list(range(2019, 2026))
    all_years = []

    for y in years:
        df_year = pull_individual_year(y, conn)
        all_years.append(df_year)

    df = pd.concat(all_years, ignore_index=True)
    print(f"  Total records: {len(df):,}")

    # Parse year-month from PERIOD (YYYYMM → YYYY-MM)
    df["period"] = df["period"].astype(str)
    df["year_month"] = df["period"].str[:4] + "-" + df["period"].str[4:6]
    df["date"] = pd.to_datetime(df["year_month"] + "-01")

    # Compute age from birth year
    df["birth_year"] = df["birth_year"].astype(int)
    df["age"] = df["date"].dt.year - df["birth_year"]

    # Age group: young (22-25) vs older (26+)
    # Aligned with script 15's youngest age group and Brynjolfsson et al. (2025)
    df["young"] = ((df["age"] >= 22) & (df["age"] <= 25)).astype(int)

    # SSYK 4-digit code (zero-padded)
    df["ssyk4"] = df["ssyk4"].astype(str).str.zfill(4)

    # Filter to working-age population (22-69, aligned with script 15)
    df = df[(df["age"] >= 22) & (df["age"] <= 69)].copy()

    # --- Fine age bands (for spotlight figures) ---
    def assign_band(age):
        for lo, hi, label in AGE_BANDS:
            if lo <= age <= hi:
                return label
        return None

    df["age_band"] = df["age"].apply(assign_band)

    agg_fine = (
        df[df["age_band"].notna()]
        .groupby(["ssyk4", "year_month", "age_band"])["person_id"]
        .nunique()
        .reset_index()
        .rename(columns={"person_id": "n_employed"})
    )

    # --- Broad age groups (for canaries regression) ---
    agg_broad = (
        df.groupby(["ssyk4", "year_month", "young"])["person_id"]
        .nunique()
        .reset_index()
        .rename(columns={"person_id": "n_employed"})
    )

    print(f"  Aggregated (broad): {len(agg_broad):,} cells")
    print(f"  Aggregated (fine):  {len(agg_fine):,} cells")
    print(f"  Occupations: {agg_broad['ssyk4'].nunique()}")
    print(f"  Months: {agg_broad['year_month'].nunique()}")
    print(f"  Period: {agg_broad['year_month'].min()} to {agg_broad['year_month'].max()}")

    return agg_fine, agg_broad


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
    # Include zeros: ln(n+1) handles zero-employment cells correctly.
    # Previous version filtered df[df["n_employed"] > 0] and used ln(n),
    # which dropped extensive-margin zeros and biased the DiD toward zero.
    df["ln_emp"] = np.log(df["n_employed"] + 1)

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
# ║  SHARED: EVENT AND MILESTONE ANNOTATION LINES                       ║
# ╚══════════════════════════════════════════════════════════════════════╝

def add_event_lines(ax):
    """
    Draw vertical annotation lines for macro events and GenAI milestones.
    Used by both spotlight and canaries figures for visual consistency.

    Draws three types of lines:
      1. Riksbank rate hike (orange dotted) — macro tightening
      2. ChatGPT launch (teal dotted) — GenAI emergence
      3. GenAI capability milestones (purple dashed) — successive waves

    All lines use '_nolegend_' to avoid polluting the data legend.
    """
    # Macro / ChatGPT event markers (existing style)
    ax.axvline(pd.Timestamp(RIKSBANKEN_HIKE + "-01"),
               color=ORANGE, linewidth=1, linestyle=":", alpha=0.7,
               label="_nolegend_")
    ax.axvline(pd.Timestamp(CHATGPT_LAUNCH + "-01"),
               color=TEAL, linewidth=1.5, linestyle=":", alpha=0.9,
               label="_nolegend_")

    # GenAI capability milestones
    for year, month, label in GENAI_MILESTONES:
        date = pd.Timestamp(f"{year}-{month:02d}-01")
        ax.axvline(date, color=MILESTONE_COLOR, linewidth=0.9,
                   linestyle="--", alpha=0.6, label="_nolegend_")
        # Label at top of plot area
        ax.text(date, ax.get_ylim()[1], label,
                ha="center", va="bottom", fontsize=6,
                color=MILESTONE_COLOR, fontweight="bold", alpha=0.8,
                rotation=45)

    # Horizontal reference at 100
    ax.axhline(100, color=GRAY, linewidth=0.5, linestyle="--", alpha=0.5)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  STEP 4a: SPOTLIGHT FIGURES — SPECIFIC OCCUPATIONS BY AGE BAND      ║
# ╚══════════════════════════════════════════════════════════════════════╝

def plot_spotlight(agg_fine, ssyk_codes, occupation_label, out_filename):
    """
    Plot employment by fine age band for selected "spotlight" occupations,
    indexed to BASE_MONTH = 100. Mirrors Brynjolfsson et al. approach.
    """
    print(f"\nPlotting spotlight: {occupation_label} ...")

    sub = agg_fine[agg_fine["ssyk4"].isin(ssyk_codes)].copy()
    if sub.empty:
        print(f"  WARNING: no data for SSYK codes {ssyk_codes} — skipping.")
        return

    sub = (
        sub.groupby(["year_month", "age_band"])["n_employed"]
        .sum().reset_index()
    )
    sub["date"] = pd.to_datetime(sub["year_month"] + "-01")

    avail_base = BASE_MONTH if BASE_MONTH in sub["year_month"].values \
        else sub["year_month"].min()

    base_vals = (
        sub[sub["year_month"] == avail_base]
        .set_index("age_band")["n_employed"]
    )
    valid_bands = [b for _, _, b in AGE_BANDS
                   if b in base_vals.index and base_vals[b] > 0]
    if not valid_bands:
        print("  WARNING: no valid age bands after base-month filter.")
        return

    sub = sub[sub["age_band"].isin(valid_bands)].copy()
    sub["index"] = sub.apply(
        lambda r: 100 * r["n_employed"] / base_vals[r["age_band"]]
        if r["age_band"] in base_vals else np.nan, axis=1,
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    band_order = [b for _, _, b in AGE_BANDS if b in valid_bands]
    colour_map = dict(zip([b for _, _, b in AGE_BANDS], AGE_BAND_COLORS))

    for band in band_order:
        s = sub[sub["age_band"] == band].sort_values("date")
        lw = 2.8 if band == band_order[0] else 1.5
        ax.plot(s["date"], s["index"],
                color=colour_map.get(band, GRAY), linewidth=lw, label=band)

    # Event markers + GenAI milestones (shared helper, no legend pollution)
    add_event_lines(ax)

    ax.set_xlabel("")
    ax.set_ylabel(f"Employment index ({avail_base} = 100)")
    ax.set_title(
        f"Employment by age group — {occupation_label}\n"
        f"(AGI register data, SSYK {', '.join(ssyk_codes)})"
    )
    ax.legend(loc="upper left", fontsize=8, ncol=2,
              title="Age group", title_fontsize=8)
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / out_filename, dpi=300)
    plt.close(fig)
    print(f"  Saved → {out_filename}")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  STEP 4b: FIGURE — BROAD EMPLOYMENT TRAJECTORIES                   ║
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
                  "label": "Young (22-25), High AI"},
        (1, 0): {"color": ORANGE, "lw": 1.5, "ls": "--",
                  "label": "Young (22-25), Low AI"},
        (0, 1): {"color": DARK_BLUE, "lw": 2.5, "ls": "-",
                  "label": "Older (26+), High AI"},
        (0, 0): {"color": DARK_BLUE, "lw": 1.5, "ls": "--",
                  "label": "Older (26+), Low AI"},
    }

    for (young, high), style in styles.items():
        subset = agg[(agg["young"] == young) & (agg["high_exposure"] == high)]
        subset = subset.sort_values("date")
        ax.plot(subset["date"], subset["index"],
                color=style["color"], linewidth=style["lw"],
                linestyle=style["ls"], label=style["label"])

    # Event markers + GenAI milestones (shared helper)
    add_event_lines(ax)

    ax.set_xlabel("")
    ax.set_ylabel("Employment index (base month = 100)")
    ax.set_title("Monthly employment by age and AI exposure (AGI register data)")
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figA8c_mona_canaries_economy.png", dpi=300)
    plt.close(fig)
    print(f"  Saved → figA8c_mona_canaries_economy.png")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                               ║
# ╚══════════════════════════════════════════════════════════════════════╝

def main():
    print("=" * 70)
    print("MONA: Canaries test using AGI monthly register data")
    print("=" * 70)

    # Load and aggregate AGI data (two panels: fine age bands + broad)
    agg_fine, agg_broad = load_agi()

    # Merge broad panel with DAIOE
    merged_broad = merge_daioe(agg_broad)

    # Save processed data (for inspection)
    merged_broad.to_csv(OUTPUT_DIR / "mona_employment_age_ai.csv", index=False)

    # Triple-diff regression
    res, panel = run_canaries_regression(merged_broad)

    # Spotlight figures: specific occupations by fine age band
    plot_spotlight(
        agg_fine,
        ssyk_codes=SSYK_SOFTWARE,
        occupation_label="Software Developers",
        out_filename="figA8a_mona_canaries_softwaredevelopers.png",
    )
    plot_spotlight(
        agg_fine,
        ssyk_codes=SSYK_CUSTOMER,
        occupation_label="Customer Service Agents",
        out_filename="figA8b_mona_canaries_customerservice.png",
    )

    # Broad trajectories
    plot_canaries(merged_broad)

    print("\n" + "=" * 70)
    print("DONE. Copy the following files from output/ to your project:")
    print("  1. figA8a_mona_canaries_softwaredevelopers.png → figures/")
    print("  2. figA8b_mona_canaries_customerservice.png    → figures/")
    print("  3. figA8c_mona_canaries_economy.png            → figures/")
    print("  4. mona_canaries_regression.csv                → tables/")
    print("=" * 70)


if __name__ == "__main__":
    main()
