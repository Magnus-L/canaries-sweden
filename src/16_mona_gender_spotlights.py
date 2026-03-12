#!/usr/bin/env python3
"""
15_mona_gender_and_spotlights.py -- Gender heterogeneity + occupation spotlights.

======================================================================
  THIS SCRIPT RUNS IN SCB's MONA SECURE ENVIRONMENT.
  Uses monthly AGI register data via the same SQL connection as script 14.
  Do NOT run outside MONA -- the data is not available externally.
======================================================================

Purpose:
  Extend the main employer-level DiD (script 14) with:

  PART A — Gender heterogeneity
    A1. Descriptive figure: monthly employment by age x AI exposure x gender
        (3-month moving average), as a side-by-side panel (Men | Women).
    A2. Regression: the main Brynjolfsson-style DiD run separately for
        men and women. Same design as script 14 (employer x quartile +
        employer x month FE), just filtered by gender.

  PART B — Occupation spotlights
    B1. Descriptive figures: indexed employment trajectories by age group
        for four selected occupations (two teleworkable, two non-teleworkable).
    B2. Regression: within-occupation DiD comparing young vs old workers
        at the same employer. This uses a DIFFERENT design from script 14
        because within a single occupation all workers share the same AI
        exposure quartile -- so we use age as the treatment dimension.

  The occupation spotlight regression:
    ln(n_emp_{f,a,t} + 1) = alpha_{f,a} + beta_{f,t} + gamma*PostGPT_t*Young_a + e

    f = employer, a = age binary (young vs old), t = month.
    employer x age FE absorb that firms always have more old than young.
    employer x month FE absorb ALL firm-level shocks.
    gamma captures: within the same firm, did young workers in this
    specific occupation decline more than older workers after ChatGPT?

Selected spotlight occupations:
  High genAI + teleworkable:
    2512 Software developers (Mjukvaru- och systemutvecklare) — genAI p99
    4112 Payroll administrators (Lone- och personaladministratorer) — genAI p96
  High genAI + non-teleworkable:
    4222 Customer service (Kundtjanstpersonal) — genAI p83
    4225 Receptionists (Kontorsreceptionister) — genAI p76

Data requirements:
  Same AGI + Individ tables as script 14, plus:
  - ind.Kon (gender: 1=man, 2=kvinna in SCB coding)

Output files (export from MONA):
  A1: gender_canaries_panel.png         — descriptive figure
  A2: gender_did_results.csv            — DiD coefficients by gender x age
  B1: spotlight_{ssyk4}.png             — descriptive figure per occupation
  B2: spotlight_regression_results.csv  — within-occupation DiD results
  B2: spotlight_es_{ssyk4}.png          — event study per occupation (if N allows)
      spotlight_summary.txt             — diagnostics and sample sizes
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

# --- MONA connection (same as script 14) ---
import pyodbc
conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=monasql.micro.intra;"
    "DATABASE=P1207;"
    "Trusted_Connection=yes;"
)

# --- DAIOE quartile file (same as script 14) ---
DAIOE_PATH = r"\\micro.intra\Projekt\P1207$\P1207_Gem\Lydia P1207\daioe_quartiles.csv"

# --- Output directory ---
OUTPUT_DIR = Path("output_15")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Treatment dates ---
RIKSBANK_YM = "2022-04"
CHATGPT_YM = "2022-12"
BASE_MONTH = "2022-10"  # Index base for descriptive figures (just before ChatGPT)

# --- Reference period for event studies ---
REF_HALFYEAR = "2022H1"

# --- Age group definitions (same as script 14) ---
AGE_GROUPS = {
    "22-25": (22, 25),
    "26-30": (26, 30),
    "31-34": (31, 34),
    "35-40": (35, 40),
    "41-49": (41, 49),
    "50+":   (50, 69),
}

# --- Age binary for occupation spotlights ---
# Young: 22-30 (broader than main 22-25 for cell size)
# Old: 41+ (combining 41-49 and 50+)
# Sensitivity: also try 22-25 vs 50+ for direct comparability with main result
YOUNG_RANGE = (22, 30)
OLD_RANGE = (41, 69)

# --- Minimum employer size ---
MIN_EMPLOYER_SIZE = 5

# --- Spotlight occupations ---
SPOTLIGHT_OCCS = {
    "2512": "Software developers (teleworkable)",
    "4112": "Payroll administrators (teleworkable)",
    "4222": "Customer service (non-teleworkable)",
    "4225": "Receptionists (non-teleworkable)",
}

# --- Colours (same as paper) ---
ORANGE = "#E8873A"
TEAL = "#2E7D6F"
GRAY = "#8C8C8C"
DARK_BLUE = "#1B3A5C"
DARK_TEXT = "#2C2C2C"
LIGHT_GRAY = "#D0D0D0"


# ======================================================================
#   STEP 1: PULL DATA (same as script 14, but with gender added)
# ======================================================================

def pull_year(year, conn):
    """
    Pull one year of AGI data, aggregated to
    employer x year_month x ssyk4 x age_group x gender cells.

    Key change from script 14: adds ind.Kon AS gender to the query.
    Kon = 1 (man), 2 (kvinna) in SCB's coding.
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
        monthly_queries.append(f"""
            SELECT
                agi.P1207_LOPNR_PEORGNR AS employer_id,
                agi.PERIOD AS period,
                ind.Ssyk4_2012_J16 AS ssyk4,
                ind.FodelseAr AS birth_year,
                ind.Kon AS gender,
                agi.P1207_LOPNR_PERSONNR AS person_id
            FROM dbo.Arb_AGIIndivid{ym}{suffix} agi
            LEFT JOIN dbo.Individ_{individ_year} ind
                ON agi.P1207_LOPNR_PERSONNR = ind.P1207_LopNr_PersonNr
        """)

    union_query = "\nUNION ALL\n".join(monthly_queries)

    # Aggregate in SQL for efficiency: count distinct persons per cell
    # NOTE: gender is added to both SELECT and GROUP BY
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
            gender,
            CAST(LEFT(period,4) AS INT) - birth_year AS age
        FROM base
        WHERE birth_year IS NOT NULL
          AND gender IS NOT NULL
    )
    SELECT
        employer_id,
        LEFT(period,4) + '-' + SUBSTRING(period,5,2) AS year_month,
        ssyk4,
        gender,
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
        gender,
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
    Pull all years 2019-2025, merge DAIOE quartiles, apply employer size filter.

    Returns the raw panel at the employer x ssyk4 x age_group x gender x month
    level. This is the finest grain -- each analysis function re-aggregates
    from here as needed.
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
    # Count distinct persons per employer (across all months)
    emp_size = raw.groupby("employer_id")["n_emp"].sum()
    large_emp = emp_size[emp_size >= MIN_EMPLOYER_SIZE].index
    n_before = raw["employer_id"].nunique()
    raw = raw[raw["employer_id"].isin(large_emp)].copy()
    print(f"  Employers: {n_before:,} -> {raw['employer_id'].nunique():,} "
          f"(>={MIN_EMPLOYER_SIZE} total person-months)")

    # --- Gender labels ---
    raw["gender_label"] = raw["gender"].map({1: "Men", 2: "Women"})
    raw = raw[raw["gender_label"].notna()].copy()

    # --- Save raw panel for resuming later ---
    out = OUTPUT_DIR / "raw_panel.parquet"
    raw.to_parquet(out, index=False)
    print(f"  Saved raw panel -> {out.name} ({len(raw):,} rows)")

    # --- Summary ---
    print(f"\n  Period: {raw['year_month'].min()} to {raw['year_month'].max()}")
    print(f"  Months: {raw['year_month'].nunique()}")
    print(f"  Employers: {raw['employer_id'].nunique():,}")
    print(f"  Occupations: {raw['ssyk4'].nunique()}")
    print(f"  Gender split: {raw.groupby('gender_label')['n_emp'].sum().to_dict()}")

    return raw


# ======================================================================
#   BALANCED PANEL CONSTRUCTION (zero-fill)
# ======================================================================

def _balance_panel_for_age(agg, age_label, all_months):
    """
    Build balanced panel for one age group within a gender-filtered
    aggregation: every (employer, quartile) ever observed × all months.
    Missing cells filled with n_emp = 0.

    Applies identification restriction: employers with both Q4 and Q1-Q3.

    WHY: Without zero-filling, firms that shed ALL workers of an age group
    in a quartile disappear from the data. This drops the strongest
    treatment cases and biases the DiD toward zero.
    """
    sub = agg[agg["age_group"] == age_label].copy()

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

    # Cross join: employer-quartile pairs × all months
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
    print(f"      Balanced: {len(balanced):,} cells "
          f"({n_zeros:,} zeros = {pct_zero:.1f}%), "
          f"{len(valid_emps):,} employers")

    return balanced


def _balance_spotlight_panel(panel, all_months):
    """
    Build balanced panel for one occupation's spotlight regression:
    every (employer, age_binary) ever observed × all months.
    Missing cells filled with n_emp = 0.

    The identification restriction (employers with both young and old)
    is already applied before calling this function.
    """
    # Unique employer-age pairs
    emp_age = (
        panel.groupby(["employer_id", "age_binary"])
        .size()
        .reset_index()[["employer_id", "age_binary"]]
    )

    # Cross join × all months
    months_df = pd.DataFrame({"year_month": all_months})
    emp_age["_k"] = 1
    months_df["_k"] = 1
    balanced = emp_age.merge(months_df, on="_k").drop(columns="_k")

    # Left join actual counts
    balanced = balanced.merge(
        panel[["employer_id", "age_binary", "year_month", "n_emp"]],
        on=["employer_id", "age_binary", "year_month"],
        how="left",
    )
    balanced["n_emp"] = balanced["n_emp"].fillna(0).astype(int)

    n_zeros = (balanced["n_emp"] == 0).sum()
    pct_zero = 100 * n_zeros / len(balanced) if len(balanced) > 0 else 0
    print(f"    Balanced: {len(balanced):,} cells "
          f"({n_zeros:,} zeros = {pct_zero:.1f}%)")

    return balanced


# ======================================================================
#   PART A1: GENDER DESCRIPTIVE FIGURE
# ======================================================================

def plot_gender_canaries(raw):
    """
    Side-by-side panels showing the canaries pattern separately for men
    and women.

    Each panel has 4 lines:
      - Young (22-25), High AI (Q4)     [orange solid]
      - Young (22-25), Low AI (Q1-Q3)   [orange dashed]
      - Older (26+), High AI (Q4)       [dark blue solid]
      - Older (26+), Low AI (Q1-Q3)     [dark blue dashed]

    3-month moving average, indexed to 100 at BASE_MONTH.
    This directly shows whether the canaries effect differs by gender.
    """
    print("\n" + "=" * 70)
    print("PART A1: Gender descriptive figure")
    print("=" * 70)

    df = raw.copy()

    # Binary age: young (22-25) vs older (26+)
    df["age_binary"] = np.where(df["age_group"] == "22-25", "Young (22-25)", "Older (26+)")

    # Binary AI exposure: Q4 vs rest
    df["ai_binary"] = np.where(df["exposure_quartile"] == 4, "High AI", "Low AI")

    # Aggregate to gender x age_binary x ai_binary x month
    agg = (
        df.groupby(["gender_label", "age_binary", "ai_binary", "year_month"])
        ["n_emp"].sum()
        .reset_index()
    )
    agg["date"] = pd.to_datetime(agg["year_month"] + "-01")
    agg = agg.sort_values("date")

    # 3-month moving average
    agg["n_emp_ma"] = (
        agg.groupby(["gender_label", "age_binary", "ai_binary"])["n_emp"]
        .transform(lambda x: x.rolling(3, min_periods=1, center=True).mean())
    )

    # Index to 100 at base month
    base = agg[agg["year_month"] == BASE_MONTH].copy()
    base = base.rename(columns={"n_emp_ma": "base_val"})
    agg = agg.merge(
        base[["gender_label", "age_binary", "ai_binary", "base_val"]],
        on=["gender_label", "age_binary", "ai_binary"],
        how="left",
    )
    agg["index"] = 100 * agg["n_emp_ma"] / agg["base_val"]

    # Plot: 1x2 panel
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    line_styles = {
        ("Young (22-25)", "High AI"): {"color": ORANGE, "ls": "-", "lw": 2.2},
        ("Young (22-25)", "Low AI"):  {"color": ORANGE, "ls": "--", "lw": 1.5},
        ("Older (26+)", "High AI"):   {"color": DARK_BLUE, "ls": "-", "lw": 2.2},
        ("Older (26+)", "Low AI"):    {"color": DARK_BLUE, "ls": "--", "lw": 1.5},
    }

    for ax, gender in zip(axes, ["Men", "Women"]):
        sub = agg[agg["gender_label"] == gender]

        for (age_b, ai_b), style in line_styles.items():
            line = sub[(sub["age_binary"] == age_b) & (sub["ai_binary"] == ai_b)]
            label = f"{age_b}, {ai_b}"
            ax.plot(line["date"], line["index"], label=label, **style)

        # Reference lines
        ax.axhline(100, color=LIGHT_GRAY, ls="--", lw=0.8)
        ax.axvline(pd.Timestamp(RIKSBANK_YM + "-01"), color="salmon",
                   ls=":", lw=1, alpha=0.7)
        ax.axvline(pd.Timestamp(CHATGPT_YM + "-01"), color=TEAL,
                   ls=":", lw=1, alpha=0.7)

        ax.set_title(gender, fontsize=13, fontweight="bold")
        ax.set_ylabel("Employment index (base = 100)" if ax == axes[0] else "")
        ax.tick_params(axis="x", rotation=45)

    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(
        "Monthly employment by age and AI exposure, by gender (3-month MA)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out = OUTPUT_DIR / "gender_canaries_panel.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out.name}")


# ======================================================================
#   PART A2: GENDER DiD REGRESSION
# ======================================================================

def run_gender_did(raw):
    """
    Run the main Brynjolfsson-style DiD separately for men and women.

    Same design as script 14:
      ln(n_emp_{f,q,t} + 1) = a_{f,q} + b_{f,t}
                               + g1*PostRB_t*High_q
                               + g2*PostGPT_t*High_q + e

    Employer x quartile FE + employer x month FE.
    Run for each gender x age_group combination.

    Economic question: Is the canaries effect equally strong for men
    and women, or does it concentrate in one gender? This matters for
    policy targeting.
    """
    print("\n" + "=" * 70)
    print("PART A2: Gender DiD regression")
    print("=" * 70)

    # Aggregate to employer x quartile x age_group x gender x month
    agg = (
        raw.groupby([
            "employer_id", "exposure_quartile", "age_group",
            "gender_label", "year_month"
        ])["n_emp"].sum()
        .reset_index()
    )

    all_months = sorted(agg["year_month"].unique())
    all_results = []

    for gender in ["Men", "Women"]:
        agg_g = agg[agg["gender_label"] == gender].copy()
        print(f"\n  === {gender} ===")

        for age_label in AGE_GROUPS:
            # Build balanced panel with zero-filled cells
            sub = _balance_panel_for_age(agg_g, age_label, all_months)

            if len(sub) < 100:
                print(f"    {age_label}: too few obs ({len(sub)}), skipping")
                continue

            # Treatment variables
            sub["post_rb"] = (sub["year_month"] >= RIKSBANK_YM).astype(int)
            sub["post_gpt"] = (sub["year_month"] >= CHATGPT_YM).astype(int)
            sub["high"] = (sub["exposure_quartile"] == 4).astype(int)
            sub["post_rb_x_high"] = sub["post_rb"] * sub["high"]
            sub["post_gpt_x_high"] = sub["post_gpt"] * sub["high"]
            sub["ln_emp"] = np.log(sub["n_emp"] + 1)

            # FE identifiers
            sub["fe_emp_q"] = (
                sub["employer_id"].astype(str) + "_" +
                sub["exposure_quartile"].astype(str)
            )
            sub["fe_emp_t"] = (
                sub["employer_id"].astype(str) + "_" +
                sub["year_month"]
            )

            result = _run_panel_ols(sub, f"{gender}_{age_label}")
            if result is not None:
                result["gender"] = gender
                result["age_group"] = age_label
                all_results.append(result)
                print(f"    {age_label}: g2={result['gamma2_gpt_high']:+.4f} "
                      f"(SE={result['se2']:.4f}, p={result['pval2']:.4f}), "
                      f"n={result['n_obs']:,}")

    if all_results:
        results_df = pd.DataFrame(all_results)
        out = OUTPUT_DIR / "gender_did_results.csv"
        results_df.to_csv(out, index=False)
        print(f"\n  Saved -> {out.name}")

        # Print comparison table
        print("\n  === GENDER COMPARISON (gamma2 = PostGPT x High) ===")
        pivot = results_df.pivot(
            index="age_group", columns="gender",
            values=["gamma2_gpt_high", "pval2"],
        )
        print(pivot.to_string())
        return results_df

    return pd.DataFrame()


def _run_panel_ols(sub, label):
    """
    Estimate DiD with employer x quartile + employer x month FE.
    Uses linearmodels PanelOLS with absorbed effects.
    Falls back to within-transformation if linearmodels fails.
    """
    t0 = time.time()

    # --- Approach A: linearmodels ---
    try:
        from linearmodels.panel import PanelOLS

        panel = sub.copy()
        panel["date"] = pd.to_datetime(panel["year_month"] + "-01")
        panel = panel.set_index(["fe_emp_q", "date"])

        other_fe = pd.DataFrame(
            {"fe_emp_t": panel["fe_emp_t"]},
            index=panel.index,
        )

        mod = PanelOLS(
            dependent=panel["ln_emp"],
            exog=panel[["post_rb_x_high", "post_gpt_x_high"]],
            entity_effects=True,
            time_effects=False,
            other_effects=other_fe,
        )
        res = mod.fit(cov_type="clustered", cluster_entity=True)

        return {
            "label": label,
            "method": "PanelOLS",
            "n_obs": int(res.nobs),
            "gamma1_rb_high": res.params["post_rb_x_high"],
            "se1": res.std_errors["post_rb_x_high"],
            "pval1": res.pvalues["post_rb_x_high"],
            "gamma2_gpt_high": res.params["post_gpt_x_high"],
            "se2": res.std_errors["post_gpt_x_high"],
            "pval2": res.pvalues["post_gpt_x_high"],
        }

    except Exception as e:
        print(f"      linearmodels failed ({e}), trying within-transformation")

    # --- Approach B: within-transformation ---
    try:
        import statsmodels.api as sm

        panel = sub.copy()
        for col in ["ln_emp", "post_rb_x_high", "post_gpt_x_high"]:
            panel[f"{col}_dm1"] = panel.groupby("fe_emp_q")[col].transform(
                lambda x: x - x.mean()
            )
        for col in ["ln_emp", "post_rb_x_high", "post_gpt_x_high"]:
            panel[f"{col}_dm"] = panel.groupby("fe_emp_t")[f"{col}_dm1"].transform(
                lambda x: x - x.mean()
            )

        y = panel["ln_emp_dm"].values
        X = panel[["post_rb_x_high_dm", "post_gpt_x_high_dm"]].values

        mod = sm.OLS(y, X)
        res = mod.fit(
            cov_type="cluster",
            cov_kwds={"groups": panel["employer_id"].values},
        )

        return {
            "label": label,
            "method": "within-transformation",
            "n_obs": len(panel),
            "gamma1_rb_high": res.params[0],
            "se1": res.bse[0],
            "pval1": res.pvalues[0],
            "gamma2_gpt_high": res.params[1],
            "se2": res.bse[1],
            "pval2": res.pvalues[1],
        }

    except Exception as e:
        print(f"      within-transformation also failed: {e}")
        return None


# ======================================================================
#   PART B1: OCCUPATION SPOTLIGHT DESCRIPTIVE FIGURES
# ======================================================================

def plot_spotlight_descriptives(raw):
    """
    For each spotlight occupation, plot indexed employment trajectories
    by age group (3-month MA). This shows whether the canaries pattern
    holds within specific occupations.

    Produces one figure per occupation: lines for each age group,
    vertical markers for Riksbank and ChatGPT.

    Economic intuition: The aggregate regression shows Q4 vs Q1-Q3.
    These spotlights make the finding concrete — "here is what happened
    to young software developers at Swedish firms."
    """
    print("\n" + "=" * 70)
    print("PART B1: Occupation spotlight descriptive figures")
    print("=" * 70)

    # Age group colours (gradient from orange=young to blue=old)
    age_colors = {
        "22-25": ORANGE,
        "26-30": "#D4A05A",
        "31-34": GRAY,
        "35-40": "#6B9DAA",
        "41-49": TEAL,
        "50+": DARK_BLUE,
    }

    for ssyk4, occ_label in SPOTLIGHT_OCCS.items():
        print(f"\n  {ssyk4} — {occ_label}")
        sub = raw[raw["ssyk4"] == ssyk4].copy()

        if sub.empty:
            print(f"    No data for SSYK {ssyk4}, skipping")
            continue

        # Aggregate to age_group x month (across employers and gender)
        agg = (
            sub.groupby(["age_group", "year_month"])["n_emp"].sum()
            .reset_index()
        )
        agg["date"] = pd.to_datetime(agg["year_month"] + "-01")
        agg = agg.sort_values("date")

        # 3-month moving average
        agg["n_emp_ma"] = (
            agg.groupby("age_group")["n_emp"]
            .transform(lambda x: x.rolling(3, min_periods=1, center=True).mean())
        )

        # Index to 100 at base month
        base = agg[agg["year_month"] == BASE_MONTH].copy()
        base = base.rename(columns={"n_emp_ma": "base_val"})
        agg = agg.merge(
            base[["age_group", "base_val"]], on="age_group", how="left",
        )
        agg["index"] = 100 * agg["n_emp_ma"] / agg["base_val"]

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5.5))

        for age_label in AGE_GROUPS:
            line = agg[agg["age_group"] == age_label]
            if line.empty:
                continue
            lw = 2.5 if age_label in ("22-25", "50+") else 1.2
            ax.plot(line["date"], line["index"],
                    label=age_label, color=age_colors.get(age_label, GRAY),
                    linewidth=lw)

        ax.axhline(100, color=LIGHT_GRAY, ls="--", lw=0.8)
        ax.axvline(pd.Timestamp(RIKSBANK_YM + "-01"), color="salmon",
                   ls=":", lw=1, alpha=0.7, label="Riksbank rate hike")
        ax.axvline(pd.Timestamp(CHATGPT_YM + "-01"), color=TEAL,
                   ls=":", lw=1, alpha=0.7, label="ChatGPT launch")

        ax.set_title(f"{ssyk4} — {occ_label}\n"
                     "Monthly employment by age group (3-month MA)",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("Employment index (base = 100)")
        ax.legend(fontsize=8, ncol=2)
        ax.tick_params(axis="x", rotation=45)

        # Print cell sizes for diagnostics
        n_total = sub["n_emp"].sum()
        n_emp_unique = sub["employer_id"].nunique()
        ax.text(0.02, 0.02,
                f"N = {n_total:,} person-months, {n_emp_unique:,} employers",
                transform=ax.transAxes, fontsize=7, color=GRAY)

        fig.tight_layout()
        out = OUTPUT_DIR / f"spotlight_{ssyk4}.png"
        fig.savefig(out, dpi=300)
        plt.close()
        print(f"    Saved -> {out.name}")
        print(f"    Total person-months: {n_total:,}, employers: {n_emp_unique:,}")

        # Also save per-age-group counts for diagnostics
        age_counts = sub.groupby("age_group")["n_emp"].sum()
        for ag, n in age_counts.items():
            print(f"      {ag}: {n:,}")


# ======================================================================
#   PART B2: OCCUPATION SPOTLIGHT REGRESSIONS
# ======================================================================

def run_spotlight_regressions(raw):
    """
    Within-occupation DiD: does the canaries pattern hold at the
    occupation level?

    Design for each occupation:
      ln(n_emp_{f,a,t} + 1) = alpha_{f,a} + beta_{f,t}
                               + gamma*PostGPT_t*Young_a + e

      f = employer, a = age binary (young/old), t = month
      employer x age FE + employer x month FE
      gamma = within-firm change in young vs old after ChatGPT

    NOTE: This is a different identification strategy than the main
    regression (which uses AI exposure variation). Here we hold
    the occupation fixed and use age as the treatment dimension.
    Requires that employers have BOTH young and old workers in
    this specific occupation — so cell sizes may be small.

    If cell sizes are insufficient (< 500 employer-age-month cells
    or < 50 employers), the regression is skipped and only the
    descriptive figure (Part B1) is available.
    """
    print("\n" + "=" * 70)
    print("PART B2: Occupation spotlight regressions")
    print("=" * 70)

    all_results = []

    for ssyk4, occ_label in SPOTLIGHT_OCCS.items():
        print(f"\n  {ssyk4} — {occ_label}")
        sub = raw[raw["ssyk4"] == ssyk4].copy()

        if sub.empty:
            print(f"    No data, skipping")
            continue

        # Create age binary: young (22-30) vs old (41+)
        # Aggregate 22-25 + 26-30 into young; 41-49 + 50+ into old
        sub["age_binary"] = None
        sub.loc[sub["age_group"].isin(["22-25", "26-30"]), "age_binary"] = "Young"
        sub.loc[sub["age_group"].isin(["41-49", "50+"]), "age_binary"] = "Old"
        sub = sub[sub["age_binary"].notna()].copy()

        # Aggregate to employer x age_binary x month
        # (collapse across gender -- we already tested gender in Part A)
        panel = (
            sub.groupby(["employer_id", "age_binary", "year_month"])
            ["n_emp"].sum()
            .reset_index()
        )

        # Filter to employers that have BOTH young and old in at least some months
        # This is necessary for the FE to be identified
        emp_age_coverage = panel.groupby("employer_id")["age_binary"].nunique()
        both_ages = emp_age_coverage[emp_age_coverage == 2].index
        panel = panel[panel["employer_id"].isin(both_ages)].copy()

        # Build balanced panel with zero-filled cells
        all_months = sorted(raw["year_month"].unique())
        panel = _balance_spotlight_panel(panel, all_months)

        n_cells = len(panel)
        n_employers = panel["employer_id"].nunique()
        print(f"    Panel: {n_cells:,} cells, {n_employers:,} employers "
              f"(with both young and old, balanced)")

        # Check minimum requirements
        if n_cells < 500 or n_employers < 50:
            print(f"    WARNING: Too few observations for reliable regression.")
            print(f"    Skipping regression. Use descriptive figure (B1) instead.")
            all_results.append({
                "ssyk4": ssyk4,
                "occupation": occ_label,
                "method": "SKIPPED (small N)",
                "n_cells": n_cells,
                "n_employers": n_employers,
                "gamma_gpt_young": np.nan,
                "se": np.nan,
                "pval": np.nan,
            })
            continue

        # Treatment variables
        panel["post_gpt"] = (panel["year_month"] >= CHATGPT_YM).astype(int)
        panel["young"] = (panel["age_binary"] == "Young").astype(int)
        panel["post_gpt_x_young"] = panel["post_gpt"] * panel["young"]
        panel["ln_emp"] = np.log(panel["n_emp"] + 1)

        # Also add PostRB x Young for completeness
        panel["post_rb"] = (panel["year_month"] >= RIKSBANK_YM).astype(int)
        panel["post_rb_x_young"] = panel["post_rb"] * panel["young"]

        # FE identifiers
        panel["fe_emp_age"] = (
            panel["employer_id"].astype(str) + "_" + panel["age_binary"]
        )
        panel["fe_emp_t"] = (
            panel["employer_id"].astype(str) + "_" + panel["year_month"]
        )

        # --- Run PanelOLS ---
        result = _run_spotlight_ols(panel, ssyk4, occ_label)
        if result is not None:
            all_results.append(result)

        # --- Event study (if enough data) ---
        if n_employers >= 100:
            _run_spotlight_event_study(panel, ssyk4, occ_label)
        else:
            print(f"    Skipping event study (need >= 100 employers, have {n_employers})")

    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        out = OUTPUT_DIR / "spotlight_regression_results.csv"
        results_df.to_csv(out, index=False)
        print(f"\n  Saved -> {out.name}")
        print("\n  === SPOTLIGHT REGRESSION SUMMARY ===")
        print(results_df[["ssyk4", "occupation", "gamma_gpt_young", "se", "pval",
                          "n_employers"]].to_string(index=False))
        return results_df

    return pd.DataFrame()


def _run_spotlight_ols(panel, ssyk4, occ_label):
    """
    Run the within-occupation DiD for one SSYK code.
    employer x age FE + employer x month FE.
    """
    t0 = time.time()

    try:
        from linearmodels.panel import PanelOLS

        p = panel.copy()
        p["date"] = pd.to_datetime(p["year_month"] + "-01")
        p = p.set_index(["fe_emp_age", "date"])

        other_fe = pd.DataFrame(
            {"fe_emp_t": p["fe_emp_t"]},
            index=p.index,
        )

        mod = PanelOLS(
            dependent=p["ln_emp"],
            exog=p[["post_rb_x_young", "post_gpt_x_young"]],
            entity_effects=True,
            time_effects=False,
            other_effects=other_fe,
        )
        res = mod.fit(cov_type="clustered", cluster_entity=True)

        elapsed = time.time() - t0
        gamma_rb = res.params["post_rb_x_young"]
        gamma_gpt = res.params["post_gpt_x_young"]

        print(f"    [PanelOLS, {elapsed:.0f}s]")
        print(f"    g_rb  (PostRB x Young)  = {gamma_rb:+.4f} "
              f"(SE={res.std_errors['post_rb_x_young']:.4f}, "
              f"p={res.pvalues['post_rb_x_young']:.4f})")
        print(f"    g_gpt (PostGPT x Young) = {gamma_gpt:+.4f} "
              f"(SE={res.std_errors['post_gpt_x_young']:.4f}, "
              f"p={res.pvalues['post_gpt_x_young']:.4f})")

        return {
            "ssyk4": ssyk4,
            "occupation": occ_label,
            "method": "PanelOLS",
            "n_cells": int(res.nobs),
            "n_employers": panel["employer_id"].nunique(),
            "gamma_rb_young": gamma_rb,
            "se_rb": res.std_errors["post_rb_x_young"],
            "pval_rb": res.pvalues["post_rb_x_young"],
            "gamma_gpt_young": gamma_gpt,
            "se": res.std_errors["post_gpt_x_young"],
            "pval": res.pvalues["post_gpt_x_young"],
        }

    except Exception as e:
        print(f"    PanelOLS failed: {e}")

    # --- Fallback: within-transformation ---
    try:
        import statsmodels.api as sm

        p = panel.copy()
        for col in ["ln_emp", "post_rb_x_young", "post_gpt_x_young"]:
            p[f"{col}_dm1"] = p.groupby("fe_emp_age")[col].transform(
                lambda x: x - x.mean()
            )
        for col in ["ln_emp", "post_rb_x_young", "post_gpt_x_young"]:
            p[f"{col}_dm"] = p.groupby("fe_emp_t")[f"{col}_dm1"].transform(
                lambda x: x - x.mean()
            )

        y = p["ln_emp_dm"].values
        X = p[["post_rb_x_young_dm", "post_gpt_x_young_dm"]].values

        mod = sm.OLS(y, X)
        res = mod.fit(
            cov_type="cluster",
            cov_kwds={"groups": p["employer_id"].values},
        )

        print(f"    [within-transformation]")
        print(f"    g_gpt (PostGPT x Young) = {res.params[1]:+.4f} "
              f"(SE={res.bse[1]:.4f}, p={res.pvalues[1]:.4f})")

        return {
            "ssyk4": ssyk4,
            "occupation": occ_label,
            "method": "within-transformation",
            "n_cells": len(p),
            "n_employers": panel["employer_id"].nunique(),
            "gamma_rb_young": res.params[0],
            "se_rb": res.bse[0],
            "pval_rb": res.pvalues[0],
            "gamma_gpt_young": res.params[1],
            "se": res.bse[1],
            "pval": res.pvalues[1],
        }

    except Exception as e:
        print(f"    Within-transformation also failed: {e}")
        return {
            "ssyk4": ssyk4,
            "occupation": occ_label,
            "method": "FAILED",
            "n_cells": len(panel),
            "n_employers": panel["employer_id"].nunique(),
            "gamma_gpt_young": np.nan,
            "se": np.nan,
            "pval": np.nan,
        }


def _run_spotlight_event_study(panel, ssyk4, occ_label):
    """
    Half-year event study for one occupation: interact half-year
    dummies with Young indicator. Reference: 2022H1.

    This shows (a) whether pre-trends are flat (key for credibility)
    and (b) the time path of the young-old divergence.
    """
    print(f"    Running event study for {ssyk4}...")

    p = panel.copy()
    year = p["year_month"].str[:4]
    month = p["year_month"].str[5:7].astype(int)
    p["halfyear"] = year + np.where(month <= 6, "H1", "H2")

    all_periods = sorted(p["halfyear"].unique())
    event_periods = [pp for pp in all_periods if pp != REF_HALFYEAR]

    # Create interaction dummies
    for pp in event_periods:
        p[f"hy_{pp}"] = ((p["halfyear"] == pp).astype(int) * p["young"])
    interaction_cols = [f"hy_{pp}" for pp in event_periods]

    try:
        from linearmodels.panel import PanelOLS

        p2 = p.copy()
        p2["date"] = pd.to_datetime(p2["year_month"] + "-01")
        p2 = p2.set_index(["fe_emp_age", "date"])

        mod = PanelOLS(
            dependent=p2["ln_emp"],
            exog=p2[interaction_cols],
            entity_effects=True,
            time_effects=True,
        )
        res = mod.fit(cov_type="clustered", cluster_entity=True)

        # Collect coefficients
        es_data = []
        for pp in event_periods:
            col = f"hy_{pp}"
            es_data.append({
                "period": pp,
                "coef": res.params[col],
                "se": res.std_errors[col],
            })
        # Add reference period
        es_data.append({"period": REF_HALFYEAR, "coef": 0.0, "se": 0.0})

        es_df = pd.DataFrame(es_data).sort_values("period")
        es_df["ci_lo"] = es_df["coef"] - 1.96 * es_df["se"]
        es_df["ci_hi"] = es_df["coef"] + 1.96 * es_df["se"]

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(len(es_df))
        ax.fill_between(x, es_df["ci_lo"], es_df["ci_hi"],
                        alpha=0.15, color=ORANGE)
        ax.plot(x, es_df["coef"], "o-", color=ORANGE, lw=2, ms=6)
        ax.axhline(0, color=DARK_TEXT, lw=0.8, alpha=0.5)

        # Mark reference and ChatGPT
        ref_idx = list(es_df["period"]).index(REF_HALFYEAR)
        ax.axvline(ref_idx, color=TEAL, ls="--", lw=1, alpha=0.7)
        if "2022H2" in list(es_df["period"]):
            gpt_idx = list(es_df["period"]).index("2022H2")
            ax.axvline(gpt_idx, color=GRAY, ls=":", lw=1, alpha=0.7)

        ax.set_xticks(list(x))
        ax.set_xticklabels(es_df["period"], rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Coefficient (Young x period)")
        ax.set_title(f"Event study: {ssyk4} — {occ_label}\n"
                     "(Young vs Old, employer x age + employer x month FE)")

        fig.tight_layout()
        out = OUTPUT_DIR / f"spotlight_es_{ssyk4}.png"
        fig.savefig(out, dpi=300)
        plt.close()
        print(f"    Saved -> {out.name}")

        # Save CSV
        es_df.to_csv(OUTPUT_DIR / f"spotlight_es_{ssyk4}.csv", index=False)

    except Exception as e:
        print(f"    Event study failed: {e}")


# ======================================================================
#   SUMMARY
# ======================================================================

def write_summary(raw, gender_results, spotlight_results):
    """Write a comprehensive diagnostics file."""
    out = OUTPUT_DIR / "spotlight_summary.txt"
    with open(out, "w") as f:
        f.write("SCRIPT 15 — GENDER HETEROGENEITY + OCCUPATION SPOTLIGHTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Data period: {raw['year_month'].min()} to {raw['year_month'].max()}\n")
        f.write(f"Total rows: {len(raw):,}\n")
        f.write(f"Employers: {raw['employer_id'].nunique():,}\n")
        f.write(f"Occupations: {raw['ssyk4'].nunique()}\n\n")

        f.write("--- Gender distribution ---\n")
        for g, n in raw.groupby("gender_label")["n_emp"].sum().items():
            f.write(f"  {g}: {n:,}\n")

        if not gender_results.empty:
            f.write("\n--- Gender DiD results ---\n")
            f.write(gender_results.to_string(index=False))
            f.write("\n")

        f.write("\n--- Spotlight occupations ---\n")
        for ssyk4, label in SPOTLIGHT_OCCS.items():
            sub = raw[raw["ssyk4"] == ssyk4]
            f.write(f"  {ssyk4} ({label}): {sub['n_emp'].sum():,} person-months, "
                    f"{sub['employer_id'].nunique():,} employers\n")

        if not spotlight_results.empty:
            f.write("\n--- Spotlight regression results ---\n")
            f.write(spotlight_results.to_string(index=False))
            f.write("\n")

        f.write("\n--- FE structure ---\n")
        f.write("Gender DiD: employer x quartile + employer x month (same as script 15)\n")
        f.write("Spotlights: employer x age_binary + employer x month\n")
        f.write("SEs clustered by employer (gender) / employer-age entity (spotlights)\n")
        f.write("Panel: balanced (zero-filled), restricted to employers with Q4 and Q1-Q3\n")
        f.write("Spotlight panel: balanced (zero-filled), restricted to employers with both ages\n")

    print(f"\n  Saved summary -> {out.name}")


# ======================================================================
#   MAIN
# ======================================================================

def main():
    print("=" * 70)
    print("SCRIPT 15: Gender heterogeneity + occupation spotlights")
    print("=" * 70)

    # Step 1: Pull all data
    raw = load_all_data()

    # Part A1: Gender descriptive figure
    plot_gender_canaries(raw)

    # Part A2: Gender DiD regression
    gender_results = run_gender_did(raw)

    # Part B1: Occupation spotlight descriptives
    plot_spotlight_descriptives(raw)

    # Part B2: Occupation spotlight regressions
    spotlight_results = run_spotlight_regressions(raw)

    # Summary
    write_summary(raw, gender_results, spotlight_results)

    print("\n" + "=" * 70)
    print("DONE. Export these files from MONA:")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("  A1: gender_canaries_panel.png")
    print("  A2: gender_did_results.csv")
    print("  B1: spotlight_2512.png, spotlight_4112.png, ...")
    print("  B2: spotlight_regression_results.csv")
    print("  B2: spotlight_es_2512.png, spotlight_es_2512.csv, ...")
    print("      spotlight_summary.txt")
    print("=" * 70)


if __name__ == "__main__":
    main()
