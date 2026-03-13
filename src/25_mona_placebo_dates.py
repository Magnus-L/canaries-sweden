#!/usr/bin/env python3
"""
25_mona_placebo_dates.py -- Placebo-date event study for canaries paper.

======================================================================
  THIS SCRIPT IS DESIGNED TO RUN IN SCB's MONA SECURE ENVIRONMENT
  Do NOT run outside MONA -- the data is not available externally.
======================================================================

PURPOSE:
  Re-estimate the event study from script 18 with FAKE treatment dates
  as a placebo test. If the post-ChatGPT acceleration in employment
  composition shifts is real (not a pre-existing trend), then moving
  the treatment date earlier should produce NO significant acceleration.

  Two placebo dates are tested:
    1. November 2021 -- one full year before the real ChatGPT launch.
       Reference period: 2021H1. Post-treatment: 2021H2 onward.
       Logic: if pre-existing trends drive the result, we should see
       acceleration here too. If not, the 2021H2-2022H1 window should
       show flat coefficients.

    2. July 2022 -- between the Riksbank rate hike (April 2022) and
       ChatGPT launch (November 2022). Reference period: 2022H1.
       Post-treatment: 2022H2 onward.
       Logic: this tests whether the Riksbank hike (a macro shock)
       triggers the age-composition pattern. If the pattern is truly
       AI-driven, it should NOT appear 5 months before ChatGPT.

INTERPRETATION:
  - If placebo event studies show no acceleration at the fake treatment
    date, this confirms that the post-2024 pattern is not a pre-existing
    trend continuing through the real treatment.
  - If a pattern DOES appear at the placebo date, it would suggest a
    confounding trend or that the Riksbank hike drives composition shifts
    (undermining the AI interpretation).

SPECIFICATION:
  Identical to script 18 (corrected FE):
    Entity FE:  employer x quartile  (PanelOLS entity_effects)
    Other FE:   employer x month     (PanelOLS other_effects)
    Interaction: halfyear x high (Q4 vs Q1-Q3)
    SEs:        clustered by entity (employer x quartile)

ESTIMATED RUNTIME:
  ~60-120 min total (2 placebo dates x 6 age groups each).

OUTPUT FILES (in output_25/):
  - placebo_nov2021_es_all.csv       -- event study coefficients (Nov 2021 placebo)
  - placebo_jul2022_es_all.csv       -- event study coefficients (Jul 2022 placebo)
  - placebo_nov2021_es_*.png         -- figures per age group (Nov 2021 placebo)
  - placebo_jul2022_es_*.png         -- figures per age group (Jul 2022 placebo)
  - placebo_pretrends.txt            -- pre-trend test results for both placebos
  - placebo_summary.txt              -- comparison summary
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

# --- MONA SQL connection (same as scripts 14-18) ---
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
OUTPUT_DIR = Path("output_25")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Age group definitions (same as script 18 / Lydia's runs) ---
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

# --- Colours ---
ORANGE = "#E8873A"
TEAL = "#2E7D6F"
GRAY = "#8C8C8C"

# --- Placebo configurations ---
# Each entry: (label, placebo_halfyear, reference_halfyear, description)
PLACEBO_CONFIGS = [
    (
        "nov2021",
        "2021H2",      # fake treatment starts here
        "2021H1",      # reference period = half-year before fake treatment
        "Placebo: Nov 2021 (1 year before ChatGPT)",
    ),
    (
        "jul2022",
        "2022H2",      # fake treatment starts here
        "2022H1",      # reference period = same as real spec, but pre-ChatGPT
        "Placebo: Jul 2022 (between Riksbank hike and ChatGPT)",
    ),
]


# ======================================================================
#   STEP 1: LOAD DATA (same pipeline as script 18)
# ======================================================================

def pull_year(year, conn):
    """
    Pull one year of AGI data with cascading SSYK lookup (2023 -> 2022 -> 2021).
    Same approach as scripts 14 and 15 for data consistency.
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
    Load AGI data year by year (same pipeline as scripts 14-15),
    merge DAIOE quartiles, assign age groups,
    aggregate to employer x quartile x age_group x month cells.
    """
    print("=" * 70)
    print("STEP 1: Loading and preparing data")
    print("=" * 70)

    # Pull year by year (consistent with scripts 14 and 15)
    frames = []
    t0 = time.time()
    for year in range(2019, 2026):
        frames.append(pull_year(year, conn))
    df = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {len(df):,} records in {time.time()-t0:.0f}s")

    # Drop rows without SSYK or age group
    df = df[df["ssyk4"].notna() & (df["ssyk4"] != "None")].copy()
    df = df[df["age_group"].notna()].copy()

    # SSYK4 as string
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

    # Filter small employers
    emp_size = df.groupby("employer_id")["person_count"].sum()
    large_emp = emp_size[emp_size >= MIN_EMPLOYER_SIZE].index
    df = df[df["employer_id"].isin(large_emp)].copy()
    print(f"  Employers with >={MIN_EMPLOYER_SIZE} workers: "
          f"{df['employer_id'].nunique():,}")

    # Aggregate to employer x quartile x age_group x month
    # (already aggregated in SQL, but re-aggregate to collapse across ssyk4)
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


# ======================================================================
#   STEP 1b: BALANCED PANEL (ZERO-FILL)
# ======================================================================

def balance_panel_for_age(agg, age_label):
    """
    Build balanced panel for one age group: every (employer, quartile)
    combination the employer is ever observed in x all months.
    Missing cells filled with n_emp = 0.

    Applies identification restriction: employers with both Q4 and Q1-Q3.

    WHY: Without zero-filling, firms that shed ALL workers of an age group
    in a quartile disappear from the data. This drops the strongest
    treatment cases and biases the DiD toward zero.
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
#   STEP 2: PLACEBO EVENT STUDY
# ======================================================================

def assign_halfyear(ym):
    """Map year-month string to half-year label, e.g. '2022-03' -> '2022H1'."""
    year = ym[:4]
    month = int(ym[5:7])
    half = "H1" if month <= 6 else "H2"
    return f"{year}{half}"


def pretrend_joint_test(res, event_periods, ref_period):
    """Joint chi2 test: H0 = all pre-treatment event-study coefficients are zero."""
    from scipy import stats as sp_stats

    pre_cols = [f"hy_{p}" for p in event_periods if p < ref_period]
    if not pre_cols:
        return None

    beta = res.params[pre_cols].values
    try:
        V = res.cov.loc[pre_cols, pre_cols].values        # linearmodels
    except AttributeError:
        V = res.cov_params().loc[pre_cols, pre_cols].values  # statsmodels

    k = len(pre_cols)
    wald = float(beta @ np.linalg.solve(V, beta))
    p = 1 - sp_stats.chi2.cdf(wald, k)
    return {"chi2": round(wald, 2), "df": k, "p_value": round(p, 4)}


def run_placebo_event_study(agg, placebo_halfyear, ref_halfyear, description):
    """
    Half-year event study with a FAKE treatment date (placebo test).

    Uses the SAME FE specification as script 18 (corrected):
      entity FE = employer x quartile  (PanelOLS entity_effects)
      other FE  = employer x month     (PanelOLS other_effects)

    The only difference from script 18 is the reference period and the
    interpretive framing: we expect NO acceleration at the placebo date.

    Parameters
    ----------
    agg : pd.DataFrame
        Aggregated panel from load_and_prepare().
    placebo_halfyear : str
        The half-year where the fake treatment "starts" (e.g. "2021H2").
        This is the first post-treatment period in the placebo design.
    ref_halfyear : str
        The reference period (omitted dummy), e.g. "2021H1".
    description : str
        Human-readable label for console output.

    Returns
    -------
    es_df : pd.DataFrame
        Event study coefficients for all age groups.
    pretrend_tests : list of dict
        Pre-trend test results per age group.
    """
    from linearmodels.panel import PanelOLS

    print("\n" + "=" * 70)
    print(f"PLACEBO EVENT STUDY: {description}")
    print(f"  Fake treatment start: {placebo_halfyear}")
    print(f"  Reference period:     {ref_halfyear}")
    print(f"  FE: employer x quartile + employer x month")
    print("=" * 70)

    agg = agg.copy()

    # Compute half-year labels
    agg["halfyear"] = agg["year_month"].apply(assign_halfyear)
    all_periods = sorted(agg["halfyear"].unique())
    event_periods = [p for p in all_periods if p != ref_halfyear]

    print(f"  Half-year periods: {all_periods}")
    print(f"  Reference period: {ref_halfyear}")
    print(f"  Event dummies: {len(event_periods)}")

    all_es_results = []
    pretrend_tests = []

    for age_label in AGE_GROUPS:
        print(f"\n--- Placebo event study: {age_label} ---")
        t0 = time.time()

        # Build balanced panel with zero-filled cells
        sub = balance_panel_for_age(agg, age_label)
        if len(sub) < 100:
            print(f"  Too few observations, skipping")
            continue

        print(f"  Observations: {len(sub):,}")
        print(f"  Employers: {sub['employer_id'].nunique():,}")
        print(f"  Zero cells: {(sub['n_emp'] == 0).sum():,}")

        # Compute variables on the balanced panel
        sub["ln_emp"] = np.log(sub["n_emp"] + 1)
        sub["high"] = (sub["exposure_quartile"] == 4).astype(int)
        sub["halfyear"] = sub["year_month"].apply(assign_halfyear)
        sub["fe_emp_q"] = (
            sub["employer_id"].astype(str) + "_" +
            sub["exposure_quartile"].astype(str)
        )
        sub["fe_emp_t"] = (
            sub["employer_id"].astype(str) + "_" +
            sub["year_month"]
        )

        # Create interaction dummies: halfyear x high
        for p in event_periods:
            sub[f"hy_{p}"] = ((sub["halfyear"] == p).astype(int) * sub["high"])

        interaction_cols = [f"hy_{p}" for p in event_periods]

        # --- PanelOLS with CORRECT FE (same as script 18) ---
        # Entity = employer x quartile
        # Other  = employer x month
        panel = sub.copy()
        panel["date"] = pd.to_datetime(panel["year_month"] + "-01")

        # Set index: entity = employer x quartile, time = date
        panel = panel.set_index(["fe_emp_q", "date"])

        # employer x month as other_effects
        other_fe = pd.DataFrame(
            {"fe_emp_t": panel["fe_emp_t"]},
            index=panel.index,
        )

        try:
            mod = PanelOLS(
                dependent=panel["ln_emp"],
                exog=panel[interaction_cols],
                entity_effects=True,
                time_effects=False,
                other_effects=other_fe,
            )
            res = mod.fit(cov_type="clustered", cluster_entity=True)

            elapsed = time.time() - t0
            print(f"  Estimated in {elapsed:.0f}s")

            # Collect coefficients
            for p in event_periods:
                col = f"hy_{p}"
                all_es_results.append({
                    "age_group": age_label,
                    "period": p,
                    "coef": res.params[col],
                    "se": res.std_errors[col],
                    "pval": res.pvalues[col],
                })

            # Add reference period (zero by construction)
            all_es_results.append({
                "age_group": age_label,
                "period": ref_halfyear,
                "coef": 0.0,
                "se": 0.0,
                "pval": 1.0,
            })

            # Pre-trend joint test (relative to the PLACEBO treatment)
            # Pre-treatment = periods before placebo_halfyear
            pre_cols = [f"hy_{p}" for p in event_periods if p < placebo_halfyear]
            if pre_cols:
                beta = res.params[pre_cols].values
                try:
                    V = res.cov.loc[pre_cols, pre_cols].values
                except AttributeError:
                    V = res.cov_params().loc[pre_cols, pre_cols].values
                from scipy import stats as sp_stats
                k = len(pre_cols)
                wald = float(beta @ np.linalg.solve(V, beta))
                p_val = 1 - sp_stats.chi2.cdf(wald, k)
                pt = {"chi2": round(wald, 2), "df": k, "p_value": round(p_val, 4)}
                print(f"  Pre-trend test (pre-placebo): chi2({pt['df']}) = "
                      f"{pt['chi2']:.2f}, p = {pt['p_value']:.4f}")
                pretrend_tests.append({"age_group": age_label, **pt})

        except Exception as e:
            print(f"  ERROR: {e}")
            print("  This age group failed -- check memory/data.")
            continue

    # --- Save results ---
    es_df = pd.DataFrame()
    if all_es_results:
        es_df = pd.DataFrame(all_es_results)

    return es_df, pretrend_tests


# ======================================================================
#   STEP 3: FIGURES
# ======================================================================

def plot_placebo_event_studies(es_df, placebo_label, placebo_halfyear,
                               ref_halfyear, description):
    """
    One figure per age group for a given placebo specification.
    Vertical dashed line marks the placebo treatment date.
    """
    if es_df.empty:
        return

    print(f"\n  Plotting figures for {placebo_label}...")

    for age_label in AGE_GROUPS:
        sub = es_df[es_df["age_group"] == age_label].copy()
        if sub.empty:
            continue

        sub = sub.sort_values("period")

        # Numeric x-axis
        periods_sorted = sorted(sub["period"].unique())
        x_map = {p: i for i, p in enumerate(periods_sorted)}
        sub["x"] = sub["period"].map(x_map)
        ref_x = x_map.get(ref_halfyear, None)
        placebo_x = x_map.get(placebo_halfyear, None)

        fig, ax = plt.subplots(figsize=(10, 5))

        # Confidence intervals
        sub["ci_lo"] = sub["coef"] - 1.96 * sub["se"]
        sub["ci_hi"] = sub["coef"] + 1.96 * sub["se"]

        ax.fill_between(sub["x"], sub["ci_lo"], sub["ci_hi"],
                        alpha=0.2, color=ORANGE)
        ax.plot(sub["x"], sub["coef"], "o-", color=ORANGE, linewidth=2,
                markersize=6)
        ax.axhline(0, color="black", linewidth=0.5)

        # Reference period marker
        if ref_x is not None:
            ax.axvline(ref_x, color=GRAY, linestyle="--", linewidth=1,
                       label=f"Reference: {ref_halfyear}")

        # Placebo treatment marker
        if placebo_x is not None:
            ax.axvline(placebo_x, color="red", linestyle="--", linewidth=1.5,
                       label=f"Placebo treatment: {placebo_halfyear}")

        ax.set_xticks(range(len(periods_sorted)))
        ax.set_xticklabels(periods_sorted, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Coefficient (ln employment)", fontsize=11)
        ax.set_title(f"Placebo event study: {age_label}\n{description}",
                     fontsize=12)
        ax.legend(fontsize=9)

        fig.tight_layout()
        fname = (f"placebo_{placebo_label}_es_"
                 f"{age_label.replace('+', 'plus')}.png")
        fig.savefig(OUTPUT_DIR / fname, dpi=150)
        plt.close(fig)
        print(f"    Saved {fname}")

    # --- Combined panel (2x3) ---
    age_list = list(AGE_GROUPS.keys())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)

    for idx, age_label in enumerate(age_list):
        ax = axes[idx // 3, idx % 3]
        sub = es_df[es_df["age_group"] == age_label].copy()
        if sub.empty:
            ax.set_title(age_label)
            continue

        sub = sub.sort_values("period")
        periods_sorted = sorted(sub["period"].unique())
        x_map = {p: i for i, p in enumerate(periods_sorted)}
        sub["x"] = sub["period"].map(x_map)
        ref_x = x_map.get(ref_halfyear, None)
        placebo_x = x_map.get(placebo_halfyear, None)

        sub["ci_lo"] = sub["coef"] - 1.96 * sub["se"]
        sub["ci_hi"] = sub["coef"] + 1.96 * sub["se"]

        ax.fill_between(sub["x"], sub["ci_lo"], sub["ci_hi"],
                        alpha=0.2, color=ORANGE)
        ax.plot(sub["x"], sub["coef"], "o-", color=ORANGE, linewidth=1.5,
                markersize=4)
        ax.axhline(0, color="black", linewidth=0.5)
        if ref_x is not None:
            ax.axvline(ref_x, color=GRAY, linestyle="--", linewidth=0.8)
        if placebo_x is not None:
            ax.axvline(placebo_x, color="red", linestyle="--", linewidth=1.2,
                       label="Placebo treatment")

        ax.set_xticks(range(len(periods_sorted)))
        ax.set_xticklabels(periods_sorted, rotation=45, ha="right", fontsize=7)
        ax.set_title(age_label, fontsize=11)

    fig.suptitle(f"Placebo event study: {description}\n"
                 f"(employer x quartile + employer x month FE)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"placebo_{placebo_label}_es_panel.png", dpi=150)
    plt.close(fig)
    print(f"    Saved combined panel figure")


# ======================================================================
#   MAIN
# ======================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("25_mona_placebo_dates.py")
    print("Placebo-date event study for canaries paper")
    print("Testing fake treatment dates to rule out pre-existing trends")
    print("=" * 70)

    t_start = time.time()

    # Step 1: Load data ONCE (reused for both placebo dates)
    agg = load_and_prepare()

    # Step 2: Run placebo event studies
    all_pretrend_results = []
    placebo_results = {}

    for label, placebo_hy, ref_hy, desc in PLACEBO_CONFIGS:
        es_df, pt_tests = run_placebo_event_study(
            agg,
            placebo_halfyear=placebo_hy,
            ref_halfyear=ref_hy,
            description=desc,
        )

        # Save coefficients
        if not es_df.empty:
            fname = f"placebo_{label}_es_all.csv"
            es_df.to_csv(OUTPUT_DIR / fname, index=False)
            print(f"\n  Saved coefficients -> {fname}")

        # Plot figures
        plot_placebo_event_studies(es_df, label, placebo_hy, ref_hy, desc)

        # Collect pre-trend results
        for pt in pt_tests:
            all_pretrend_results.append({"placebo": label, **pt})

        placebo_results[label] = {
            "es_df": es_df,
            "pretrend_tests": pt_tests,
            "description": desc,
            "placebo_halfyear": placebo_hy,
            "ref_halfyear": ref_hy,
        }

    # Step 3: Save pre-trend results
    if all_pretrend_results:
        pt_all = pd.DataFrame(all_pretrend_results)
        pt_all.to_csv(OUTPUT_DIR / "placebo_pretrends.csv", index=False)
        pt_text = pt_all.to_string(index=False)
        (OUTPUT_DIR / "placebo_pretrends.txt").write_text(pt_text)
        print(f"\n  Saved pre-trend tests -> placebo_pretrends.txt")

    # Step 4: Comparison summary
    elapsed = time.time() - t_start
    summary_lines = [
        "=" * 60,
        "PLACEBO DATE TEST SUMMARY",
        "=" * 60,
        f"Script: 25_mona_placebo_dates.py",
        f"FE: employer x quartile (entity) + employer x month (other_effects)",
        f"Runtime: {elapsed/60:.1f} minutes",
        "",
        "PURPOSE:",
        "  Test whether the post-ChatGPT employment composition shift",
        "  (young workers in high-AI-exposure occupations) could be",
        "  explained by a pre-existing trend rather than AI adoption.",
        "",
        "  If no acceleration appears at the placebo dates, this confirms",
        "  the pattern is specific to the post-ChatGPT period.",
        "",
    ]

    for label, info in placebo_results.items():
        summary_lines.append(f"--- {info['description']} ---")
        summary_lines.append(
            f"  Fake treatment: {info['placebo_halfyear']}  "
            f"| Reference: {info['ref_halfyear']}"
        )
        summary_lines.append("  PRE-TREND TESTS (joint chi2, "
                             "H0: all pre-placebo coefficients = 0):")

        pt_tests = info["pretrend_tests"]
        if pt_tests:
            for pt in pt_tests:
                status = "PASS" if pt["p_value"] > 0.05 else "FAIL"
                summary_lines.append(
                    f"    {pt['age_group']:>8s}: chi2({pt['df']}) = "
                    f"{pt['chi2']:>7.2f}, p = {pt['p_value']:.4f}  [{status}]"
                )
        else:
            summary_lines.append("    No pre-trend tests computed.")

        # Report key coefficients for youngest group
        es_df = info["es_df"]
        if not es_df.empty:
            young = es_df[es_df["age_group"] == "22-25"].sort_values("period")
            if not young.empty:
                summary_lines.append("")
                summary_lines.append("  KEY COEFFICIENTS (22-25 age group):")
                for _, row in young.iterrows():
                    sig = ""
                    if row["pval"] < 0.01:
                        sig = "***"
                    elif row["pval"] < 0.05:
                        sig = "**"
                    elif row["pval"] < 0.10:
                        sig = "*"
                    summary_lines.append(
                        f"    {row['period']}: {row['coef']:+.4f} "
                        f"(SE={row['se']:.4f}){sig}"
                    )
        summary_lines.append("")

    summary_lines.extend([
        "INTERPRETATION GUIDE:",
        "  - Flat or near-zero coefficients at the placebo date = GOOD.",
        "    This means the post-ChatGPT pattern is not a continuation",
        "    of a pre-existing trend.",
        "  - Significant negative coefficients for young workers at the",
        "    placebo date = CONCERNING. Would suggest pre-existing trend",
        "    or confounding (e.g. Riksbank hike for Jul 2022 placebo).",
        "  - Compare the magnitude and trajectory of coefficients here",
        "    with the real event study (script 18) to assess whether",
        "    there is a structural break at the ChatGPT launch.",
    ])

    summary_text = "\n".join(summary_lines)
    (OUTPUT_DIR / "placebo_summary.txt").write_text(summary_text)
    print(f"\n{summary_text}")
    print(f"\n  All output in: {OUTPUT_DIR}/")
