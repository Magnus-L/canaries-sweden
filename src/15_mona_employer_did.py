#!/usr/bin/env python3
"""
15_mona_employer_did.py -- Brynjolfsson-style employer-level DiD.

======================================================================
  THIS SCRIPT IS DESIGNED TO RUN IN SCB's MONA SECURE ENVIRONMENT
  It queries monthly AGI (employer declaration) register data via SQL.
  Do NOT run outside MONA -- the data is not available externally.
======================================================================

Purpose:
  Formally test whether young workers (22-25) in high-AI-exposure occupations
  experienced disproportionate employment declines after ChatGPT. This upgrades
  the descriptive Figure 2 in the paper to a causally identified result.

Design:
  Mirrors Brynjolfsson, Chandar & Chen (2025), Eq. 4.1:

    ln(E[y_{f,q,t}]) = a_{f,q} + b_{f,t} + g1*PostRB_t*HighQ4_q
                        + g2*PostGPT_t*HighQ4_q + e_{f,q,t}

  where f = employer, q = DAIOE quartile, t = month.

  Employer x quartile FE absorb baseline differences within firms.
  Employer x month FE absorb ALL firm-level macro shocks (interest rates,
  energy crisis, seasonal hiring, etc.).

  Run SEPARATELY for each age group. The "canaries" finding is that
  g2 is negative and significant for ages 22-25, but not for older groups.

Data access:
  SQL via pyodbc to MONA SQL Server (P1207). Year-by-year queries to
  dbo.Arb_AGIIndivid{ym}{suffix} tables, joined to Individ tables for
  SSYK codes. For years >= 2023, uses cascading SSYK lookup:
  COALESCE(Individ_2023, Individ_2022, Individ_2021).

Panel construction:
  - SQL aggregates to employer x ssyk4 x age_group x month cells
  - Python merges DAIOE quartiles, re-aggregates to employer x quartile
  - Build BALANCED panel: every (employer, quartile) ever observed x all months
  - Zero-fill missing cells (n_emp = 0) -- captures extensive margin
  - Restrict to employers with workers in both Q4 and Q1-Q3 (identification)

Estimator:
  - Primary: OLS on ln(count+1) with high-dimensional FE via linearmodels
    (linearmodels.panel.PanelOLS or absorbed-FE approach)
  - Robustness: Poisson PML via pyfixest (handles zeros naturally, consistent
    under heteroskedasticity; Santos Silva & Tenreyro 2006, Chen & Roth 2024)
  - If linearmodels unavailable: manual within-transformation with pandas

Input (on MONA SQL Server):
  1. dbo.Arb_AGIIndivid{ym}{suffix} -- monthly AGI tables
  2. dbo.Individ_{year} -- individual register tables (SSYK, birth year)
  3. daioe_quartiles.csv on network share (ssyk4, exposure_quartile)

Output files (export from MONA):
  1. canaries_did_results.csv       -- DiD coefficient table by age group
  2. canaries_eventstudy_*.csv      -- half-year event study coefficients
  3. canaries_es_young.png          -- event study figure for 22-25
  4. canaries_es_older.png          -- event study figure for 26-30, 41-49
  5. canaries_summary.txt           -- sample sizes and diagnostics
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import time
import pyodbc
warnings.filterwarnings("ignore")

# Database connection to MONA SQL Server
conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=monasql.micro.intra;"
    "DATABASE=P1207;"
    "Trusted_Connection=yes;"
)


# ======================================================================
#   CONFIGURATION -- ADJUST THESE FOR YOUR MONA ENVIRONMENT
# ======================================================================

# Path to DAIOE quartiles on MONA network share
DAIOE_PATH = r"\\micro.intra\Projekt\P1207$\P1207_Gem\Lydia P1207\daioe_quartiles.csv"

# Output paths (saves alongside script 09 output)
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Treatment dates
RIKSBANK_YM = "2022-04"
CHATGPT_YM = "2022-12"

# Reference period for event study (just before ChatGPT)
REF_HALFYEAR = "2022H1"

# Age group definitions (run regressions separately for each)
# Updated to match the 6-group definition used in the paper (Table A3)
AGE_GROUPS = {
    "22-25": (22, 25),
    "26-30": (26, 30),
    "31-34": (31, 34),
    "35-40": (35, 40),
    "41-49": (41, 49),
    "50+":   (50, 69),
}

# Minimum employer size (helps with computation -- drop tiny employers)
MIN_EMPLOYER_SIZE = 5

# Colours
ORANGE = "#E8873A"
TEAL = "#2E7D6F"
GRAY = "#8C8C8C"
DARK_BLUE = "#1B3A5C"
DARK_TEXT = "#2C2C2C"


# ======================================================================
#   STEP 1a: SQL DATA PULL -- YEAR BY YEAR
# ======================================================================

def pull_and_aggregate_year_employer(year, conn):
    """
    Query one year of AGI data from MONA SQL Server, aggregated to
    employer x ssyk4 x age_group x month cells with COUNT(DISTINCT person_id).

    For years >= 2023, uses cascading SSYK lookup:
      COALESCE(Individ_2023, Individ_2022, Individ_2021)
    This recovers individuals who lack a 2023 SSYK code but had one
    in an earlier register year (e.g. workers with employment gaps).

    For years < 2023, uses a single LEFT JOIN to Individ_{year}.

    Table naming: _def suffix for years < 2025, _prel for 2025 (max 6 months).
    """
    print(f"  Processing year {year}...")

    # Primary SSYK source: 2023 (or current year if before 2023)
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

        # Cascading SSYK lookup for years >= 2023
        if individ_year >= 2023:
            monthly_queries.append(
                f"""
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
                """
            )
        else:
            # For years <= 2022, use the year's own Individ table
            monthly_queries.append(
                f"""
                SELECT
                    agi.P1207_LOPNR_PEORGNR AS employer_id,
                    agi.PERIOD AS period,
                    ind.Ssyk4_2012_J16 AS ssyk4,
                    ind.FodelseAr AS birth_year,
                    agi.P1207_LOPNR_PERSONNR AS person_id
                FROM dbo.Arb_AGIIndivid{ym}{suffix} agi
                LEFT JOIN dbo.Individ_{individ_year} ind
                    ON agi.P1207_LOPNR_PERSONNR = ind.P1207_LopNr_PersonNr
                """
            )

    # Combine all months for this year, compute age, assign age groups,
    # and aggregate to employer x ssyk4 x age_group x month cells
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


# ======================================================================
#   STEP 1b: LOAD AND PREPARE (merge DAIOE, re-aggregate)
# ======================================================================

def load_and_prepare():
    """
    Pull AGI data year by year via SQL (2019-2025), merge DAIOE quartiles,
    and aggregate to employer x quartile x age_group x month cells.

    The SQL step aggregates to employer x ssyk4 x age_group x month
    (with COUNT(DISTINCT person_id)) for performance. Then Python merges
    DAIOE quartiles on ssyk4 and re-aggregates to employer x quartile x
    age_group x month -- the unit of analysis in Brynjolfsson et al. (2025).
    """
    print("=" * 70)
    print("STEP 1: Loading and preparing data via SQL")
    print("=" * 70)

    # --- Pull year by year from SQL ---
    years = list(range(2019, 2026))
    all_years = []

    for y in years:
        df_year = pull_and_aggregate_year_employer(y, conn)
        all_years.append(df_year)
        print(f"    Year {y}: {len(df_year):,} cells")

    agg = pd.concat(all_years, ignore_index=True)
    print(f"\n  Rows after SQL aggregation: {len(agg):,}")

    # --- Merge DAIOE quartiles ---
    print("  Merging DAIOE quartiles...")
    daioe = pd.read_csv(DAIOE_PATH)
    daioe["ssyk4"] = daioe["ssyk4"].astype(str).str.zfill(4)

    # Parse quartile labels to numeric (handles "Q1 (lowest)", "Q2", etc.)
    if daioe["exposure_quartile"].dtype == object:
        daioe["exposure_quartile"] = (
            daioe["exposure_quartile"]
            .str.strip()
            .str.extract(r"Q(\d)")
            .astype(int)
        )

    agg["ssyk4"] = agg["ssyk4"].astype(str).str.zfill(4)

    agg = agg.merge(
        daioe[["ssyk4", "exposure_quartile"]],
        on="ssyk4",
        how="inner",
    )
    print(f"  Rows after DAIOE merge: {len(agg):,}")

    # --- Re-aggregate to employer x quartile x age_group x month ---
    # (SQL grouped by ssyk4; now collapse across occupations within quartile)
    agg = (
        agg.groupby(
            ["employer_id", "year_month", "exposure_quartile", "age_group"]
        )["n_emp"]
        .sum()
        .reset_index()
    )

    # --- Filter small employers ---
    emp_size = agg.groupby("employer_id")["n_emp"].sum()
    large_emp = emp_size[emp_size >= MIN_EMPLOYER_SIZE].index
    n_before = agg["employer_id"].nunique()
    agg = agg[agg["employer_id"].isin(large_emp)].copy()
    print(f"  Employers: {n_before:,} total -> {agg['employer_id'].nunique():,} "
          f"(>={MIN_EMPLOYER_SIZE} total workers)")

    print(f"\n  Final panel cells: {len(agg):,}")
    print(f"  Employers: {agg['employer_id'].nunique():,}")
    print(f"  Months: {agg['year_month'].nunique()}")
    print(f"  Period: {agg['year_month'].min()} to {agg['year_month'].max()}")
    print(f"\n  Quartile distribution:")
    for q in sorted(agg["exposure_quartile"].unique()):
        n = agg[agg["exposure_quartile"] == q]["n_emp"].sum()
        print(f"    Q{q}: {n:,} person-months")

    return agg


# ======================================================================
#   STEP 1c: BUILD BALANCED PANEL (ZERO-FILL)
# ======================================================================

def balance_panel_for_age(agg, age_label):
    """
    Build balanced panel for one age group: every (employer, quartile)
    combination the employer is ever observed in × all months.
    Missing cells filled with n_emp = 0.

    Also applies identification restriction from the paper: keep only
    employers observed in both Q4 and at least one of Q1-Q3. This
    ensures that the employer×quartile FE actually absorbs within-
    employer variation — single-quartile employers contribute nothing.

    WHY THIS MATTERS: Without zero-filling, the groupby aggregation in
    load_and_prepare() only creates cells where employment > 0. Firms
    that shed ALL workers of an age group in a quartile-month simply
    disappear from the data. This drops the strongest treatment cases
    (extensive margin) and biases the DiD toward zero.
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
    all_months = pd.DataFrame({"year_month": sorted(agg["year_month"].unique())})
    emp_q["_k"] = 1
    all_months["_k"] = 1
    balanced = emp_q.merge(all_months, on="_k").drop(columns="_k")

    # Left join actual employment counts
    balanced = balanced.merge(
        sub[["employer_id", "exposure_quartile", "year_month", "n_emp"]],
        on=["employer_id", "exposure_quartile", "year_month"],
        how="left",
    )
    balanced["n_emp"] = balanced["n_emp"].fillna(0).astype(int)
    balanced["age_group"] = age_label

    n_zeros = (balanced["n_emp"] == 0).sum()
    pct_zero = 100 * n_zeros / len(balanced)
    print(f"  Balanced panel ({age_label}): {len(balanced):,} cells "
          f"({n_zeros:,} zeros = {pct_zero:.1f}%)")
    print(f"  Employers with both Q4 and Q1-Q3: {len(valid_emps):,}")

    return balanced


# ======================================================================
#   STEP 2: MAIN DiD BY AGE GROUP
# ======================================================================

def run_did_by_age(agg):
    """
    For each age group, estimate:

      ln(n_emp_{f,q,t} + 1) = a_{f,q} + b_{f,t}
                               + g1*PostRB_t*High_q
                               + g2*PostGPT_t*High_q + e

    where High = (quartile == 4).

    Employer x quartile FE (a_{f,q}) absorb time-invariant differences.
    Employer x month FE (b_{f,t}) absorb ALL firm-level shocks.

    Identification: within-firm, within-month variation across quartiles.

    Returns a DataFrame of coefficients for all age groups.
    """
    print("\n" + "=" * 70)
    print("STEP 2: DiD regressions by age group")
    print("=" * 70)

    all_results = []

    for age_label, (age_lo, age_hi) in AGE_GROUPS.items():
        print(f"\n--- Age group: {age_label} ---")

        # Build balanced panel with zero-filled cells
        # NOTE: balance_panel_for_age() returns only employer_id,
        # exposure_quartile, year_month, n_emp, age_group.
        # Treatment variables and FE identifiers must be created AFTER
        # balancing, not before (otherwise they are dropped by the
        # cross-join + left-merge inside balance_panel_for_age).
        sub = balance_panel_for_age(agg, age_label)

        if len(sub) < 100:
            print(f"  Too few observations ({len(sub)}), skipping")
            continue

        # Create treatment variables on the balanced panel
        sub["post_rb"] = (sub["year_month"] >= RIKSBANK_YM).astype(int)
        sub["post_gpt"] = (sub["year_month"] >= CHATGPT_YM).astype(int)
        sub["high"] = (sub["exposure_quartile"] == 4).astype(int)
        sub["post_rb_x_high"] = sub["post_rb"] * sub["high"]
        sub["post_gpt_x_high"] = sub["post_gpt"] * sub["high"]

        # Log outcome (add 1 to handle zeros from balanced panel)
        sub["ln_emp"] = np.log(sub["n_emp"] + 1)

        # Create FE group identifiers
        sub["fe_emp_q"] = (
            sub["employer_id"].astype(str) + "_" +
            sub["exposure_quartile"].astype(str)
        )
        sub["fe_emp_t"] = (
            sub["employer_id"].astype(str) + "_" +
            sub["year_month"]
        )

        print(f"  Observations: {len(sub):,}")
        print(f"  Employers: {sub['employer_id'].nunique():,}")
        print(f"  Mean employment: {sub['n_emp'].mean():.2f}")
        print(f"  Zero cells: {(sub['n_emp'] == 0).sum():,}")

        # --- Main: OLS on ln(count+1) ---
        result = _estimate_did(sub, age_label)
        if result is not None:
            all_results.append(result)

        # --- Robustness: Poisson (pyfixest) ---
        poisson_result = _estimate_poisson(sub, age_label)
        if poisson_result is not None:
            all_results.append(poisson_result)

    # Combine results
    if all_results:
        results_df = pd.DataFrame(all_results)
        out = OUTPUT_DIR / "canaries_did_results.csv"
        results_df.to_csv(out, index=False)
        print(f"\n  Saved DiD results -> {out.name}")
        print("\n  === SUMMARY ===")
        print(results_df.to_string(index=False))
        return results_df

    return pd.DataFrame()


def _estimate_did(sub, age_label):
    """
    Estimate the DiD for one age group. Tries three approaches:

    A. linearmodels PanelOLS with employer x quartile entity FE +
       employer x month as absorbed other_effects
    B. Manual within-transformation (demean by employer x quartile,
       then include employer x month dummies via absorption)
    C. Simple OLS with occupation + month FE (backup, weaker identification)
    """
    t0 = time.time()

    # --- Approach A: linearmodels ---
    try:
        from linearmodels.panel import PanelOLS

        panel = sub.copy()
        panel["date"] = pd.to_datetime(panel["year_month"] + "-01")

        # PanelOLS can absorb entity effects + one set of other_effects.
        # Entity = employer x quartile, Other = employer x month
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

        gamma1 = res.params["post_rb_x_high"]
        gamma2 = res.params["post_gpt_x_high"]
        se1 = res.std_errors["post_rb_x_high"]
        se2 = res.std_errors["post_gpt_x_high"]
        p1 = res.pvalues["post_rb_x_high"]
        p2 = res.pvalues["post_gpt_x_high"]

        elapsed = time.time() - t0
        print(f"  [linearmodels, {elapsed:.0f}s]")
        print(f"  g1 (PostRB x High)  = {gamma1:+.4f} (SE={se1:.4f}, p={p1:.4f})")
        print(f"  g2 (PostGPT x High) = {gamma2:+.4f} (SE={se2:.4f}, p={p2:.4f})")

        return {
            "age_group": age_label,
            "method": "PanelOLS",
            "n_obs": int(res.nobs),
            "gamma1_rb_high": gamma1,
            "se1": se1,
            "pval1": p1,
            "gamma2_gpt_high": gamma2,
            "se2": se2,
            "pval2": p2,
        }

    except ImportError:
        print("  linearmodels not available -- trying manual within-transformation")
    except Exception as e:
        print(f"  linearmodels failed: {e}")
        print("  Trying manual within-transformation...")

    # --- Approach B: Manual within-transformation ---
    # Demean by employer x quartile (entity FE) and employer x month (time FE)
    try:
        panel = sub.copy()

        # Demean outcome and regressors by employer x quartile
        for col in ["ln_emp", "post_rb_x_high", "post_gpt_x_high"]:
            panel[f"{col}_dm1"] = panel.groupby("fe_emp_q")[col].transform(
                lambda x: x - x.mean()
            )

        # Then demean by employer x month
        for col in ["ln_emp", "post_rb_x_high", "post_gpt_x_high"]:
            panel[f"{col}_dm"] = panel.groupby("fe_emp_t")[f"{col}_dm1"].transform(
                lambda x: x - x.mean()
            )

        # OLS on demeaned data (no constant needed after demeaning)
        import statsmodels.api as sm

        y = panel["ln_emp_dm"].values
        X = panel[["post_rb_x_high_dm", "post_gpt_x_high_dm"]].values

        # Simple OLS (SEs will be approximate without proper clustering correction)
        mod = sm.OLS(y, X)
        res = mod.fit(
            cov_type="cluster",
            cov_kwds={"groups": panel["employer_id"].values},
        )

        gamma1 = res.params[0]
        gamma2 = res.params[1]
        se1 = res.bse[0]
        se2 = res.bse[1]
        p1 = res.pvalues[0]
        p2 = res.pvalues[1]

        elapsed = time.time() - t0
        print(f"  [within-transformation, {elapsed:.0f}s]")
        print(f"  g1 (PostRB x High)  = {gamma1:+.4f} (SE={se1:.4f}, p={p1:.4f})")
        print(f"  g2 (PostGPT x High) = {gamma2:+.4f} (SE={se2:.4f}, p={p2:.4f})")

        return {
            "age_group": age_label,
            "method": "within-transformation",
            "n_obs": len(panel),
            "gamma1_rb_high": gamma1,
            "se1": se1,
            "pval1": p1,
            "gamma2_gpt_high": gamma2,
            "se2": se2,
            "pval2": p2,
        }

    except Exception as e:
        print(f"  Within-transformation failed: {e}")

    # --- Approach C: Backup -- occupation-level (weaker identification) ---
    print("  Falling back to occupation-level regression (Section 5 backup)")
    return _estimate_did_occupation_level(sub, age_label)


def _estimate_did_occupation_level(sub, age_label):
    """
    Backup: occupation x month panel with occupation + month FE.
    Weaker identification (no employer-level controls) but always feasible.
    """
    import statsmodels.api as sm

    occ_panel = (
        sub.groupby(["exposure_quartile", "year_month"])
        .agg(n_emp=("n_emp", "sum"))
        .reset_index()
    )
    occ_panel["high"] = (occ_panel["exposure_quartile"] == 4).astype(int)
    occ_panel["post_rb"] = (occ_panel["year_month"] >= RIKSBANK_YM).astype(int)
    occ_panel["post_gpt"] = (occ_panel["year_month"] >= CHATGPT_YM).astype(int)
    occ_panel["post_rb_x_high"] = occ_panel["post_rb"] * occ_panel["high"]
    occ_panel["post_gpt_x_high"] = occ_panel["post_gpt"] * occ_panel["high"]
    occ_panel["ln_emp"] = np.log(occ_panel["n_emp"] + 1)

    # Occupation + month dummies
    q_dummies = pd.get_dummies(occ_panel["exposure_quartile"], prefix="q", drop_first=True)
    t_dummies = pd.get_dummies(occ_panel["year_month"], prefix="t", drop_first=True)

    X = pd.concat([
        occ_panel[["post_rb_x_high", "post_gpt_x_high"]],
        q_dummies, t_dummies,
    ], axis=1).astype(float)
    X = sm.add_constant(X)

    mod = sm.OLS(occ_panel["ln_emp"].values, X)
    res = mod.fit(cov_type="HC1")

    gamma1 = res.params["post_rb_x_high"]
    gamma2 = res.params["post_gpt_x_high"]

    print(f"  [occupation-level backup]")
    print(f"  g1 (PostRB x High)  = {gamma1:+.4f} (SE={res.bse['post_rb_x_high']:.4f})")
    print(f"  g2 (PostGPT x High) = {gamma2:+.4f} (SE={res.bse['post_gpt_x_high']:.4f})")

    return {
        "age_group": age_label,
        "method": "occupation-level",
        "n_obs": len(occ_panel),
        "gamma1_rb_high": gamma1,
        "se1": res.bse["post_rb_x_high"],
        "pval1": res.pvalues["post_rb_x_high"],
        "gamma2_gpt_high": gamma2,
        "se2": res.bse["post_gpt_x_high"],
        "pval2": res.pvalues["post_gpt_x_high"],
    }


# ======================================================================
#   ROBUSTNESS: POISSON (pyfixest)
# ======================================================================

def _estimate_poisson(sub, age_label):
    """
    Poisson pseudo-maximum likelihood with high-dimensional FE.
    Uses pyfixest (Python equivalent of fixest/ppmlhdfe).

    Preferred estimator per Santos Silva & Tenreyro (2006) and
    Chen & Roth (2024, JPE Micro): handles zeros naturally, consistent
    under heteroskedasticity, no ln(y+1) transformation needed.

    Returns None if pyfixest is not available.
    """
    try:
        import pyfixest as pf
    except ImportError:
        print(f"  [Poisson] pyfixest not installed -- skipping Poisson robustness")
        print(f"  Install with: pip install pyfixest")
        return None

    t0 = time.time()
    panel = sub.copy()

    # Treatment variables (may already exist from OLS step, but safe to recreate)
    panel["post_rb"] = (panel["year_month"] >= RIKSBANK_YM).astype(int)
    panel["post_gpt"] = (panel["year_month"] >= CHATGPT_YM).astype(int)
    panel["high"] = (panel["exposure_quartile"] == 4).astype(int)
    panel["post_rb_x_high"] = panel["post_rb"] * panel["high"]
    panel["post_gpt_x_high"] = panel["post_gpt"] * panel["high"]

    # FE identifiers
    panel["fe_emp_q"] = (
        panel["employer_id"].astype(str) + "_" +
        panel["exposure_quartile"].astype(str)
    )
    panel["fe_emp_t"] = (
        panel["employer_id"].astype(str) + "_" +
        panel["year_month"]
    )

    try:
        fit = pf.fepois(
            "n_emp ~ post_rb_x_high + post_gpt_x_high | fe_emp_q + fe_emp_t",
            data=panel,
            vcov={"CRV1": "employer_id"},
        )

        gamma1 = fit.coef()["post_rb_x_high"]
        gamma2 = fit.coef()["post_gpt_x_high"]
        se1 = fit.se()["post_rb_x_high"]
        se2 = fit.se()["post_gpt_x_high"]
        p1 = fit.pvalue()["post_rb_x_high"]
        p2 = fit.pvalue()["post_gpt_x_high"]

        elapsed = time.time() - t0
        print(f"  [Poisson/pyfixest, {elapsed:.0f}s]")
        print(f"  g1 (PostRB x High)  = {gamma1:+.4f} (SE={se1:.4f}, p={p1:.4f})")
        print(f"  g2 (PostGPT x High) = {gamma2:+.4f} (SE={se2:.4f}, p={p2:.4f})")

        return {
            "age_group": age_label,
            "method": "Poisson",
            "n_obs": fit.nobs,
            "gamma1_rb_high": gamma1,
            "se1": se1,
            "pval1": p1,
            "gamma2_gpt_high": gamma2,
            "se2": se2,
            "pval2": p2,
        }

    except Exception as e:
        print(f"  [Poisson] Failed: {e}")
        return None


# ======================================================================
#   PRE-TREND JOINT TEST (used by Step 3)
# ======================================================================

def pretrend_joint_test(res, event_periods, ref_period):
    """Joint chi2 test: H0: all pre-treatment ES coefficients = 0."""
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
    p_val = 1 - sp_stats.chi2.cdf(wald, k)
    return {"chi2": round(wald, 3), "df": k, "p": round(p_val, 4)}


# ======================================================================
#   STEP 3: HALF-YEAR EVENT STUDY
# ======================================================================

def assign_halfyear(ym_series):
    """Map 'YYYY-MM' strings to 'YYYYHn' labels."""
    year = ym_series.str[:4]
    month = ym_series.str[5:7].astype(int)
    half = np.where(month <= 6, "H1", "H2")
    return year + half


def run_halfyear_event_study(agg):
    """
    Half-year event study: interact half-year dummies with High indicator,
    separately by age group. Reference: 2022H1 (pre-Riksbank).

    This traces the time path of the high-vs-low AI exposure gap,
    showing whether divergence appears (a) pre-ChatGPT, (b) post-ChatGPT,
    or (c) was already present earlier.
    """
    print("\n" + "=" * 70)
    print("STEP 3: Half-year event study")
    print("=" * 70)

    # Determine half-year periods from the raw data
    # (don't modify agg -- variables are created per age group after balancing)
    _hy = assign_halfyear(agg["year_month"])
    all_periods = sorted(_hy.unique())
    event_periods = [p for p in all_periods if p != REF_HALFYEAR]

    all_es_results = []
    pretrend_tests = []

    for age_label in AGE_GROUPS:
        print(f"\n--- Event study: {age_label} ---")

        # Build balanced panel with zero-filled cells
        sub = balance_panel_for_age(agg, age_label)

        if len(sub) < 100:
            print(f"  Too few observations, skipping")
            continue

        # Create variables on the balanced panel (same pattern as script 18)
        sub["ln_emp"] = np.log(sub["n_emp"] + 1)
        sub["high"] = (sub["exposure_quartile"] == 4).astype(int)
        sub["halfyear"] = assign_halfyear(sub["year_month"])
        sub["fe_emp_q"] = (
            sub["employer_id"].astype(str) + "_" +
            sub["exposure_quartile"].astype(str)
        )
        sub["fe_emp_t"] = (
            sub["employer_id"].astype(str) + "_" +
            sub["year_month"]
        )

        # Create interaction dummies: halfyear × high
        for p in event_periods:
            sub[f"hy_{p}"] = ((sub["halfyear"] == p).astype(int) * sub["high"])

        interaction_cols = [f"hy_{p}" for p in event_periods]

        # Try linearmodels with CORRECT FE (matching main DiD and script 18)
        try:
            from linearmodels.panel import PanelOLS

            panel = sub.copy()
            panel["date"] = pd.to_datetime(panel["year_month"] + "-01")

            # Entity = employer × quartile (same as main DiD)
            panel = panel.set_index(["fe_emp_q", "date"])

            # employer × month as other_effects (THE KEY FIX --
            # matches main DiD and script 18, absorbs all firm-level
            # time-varying shocks instead of just calendar month FE)
            other_fe = pd.DataFrame(
                {"fe_emp_t": panel["fe_emp_t"]},
                index=panel.index,
            )

            mod = PanelOLS(
                dependent=panel["ln_emp"],
                exog=panel[interaction_cols],
                entity_effects=True,
                time_effects=False,       # NOT calendar-month FE
                other_effects=other_fe,   # employer×month FE instead
            )
            res = mod.fit(cov_type="clustered", cluster_entity=True)

            for p in event_periods:
                col = f"hy_{p}"
                all_es_results.append({
                    "age_group": age_label,
                    "period": p,
                    "coef": res.params[col],
                    "se": res.std_errors[col],
                    "pval": res.pvalues[col],
                })

            # Pre-trend joint test
            pt = pretrend_joint_test(res, event_periods, REF_HALFYEAR)
            if pt:
                print(f"  Pre-trend test: chi2({pt['df']}) = {pt['chi2']:.2f}, p = {pt['p']:.4f}")
                pretrend_tests.append({"age_group": age_label, **pt})

        except (ImportError, Exception) as e:
            print(f"  linearmodels failed ({e}), using statsmodels")
            import statsmodels.api as sm

            # Simpler: quartile + month dummies
            q_dummies = pd.get_dummies(sub["exposure_quartile"], prefix="q", drop_first=True)
            t_dummies = pd.get_dummies(sub["year_month"], prefix="t", drop_first=True)
            X = pd.concat([sub[interaction_cols], q_dummies, t_dummies], axis=1).astype(float)
            X = sm.add_constant(X)

            mod = sm.OLS(sub["ln_emp"].values, X)
            res = mod.fit(cov_type="HC1")

            for p in event_periods:
                col = f"hy_{p}"
                all_es_results.append({
                    "age_group": age_label,
                    "period": p,
                    "coef": res.params[col],
                    "se": res.bse[col],
                    "pval": res.pvalues[col],
                })

            # Pre-trend joint test (statsmodels fallback)
            pt = pretrend_joint_test(res, event_periods, REF_HALFYEAR)
            if pt:
                print(f"  Pre-trend test: chi2({pt['df']}) = {pt['chi2']:.2f}, p = {pt['p']:.4f}")
                pretrend_tests.append({"age_group": age_label, **pt})

        # Add reference period
        all_es_results.append({
            "age_group": age_label,
            "period": REF_HALFYEAR,
            "coef": 0.0,
            "se": 0.0,
            "pval": 1.0,
        })

    if pretrend_tests:
        pt_df = pd.DataFrame(pretrend_tests)
        pt_out = OUTPUT_DIR / "pretrend_ftest.csv"
        pt_df.to_csv(pt_out, index=False)
        print(f"\n  Pre-trend F-tests -> {pt_out.name}")
        print(pt_df.to_string(index=False))

    if all_es_results:
        es_df = pd.DataFrame(all_es_results)
        out = OUTPUT_DIR / "canaries_es_all.csv"
        es_df.to_csv(out, index=False)
        print(f"\n  Saved event study -> {out.name}")
        return es_df

    return pd.DataFrame()


# ======================================================================
#   STEP 4: EVENT STUDY FIGURES
# ======================================================================

def plot_event_studies(es_df):
    """Create event study coefficient plots for key age groups."""
    print("\n" + "=" * 70)
    print("STEP 4: Event study figures")
    print("=" * 70)

    for age_label, color, filename in [
        ("22-25", ORANGE, "canaries_es_young.png"),
        ("26-30", TEAL, "canaries_es_25to30.png"),
        ("41-49", DARK_BLUE, "canaries_es_41to49.png"),
    ]:
        sub = es_df[es_df["age_group"] == age_label].sort_values("period")
        if sub.empty:
            print(f"  No data for {age_label}, skipping")
            continue

        sub = sub.copy()
        sub["x"] = range(len(sub))
        sub["ci_lo"] = sub["coef"] - 1.96 * sub["se"]
        sub["ci_hi"] = sub["coef"] + 1.96 * sub["se"]

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.fill_between(sub["x"], sub["ci_lo"], sub["ci_hi"],
                         alpha=0.15, color=color)
        ax.plot(sub["x"], sub["coef"], "o-", color=color, linewidth=2, markersize=6)
        ax.axhline(0, color=DARK_TEXT, linewidth=0.8, alpha=0.5)

        # Mark reference
        ref_rows = sub[sub["period"] == REF_HALFYEAR]
        if not ref_rows.empty:
            ref_x = ref_rows["x"].values[0]
            ax.axvline(ref_x, color=TEAL, linestyle="--", linewidth=1, alpha=0.7)

        # Mark ChatGPT
        gpt_rows = sub[sub["period"] == "2022H2"]
        if not gpt_rows.empty:
            gpt_x = gpt_rows["x"].values[0]
            ax.axvline(gpt_x, color=GRAY, linestyle=":", linewidth=1, alpha=0.7)

        ax.set_xticks(sub["x"])
        ax.set_xticklabels(sub["period"], rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Coefficient (High x period)")
        ax.set_title(f"Employment event study: ages {age_label}")

        fig.tight_layout()
        out = OUTPUT_DIR / filename
        fig.savefig(out, dpi=300)
        plt.close()
        print(f"  Saved -> {filename}")


# ======================================================================
#   STEP 5: DIAGNOSTICS AND SUMMARY
# ======================================================================

def write_summary(agg, did_results, es_df):
    """Write diagnostic summary and table-ready statistics."""

    # --- Main summary file ---
    out = OUTPUT_DIR / "canaries_summary.txt"
    with open(out, "w") as f:
        f.write("CANARIES REGRESSION -- SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Total panel cells (non-zero): {len(agg):,}\n")
        f.write(f"Employers: {agg['employer_id'].nunique():,}\n")
        f.write(f"Months: {agg['year_month'].nunique()}\n")
        f.write(f"Period: {agg['year_month'].min()} to {agg['year_month'].max()}\n")
        f.write(f"Min employer size: {MIN_EMPLOYER_SIZE}\n\n")

        f.write("--- Age group sizes (non-zero cells) ---\n")
        for ag in AGE_GROUPS:
            n = len(agg[agg["age_group"] == ag])
            f.write(f"  {ag}: {n:,} cells\n")

        f.write("\n--- Quartile distribution ---\n")
        for q in sorted(agg["exposure_quartile"].unique()):
            n = agg[agg["exposure_quartile"] == q]["n_emp"].sum()
            f.write(f"  Q{q}: {n:,} person-months\n")

        if not did_results.empty:
            f.write("\n--- DiD Results (OLS + Poisson) ---\n")
            f.write(did_results.to_string(index=False))
            f.write("\n")

        f.write("\nFE structure: employer x quartile + employer x month\n")
        f.write("SEs clustered by employer\n")
        f.write("Panel: balanced (zero-filled), restricted to employers with Q4 and Q1-Q3\n")
        f.write("Main estimator: OLS on ln(count+1)\n")
        f.write("Robustness: Poisson PML via pyfixest (if available)\n")

    print(f"\n  Saved summary -> {out.name}")

    # --- Table-ready summary statistics (for appendix Table A2) ---
    _write_table_sumstats(agg)


def _write_table_sumstats(agg):
    """
    Produce summary statistics for the appendix employment table,
    broken out by DAIOE quartile: employers, mean workers per cell,
    and zero-cell shares. Uses balanced panels per age group.
    """
    out = OUTPUT_DIR / "sumstats_employment_table.txt"
    quartiles = sorted(agg["exposure_quartile"].unique())
    months = sorted(agg["year_month"].unique())
    n_months = len(months)

    lines = []
    lines.append("EMPLOYMENT SUMMARY STATISTICS FOR APPENDIX TABLE")
    lines.append("=" * 60)
    lines.append(f"Months: {n_months}  ({months[0]} to {months[-1]})")

    # Employers per quartile
    lines.append("\n--- Employers per quartile ---")
    for q in quartiles:
        n = agg.loc[agg["exposure_quartile"] == q, "employer_id"].nunique()
        lines.append(f"  Q{q}: {n:>10,}")
    lines.append(f"  All: {agg['employer_id'].nunique():>10,}  "
                 f"(employers can span multiple quartiles)")

    # Mean workers per cell and zero shares, by quartile x age group
    # Uses balanced panels (same as regression sample)
    lines.append("\n--- Balanced panel statistics (employers with Q4 and Q1-Q3) ---")

    header = f"  {'Age group':<12}" + "".join(f"{'Q'+str(q):>10}" for q in quartiles) + f"{'All':>10}"

    lines.append("\n  Mean workers per cell:")
    lines.append(header)
    lines.append("  " + "-" * (12 + 10 * (len(quartiles) + 1)))
    for age_label in AGE_GROUPS:
        balanced = balance_panel_for_age(agg, age_label)
        row = f"  {age_label:<12}"
        for q in quartiles:
            mean_val = balanced.loc[balanced["exposure_quartile"] == q, "n_emp"].mean()
            row += f"{mean_val:>10.2f}"
        row += f"{balanced['n_emp'].mean():>10.2f}"
        lines.append(row)

    lines.append("\n  Share zero-employment cells (%):")
    lines.append(header)
    lines.append("  " + "-" * (12 + 10 * (len(quartiles) + 1)))
    for age_label in AGE_GROUPS:
        balanced = balance_panel_for_age(agg, age_label)
        row = f"  {age_label:<12}"
        for q in quartiles:
            bq = balanced[balanced["exposure_quartile"] == q]
            pct = 100 * (bq["n_emp"] == 0).sum() / len(bq) if len(bq) > 0 else 0
            row += f"{pct:>9.1f}%"
        pct_all = 100 * (balanced["n_emp"] == 0).sum() / len(balanced)
        row += f"{pct_all:>9.1f}%"
        lines.append(row)

    lines.append("\n  Balanced panel cells (total):")
    lines.append(header)
    lines.append("  " + "-" * (12 + 10 * (len(quartiles) + 1)))
    for age_label in AGE_GROUPS:
        balanced = balance_panel_for_age(agg, age_label)
        row = f"  {age_label:<12}"
        for q in quartiles:
            n = len(balanced[balanced["exposure_quartile"] == q])
            row += f"{n:>10,}"
        row += f"{len(balanced):>10,}"
        lines.append(row)

    lines.append("")
    text = "\n".join(lines)
    out.write_text(text)
    print(f"\n  Saved table statistics -> {out.name}")
    print(text)


# ======================================================================
#   MAIN
# ======================================================================

def main():
    print("=" * 70)
    print("CANARIES REGRESSION -- Brynjolfsson-style employer-level DiD")
    print("Python version for MONA")
    print("=" * 70)

    # Step 1: Load and prepare
    agg = load_and_prepare()

    # Step 2: Main DiD by age group
    did_results = run_did_by_age(agg)

    # Step 3: Half-year event study
    es_df = run_halfyear_event_study(agg)

    # Step 4: Event study figures
    if not es_df.empty:
        plot_event_studies(es_df)

    # Step 5: Summary
    write_summary(agg, did_results, es_df)

    print("\n" + "=" * 70)
    print("DONE. Export these files from MONA:")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("  1. canaries_did_results.csv    -- DiD coefficients (OLS + Poisson)")
    print("  2. canaries_es_all.csv         -- event study coefficients")
    print("  3. canaries_es_young.png       -- event study figure (22-25)")
    print("  4. canaries_es_25to30.png      -- event study figure (25-30)")
    print("  5. canaries_es_41to49.png      -- event study figure (41-49)")
    print("  6. canaries_summary.txt        -- sample sizes and diagnostics")
    print("  7. pretrend_ftest.csv          -- joint pre-trend tests by age group")
    print("")
    print("  NOTE: Panel is now balanced (zero-filled) and restricted to")
    print("  employers with workers in both Q4 and Q1-Q3 occupations.")
    print("  Results include both OLS on ln(count+1) and Poisson robustness.")
    print("=" * 70)


if __name__ == "__main__":
    main()
