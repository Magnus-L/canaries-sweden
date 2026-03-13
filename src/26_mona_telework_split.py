#!/usr/bin/env python3
"""
26_mona_telework_split.py -- DiD + event study split by Dingel-Neiman teleworkability.

======================================================================
  THIS SCRIPT IS DESIGNED TO RUN IN SCB's MONA SECURE ENVIRONMENT
  Do NOT run outside MONA -- the data is not available externally.
======================================================================

PURPOSE:
  Re-estimate the employer-level DiD AND event study separately for
  teleworkable vs non-teleworkable occupations, using Dingel & Neiman
  (2020) teleworkability scores crosswalked to SSYK 2012.

  This addresses the concern that AI-exposed occupations overlap with
  teleworkable occupations, and that the "canaries" effect might reflect
  remote-work-driven labour market restructuring rather than AI per se.
  If AI exposure and teleworkability are confounded, splitting the sample
  should reveal where the effect lives.

  The local Platsbanken analysis (script 09) found the ChatGPT effect is
  zero in teleworkable occupations (p=0.917) but -0.233*** (p=0.001) in
  non-teleworkable ones. This script tests whether the same pattern holds
  in the employer-level MONA employment data.

CROSSWALK:
  The script first tries to load a pre-built file mapping SSYK4 -> binary
  teleworkable indicator. If not found, it includes a builder function
  that Lydia can run from raw crosswalk files (SOC -> ISCO -> SSYK).

SPECIFICATIONS:
  DiD:  ln(n_emp+1) = a_{f,q} + b_{f,t} + g1*PostRB*High + g2*PostGPT*High
  ES:   ln(n_emp+1) = a_{f,q} + b_{f,t} + sum_h(delta_h * I(h) * High)

  FE: employer x quartile (entity) + employer x month (other_effects)
  SEs: clustered by entity (employer x quartile)

  Identical to scripts 15 and 18.

ESTIMATED RUNTIME:
  ~60-120 min (two subsamples x 6 age groups each for both DiD and ES).

OUTPUT FILES (in output_26/):
  1. telework_did_results.csv       -- DiD coefficients for both subsamples
  2. telework_es_all.csv            -- event study coefficients
  3. telework_pretrends.csv         -- pre-trend joint F-test results
  4. telework_es_*.png              -- event study figures per age group
  5. telework_comparison_panel.png  -- side-by-side comparison panel
  6. telework_summary.txt           -- diagnostics and interpretation
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

# --- Teleworkability file (pre-built SSYK4 -> teleworkable 0/1) ---
# Try this path first. If not found, the script will attempt to build
# from raw crosswalk files (see build_telework_mapping()).
TELEWORK_PATH = r"\\micro.intra\Projekt\P1207$\P1207_Gem\Lydia P1207\dingel_neiman_ssyk4.csv"

# --- Output directory ---
OUTPUT_DIR = Path("output_26")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Treatment dates ---
RIKSBANK_YM = "2022-04"
CHATGPT_YM = "2022-12"

# --- Reference period (omitted dummy) ---
REF_HALFYEAR = "2022H1"

# --- Age group definitions (same as scripts 15 and 18) ---
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
DARK_BLUE = "#1B3A5C"


# ======================================================================
#   TELEWORKABILITY LOADING / BUILDING
# ======================================================================

def load_telework_mapping():
    """
    Load SSYK4 -> teleworkable (0/1) mapping.

    Strategy:
      1. Try the pre-built file on the MONA network share.
      2. If not found, try to build from raw crosswalk files.
      3. If raw files also missing, raise with instructions for Lydia.

    The binary split uses the median of the continuous Dingel-Neiman
    score at SSYK4 level (same approach as script 09 for Platsbanken).

    Returns: DataFrame with columns [ssyk4, teleworkable] where
             teleworkable is 0 or 1 (binary, median-split).
    """
    # --- Attempt 1: pre-built file ---
    try:
        tw = pd.read_csv(TELEWORK_PATH)
        tw["ssyk4"] = tw["ssyk4"].astype(str).str.zfill(4)
        print(f"  Loaded telework mapping from network share: {len(tw)} SSYK codes")

        # If the file already has binary values, use as-is
        if set(tw["teleworkable"].unique()) <= {0, 1, 0.0, 1.0}:
            tw["teleworkable"] = tw["teleworkable"].astype(int)
            print(f"  Binary classification (pre-built)")
        else:
            # Continuous scores: split at median
            median_tw = tw["teleworkable"].median()
            tw["teleworkable"] = (tw["teleworkable"] >= median_tw).astype(int)
            print(f"  Median-split at {median_tw:.3f}: "
                  f"{tw['teleworkable'].sum()} teleworkable, "
                  f"{(1 - tw['teleworkable']).sum()} non-teleworkable")
        return tw

    except FileNotFoundError:
        print(f"  Pre-built telework file not found at: {TELEWORK_PATH}")
        print(f"  Attempting to build from raw crosswalk files...")

    # --- Attempt 2: build from raw files ---
    return build_telework_mapping()


def build_telework_mapping():
    """
    Build SSYK4 -> teleworkable mapping from raw crosswalk files.

    Crosswalk chain: O*NET-SOC -> SOC 2010 -> ISCO-08 -> SSYK 2012
    (same approach as script 09_remote_work_robustness.py)

    Expected raw files on MONA:
      - dingel_neiman_telework.csv (from GitHub, O*NET-SOC level)
      - isco_soc_crosswalk2.xls   (BLS SOC 2010 -> ISCO-08)
      - ssyk2012_isco08.xlsx      (SCB SSYK 2012 -> ISCO-08)

    If these are not available, the function raises an error with
    instructions for Lydia to prepare the file.
    """
    # Possible locations for raw files on MONA
    raw_paths = [
        Path(r"\\micro.intra\Projekt\P1207$\P1207_Gem\Lydia P1207"),
        Path("."),
    ]

    dn_file = None
    bls_file = None
    scb_file = None

    for base in raw_paths:
        if dn_file is None and (base / "dingel_neiman_telework.csv").exists():
            dn_file = base / "dingel_neiman_telework.csv"
        if bls_file is None and (base / "isco_soc_crosswalk2.xls").exists():
            bls_file = base / "isco_soc_crosswalk2.xls"
        if scb_file is None and (base / "ssyk2012_isco08.xlsx").exists():
            scb_file = base / "ssyk2012_isco08.xlsx"

    if dn_file is None or bls_file is None or scb_file is None:
        missing = []
        if dn_file is None:
            missing.append("dingel_neiman_telework.csv")
        if bls_file is None:
            missing.append("isco_soc_crosswalk2.xls")
        if scb_file is None:
            missing.append("ssyk2012_isco08.xlsx")

        raise FileNotFoundError(
            f"Cannot build telework mapping -- missing files: {missing}\n\n"
            f"INSTRUCTIONS FOR LYDIA:\n"
            f"  Option A (recommended): Upload the pre-built file to:\n"
            f"    {TELEWORK_PATH}\n"
            f"  The file should have columns: ssyk4, teleworkable\n"
            f"  where teleworkable is 0 or 1 (median-split of Dingel-Neiman scores).\n"
            f"  Magnus has this file locally at:\n"
            f"    tables/telework_ssyk_mapping.csv\n\n"
            f"  Option B: Upload these three raw files to the network share:\n"
            f"    1. dingel_neiman_telework.csv (from GitHub: jdingel/DingelNeiman-workathome)\n"
            f"    2. isco_soc_crosswalk2.xls    (from BLS)\n"
            f"    3. ssyk2012_isco08.xlsx        (from SCB)\n"
        )

    # --- Build the crosswalk ---
    print(f"  Building crosswalk from raw files...")

    # 1. Dingel-Neiman: O*NET-SOC -> SOC 2010 (truncate)
    dn = pd.read_csv(dn_file)
    dn["soc2010"] = dn["onetsoccode"].str[:7]
    soc_tw = dn.groupby("soc2010")["teleworkable"].mean().reset_index()
    print(f"    Dingel-Neiman: {len(soc_tw)} SOC 2010 codes")

    # 2. BLS: SOC 2010 -> ISCO-08
    bls = pd.read_excel(bls_file, sheet_name="2010 SOC to ISCO-08",
                        header=None, skiprows=8)
    bls.columns = ["soc2010", "soc_title", "part", "isco08", "isco_title", "comment"]
    bls = bls[["soc2010", "isco08"]].dropna()
    bls["soc2010"] = bls["soc2010"].astype(str).str.strip()
    bls["isco08"] = bls["isco08"].astype(str).str.strip().str.zfill(4)
    print(f"    BLS crosswalk: {len(bls)} SOC-ISCO pairs")

    # 3. Merge SOC -> ISCO with telework scores
    soc_isco_tw = bls.merge(soc_tw, on="soc2010", how="inner")
    isco_tw = soc_isco_tw.groupby("isco08")["teleworkable"].mean().reset_index()
    print(f"    ISCO codes with telework score: {len(isco_tw)}")

    # 4. SCB: SSYK -> ISCO (invert to ISCO -> SSYK)
    scb = pd.read_excel(scb_file, sheet_name="Nyckel", header=None, skiprows=5)
    scb.columns = ["ssyk2012", "ssyk_title", "isco08", "isco_title", "notes"]
    scb = scb[["ssyk2012", "isco08"]].dropna()
    scb["ssyk2012"] = scb["ssyk2012"].astype(str).str.strip().str.zfill(4)
    scb["isco08"] = scb["isco08"].astype(str)
    scb = scb.assign(isco08=scb["isco08"].str.split(r",\s*")).explode("isco08")
    scb["isco08"] = scb["isco08"].str.strip().str.zfill(4)
    print(f"    SCB crosswalk: {len(scb)} ISCO-SSYK pairs")

    # 5. Merge ISCO -> SSYK with telework scores
    isco_ssyk_tw = scb.merge(isco_tw, on="isco08", how="inner")
    ssyk_tw = (
        isco_ssyk_tw.groupby("ssyk2012")["teleworkable"]
        .mean()
        .reset_index()
        .rename(columns={"ssyk2012": "ssyk4"})
    )
    print(f"    SSYK codes with telework score: {len(ssyk_tw)}")

    # 6. Median split
    median_tw = ssyk_tw["teleworkable"].median()
    ssyk_tw["teleworkable"] = (ssyk_tw["teleworkable"] >= median_tw).astype(int)
    print(f"    Median-split at {median_tw:.3f}: "
          f"{ssyk_tw['teleworkable'].sum()} teleworkable, "
          f"{(1 - ssyk_tw['teleworkable']).sum()} non-teleworkable")

    return ssyk_tw


# ======================================================================
#   STEP 1: LOAD DATA (same pipeline as scripts 14-18)
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
    Load AGI data year by year (same pipeline as scripts 14-18),
    merge DAIOE quartiles, assign age groups,
    aggregate to employer x quartile x age_group x month cells.

    IMPORTANT: This version also keeps ssyk4 in the pre-aggregation data
    so we can merge the telework classification before collapsing.
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

    # NOTE: We do NOT aggregate across ssyk4 yet -- we need ssyk4 for
    # the telework merge. Aggregation happens after the telework split.
    return df


def aggregate_to_panel(df):
    """
    Aggregate to employer x quartile x age_group x month cells.
    This is the standard aggregation step from scripts 15/18, factored
    out so we can call it on each telework subsample.
    """
    agg = (
        df.groupby(["employer_id", "exposure_quartile", "age_group", "year_month"])
        ["person_count"]
        .sum()
        .reset_index()
        .rename(columns={"person_count": "n_emp"})
    )
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
#   STEP 2: DiD ESTIMATION
# ======================================================================

def run_did_for_subsample(agg, subsample_label):
    """
    Run the main DiD for each age group on a given subsample (teleworkable
    or non-teleworkable). Same specification as script 15:

      ln(n_emp+1) = a_{f,q} + b_{f,t} + g1*PostRB*High + g2*PostGPT*High

    Returns a list of result dicts.
    """
    from linearmodels.panel import PanelOLS

    print(f"\n{'=' * 70}")
    print(f"DiD: {subsample_label}")
    print(f"{'=' * 70}")

    results = []

    for age_label in AGE_GROUPS:
        print(f"\n--- {subsample_label}, age {age_label} ---")
        t0 = time.time()

        # Build balanced panel
        sub = balance_panel_for_age(agg, age_label)
        if len(sub) < 100:
            print(f"  Too few observations, skipping")
            continue

        # Create treatment variables
        sub["post_rb"] = (sub["year_month"] >= RIKSBANK_YM).astype(int)
        sub["post_gpt"] = (sub["year_month"] >= CHATGPT_YM).astype(int)
        sub["high"] = (sub["exposure_quartile"] == 4).astype(int)
        sub["post_rb_x_high"] = sub["post_rb"] * sub["high"]
        sub["post_gpt_x_high"] = sub["post_gpt"] * sub["high"]

        # Log outcome
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

        print(f"  Observations: {len(sub):,}")
        print(f"  Employers: {sub['employer_id'].nunique():,}")
        print(f"  Zero cells: {(sub['n_emp'] == 0).sum():,}")

        # --- PanelOLS ---
        try:
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

            elapsed = time.time() - t0
            gamma1 = res.params["post_rb_x_high"]
            gamma2 = res.params["post_gpt_x_high"]
            se1 = res.std_errors["post_rb_x_high"]
            se2 = res.std_errors["post_gpt_x_high"]
            p1 = res.pvalues["post_rb_x_high"]
            p2 = res.pvalues["post_gpt_x_high"]

            print(f"  [{elapsed:.0f}s]")
            print(f"  g1 (PostRB x High)  = {gamma1:+.4f} (SE={se1:.4f}, p={p1:.4f})")
            print(f"  g2 (PostGPT x High) = {gamma2:+.4f} (SE={se2:.4f}, p={p2:.4f})")

            results.append({
                "subsample": subsample_label,
                "age_group": age_label,
                "n_obs": int(res.nobs),
                "n_employers": sub["employer_id"].nunique(),
                "gamma1_rb_high": gamma1,
                "se1": se1,
                "pval1": p1,
                "gamma2_gpt_high": gamma2,
                "se2": se2,
                "pval2": p2,
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    return results


# ======================================================================
#   STEP 3: EVENT STUDY
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


def run_event_study_for_subsample(agg, subsample_label):
    """
    Run the half-year event study for each age group on a given subsample.
    Same specification as script 18 (corrected FE):

      ln(n_emp+1) = a_{f,q} + b_{f,t} + sum_h(delta_h * I(h) * High)

    FE: employer x quartile (entity) + employer x month (other_effects)
    """
    from linearmodels.panel import PanelOLS

    print(f"\n{'=' * 70}")
    print(f"Event study: {subsample_label}")
    print(f"{'=' * 70}")

    # Compute half-year periods from the full dataset
    all_halfyears = sorted(set(assign_halfyear(ym)
                               for ym in agg["year_month"].unique()))
    event_periods = [p for p in all_halfyears if p != REF_HALFYEAR]

    print(f"  Half-year periods: {all_halfyears}")
    print(f"  Reference: {REF_HALFYEAR}")

    all_es_results = []
    pretrend_tests = []

    for age_label in AGE_GROUPS:
        print(f"\n--- {subsample_label}, age {age_label} (ES) ---")
        t0 = time.time()

        # Build balanced panel
        sub = balance_panel_for_age(agg, age_label)
        if len(sub) < 100:
            print(f"  Too few observations, skipping")
            continue

        # Variables
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

        # Interaction dummies
        for p in event_periods:
            sub[f"hy_{p}"] = ((sub["halfyear"] == p).astype(int) * sub["high"])

        interaction_cols = [f"hy_{p}" for p in event_periods]

        # --- PanelOLS ---
        try:
            panel = sub.copy()
            panel["date"] = pd.to_datetime(panel["year_month"] + "-01")
            panel = panel.set_index(["fe_emp_q", "date"])

            other_fe = pd.DataFrame(
                {"fe_emp_t": panel["fe_emp_t"]},
                index=panel.index,
            )

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
                    "subsample": subsample_label,
                    "age_group": age_label,
                    "period": p,
                    "coef": res.params[col],
                    "se": res.std_errors[col],
                    "pval": res.pvalues[col],
                })

            # Reference period (zero by construction)
            all_es_results.append({
                "subsample": subsample_label,
                "age_group": age_label,
                "period": REF_HALFYEAR,
                "coef": 0.0,
                "se": 0.0,
                "pval": 1.0,
            })

            # Pre-trend test
            pt = pretrend_joint_test(res, event_periods, REF_HALFYEAR)
            if pt:
                print(f"  Pre-trend: chi2({pt['df']}) = {pt['chi2']:.2f}, "
                      f"p = {pt['p_value']:.4f}")
                pretrend_tests.append({
                    "subsample": subsample_label,
                    "age_group": age_label,
                    **pt,
                })

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    return all_es_results, pretrend_tests


# ======================================================================
#   STEP 4: FIGURES
# ======================================================================

def plot_telework_comparison(es_df):
    """
    Plot event study comparison: teleworkable vs non-teleworkable,
    one panel per age group (2x3 grid).
    """
    if es_df.empty:
        return

    print(f"\n{'=' * 70}")
    print("STEP 4: Plotting telework comparison figures")
    print(f"{'=' * 70}")

    age_list = list(AGE_GROUPS.keys())
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=False)

    for idx, age_label in enumerate(age_list):
        ax = axes[idx // 3, idx % 3]

        for subsample, color, label_short in [
            ("Non-teleworkable", ORANGE, "Non-telework"),
            ("Teleworkable", TEAL, "Telework"),
        ]:
            sub = es_df[
                (es_df["age_group"] == age_label) &
                (es_df["subsample"] == subsample)
            ].copy()

            if sub.empty:
                continue

            sub = sub.sort_values("period")
            periods_sorted = sorted(sub["period"].unique())
            x_map = {p: i for i, p in enumerate(periods_sorted)}
            sub["x"] = sub["period"].map(x_map)

            # Offset x slightly to avoid overlap
            offset = -0.1 if subsample == "Non-teleworkable" else 0.1
            sub["x_off"] = sub["x"] + offset

            sub["ci_lo"] = sub["coef"] - 1.96 * sub["se"]
            sub["ci_hi"] = sub["coef"] + 1.96 * sub["se"]

            ax.fill_between(sub["x_off"], sub["ci_lo"], sub["ci_hi"],
                            alpha=0.15, color=color)
            ax.plot(sub["x_off"], sub["coef"], "o-", color=color,
                    linewidth=1.5, markersize=4, label=label_short)

        ax.axhline(0, color="black", linewidth=0.5)

        # Reference period marker
        ref_x = x_map.get(REF_HALFYEAR, None) if 'x_map' in dir() else None
        if ref_x is not None:
            ax.axvline(ref_x, color=GRAY, linestyle="--", linewidth=0.8)

        # Determine x-axis from any available subsample
        any_sub = es_df[es_df["age_group"] == age_label]
        if not any_sub.empty:
            ps = sorted(any_sub["period"].unique())
            ax.set_xticks(range(len(ps)))
            ax.set_xticklabels(ps, rotation=45, ha="right", fontsize=7)

        ax.set_title(age_label, fontsize=11)
        if idx == 0:
            ax.legend(fontsize=8, loc="lower left")

    fig.suptitle(
        "Event study by teleworkability (Dingel-Neiman 2020)\n"
        "FE: employer x quartile + employer x month",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "telework_comparison_panel.png", dpi=150)
    plt.close(fig)
    print("  Saved: telework_comparison_panel.png")

    # --- Individual age group figures ---
    for age_label in age_list:
        fig, ax = plt.subplots(figsize=(10, 5))

        for subsample, color, label_short in [
            ("Non-teleworkable", ORANGE, "Non-teleworkable"),
            ("Teleworkable", TEAL, "Teleworkable"),
        ]:
            sub = es_df[
                (es_df["age_group"] == age_label) &
                (es_df["subsample"] == subsample)
            ].copy()

            if sub.empty:
                continue

            sub = sub.sort_values("period")
            periods_sorted = sorted(sub["period"].unique())
            x_map = {p: i for i, p in enumerate(periods_sorted)}
            sub["x"] = sub["period"].map(x_map)

            offset = -0.1 if subsample == "Non-teleworkable" else 0.1
            sub["x_off"] = sub["x"] + offset

            sub["ci_lo"] = sub["coef"] - 1.96 * sub["se"]
            sub["ci_hi"] = sub["coef"] + 1.96 * sub["se"]

            ax.fill_between(sub["x_off"], sub["ci_lo"], sub["ci_hi"],
                            alpha=0.15, color=color)
            ax.plot(sub["x_off"], sub["coef"], "o-", color=color,
                    linewidth=2, markersize=6, label=label_short)

        ax.axhline(0, color="black", linewidth=0.5)
        ref_x = x_map.get(REF_HALFYEAR, None) if 'x_map' in dir() else None
        if ref_x is not None:
            ax.axvline(ref_x, color=GRAY, linestyle="--", linewidth=1,
                       label=f"Reference: {REF_HALFYEAR}")

        any_sub = es_df[es_df["age_group"] == age_label]
        if not any_sub.empty:
            ps = sorted(any_sub["period"].unique())
            ax.set_xticks(range(len(ps)))
            ax.set_xticklabels(ps, rotation=45, ha="right", fontsize=9)

        ax.set_ylabel("Coefficient (ln employment)", fontsize=11)
        ax.set_title(f"Event study: {age_label} -- telework split", fontsize=13)
        ax.legend(fontsize=9)

        fig.tight_layout()
        safe_label = age_label.replace("+", "plus")
        fig.savefig(OUTPUT_DIR / f"telework_es_{safe_label}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: telework_es_{safe_label}.png")


# ======================================================================
#   MAIN
# ======================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("26_mona_telework_split.py")
    print("DiD + event study split by Dingel-Neiman teleworkability")
    print("=" * 70)

    t_start = time.time()

    # Step 1: Load all data (keep ssyk4 for telework merge)
    df = load_and_prepare()

    # Step 1b: Load teleworkability mapping
    print("\n  Loading teleworkability mapping...")
    tw = load_telework_mapping()

    # Step 1c: Merge telework classification onto the micro data
    n_before = len(df)
    df = df.merge(tw[["ssyk4", "teleworkable"]], on="ssyk4", how="inner")
    n_after = len(df)
    match_rate = 100 * n_after / n_before if n_before > 0 else 0
    print(f"  Telework merge: {n_before:,} -> {n_after:,} records "
          f"({match_rate:.1f}% match rate)")

    # Summarise the split
    tw_count = df[df["teleworkable"] == 1]["person_count"].sum()
    ntw_count = df[df["teleworkable"] == 0]["person_count"].sum()
    print(f"  Teleworkable:     {tw_count:,} person-months")
    print(f"  Non-teleworkable: {ntw_count:,} person-months")

    # Step 1d: Split and aggregate each subsample separately
    df_tw = df[df["teleworkable"] == 1].copy()
    df_ntw = df[df["teleworkable"] == 0].copy()

    agg_tw = aggregate_to_panel(df_tw)
    agg_ntw = aggregate_to_panel(df_ntw)

    print(f"\n  Teleworkable panel:     {len(agg_tw):,} cells")
    print(f"  Non-teleworkable panel: {len(agg_ntw):,} cells")

    # Step 2: DiD for each subsample
    did_results = []
    did_results.extend(run_did_for_subsample(agg_tw, "Teleworkable"))
    did_results.extend(run_did_for_subsample(agg_ntw, "Non-teleworkable"))

    if did_results:
        did_df = pd.DataFrame(did_results)
        did_df.to_csv(OUTPUT_DIR / "telework_did_results.csv", index=False)
        print(f"\n  Saved DiD results -> telework_did_results.csv")
        print("\n  === DiD SUMMARY ===")
        print(did_df.to_string(index=False))

    # Step 3: Event study for each subsample
    es_results_tw, pt_tw = run_event_study_for_subsample(agg_tw, "Teleworkable")
    es_results_ntw, pt_ntw = run_event_study_for_subsample(agg_ntw, "Non-teleworkable")

    all_es = es_results_tw + es_results_ntw
    all_pt = pt_tw + pt_ntw

    if all_es:
        es_df = pd.DataFrame(all_es)
        es_df.to_csv(OUTPUT_DIR / "telework_es_all.csv", index=False)
        print(f"\n  Saved event study coefficients -> telework_es_all.csv")

    if all_pt:
        pt_df = pd.DataFrame(all_pt)
        pt_df.to_csv(OUTPUT_DIR / "telework_pretrends.csv", index=False)
        print(f"  Saved pre-trend tests -> telework_pretrends.csv")

    # Step 4: Figures
    if all_es:
        plot_telework_comparison(es_df)

    # Step 5: Summary
    elapsed = time.time() - t_start
    summary_lines = [
        "=" * 60,
        "TELEWORK SPLIT SUMMARY",
        "=" * 60,
        f"Script: 26_mona_telework_split.py",
        f"Telework classification: Dingel-Neiman (2020), median-split",
        f"FE: employer x quartile (entity) + employer x month (other_effects)",
        f"Runtime: {elapsed/60:.1f} minutes",
        "",
    ]

    if did_results:
        summary_lines.append("--- DiD RESULTS (gamma2 = Post-ChatGPT x High) ---")
        summary_lines.append(f"{'Subsample':<20s} {'Age':>6s} {'g2':>8s} "
                             f"{'SE':>8s} {'p':>8s} {'N':>10s}")
        for r in did_results:
            sig = "***" if r["pval2"] < 0.01 else "**" if r["pval2"] < 0.05 \
                  else "*" if r["pval2"] < 0.10 else ""
            summary_lines.append(
                f"{r['subsample']:<20s} {r['age_group']:>6s} "
                f"{r['gamma2_gpt_high']:>+8.4f}{sig:<3s} "
                f"({r['se2']:.4f}) "
                f"{r['pval2']:>8.4f} {r['n_obs']:>10,d}"
            )
        summary_lines.append("")

    if all_pt:
        summary_lines.append("--- PRE-TREND TESTS ---")
        for pt in all_pt:
            status = "PASS" if pt["p_value"] > 0.05 else "FAIL"
            summary_lines.append(
                f"  {pt['subsample']:<20s} {pt['age_group']:>6s}: "
                f"chi2({pt['df']}) = {pt['chi2']:>7.2f}, "
                f"p = {pt['p_value']:.4f}  [{status}]"
            )

    summary_text = "\n".join(summary_lines)
    (OUTPUT_DIR / "telework_summary.txt").write_text(summary_text)
    print(f"\n{summary_text}")
    print(f"\n  All output in: {OUTPUT_DIR}/")
