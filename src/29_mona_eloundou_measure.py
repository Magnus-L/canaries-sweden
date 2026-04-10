#!/usr/bin/env python3
"""
29_mona_eloundou_measure.py -- Robustness: Eloundou GPT exposure instead of DAIOE.

======================================================================
  THIS SCRIPT IS DESIGNED TO RUN IN SCB's MONA SECURE ENVIRONMENT
  Do NOT run outside MONA -- the data is not available externally.
======================================================================

WHY THIS SCRIPT EXISTS:
  The main employment results (scripts 15, 18) use DAIOE quartiles to
  classify occupations by AI exposure. This script replaces DAIOE with
  the Eloundou et al. (2024) GPT exposure score to test whether the
  employment canaries result is robust across AI exposure measures.

  If results hold with Eloundou exposure, the finding is not an artifact
  of our particular exposure measure (DAIOE). This is essential given
  the variance across AI exposure metrics documented by Gimbel et al. (2026).

SPECIFICATION:
  Identical to script 18 (corrected FE event study):
    entity FE = employer x quartile  (PanelOLS entity_effects)
    other FE  = employer x month     (PanelOLS other_effects)

  Only change: exposure quartiles come from Eloundou GPT scores
  (crosswalked SOC -> ISCO -> SSYK) instead of DAIOE.

EXPOSURE FILE:
  Tries to load from:
    1. \\\\micro.intra\\Projekt\\P1207$\\P1207_Gem\\Lydia P1207\\eloundou_ssyk4.csv
  If not found, tries to build from raw Eloundou scores + crosswalk files.
  If that also fails, prints instructions for Lydia.

  Expected columns: ssyk4 (4-digit SSYK code), eloundou_quartile (1-4).

OUTPUT FILES:
  1. eloundou_es_all.csv          -- event study coefficients
  2. eloundou_did_results.csv     -- DiD coefficients
  3. eloundou_pretrends.txt       -- pre-trend joint F-test results
  4. eloundou_es_*.png            -- figures per age group
  5. eloundou_comparison.txt      -- notes for comparison with DAIOE results
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

# --- MONA SQL connection (same as scripts 14-15, 18) ---
import pyodbc
conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=monasql.micro.intra;"
    "DATABASE=P1207;"
    "Trusted_Connection=yes;"
)

# --- Eloundou exposure file (crosswalked to SSYK4) ---
ELOUNDOU_PATH = r"\\micro.intra\Projekt\P1207$\P1207_Gem\Lydia P1207\eloundou_ssyk4.csv"

# --- Fallback: raw Eloundou scores + crosswalk files ---
ELOUNDOU_RAW_PATH = r"\\micro.intra\Projekt\P1207$\P1207_Gem\Lydia P1207\eloundou_raw.csv"
SOC_ISCO_CROSSWALK = r"\\micro.intra\Projekt\P1207$\P1207_Gem\Lydia P1207\soc_isco_crosswalk.csv"
ISCO_SSYK_CROSSWALK = r"\\micro.intra\Projekt\P1207$\P1207_Gem\Lydia P1207\isco_ssyk_crosswalk.csv"

# --- Output directory ---
OUTPUT_DIR = Path("output_29")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Treatment dates ---
RIKSBANK_YM = "2022-04"
CHATGPT_YM = "2022-12"

# --- Reference period (omitted dummy) ---
REF_HALFYEAR = "2022H1"

# --- Age group definitions (same as script 15 / Lydia's runs) ---
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


# ======================================================================
#   ELOUNDOU EXPOSURE LOADING
# ======================================================================

def load_eloundou_quartiles():
    """
    Load Eloundou GPT exposure quartiles at the SSYK4 level.

    Strategy:
      1. Try pre-built file (eloundou_ssyk4.csv)
      2. If not found, try building from raw scores + crosswalks
      3. If that fails, print instructions for Lydia and exit

    Returns
    -------
    pd.DataFrame with columns: ssyk4 (str, zero-padded), exposure_quartile (int 1-4)
    """
    # --- Strategy 1: Pre-built file ---
    eloundou_path = Path(ELOUNDOU_PATH)
    if eloundou_path.exists():
        print(f"  Loading Eloundou quartiles from: {ELOUNDOU_PATH}")
        df = pd.read_csv(ELOUNDOU_PATH)
        df["ssyk4"] = df["ssyk4"].astype(str).str.zfill(4)

        # Handle column naming flexibility
        if "eloundou_quartile" in df.columns:
            df = df.rename(columns={"eloundou_quartile": "exposure_quartile"})
        elif "exposure_quartile" not in df.columns:
            # Try to find any quartile column
            q_cols = [c for c in df.columns if "quartile" in c.lower()]
            if q_cols:
                df = df.rename(columns={q_cols[0]: "exposure_quartile"})
            else:
                raise ValueError(
                    f"Cannot find quartile column in {ELOUNDOU_PATH}. "
                    f"Columns: {list(df.columns)}"
                )

        # Ensure integer quartiles
        if df["exposure_quartile"].dtype == object:
            q_map = {"Q1 (lowest)": 1, "Q2": 2, "Q3": 3, "Q4 (highest)": 4,
                     "Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
            df["exposure_quartile"] = df["exposure_quartile"].map(q_map)

        n_ssyk = df["ssyk4"].nunique()
        print(f"  Loaded {n_ssyk} SSYK4 codes with Eloundou quartiles")
        print(f"  Quartile distribution:\n{df['exposure_quartile'].value_counts().sort_index()}")
        return df[["ssyk4", "exposure_quartile"]].drop_duplicates()

    # --- Strategy 2: Build from raw scores + crosswalks ---
    print(f"  Pre-built file not found: {ELOUNDOU_PATH}")
    print(f"  Attempting to build from raw scores + crosswalks...")

    raw_path = Path(ELOUNDOU_RAW_PATH)
    soc_isco_path = Path(SOC_ISCO_CROSSWALK)
    isco_ssyk_path = Path(ISCO_SSYK_CROSSWALK)

    if raw_path.exists() and soc_isco_path.exists() and isco_ssyk_path.exists():
        print(f"  Found all three files, building crosswalk...")

        # Load raw Eloundou scores (expected: SOC code + exposure score)
        raw = pd.read_csv(ELOUNDOU_RAW_PATH)
        print(f"  Raw Eloundou columns: {list(raw.columns)}")

        # Identify SOC and score columns (flexible naming)
        soc_col = None
        score_col = None
        for c in raw.columns:
            cl = c.lower()
            if "soc" in cl or "onet" in cl or "o_net" in cl:
                soc_col = c
            if "exposure" in cl or "score" in cl or "beta" in cl or "zeta" in cl:
                score_col = c
        if soc_col is None or score_col is None:
            print(f"  ERROR: Cannot identify SOC/score columns in raw file.")
            print(f"  Columns: {list(raw.columns)}")
            _print_instructions()
            raise SystemExit(1)

        # Load crosswalks
        soc_isco = pd.read_csv(SOC_ISCO_CROSSWALK)
        isco_ssyk = pd.read_csv(ISCO_SSYK_CROSSWALK)

        # Standardise SOC codes (remove dots/dashes)
        raw[soc_col] = raw[soc_col].astype(str).str.replace(r"[\.\-]", "", regex=True)

        # Identify crosswalk columns
        soc_cw_col = [c for c in soc_isco.columns if "soc" in c.lower()][0]
        isco_cw_col_1 = [c for c in soc_isco.columns if "isco" in c.lower()][0]
        soc_isco[soc_cw_col] = soc_isco[soc_cw_col].astype(str).str.replace(
            r"[\.\-]", "", regex=True)

        isco_cw_col_2 = [c for c in isco_ssyk.columns if "isco" in c.lower()][0]
        ssyk_cw_col = [c for c in isco_ssyk.columns if "ssyk" in c.lower()][0]

        # Merge: SOC -> ISCO -> SSYK
        merged = raw.merge(soc_isco, left_on=soc_col, right_on=soc_cw_col, how="inner")
        merged = merged.merge(isco_ssyk, left_on=isco_cw_col_1,
                              right_on=isco_cw_col_2, how="inner")

        # Average exposure within SSYK4
        merged[ssyk_cw_col] = merged[ssyk_cw_col].astype(str).str.zfill(4)
        ssyk_scores = (
            merged.groupby(ssyk_cw_col)[score_col]
            .mean()
            .reset_index()
            .rename(columns={ssyk_cw_col: "ssyk4", score_col: "eloundou_score"})
        )

        # Assign quartiles
        ssyk_scores["exposure_quartile"] = pd.qcut(
            ssyk_scores["eloundou_score"], 4, labels=[1, 2, 3, 4]
        ).astype(int)

        n_ssyk = ssyk_scores["ssyk4"].nunique()
        print(f"  Built Eloundou quartiles for {n_ssyk} SSYK4 codes")
        print(f"  Quartile distribution:\n"
              f"{ssyk_scores['exposure_quartile'].value_counts().sort_index()}")

        # Save for future use
        ssyk_scores[["ssyk4", "exposure_quartile"]].to_csv(
            ELOUNDOU_PATH, index=False)
        print(f"  Saved to: {ELOUNDOU_PATH}")

        return ssyk_scores[["ssyk4", "exposure_quartile"]]

    # --- Strategy 3: Print instructions ---
    _print_instructions()
    raise SystemExit(1)


def _print_instructions():
    """Print instructions for Lydia to prepare the Eloundou file."""
    print("\n" + "=" * 70)
    print("INSTRUCTIONS FOR LYDIA: Prepare Eloundou exposure file")
    print("=" * 70)
    print("""
The script needs a file mapping SSYK4 codes to Eloundou GPT exposure quartiles.

OPTION A (preferred): Create the file outside MONA and import it.
  1. Download Eloundou et al. (2024) exposure scores from:
     https://github.com/openai/gpt-exposure
     (or from the replication package)
  2. Use the BLS SOC -> ISCO crosswalk (in data/raw/ on the local machine)
  3. Use the SCB SSYK -> ISCO crosswalk (in data/raw/ on the local machine)
  4. Crosswalk: SOC -> ISCO -> SSYK4
  5. Average exposure scores within each SSYK4 code
  6. Assign quartiles (pd.qcut into 4 groups, 1=lowest, 4=highest)
  7. Save as CSV with columns: ssyk4, eloundou_quartile
  8. Import to MONA at:
     \\\\micro.intra\\Projekt\\P1207$\\P1207_Gem\\Lydia P1207\\eloundou_ssyk4.csv

OPTION B: Place the raw files on the MONA share.
  Put these three files in \\\\micro.intra\\Projekt\\P1207$\\P1207_Gem\\Lydia P1207\\:
  1. eloundou_raw.csv  (columns: soc_code, exposure_score)
  2. soc_isco_crosswalk.csv  (columns: soc_code, isco_code)
  3. isco_ssyk_crosswalk.csv  (columns: isco_code, ssyk4)
  The script will then build the quartiles automatically.

FILE FORMAT for eloundou_ssyk4.csv:
  ssyk4,eloundou_quartile
  0110,2
  0210,3
  1111,4
  ...
  (4-digit SSYK, quartile 1-4 where 4 = highest GPT exposure)
""")


# ======================================================================
#   STEP 1: LOAD DATA (same pipeline as script 18)
# ======================================================================

def pull_year(year, conn):
    """
    Pull one year of AGI data with cascading SSYK lookup (2023 -> 2022 -> 2021).
    Same approach as scripts 14, 15, 18 for data consistency.
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
    Load AGI data year by year (same pipeline as script 18),
    merge ELOUNDOU quartiles (instead of DAIOE), assign age groups,
    aggregate to employer x quartile x age_group x month cells.
    """
    print("=" * 70)
    print("STEP 1: Loading and preparing data")
    print("  EXPOSURE MEASURE: Eloundou GPT exposure (not DAIOE)")
    print("=" * 70)

    # Pull year by year (consistent with script 18)
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

    # Merge ELOUNDOU quartiles (KEY DIFFERENCE from script 18)
    print("  Loading Eloundou GPT exposure quartiles...")
    eloundou = load_eloundou_quartiles()
    df = df.merge(eloundou[["ssyk4", "exposure_quartile"]], on="ssyk4", how="inner")
    print(f"  After Eloundou merge: {len(df):,} records")

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
#   STEP 2: CORRECTED EVENT STUDY
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


def run_corrected_event_study(agg, ref_halfyear=None):
    """
    Half-year event study with the CORRECT FE specification:
      entity FE = employer x quartile  (manual within-transformation)
      other FE  = employer x month     (manual within-transformation)

    Uses manual double-demeaning (Frisch-Waugh-Lovell) instead of
    PanelOLS, because PanelOLS with high-dimensional other_effects
    hangs silently on large panels without raising an exception
    (same issue fixed in scripts 15 and 16). The within-transformation
    is mathematically equivalent and scales via vectorised pandas groupby.

    Parameters
    ----------
    ref_halfyear : str, optional
        Reference period to omit (e.g. "2022H1" or "2021H2").
        Defaults to global REF_HALFYEAR.
    """
    import statsmodels.api as sm

    if ref_halfyear is None:
        ref_halfyear = REF_HALFYEAR

    print("\n" + "=" * 70)
    print(f"STEP 2: Corrected event study (ref = {ref_halfyear})")
    print("  FE: employer x quartile + employer x month")
    print("  EXPOSURE MEASURE: Eloundou GPT exposure")
    print("=" * 70)

    agg = agg.copy()

    # Compute half-year labels on raw agg (needed for event period list)
    agg["halfyear"] = agg["year_month"].apply(assign_halfyear)
    all_periods = sorted(agg["halfyear"].unique())
    event_periods = [p for p in all_periods if p != ref_halfyear]

    print(f"  Half-year periods: {all_periods}")
    print(f"  Reference period: {ref_halfyear}")
    print(f"  Event dummies: {len(event_periods)}")

    all_es_results = []
    pretrend_tests = []

    for age_label in AGE_GROUPS:
        print(f"\n--- Event study: {age_label} ---")
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

        # --- Manual double-demean (replaces PanelOLS to avoid silent hang) ---
        # Step 1: demean by employer x quartile (entity FE)
        # Step 2: demean again by employer x month (other FE)
        # FWL theorem says this is equivalent to including both FE in OLS.
        panel = sub.copy()

        cols_to_demean = ["ln_emp"] + interaction_cols
        for col in cols_to_demean:
            panel[f"{col}_dm1"] = panel.groupby("fe_emp_q")[col].transform(
                lambda x: x - x.mean()
            )
        for col in cols_to_demean:
            panel[f"{col}_dm"] = panel.groupby("fe_emp_t")[f"{col}_dm1"].transform(
                lambda x: x - x.mean()
            )

        try:
            y = panel["ln_emp_dm"].values
            # Pass X as DataFrame so statsmodels preserves column names
            X = panel[[f"{c}_dm" for c in interaction_cols]].copy()
            X.columns = interaction_cols  # restore original names for indexing
            mod = sm.OLS(y, X)
            res = mod.fit(
                cov_type="cluster",
                cov_kwds={"groups": panel["employer_id"].values},
            )

            elapsed = time.time() - t0
            print(f"  Estimated [within-transformation] in {elapsed:.0f}s")

            # Collect coefficients (statsmodels uses res.bse, not res.std_errors)
            for p in event_periods:
                col = f"hy_{p}"
                all_es_results.append({
                    "age_group": age_label,
                    "period": p,
                    "coef": res.params[col],
                    "se": res.bse[col],
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

            # Pre-trend joint test
            pt = pretrend_joint_test(res, event_periods, ref_halfyear)
            if pt:
                print(f"  Pre-trend test: chi2({pt['df']}) = {pt['chi2']:.2f}, "
                      f"p = {pt['p_value']:.4f}")
                pretrend_tests.append({"age_group": age_label, **pt})

        except Exception as e:
            print(f"  ERROR: {e}")
            print("  This age group failed -- check memory/data.")
            continue

    # --- Save results ---
    suffix = f"_ref{ref_halfyear}"
    if all_es_results:
        es_df = pd.DataFrame(all_es_results)
        fname = f"eloundou_es_all{suffix}.csv"
        es_df.to_csv(OUTPUT_DIR / fname, index=False)
        print(f"\n  Saved event study coefficients -> {fname}")

    if pretrend_tests:
        pt_df = pd.DataFrame(pretrend_tests)
        pt_df.to_csv(OUTPUT_DIR / f"eloundou_pretrends{suffix}.csv", index=False)
        pt_path = OUTPUT_DIR / f"eloundou_pretrends{suffix}.txt"
        pt_df.to_string(pt_path, index=False)
        print(f"  Saved pre-trend tests -> eloundou_pretrends{suffix}.txt")
        print("\n  === PRE-TREND TEST RESULTS ===")
        print(pt_df.to_string(index=False))

    return es_df if all_es_results else pd.DataFrame(), pretrend_tests


# ======================================================================
#   STEP 3: FIGURES
# ======================================================================

def plot_event_studies(es_df):
    """One figure per age group, plus a combined panel."""
    if es_df.empty:
        return

    print("\n" + "=" * 70)
    print("STEP 3: Plotting event studies")
    print("=" * 70)

    for age_label in AGE_GROUPS:
        sub = es_df[es_df["age_group"] == age_label].copy()
        if sub.empty:
            continue

        sub = sub.sort_values("period")

        # Numeric x-axis
        periods_sorted = sorted(sub["period"].unique())
        x_map = {p: i for i, p in enumerate(periods_sorted)}
        sub["x"] = sub["period"].map(x_map)
        ref_x = x_map.get(REF_HALFYEAR, None)

        fig, ax = plt.subplots(figsize=(10, 5))

        # Confidence intervals
        sub["ci_lo"] = sub["coef"] - 1.96 * sub["se"]
        sub["ci_hi"] = sub["coef"] + 1.96 * sub["se"]

        ax.fill_between(sub["x"], sub["ci_lo"], sub["ci_hi"],
                        alpha=0.2, color=TEAL)
        ax.plot(sub["x"], sub["coef"], "o-", color=TEAL, linewidth=2,
                markersize=6)
        ax.axhline(0, color="black", linewidth=0.5)

        # Reference period marker
        if ref_x is not None:
            ax.axvline(ref_x, color=GRAY, linestyle="--", linewidth=1,
                       label=f"Reference: {REF_HALFYEAR}")

        ax.set_xticks(range(len(periods_sorted)))
        ax.set_xticklabels(periods_sorted, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Coefficient (ln employment)", fontsize=11)
        ax.set_title(f"Event study: {age_label} (Eloundou GPT exposure)",
                     fontsize=13)
        ax.legend(fontsize=9)

        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"eloundou_es_{age_label.replace('+','plus')}.png",
                    dpi=150)
        plt.close(fig)
        print(f"  Saved figure for {age_label}")

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
        ref_x = x_map.get(REF_HALFYEAR, None)

        sub["ci_lo"] = sub["coef"] - 1.96 * sub["se"]
        sub["ci_hi"] = sub["coef"] + 1.96 * sub["se"]

        ax.fill_between(sub["x"], sub["ci_lo"], sub["ci_hi"],
                        alpha=0.2, color=TEAL)
        ax.plot(sub["x"], sub["coef"], "o-", color=TEAL, linewidth=1.5,
                markersize=4)
        ax.axhline(0, color="black", linewidth=0.5)
        if ref_x is not None:
            ax.axvline(ref_x, color=GRAY, linestyle="--", linewidth=0.8)

        ax.set_xticks(range(len(periods_sorted)))
        ax.set_xticklabels(periods_sorted, rotation=45, ha="right", fontsize=7)
        ax.set_title(age_label, fontsize=11)

    fig.suptitle("Eloundou GPT exposure: event study "
                 "(employer x quartile + employer x month FE)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "eloundou_es_panel.png", dpi=150)
    plt.close(fig)
    print("  Saved combined panel figure")


# ======================================================================
#   STEP 4: DiD (simple post-treatment dummy, same as script 15)
# ======================================================================

def run_did(agg):
    """
    Simple DiD with post-treatment dummy (post x high_exposure).
    Same specification as script 15 Step 2, but using Eloundou exposure.

    Uses manual double-demeaning instead of PanelOLS for the same reason
    as run_corrected_event_study above (silent hangs on large panels).
    """
    import statsmodels.api as sm

    print("\n" + "=" * 70)
    print("STEP 4: DiD (post x high_exposure)")
    print("  FE: employer x quartile + employer x month")
    print("  EXPOSURE MEASURE: Eloundou GPT exposure")
    print("=" * 70)

    agg = agg.copy()
    agg["halfyear"] = agg["year_month"].apply(assign_halfyear)

    # Post = after ChatGPT (2022H2 onwards)
    agg["post"] = (agg["year_month"] >= "2022-07").astype(int)

    did_results = []

    for age_label in AGE_GROUPS:
        print(f"\n--- DiD: {age_label} ---")
        t0 = time.time()

        sub = balance_panel_for_age(agg, age_label)
        if len(sub) < 100:
            print(f"  Too few observations, skipping")
            continue

        sub["ln_emp"] = np.log(sub["n_emp"] + 1)
        sub["high"] = (sub["exposure_quartile"] == 4).astype(int)
        sub["post"] = (sub["year_month"] >= "2022-07").astype(int)
        sub["post_high"] = sub["post"] * sub["high"]
        sub["fe_emp_q"] = (
            sub["employer_id"].astype(str) + "_" +
            sub["exposure_quartile"].astype(str)
        )
        sub["fe_emp_t"] = (
            sub["employer_id"].astype(str) + "_" +
            sub["year_month"]
        )

        # --- Manual double-demean (FWL) — replaces PanelOLS to avoid hang ---
        panel = sub.copy()

        for col in ["ln_emp", "post_high"]:
            panel[f"{col}_dm1"] = panel.groupby("fe_emp_q")[col].transform(
                lambda x: x - x.mean()
            )
        for col in ["ln_emp", "post_high"]:
            panel[f"{col}_dm"] = panel.groupby("fe_emp_t")[f"{col}_dm1"].transform(
                lambda x: x - x.mean()
            )

        try:
            y = panel["ln_emp_dm"].values
            X = panel[["post_high_dm"]].copy()
            X.columns = ["post_high"]  # restore name for indexing
            mod = sm.OLS(y, X)
            res = mod.fit(
                cov_type="cluster",
                cov_kwds={"groups": panel["employer_id"].values},
            )

            elapsed = time.time() - t0
            print(f"  Estimated [within-transformation] in {elapsed:.0f}s")
            print(f"  post_high: {res.params['post_high']:.4f} "
                  f"(SE: {res.bse['post_high']:.4f}, "
                  f"p: {res.pvalues['post_high']:.4f})")

            did_results.append({
                "age_group": age_label,
                "coef": res.params["post_high"],
                "se": res.bse["post_high"],
                "pval": res.pvalues["post_high"],
                "n_obs": int(res.nobs),
                "n_entities": int(panel["employer_id"].nunique()),
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    if did_results:
        did_df = pd.DataFrame(did_results)
        did_df.to_csv(OUTPUT_DIR / "eloundou_did_results.csv", index=False)
        print(f"\n  Saved DiD results -> eloundou_did_results.csv")
        print("\n  === DiD RESULTS (Eloundou exposure) ===")
        print(did_df.to_string(index=False))

    return did_results


# ======================================================================
#   MAIN
# ======================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("29_mona_eloundou_measure.py")
    print("Robustness: Eloundou GPT exposure instead of DAIOE")
    print("Event study with employer x quartile + employer x month FE")
    print("(matching the main DiD specification)")
    print("=" * 70)

    t_start = time.time()

    # Step 1: Load data (with Eloundou exposure)
    agg = load_and_prepare()

    # Step 2a: Primary event study (reference = 2022H1)
    es_df, pt_tests = run_corrected_event_study(agg, ref_halfyear="2022H1")
    plot_event_studies(es_df)

    # Step 2b: Robustness -- alternative reference period (2021H2)
    print("\n" + "=" * 70)
    print("ROBUSTNESS: Re-running with reference period 2021H2")
    print("=" * 70)
    es_df_alt, pt_tests_alt = run_corrected_event_study(agg, ref_halfyear="2021H2")

    # Step 3: DiD
    did_results = run_did(agg)

    elapsed = time.time() - t_start
    print(f"\n  Total runtime: {elapsed/60:.1f} minutes")

    # Comparison summary
    summary = [
        "=" * 60,
        "ELOUNDOU GPT EXPOSURE -- COMPARISON WITH DAIOE",
        "=" * 60,
        f"Script: 29_mona_eloundou_measure.py",
        f"Exposure measure: Eloundou et al. (2024) GPT exposure scores",
        f"Crosswalk: SOC -> ISCO -> SSYK4",
        f"FE: employer x quartile (entity) + employer x month (other_effects)",
        f"Runtime: {elapsed/60:.1f} minutes",
        "",
        "RATIONALE:",
        "  Gimbel et al. (2026) show that AI exposure metrics agree on which",
        "  occupations have LOW exposure but disagree on magnitude at the top.",
        "  If the canaries result holds with Eloundou (a different measure",
        "  from a different methodology), it is robust to measure choice.",
        "",
        "--- PRIMARY EVENT STUDY (ref = 2022H1) ---",
        "PRE-TREND TESTS (joint chi2, H0: all pre-period coefficients = 0):",
    ]
    for pt in pt_tests:
        status = "PASS" if pt["p_value"] > 0.05 else "FAIL"
        summary.append(
            f"  {pt['age_group']:>8s}: chi2({pt['df']}) = {pt['chi2']:>7.2f}, "
            f"p = {pt['p_value']:.4f}  [{status}]"
        )
    summary.append("")
    summary.append("--- ROBUSTNESS EVENT STUDY (ref = 2021H2) ---")
    summary.append("PRE-TREND TESTS:")
    for pt in pt_tests_alt:
        status = "PASS" if pt["p_value"] > 0.05 else "FAIL"
        summary.append(
            f"  {pt['age_group']:>8s}: chi2({pt['df']}) = {pt['chi2']:>7.2f}, "
            f"p = {pt['p_value']:.4f}  [{status}]"
        )
    summary.append("")
    summary.append("DiD RESULTS (post = 2022H2+):")
    for dr in did_results:
        sig = "***" if dr["pval"] < 0.01 else "**" if dr["pval"] < 0.05 else "*" if dr["pval"] < 0.1 else ""
        summary.append(
            f"  {dr['age_group']:>8s}: coef = {dr['coef']:>8.4f} "
            f"(SE = {dr['se']:.4f}, p = {dr['pval']:.4f}) {sig}"
        )
    summary.append("")
    summary.append("INTERPRETATION:")
    summary.append("  Compare sign, magnitude, and significance with DAIOE results")
    summary.append("  (script 18 / output_18). Key questions:")
    summary.append("  1. Does the 22-25 age group still show negative effect?")
    summary.append("  2. Is the 50+ group still positive or null?")
    summary.append("  3. Does the age gradient (young negative, old positive) hold?")
    summary.append("  4. Are pre-trends comparable?")
    summary.append("")
    summary.append("  If patterns match: result is robust to exposure measure.")
    summary.append("  If patterns differ: discuss which occupations are classified")
    summary.append("  differently by DAIOE vs Eloundou and why.")

    summary_text = "\n".join(summary)
    (OUTPUT_DIR / "eloundou_comparison.txt").write_text(summary_text)
    print(f"\n{summary_text}")
    print(f"\n  All output in: {OUTPUT_DIR}/")
