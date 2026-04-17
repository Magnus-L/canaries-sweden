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
  Identical to script 15 (employer-level DiD):
    ln(n_emp_{f,q,t} + 1) = a_{f,q} + b_{f,t}
                             + g1*PostRB_t*High_q
                             + g2*PostGPT_t*High_q + e

  where f = employer, q = exposure quartile, t = month.
  Only change: quartiles from Eloundou GPT scores (SOC -> ISCO -> SSYK)
  instead of DAIOE.

  Event study skipped -- Lydia already has that output from a prior run.

ESTIMATOR (mirrors script 15's three-approach fallback):
  A. linearmodels PanelOLS with employer x quartile entity FE +
     employer x month as absorbed other_effects
  B. Manual within-transformation (double-demean, then OLS)
  C. Occupation-level backup (weaker identification)

EXPOSURE FILE:
  Tries to load from:
    1. \\\\micro.intra\\Projekt\\P1207$\\P1207_Gem\\Lydia P1207\\eloundou_ssyk4.csv
  If not found, tries to build from raw Eloundou scores + crosswalk files.
  If that also fails, prints instructions for Lydia.

  Expected columns: ssyk4 (4-digit SSYK code), eloundou_quartile (1-4).

OUTPUT FILES:
  1. eloundou_did_results.csv     -- DiD coefficients by age group
  2. eloundou_comparison.txt      -- summary for comparison with DAIOE
"""

import sys
import time
from pathlib import Path

# --- _Tee logger: BatchClient does not capture stdout/stderr ---
class _Tee:
    def __init__(self, log_path):
        self._stdout = sys.stdout
        self._log = open(log_path, "w", encoding="utf-8", buffering=1)
    def write(self, text):
        try:
            self._stdout.write(text)
        except UnicodeEncodeError:
            self._stdout.write(text.encode("cp1252", errors="replace").decode("cp1252"))
        self._log.write(text)
        self._log.flush()
    def flush(self):
        self._stdout.flush()
        self._log.flush()

_SCRIPT_DIR = Path(__file__).resolve().parent
_LOG_PATH = _SCRIPT_DIR / f"{Path(__file__).stem}_log.txt"
try:
    sys.stdout = _Tee(str(_LOG_PATH))
    sys.stderr = sys.stdout
    print(f"Log: {_LOG_PATH}")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"sys.argv: {sys.argv}")
except Exception as e:
    print(f"WARNING: Could not create log file: {e}")

import pandas as pd
import numpy as np
import warnings
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
#   CONFIGURATION
# ======================================================================

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

# --- Age group definitions (same as script 15) ---
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

        if "eloundou_quartile" in df.columns:
            df = df.rename(columns={"eloundou_quartile": "exposure_quartile"})
        elif "exposure_quartile" not in df.columns:
            q_cols = [c for c in df.columns if "quartile" in c.lower()]
            if q_cols:
                df = df.rename(columns={q_cols[0]: "exposure_quartile"})
            else:
                raise ValueError(
                    f"Cannot find quartile column in {ELOUNDOU_PATH}. "
                    f"Columns: {list(df.columns)}"
                )

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

        raw = pd.read_csv(ELOUNDOU_RAW_PATH)
        print(f"  Raw Eloundou columns: {list(raw.columns)}")

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

        soc_isco = pd.read_csv(SOC_ISCO_CROSSWALK)
        isco_ssyk = pd.read_csv(ISCO_SSYK_CROSSWALK)

        raw[soc_col] = raw[soc_col].astype(str).str.replace(r"[\.\-]", "", regex=True)

        soc_cw_col = [c for c in soc_isco.columns if "soc" in c.lower()][0]
        isco_cw_col_1 = [c for c in soc_isco.columns if "isco" in c.lower()][0]
        soc_isco[soc_cw_col] = soc_isco[soc_cw_col].astype(str).str.replace(
            r"[\.\-]", "", regex=True)

        isco_cw_col_2 = [c for c in isco_ssyk.columns if "isco" in c.lower()][0]
        ssyk_cw_col = [c for c in isco_ssyk.columns if "ssyk" in c.lower()][0]

        merged = raw.merge(soc_isco, left_on=soc_col, right_on=soc_cw_col, how="inner")
        merged = merged.merge(isco_ssyk, left_on=isco_cw_col_1,
                              right_on=isco_cw_col_2, how="inner")

        merged[ssyk_cw_col] = merged[ssyk_cw_col].astype(str).str.zfill(4)
        ssyk_scores = (
            merged.groupby(ssyk_cw_col)[score_col]
            .mean()
            .reset_index()
            .rename(columns={ssyk_cw_col: "ssyk4", score_col: "eloundou_score"})
        )

        ssyk_scores["exposure_quartile"] = pd.qcut(
            ssyk_scores["eloundou_score"], 4, labels=[1, 2, 3, 4]
        ).astype(int)

        n_ssyk = ssyk_scores["ssyk4"].nunique()
        print(f"  Built Eloundou quartiles for {n_ssyk} SSYK4 codes")
        print(f"  Quartile distribution:\n"
              f"{ssyk_scores['exposure_quartile'].value_counts().sort_index()}")

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
#   STEP 1: SQL DATA PULL (same as script 15 / 18)
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
    Load AGI data year by year (same pipeline as script 15/18),
    merge ELOUNDOU quartiles (instead of DAIOE), assign age groups,
    aggregate to employer x quartile x age_group x month cells.
    """
    print("=" * 70)
    print("STEP 1: Loading and preparing data")
    print("  EXPOSURE MEASURE: Eloundou GPT exposure (not DAIOE)")
    print("=" * 70)

    frames = []
    t0 = time.time()
    for year in range(2019, 2026):
        frames.append(pull_year(year, conn))
    df = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {len(df):,} records in {time.time()-t0:.0f}s")

    df = df[df["ssyk4"].notna() & (df["ssyk4"] != "None")].copy()
    df = df[df["age_group"].notna()].copy()
    df["ssyk4"] = df["ssyk4"].astype(str).str.zfill(4)

    # Merge ELOUNDOU quartiles
    print("  Loading Eloundou GPT exposure quartiles...")
    eloundou = load_eloundou_quartiles()
    df = df.merge(eloundou[["ssyk4", "exposure_quartile"]], on="ssyk4", how="inner")
    print(f"  After Eloundou merge: {len(df):,} records")

    df = df.rename(columns={"n_emp": "person_count"})

    # Filter small employers
    emp_size = df.groupby("employer_id")["person_count"].sum()
    large_emp = emp_size[emp_size >= MIN_EMPLOYER_SIZE].index
    df = df[df["employer_id"].isin(large_emp)].copy()
    print(f"  Employers with >={MIN_EMPLOYER_SIZE} workers: "
          f"{df['employer_id'].nunique():,}")

    # Aggregate to employer x quartile x age_group x month
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

    # Incremental save: cache aggregated panel
    cache_path = OUTPUT_DIR / "eloundou_agg_cache.csv"
    agg.to_csv(cache_path, index=False)
    print(f"  Cached aggregated panel -> {cache_path.name}")

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
    """
    sub = agg[agg["age_group"] == age_label].copy()
    all_months = sorted(agg["year_month"].unique())

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
#   STEP 2: DiD BY AGE GROUP (mirrors script 15's three-approach logic)
# ======================================================================

def run_did_by_age(agg):
    """
    For each age group, estimate:

      ln(n_emp_{f,q,t} + 1) = a_{f,q} + b_{f,t}
                               + g1*PostRB_t*High_q
                               + g2*PostGPT_t*High_q + e

    Uses script 15's three-approach fallback:
      A. linearmodels PanelOLS
      B. Manual within-transformation (double-demean)
      C. Occupation-level backup
    """
    print("\n" + "=" * 70)
    print("STEP 2: DiD regressions by age group")
    print("  EXPOSURE MEASURE: Eloundou GPT exposure")
    print(f"  Riksbank cutoff: {RIKSBANK_YM}")
    print(f"  ChatGPT cutoff:  {CHATGPT_YM}")
    print("=" * 70)

    all_results = []

    for age_label, (age_lo, age_hi) in AGE_GROUPS.items():
        print(f"\n--- Age group: {age_label} ---")

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

        sub["ln_emp"] = np.log(sub["n_emp"] + 1)

        # FE group identifiers
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
        out = OUTPUT_DIR / "eloundou_did_results.csv"
        results_df.to_csv(out, index=False)
        print(f"\n  Saved DiD results -> {out.name}")
        print("\n  === SUMMARY ===")
        print(results_df.to_string(index=False))
        return results_df

    return pd.DataFrame()


def _estimate_did(sub, age_label):
    """
    Estimate the DiD for one age group. Tries three approaches
    (mirrors script 15 exactly):

    A. linearmodels PanelOLS with employer x quartile entity FE +
       employer x month as absorbed other_effects
    B. Manual within-transformation (demean by employer x quartile,
       then by employer x month)
    C. Simple OLS with occupation + month FE (backup, weaker identification)
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
    try:
        panel = sub.copy()

        for col in ["ln_emp", "post_rb_x_high", "post_gpt_x_high"]:
            panel[f"{col}_dm1"] = panel.groupby("fe_emp_q")[col].transform(
                lambda x: x - x.mean()
            )

        for col in ["ln_emp", "post_rb_x_high", "post_gpt_x_high"]:
            panel[f"{col}_dm"] = panel.groupby("fe_emp_t")[f"{col}_dm1"].transform(
                lambda x: x - x.mean()
            )

        import statsmodels.api as sm

        y = panel["ln_emp_dm"].values
        X = panel[["post_rb_x_high_dm", "post_gpt_x_high_dm"]].values

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

    # Month dummies
    month_dummies = pd.get_dummies(occ_panel["year_month"], prefix="m", drop_first=True)
    # Quartile dummies
    q_dummies = pd.get_dummies(occ_panel["exposure_quartile"], prefix="q", drop_first=True)

    X = pd.concat([
        occ_panel[["post_rb_x_high", "post_gpt_x_high"]],
        month_dummies,
        q_dummies,
    ], axis=1).astype(float)
    X = sm.add_constant(X)

    mod = sm.OLS(occ_panel["ln_emp"], X)
    res = mod.fit(cov_type="HC1")

    gamma1 = res.params["post_rb_x_high"]
    gamma2 = res.params["post_gpt_x_high"]
    se1 = res.bse["post_rb_x_high"]
    se2 = res.bse["post_gpt_x_high"]
    p1 = res.pvalues["post_rb_x_high"]
    p2 = res.pvalues["post_gpt_x_high"]

    print(f"  [occupation-level backup]")
    print(f"  g1 (PostRB x High)  = {gamma1:+.4f} (SE={se1:.4f}, p={p1:.4f})")
    print(f"  g2 (PostGPT x High) = {gamma2:+.4f} (SE={se2:.4f}, p={p2:.4f})")

    return {
        "age_group": age_label,
        "method": "occupation-level",
        "n_obs": len(occ_panel),
        "gamma1_rb_high": gamma1,
        "se1": se1,
        "pval1": p1,
        "gamma2_gpt_high": gamma2,
        "se2": se2,
        "pval2": p2,
    }


def _estimate_poisson(sub, age_label):
    """Robustness: Poisson PML via pyfixest (handles zeros naturally)."""
    try:
        import pyfixest as pf

        panel = sub.copy()
        panel["high_post_rb"] = panel["post_rb_x_high"]
        panel["high_post_gpt"] = panel["post_gpt_x_high"]

        mod = pf.feols(
            "n_emp ~ high_post_rb + high_post_gpt | fe_emp_q + fe_emp_t",
            data=panel,
            vcov={"CRV1": "employer_id"},
        )

        gamma1 = mod.coef()["high_post_rb"]
        gamma2 = mod.coef()["high_post_gpt"]
        se1 = mod.se()["high_post_rb"]
        se2 = mod.se()["high_post_gpt"]
        p1 = mod.pvalue()["high_post_rb"]
        p2 = mod.pvalue()["high_post_gpt"]

        print(f"  [pyfixest Poisson]")
        print(f"  g1 (PostRB x High)  = {gamma1:+.4f} (SE={se1:.4f}, p={p1:.4f})")
        print(f"  g2 (PostGPT x High) = {gamma2:+.4f} (SE={se2:.4f}, p={p2:.4f})")

        return {
            "age_group": age_label,
            "method": "pyfixest",
            "n_obs": int(mod.nobs),
            "gamma1_rb_high": gamma1,
            "se1": se1,
            "pval1": p1,
            "gamma2_gpt_high": gamma2,
            "se2": se2,
            "pval2": p2,
        }

    except ImportError:
        print("  pyfixest not available -- skipping Poisson robustness")
        return None
    except Exception as e:
        print(f"  pyfixest failed: {e}")
        return None


# ======================================================================
#   MAIN
# ======================================================================

if __name__ == "__main__":
    try:
        print("\n" + "=" * 70)
        print("29_mona_eloundou_measure.py")
        print("Robustness: Eloundou GPT exposure instead of DAIOE")
        print("DiD only (event study skipped -- already generated)")
        print("Regression logic mirrors script 15 (PanelOLS -> manual -> backup)")
        print("=" * 70)

        t_start = time.time()

        # Step 1: Load data (with Eloundou exposure)
        agg = load_and_prepare()

        # Step 2: DiD by age group
        results_df = run_did_by_age(agg)

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
            f"FE: employer x quartile (entity) + employer x month (other)",
            f"Estimator: same three-approach fallback as script 15",
            f"Runtime: {elapsed/60:.1f} minutes",
            "",
            "RATIONALE:",
            "  Gimbel et al. (2026) show that AI exposure metrics agree on which",
            "  occupations have LOW exposure but disagree on magnitude at the top.",
            "  If the canaries result holds with Eloundou (a different measure",
            "  from a different methodology), it is robust to measure choice.",
            "",
            "DiD RESULTS (two-treatment, mirrors script 15):",
            "  gamma1 = Riksbank x High (post 2022-04)",
            "  gamma2 = ChatGPT  x High (post 2022-12)  <-- canaries coef",
        ]

        if not results_df.empty:
            for _, row in results_df.iterrows():
                sig1 = "***" if row["pval1"] < 0.01 else "**" if row["pval1"] < 0.05 else "*" if row["pval1"] < 0.1 else ""
                sig2 = "***" if row["pval2"] < 0.01 else "**" if row["pval2"] < 0.05 else "*" if row["pval2"] < 0.1 else ""
                summary.append(
                    f"  {row['age_group']:>8s} ({row['method']}):  "
                    f"g1 = {row['gamma1_rb_high']:>+8.4f} "
                    f"(SE = {row['se1']:.4f}, p = {row['pval1']:.4f}) {sig1}"
                )
                summary.append(
                    f"  {'':>8s}  {'':>{len(row['method'])+3}s}"
                    f"g2 = {row['gamma2_gpt_high']:>+8.4f} "
                    f"(SE = {row['se2']:.4f}, p = {row['pval2']:.4f}) {sig2}"
                )

        summary.extend([
            "",
            "INTERPRETATION:",
            "  Compare sign, magnitude, and significance with DAIOE results",
            "  (script 15 / canaries_did_results.csv). Key questions:",
            "  1. Does the 22-25 age group still show negative effect?",
            "  2. Is the 50+ group still positive or null?",
            "  3. Does the age gradient (young negative, old positive) hold?",
            "",
            "  If patterns match: result is robust to exposure measure.",
            "  If patterns differ: discuss which occupations are classified",
            "  differently by DAIOE vs Eloundou and why.",
        ])

        summary_text = "\n".join(summary)
        (OUTPUT_DIR / "eloundou_comparison.txt").write_text(summary_text)
        print(f"\n{summary_text}")
        print(f"\n  All output in: {OUTPUT_DIR}/")

    except BaseException as e:
        print(f"\nFATAL ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
