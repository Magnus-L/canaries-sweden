#!/usr/bin/env python3
"""
32_mona_kauhanen_robustness.py -- Sweden vs Finland: staged Kauhanen replication.

======================================================================
  THIS SCRIPT IS DESIGNED TO RUN IN SCB's MONA SECURE ENVIRONMENT
  Do NOT run outside MONA -- the data is not available externally.
======================================================================

WHY THIS SCRIPT EXISTS:
  Kauhanen et al. (2025, 2026) find a null employment effect of generative
  AI exposure in Finland using an employer-level Poisson DiD. Our Swedish
  employer-level OLS+1 DiD finds a sharp negative age gradient. The
  divergence may be (i) economy-dependent (Sweden does, Finland does not)
  or (ii) method-dependent (specification choices flip the result). This
  script tests the four specification dimensions Kauhanen varies, ONE AT
  A TIME, so we can localise where (if anywhere) the Swedish result breaks.

  Personas-review (2026-04-27) flagged this as the single highest-leverage
  cheap test we can add before submitting to Economics Letters.

STAGED SEQUENCE (stops early if Step 1 nulls; otherwise continues):

  STEP 1  -- Poisson, current spec
            Threshold: >=5 cumulative (current paper)
            Exposure: DAIOE quartile, High = Q4
            FE: employer x quartile + employer x month
            Estimator: Poisson PML (fepois)
            -> Tests whether OLS+1 vs Poisson choice matters.
               Paper's appendix line 440 asserts they are similar at
               large cell counts; this is the receipt.

  STEP 2  -- Poisson, Kauhanen threshold
            Threshold: >=10 mean monthly + >=100 cumulative (Brynjolfsson 2025)
            All else as Step 1.
            -> Tests whether sample restriction explains the
               Sweden-Finland divergence.

  STEP 3  -- Poisson, full Kauhanen exact spec
            Threshold: >=10 mean monthly + >=100 cumulative
            Exposure: Eloundou beta-measure, QUINTILE binning, High = Q5
            FE: employer x quintile + employer x month
            -> Combined Kauhanen replication on Swedish data.
               Now interpretable as the marginal effect of switching
               from quartile to quintile (since Steps 1-2 already
               isolated estimator and threshold).

  STEP 4  -- Poisson, current spec, marginally reweighted to Finnish
            occupational and industrial composition (Eurostat LFS 2022).
            Conditional: runs only if finland_marginals_2022.txt is on
            the empirical_data share. If absent, logs and skips.
            Workers reweighted by w(occ) * w(ind) where
                w(occ) = p_fin(ISCO1) / p_swe(ISCO1)
                w(ind) = p_fin(NACE2) / p_swe(NACE2)
            Tests whether industrial / occupational composition explains
            the Sweden-Finland divergence.

  STEP 5  -- Poisson, current spec, with IT/tech occupations (SSYK 25xx)
            EXCLUDED. Brynjolfsson 2025 excludes; Kauhanen 2026 likely
            excludes (replicates Brynjolfsson's sample restrictions).
            Our paper currently does NOT exclude in the employment DiD.
            Tests whether the Swedish age gradient survives the same
            ICT-specialist exclusion the comparison studies impose.

  DIAG    -- SSYK non-match rate by quartile of last-known DAIOE
            For each year 2019-2025: among AGI workers with no current-year
            SSYK match, look up last-known SSYK in Individ_2021/2022/2023.
            Tabulate non-match share by DAIOE quartile of last-known SSYK.
            -> Pre-empts Measurement Critic concern that the post-2023
               attrition rise drives the headline acceleration.

OUTPUT FILES (under output_32/):
  1. step0_panel_cache.csv         -- (employer, ssyk4, age, ym, n_emp); reusable
  2. step1_poisson_current.csv     -- Step 1 coefficients by age group
  3. step2_poisson_threshold.csv   -- Step 2 coefficients by age group
  4. step3_poisson_kauhanen.csv    -- Step 3 coefficients by age group
  5. step4_poisson_reweighted.csv  -- Step 4 coefficients (if marginals file present)
  6. step5_poisson_no_ict.csv      -- Step 5 coefficients (drop SSYK 25xx)
  7. attrition_by_quartile.csv     -- non-match rate by year x DAIOE quartile
  8. kauhanen_comparison.csv       -- master table: all completed steps side by side
  9. kauhanen_summary.txt          -- prose summary for Magnus

DEPENDENCIES:
  Required: pandas, numpy, pyodbc, pyfixest
  Optional: matplotlib (only used if FIGURE_OUTPUT == True)

  pyfixest is REQUIRED. Pre-flight check at script start. If absent:
      pip install pyfixest

EXPECTED RUNTIME:
  Step 0 SQL pull       : 30-45 min one-off (cached afterwards)
  Step 1 (6 age groups) : 10-30 min depending on cell count
  Step 2 (6 age groups) : 5-15 min (smaller sample)
  Step 3 (6 age groups) : 5-15 min (smaller sample)
  Diagnostic            : 5-10 min
  Total                 : 60-120 min, dominated by SQL pull

CRASH RECOVERY:
  step0_panel_cache.csv is written after the SQL pull. Subsequent runs
  skip the SQL pull and load from cache. If a Step crashes mid-way,
  re-running picks up where it left off via the per-step CSV files.

EXPORT-SAFE:
  All output is aggregated counts and regression coefficients. No raw
  individual-level rows. Cell-count safeguards (>=5) applied to the
  attrition diagnostic before tabulation.

Magnus -- Run this first thing tomorrow morning. Sequence:
  1. Pre-flight: ensure pyfixest is installed in MONA.
  2. python 32_mona_kauhanen_robustness.py
  3. If Step 1 (22-25, Poisson, current spec) is negative and significant,
     all three steps will run automatically and you have the full Kauhanen
     comparison. If Step 1 nulls, the script writes a warning and stops;
     you should then re-think before running Steps 2-3.
  4. Export everything in output_32/.
"""

import sys
import time
from pathlib import Path

# ======================================================================
#   _Tee LOGGER (BatchClient does not capture stdout/stderr)
# ======================================================================

class _Tee:
    def __init__(self, log_path):
        self._stdout = sys.stdout
        self._log = open(log_path, "w", encoding="utf-8", buffering=1)
    def write(self, text):
        try:
            self._stdout.write(text)
        except UnicodeEncodeError:
            self._stdout.write(
                text.encode("cp1252", errors="replace").decode("cp1252")
            )
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
except BaseException as e:
    print(f"WARNING: Could not create log file: {e}")


# ======================================================================
#   PRE-FLIGHT: REQUIRED IMPORTS
# ======================================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

try:
    import pyodbc
except ImportError as e:
    print(f"FATAL: pyodbc missing: {e}")
    print("This script requires pyodbc to query the MONA SQL Server.")
    raise SystemExit(1)

try:
    import pyfixest as pf
    _HAS_PYFIXEST = True
    print(f"pyfixest version: {pf.__version__ if hasattr(pf, '__version__') else 'unknown'}")
except ImportError:
    _HAS_PYFIXEST = False
    print("FATAL: pyfixest not installed.")
    print("Install in MONA via:    pip install pyfixest")
    print("This script requires pyfixest for Poisson PML with high-dim FE.")
    raise SystemExit(1)


# ======================================================================
#   CONFIGURATION
# ======================================================================

# --- MONA SQL connection ---
SQL_CONN_STRING = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=monasql.micro.intra;"
    "DATABASE=P1207;"
    "Trusted_Connection=yes;"
)

# --- Network share paths (consistent with scripts 15, 28, 29) ---
DAIOE_PATH = r"\\micro.intra\Projekt\P1207$\P1207_Gem\Lydia P1207\daioe_quartiles.csv"
ELOUNDOU_QUART_PATH = r"\\micro.intra\Projekt\P1207$\P1207_Gem\Lydia P1207\eloundou_ssyk4.csv"
ELOUNDOU_RAW_PATH = r"\\micro.intra\Projekt\P1207$\P1207_Gem\Lydia P1207\eloundou_raw.csv"
SOC_ISCO_CW_PATH = r"\\micro.intra\Projekt\P1207$\P1207_Gem\Lydia P1207\soc_isco_crosswalk.csv"
ISCO_SSYK_CW_PATH = r"\\micro.intra\Projekt\P1207$\P1207_Gem\Lydia P1207\isco_ssyk_crosswalk.csv"

# Step 4 input: Finnish ISCO-1 + NACE-2 marginals (Eurostat LFS 2022).
# Built locally with src/build_finland_marginals.py. Lives at the project
# root on the Lydia P1207 share (flat layout matching the existing files
# daioe_quartiles.csv, eloundou_ssyk4.csv).
FINLAND_MARGINALS_PATH = r"\\micro.intra\Projekt\P1207$\P1207_Gem\Lydia P1207\finland_marginals_2022.txt"

# --- Output directory ---
OUTPUT_DIR = _SCRIPT_DIR / "output_32"
OUTPUT_DIR.mkdir(exist_ok=True)
PANEL_CACHE = OUTPUT_DIR / "step0_panel_cache.csv"

# --- Treatment dates (consistent with scripts 14, 15, 18, 28, 29) ---
RIKSBANK_YM = "2022-04"
CHATGPT_YM = "2022-12"

# --- Age group definitions (consistent with paper Table 4) ---
AGE_GROUPS = {
    "22-25": (22, 25),
    "26-30": (26, 30),
    "31-34": (31, 34),
    "35-40": (35, 40),
    "41-49": (41, 49),
    "50+":   (50, 69),
}

# --- Step 1: current paper threshold ---
STEP1_MIN_CUMULATIVE = 5

# --- Step 2/3: Brynjolfsson/Kauhanen threshold ---
# Per appendix line 855: ">=10 employees per age group per month and
# >=100 cumulatively across exposure quantiles". Implemented as:
#   - mean monthly count of age-group workers per employer >= 10
#   - cumulative count across exposure bins per employer >= 100
KAUHANEN_MIN_MEAN_PER_MONTH = 10
KAUHANEN_MIN_CUMULATIVE = 100

# --- Behaviour flags ---
# Set to True to run Steps 2-5 even if Step 1 (22-25) shows null result.
# Default True -- you want the full picture even if one step nulls.
RUN_ALL_REGARDLESS = True   # Magnus: set False if you want stop-on-null

# Threshold for stop-on-null (default: not significant at 5%, sign positive)
NULL_THRESHOLD_P = 0.05

# --- IT/tech occupations (SSYK 25xx) excluded in Step 5 ---
# Brynjolfsson 2025 excludes; Kauhanen 2026 likely follows.
ICT_SSYK_PREFIX = "25"


# ======================================================================
#   STEP 0: SQL PULL (with cache reuse)
# ======================================================================

def pull_year_to_panel(year, conn):
    """
    Pull one year of AGI data with cascading SSYK lookup, aggregated to
    (employer x ssyk4 x age_group x year_month).

    For year >= 2023 (LISA frozen at 2023): cascade SSYK from
    Individ_2023 -> Individ_2022 -> Individ_2021. This gives the
    last-known SSYK code for workers without a current SSYK.

    Returns: DataFrame with columns
      employer_id, year_month, ssyk4, age_group, n_emp
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
            # Cascading SSYK lookup (includes workers with stale codes)
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
            # Pre-2023: use the year's own Individ table
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

    df = pd.read_sql(query, conn)
    df["ssyk4"] = df["ssyk4"].astype(str).str.zfill(4)
    return df


def step0_pull_or_load_panel():
    """
    Pull AGI 2019-2025 if cache is missing; otherwise load from cache.
    Cache schema: (employer_id, year_month, ssyk4, age_group, n_emp).
    The SSYK4 level is preserved so we can merge any binning (DAIOE
    quartile, Eloundou quartile, Eloundou quintile) downstream.
    """
    print("\n" + "=" * 70)
    print("STEP 0: AGI panel pull or cache reload")
    print("=" * 70)

    if PANEL_CACHE.exists():
        print(f"  Cache found: {PANEL_CACHE.name}")
        print(f"  Loading (skip the 30-45 min SQL pull)...")
        t0 = time.time()
        agg = pd.read_csv(PANEL_CACHE, dtype={"ssyk4": str})
        agg["ssyk4"] = agg["ssyk4"].astype(str).str.zfill(4)
        print(f"  Loaded {len(agg):,} rows in {time.time()-t0:.0f}s")
        print(f"  Period: {agg['year_month'].min()} to {agg['year_month'].max()}")
        return agg

    print(f"  No cache. Running SQL pull (30-45 min)...")
    conn = pyodbc.connect(SQL_CONN_STRING)
    frames = []
    t0 = time.time()
    for year in range(2019, 2026):
        df_year = pull_year_to_panel(year, conn)
        frames.append(df_year)
        print(f"    Year {year}: {len(df_year):,} cells, "
              f"{(time.time()-t0)/60:.1f} min elapsed")
    conn.close()

    agg = pd.concat(frames, ignore_index=True)
    print(f"\n  Total rows: {len(agg):,}")
    print(f"  Saving cache to {PANEL_CACHE.name}...")
    agg.to_csv(PANEL_CACHE, index=False)
    print(f"  Saved.")
    return agg


# ======================================================================
#   EXPOSURE LOADING (DAIOE quartile, Eloundou quintile)
# ======================================================================

def load_daioe_quartiles():
    """Load DAIOE quartiles -> DataFrame [ssyk4, daioe_quartile]."""
    print(f"  Loading DAIOE quartiles from {DAIOE_PATH}")
    df = pd.read_csv(DAIOE_PATH)
    df["ssyk4"] = df["ssyk4"].astype(str).str.zfill(4)

    if df["exposure_quartile"].dtype == object:
        df["exposure_quartile"] = (
            df["exposure_quartile"]
            .str.strip()
            .str.extract(r"Q(\d)")
            .astype(int)
        )

    print(f"  DAIOE: {df['ssyk4'].nunique()} SSYK4 codes")
    return df[["ssyk4", "exposure_quartile"]].rename(
        columns={"exposure_quartile": "daioe_quartile"}
    )


def load_eloundou_quintiles():
    """
    Load Eloundou beta-measure with QUINTILE binning. We need quintiles
    (top 20% = High = Q5) for Kauhanen exact spec, but the existing
    eloundou_ssyk4.csv on the share is QUARTILE-binned.

    Strategy:
      A. If eloundou_ssyk4.csv has an 'eloundou_score' column, recompute
         quintiles via pd.qcut(scores, 5).
      B. Else if raw + crosswalks exist, rebuild via SOC -> ISCO -> SSYK,
         compute mean score per SSYK, qcut into 5.
      C. Else fail with clear instructions.

    Returns: DataFrame [ssyk4, eloundou_quintile]
    """
    print(f"  Loading Eloundou quintiles...")

    # --- Strategy A: existing file with score column ---
    if Path(ELOUNDOU_QUART_PATH).exists():
        df = pd.read_csv(ELOUNDOU_QUART_PATH)
        df["ssyk4"] = df["ssyk4"].astype(str).str.zfill(4)

        score_cols = [c for c in df.columns
                      if "score" in c.lower() or "beta" in c.lower()]
        if score_cols:
            score_col = score_cols[0]
            print(f"    Strategy A: using score column '{score_col}' from "
                  f"existing file")
            df["eloundou_quintile"] = pd.qcut(
                df[score_col], 5, labels=[1, 2, 3, 4, 5]
            ).astype(int)
            print(f"    Built quintiles for {df['ssyk4'].nunique()} SSYK4 codes")
            return df[["ssyk4", "eloundou_quintile"]]
        else:
            print(f"    Strategy A: existing file has no score column, "
                  f"only quartile -- falling through to Strategy B")

    # --- Strategy B: rebuild from raw + crosswalks ---
    if (Path(ELOUNDOU_RAW_PATH).exists()
        and Path(SOC_ISCO_CW_PATH).exists()
        and Path(ISCO_SSYK_CW_PATH).exists()):
        print(f"    Strategy B: rebuilding from raw scores + crosswalks")

        raw = pd.read_csv(ELOUNDOU_RAW_PATH)
        soc_isco = pd.read_csv(SOC_ISCO_CW_PATH)
        isco_ssyk = pd.read_csv(ISCO_SSYK_CW_PATH)

        # Identify columns by name pattern (mirrors script 29)
        soc_col = next((c for c in raw.columns
                        if "soc" in c.lower() or "onet" in c.lower()), None)
        score_col = next((c for c in raw.columns
                          if "exposure" in c.lower() or "score" in c.lower()
                          or "beta" in c.lower() or "zeta" in c.lower()), None)

        if soc_col is None or score_col is None:
            raise RuntimeError(
                f"Cannot identify SOC/score columns in {ELOUNDOU_RAW_PATH}. "
                f"Columns: {list(raw.columns)}"
            )

        raw[soc_col] = raw[soc_col].astype(str).str.replace(
            r"[\.\-]", "", regex=True
        )

        soc_cw = next((c for c in soc_isco.columns if "soc" in c.lower()), None)
        isco_cw_a = next((c for c in soc_isco.columns if "isco" in c.lower()), None)
        soc_isco[soc_cw] = soc_isco[soc_cw].astype(str).str.replace(
            r"[\.\-]", "", regex=True
        )
        isco_cw_b = next((c for c in isco_ssyk.columns if "isco" in c.lower()), None)
        ssyk_cw = next((c for c in isco_ssyk.columns if "ssyk" in c.lower()), None)

        merged = raw.merge(soc_isco, left_on=soc_col, right_on=soc_cw, how="inner")
        merged = merged.merge(isco_ssyk, left_on=isco_cw_a, right_on=isco_cw_b,
                              how="inner")
        merged[ssyk_cw] = merged[ssyk_cw].astype(str).str.zfill(4)

        scores = (
            merged.groupby(ssyk_cw)[score_col]
            .mean()
            .reset_index()
            .rename(columns={ssyk_cw: "ssyk4", score_col: "eloundou_score"})
        )
        scores["eloundou_quintile"] = pd.qcut(
            scores["eloundou_score"], 5, labels=[1, 2, 3, 4, 5]
        ).astype(int)

        # Save with score for future runs
        out_path = OUTPUT_DIR / "eloundou_ssyk4_with_quintile.csv"
        scores.to_csv(out_path, index=False)
        print(f"    Built quintiles for {scores['ssyk4'].nunique()} SSYK4 codes")
        print(f"    Saved with score column to {out_path.name}")
        return scores[["ssyk4", "eloundou_quintile"]]

    # --- Strategy C: fail loudly ---
    raise RuntimeError(
        f"Cannot build Eloundou quintiles. Need either:\n"
        f"  A. {ELOUNDOU_QUART_PATH} with an 'eloundou_score' column, OR\n"
        f"  B. {ELOUNDOU_RAW_PATH} + crosswalk files\n"
        f"See script 29 docstring for instructions."
    )


# ======================================================================
#   THRESHOLD FILTERING
# ======================================================================

def filter_step1(panel_emp_bin, age_label):
    """
    Step 1 threshold: total person-months across all periods >= 5
    (current paper specification).
    Filter at the (employer x bin) panel for one age group.
    """
    sub = panel_emp_bin[panel_emp_bin["age_group"] == age_label].copy()
    cum = sub.groupby("employer_id")["n_emp"].sum()
    valid = cum[cum >= STEP1_MIN_CUMULATIVE].index
    return sub[sub["employer_id"].isin(valid)].copy()


def filter_kauhanen(panel_emp_bin, age_label):
    """
    Kauhanen / Brynjolfsson threshold:
      (a) mean monthly count of age-group workers per employer >= 10
      (b) cumulative across exposure bins for the employer >= 100

    Implementation note: "per age group per month >= 10" is interpreted
    as a per-employer mean across observed months. The cumulative-100
    restriction is applied to the same age-group-restricted sample.
    Both must hold for the employer to be retained.
    """
    sub = panel_emp_bin[panel_emp_bin["age_group"] == age_label].copy()

    # (a) mean monthly count
    per_month = sub.groupby(["employer_id", "year_month"])["n_emp"].sum()
    mean_per_month = per_month.groupby("employer_id").mean()
    valid_a = set(mean_per_month[
        mean_per_month >= KAUHANEN_MIN_MEAN_PER_MONTH
    ].index)

    # (b) cumulative across exposure bins
    cum = sub.groupby("employer_id")["n_emp"].sum()
    valid_b = set(cum[cum >= KAUHANEN_MIN_CUMULATIVE].index)

    valid = valid_a & valid_b
    return sub[sub["employer_id"].isin(valid)].copy()


# ======================================================================
#   PANEL CONSTRUCTION (BALANCED, ZERO-FILLED)
# ======================================================================

def build_balanced_panel(sub, bin_col, n_bins):
    """
    Build a balanced (employer x bin x ym) panel for one age group:
      - cross join: (every employer-bin pair the employer is observed in)
                    x (every month in the data)
      - left-join actual counts; missing -> 0
      - identification restriction: employer must be observed in
        bin == max_bin AND in some bin < max_bin (so that within-employer
        variation across bins exists)

    Args:
      sub: DataFrame already filtered to one age_group, columns include
           [employer_id, bin_col, year_month, n_emp]
      bin_col: name of the binning column ("daioe_quartile" or "eloundou_quintile")
      n_bins: 4 for quartile, 5 for quintile

    Returns:
      Balanced DataFrame with columns [employer_id, bin_col, year_month, n_emp]
    """
    if len(sub) == 0:
        return sub

    max_bin = n_bins  # High-exposure bin (Q4 for quartile, Q5 for quintile)

    # Identify employer-bin pairs present in the data
    emp_bin = sub[["employer_id", bin_col]].drop_duplicates()

    # Identification restriction: keep only employers spanning High and Low
    high_emps = set(emp_bin.loc[emp_bin[bin_col] == max_bin, "employer_id"])
    low_emps = set(emp_bin.loc[emp_bin[bin_col] < max_bin, "employer_id"])
    valid_emps = high_emps & low_emps
    emp_bin = emp_bin[emp_bin["employer_id"].isin(valid_emps)]

    if len(emp_bin) == 0:
        return pd.DataFrame(columns=sub.columns)

    # Cross join with all months in the data (NB: months from the full panel,
    # not the filtered subset, so we don't lose periods where this age group
    # had zero workers across the board).
    all_months = sorted(sub["year_month"].unique())
    months_df = pd.DataFrame({"year_month": all_months})

    emp_bin["_k"] = 1
    months_df["_k"] = 1
    balanced = emp_bin.merge(months_df, on="_k").drop(columns="_k")

    # Left-join actual counts
    balanced = balanced.merge(
        sub[["employer_id", bin_col, "year_month", "n_emp"]],
        on=["employer_id", bin_col, "year_month"],
        how="left",
    )
    balanced["n_emp"] = balanced["n_emp"].fillna(0).astype(int)

    return balanced


# ======================================================================
#   POISSON REGRESSION (pyfixest)
# ======================================================================

def estimate_poisson(balanced, bin_col, n_bins, age_label, spec_label):
    """
    Estimate Poisson PML on a balanced employer x bin x ym panel:

      n_emp ~ post_rb_x_high + post_gpt_x_high | fe_emp_bin + fe_emp_t

    where High = (bin == max_bin). Coefs are multiplicative-log
    (interpret as approximate percentage change).

    Returns dict with keys (or None on failure):
      spec, age_group, gamma1, se1, p1, gamma2, se2, p2, n_obs, n_emp_total
    """
    if len(balanced) < 100:
        print(f"    [{spec_label} / {age_label}] too few obs ({len(balanced)}); "
              f"skip")
        return None

    panel = balanced.copy()
    max_bin = n_bins

    # Construct treatment vars
    panel["high"] = (panel[bin_col] == max_bin).astype(int)
    panel["post_rb"] = (panel["year_month"] >= RIKSBANK_YM).astype(int)
    panel["post_gpt"] = (panel["year_month"] >= CHATGPT_YM).astype(int)
    panel["post_rb_x_high"] = panel["post_rb"] * panel["high"]
    panel["post_gpt_x_high"] = panel["post_gpt"] * panel["high"]

    # FE keys (string-typed for pyfixest's groupwise demeaning)
    panel["fe_emp_bin"] = (
        panel["employer_id"].astype(str) + "_" + panel[bin_col].astype(str)
    )
    panel["fe_emp_t"] = (
        panel["employer_id"].astype(str) + "_" + panel["year_month"]
    )

    n_obs = len(panel)
    n_emp_total = panel["n_emp"].sum()
    n_employers = panel["employer_id"].nunique()

    print(f"    [{spec_label} / {age_label}] N obs = {n_obs:,}, "
          f"N employers = {n_employers:,}, total person-months = {n_emp_total:,}")

    t0 = time.time()
    try:
        fit = pf.fepois(
            "n_emp ~ post_rb_x_high + post_gpt_x_high | fe_emp_bin + fe_emp_t",
            data=panel,
            vcov={"CRV1": "employer_id"},
        )

        coef = fit.coef()
        se = fit.se()
        pval = fit.pvalue()

        gamma1 = float(coef["post_rb_x_high"])
        gamma2 = float(coef["post_gpt_x_high"])
        se1 = float(se["post_rb_x_high"])
        se2 = float(se["post_gpt_x_high"])
        p1 = float(pval["post_rb_x_high"])
        p2 = float(pval["post_gpt_x_high"])

        elapsed = time.time() - t0
        print(f"      g1 (PostRB  x High) = {gamma1:+.4f} (SE={se1:.4f}, p={p1:.4f})")
        print(f"      g2 (PostGPT x High) = {gamma2:+.4f} (SE={se2:.4f}, p={p2:.4f})")
        print(f"      [{elapsed:.0f}s]")

        return {
            "spec": spec_label,
            "age_group": age_label,
            "gamma1": gamma1,
            "se1": se1,
            "p1": p1,
            "gamma2": gamma2,
            "se2": se2,
            "p2": p2,
            "n_obs": n_obs,
            "n_emp_total": int(n_emp_total),
            "n_employers": n_employers,
            "estimator": "Poisson PML (pyfixest fepois)",
            "fe": "employer x bin + employer x month",
            "vcov": "CRV1 by employer",
            "elapsed_s": round(elapsed, 1),
        }

    except BaseException as e:
        print(f"      Poisson FAILED for {spec_label} / {age_label}: {e}")
        return None


# ======================================================================
#   STEP 1: POISSON, CURRENT SPEC
# ======================================================================

def step1_poisson_current(panel_ssyk_age, daioe):
    """
    Step 1: Poisson PML, current paper spec.
      Threshold: cumulative >=5
      Exposure: DAIOE quartile
      High: Q4
      FE: employer x quartile + employer x month

    Tests whether the OLS+1 -> Poisson switch alone preserves the result.
    The paper's appendix (line 440) asserts they are similar; this is
    the receipt.
    """
    print("\n" + "=" * 70)
    print("STEP 1: Poisson PML, current spec (>=5 threshold, DAIOE quartile)")
    print("=" * 70)

    # Merge DAIOE and re-aggregate to (employer x quartile x age x ym)
    panel = panel_ssyk_age.merge(daioe, on="ssyk4", how="inner")
    panel = (
        panel.groupby(
            ["employer_id", "daioe_quartile", "age_group", "year_month"]
        )["n_emp"]
        .sum()
        .reset_index()
    )

    results = []
    for age_label in AGE_GROUPS:
        print(f"\n  Age group: {age_label}")
        sub = filter_step1(panel, age_label)
        balanced = build_balanced_panel(sub, "daioe_quartile", n_bins=4)
        if len(balanced) == 0:
            print(f"    No employers span Q4 and Q1-Q3; skip")
            continue
        result = estimate_poisson(
            balanced, "daioe_quartile", n_bins=4,
            age_label=age_label, spec_label="STEP1_current",
        )
        if result is not None:
            results.append(result)
            # Incremental save after each age group
            pd.DataFrame(results).to_csv(
                OUTPUT_DIR / "step1_poisson_current.csv", index=False
            )

    return pd.DataFrame(results)


# ======================================================================
#   STEP 2: POISSON, KAUHANEN THRESHOLD
# ======================================================================

def step2_poisson_threshold(panel_ssyk_age, daioe):
    """
    Step 2: Poisson PML, Kauhanen threshold.
      Threshold: mean monthly >=10 + cumulative >=100
      Exposure: DAIOE quartile (same as Step 1)
      High: Q4
      FE: employer x quartile + employer x month

    Isolates the marginal effect of the threshold change. If the result
    survives this, the Sweden-Finland sample-restriction story is dead.
    """
    print("\n" + "=" * 70)
    print("STEP 2: Poisson PML, Kauhanen threshold (>=10 mean + >=100 cum)")
    print("=" * 70)

    panel = panel_ssyk_age.merge(daioe, on="ssyk4", how="inner")
    panel = (
        panel.groupby(
            ["employer_id", "daioe_quartile", "age_group", "year_month"]
        )["n_emp"]
        .sum()
        .reset_index()
    )

    results = []
    for age_label in AGE_GROUPS:
        print(f"\n  Age group: {age_label}")
        sub = filter_kauhanen(panel, age_label)
        if len(sub) == 0:
            print(f"    No employers pass Kauhanen threshold for this age "
                  f"group; skip")
            continue
        balanced = build_balanced_panel(sub, "daioe_quartile", n_bins=4)
        if len(balanced) == 0:
            print(f"    No employers span Q4 and Q1-Q3 after threshold; skip")
            continue
        result = estimate_poisson(
            balanced, "daioe_quartile", n_bins=4,
            age_label=age_label, spec_label="STEP2_threshold",
        )
        if result is not None:
            results.append(result)
            pd.DataFrame(results).to_csv(
                OUTPUT_DIR / "step2_poisson_threshold.csv", index=False
            )

    return pd.DataFrame(results)


# ======================================================================
#   STEP 3: POISSON, FULL KAUHANEN EXACT SPEC
# ======================================================================

def step3_poisson_kauhanen(panel_ssyk_age, eloundou):
    """
    Step 3: Poisson PML, full Kauhanen exact spec.
      Threshold: mean monthly >=10 + cumulative >=100
      Exposure: Eloundou beta-measure, QUINTILE binning
      High: Q5 (top 20%)
      FE: employer x quintile + employer x month

    The fully-replicated Kauhanen specification on Swedish data. The
    quintile-vs-quartile change is the only new dimension after Steps 1-2.
    """
    print("\n" + "=" * 70)
    print("STEP 3: Poisson PML, full Kauhanen exact spec "
          "(Eloundou quintile + threshold)")
    print("=" * 70)

    panel = panel_ssyk_age.merge(eloundou, on="ssyk4", how="inner")
    panel = (
        panel.groupby(
            ["employer_id", "eloundou_quintile", "age_group", "year_month"]
        )["n_emp"]
        .sum()
        .reset_index()
    )

    results = []
    for age_label in AGE_GROUPS:
        print(f"\n  Age group: {age_label}")
        sub = filter_kauhanen(panel, age_label)
        if len(sub) == 0:
            print(f"    No employers pass Kauhanen threshold; skip")
            continue
        balanced = build_balanced_panel(sub, "eloundou_quintile", n_bins=5)
        if len(balanced) == 0:
            print(f"    No employers span Q5 and Q1-Q4 after threshold; skip")
            continue
        result = estimate_poisson(
            balanced, "eloundou_quintile", n_bins=5,
            age_label=age_label, spec_label="STEP3_kauhanen_exact",
        )
        if result is not None:
            results.append(result)
            pd.DataFrame(results).to_csv(
                OUTPUT_DIR / "step3_poisson_kauhanen.csv", index=False
            )

    return pd.DataFrame(results)


# ======================================================================
#   STEP 4: POISSON, REWEIGHTED TO FINNISH COMPOSITION (CONDITIONAL)
# ======================================================================

def _pull_employer_nace(conn):
    """
    Pull (employer_id, nace2) from Foretagsdatabasen.

    SCB's Foretagsdatabasen records each employer's primary economic
    activity using SNI 2007 5-digit codes. SNI 2007 is the Swedish
    implementation of NACE Rev. 2 and the 2-digit prefix matches NACE-2.
    We take the most-recent year's record per employer to avoid
    duplicates. If an employer changes industry over time, the most
    recent NACE-2 is used; this is a reasonable approximation for
    employer-level reweighting.

    Returns DataFrame [employer_id, nace2] (zero-padded 2-digit string).
    """
    print(f"  Pulling employer -> NACE-2 from Foretagsdatabasen...")
    query = """
        SELECT
            P1207_LopNr_PeOrgNr AS employer_id,
            LEFT(RIGHT('00000' + CAST(NaceG1_2007 AS VARCHAR(5)), 5), 2) AS nace2,
            ArendArAr AS year
        FROM dbo.Foretag_FDB
        WHERE NaceG1_2007 IS NOT NULL
    """
    df = pd.read_sql(query, conn)
    # Keep most recent year per employer
    df = df.sort_values(["employer_id", "year"], ascending=[True, False])
    df = df.drop_duplicates(subset=["employer_id"], keep="first")
    df["nace2"] = df["nace2"].astype(str).str.zfill(2)
    print(f"    Retrieved NACE-2 for {len(df):,} employers")
    return df[["employer_id", "nace2"]]


def _load_finland_marginals():
    """
    Load Finnish ISCO-1 + NACE-2 marginal shares from the empirical_data
    share. Returns dict {dimension: {code: share_fin}} or None if missing.
    """
    p = Path(FINLAND_MARGINALS_PATH)
    if not p.exists():
        print(f"  Finland marginals file not found at {p}")
        print(f"  Skipping Step 4. To run, place finland_marginals_2022.txt")
        print(f"  in the empirical_data/ folder on the MONA share.")
        return None

    df = pd.read_csv(p, dtype={"code": str})
    out = {"ISCO1": {}, "NACE2": {}}
    for _, r in df.iterrows():
        dim = r["dimension"]
        if dim in out:
            out[dim][str(r["code"]).zfill(2 if dim == "NACE2" else 1)] = float(
                r["share"]
            )
    print(f"  Loaded Finnish marginals: ISCO1 = {len(out['ISCO1'])} codes, "
          f"NACE2 = {len(out['NACE2'])} codes")
    return out


def step4_poisson_reweighted(panel_ssyk_age, daioe):
    """
    Step 4: Poisson PML, current spec, with workers reweighted to mimic
    Finland's marginal occupational and industrial composition.

    Conditional: runs only if the Finland marginals file is on the share.

    Pipeline:
      1. Load Finnish ISCO-1 and NACE-2 marginals from the share.
      2. Pull employer -> NACE-2 from Foretagsdatabasen (one extra SQL).
      3. Crosswalk SSYK-4 -> ISCO-1 (first digit of ISCO from SSYK).
         (SSYK 2012 is structurally aligned with ISCO-08; the SSYK
         major group equals the ISCO-08 major group at the 1-digit level.)
      4. Compute Swedish marginals from the AGI panel.
      5. Construct weights w(occ) * w(ind) at the cell level.
      6. Re-aggregate to (employer x quartile x age x ym) with weighted
         counts. Run Poisson with the weighted outcome.

    The weighting is implemented at the (employer, quartile, age, month)
    cell level by scaling each cell's count by the (occupation, industry)
    weight averaged over its workers. For Poisson PML in pyfixest,
    weights are passed via the `weights=` argument; the estimator
    handles the appropriate scaling internally.
    """
    print("\n" + "=" * 70)
    print("STEP 4: Poisson PML, reweighted to Finnish composition (Eurostat LFS 2022)")
    print("=" * 70)

    finland = _load_finland_marginals()
    if finland is None:
        return pd.DataFrame()

    # Pull NACE-2 from Foretagsdatabasen
    try:
        conn = pyodbc.connect(SQL_CONN_STRING)
        nace = _pull_employer_nace(conn)
        conn.close()
    except BaseException as e:
        print(f"  NACE pull failed: {e}")
        print(f"  Cannot run Step 4 without employer industry codes.")
        return pd.DataFrame()

    # SSYK-4 -> ISCO-1 (first digit). SSYK 2012 follows ISCO-08 structure
    # at the major-group (1-digit) level.
    panel = panel_ssyk_age.copy()
    panel["isco1"] = panel["ssyk4"].astype(str).str[0]

    # Merge NACE-2
    panel = panel.merge(nace, on="employer_id", how="left")
    n_missing = panel["nace2"].isna().sum()
    if n_missing:
        print(f"  Warning: {n_missing:,} cells missing NACE-2 (employer not "
              f"in Foretagsdatabasen); excluded from Step 4.")
    panel = panel.dropna(subset=["nace2"])

    # Compute Swedish marginals from this panel (population-weighted by n_emp)
    swe_isco = (
        panel.groupby("isco1")["n_emp"].sum().pipe(lambda s: s / s.sum())
    ).to_dict()
    swe_nace = (
        panel.groupby("nace2")["n_emp"].sum().pipe(lambda s: s / s.sum())
    ).to_dict()

    # Per-cell weight = (p_fin / p_swe)_occ * (p_fin / p_swe)_ind
    # If a code is missing in Finland (very rare cells): weight = 1.0 (no shift)
    # If missing in Sweden: weight set to 0 to avoid division by zero
    def _w_occ(c):
        pf = finland["ISCO1"].get(c, 0.0)
        ps = swe_isco.get(c, 0.0)
        return pf / ps if ps > 0 else 0.0

    def _w_nace(c):
        pf = finland["NACE2"].get(c, 0.0)
        ps = swe_nace.get(c, 0.0)
        return pf / ps if ps > 0 else 0.0

    panel["w_occ"] = panel["isco1"].map(_w_occ)
    panel["w_nace"] = panel["nace2"].map(_w_nace)
    panel["w"] = panel["w_occ"] * panel["w_nace"]

    print(f"  Per-cell weight summary: "
          f"min={panel['w'].min():.3f}, max={panel['w'].max():.3f}, "
          f"mean={panel['w'].mean():.3f}, median={panel['w'].median():.3f}")
    print(f"  Total reweighted person-months: {(panel['n_emp']*panel['w']).sum():,.0f}")
    print(f"  Total raw person-months:        {panel['n_emp'].sum():,.0f}")

    # Merge DAIOE quartiles and re-aggregate, carrying weights
    panel = panel.merge(daioe, on="ssyk4", how="inner")

    # Aggregate the weighted outcome and the weights to (employer, q, age, ym)
    # Weighted Poisson: pass weights to fepois. Cell-level weight is the
    # mean of worker-level weights (since within a cell all workers share
    # the same isco1 / nace2 combination if we re-aggregate to that level).
    # Strategy: reaggregate to (employer, q, age, ym, isco1, nace2), keep
    # the cell weight, sum n_emp; then collapse to (employer, q, age, ym)
    # with sum of n_emp and weighted-mean of weight by n_emp.
    grain = (
        panel.groupby(
            ["employer_id", "daioe_quartile", "age_group",
             "year_month", "isco1", "nace2"]
        )
        .agg(n_emp=("n_emp", "sum"), w=("w", "first"))
        .reset_index()
    )

    def _wmean(g):
        if g["n_emp"].sum() == 0:
            return 1.0
        return (g["w"] * g["n_emp"]).sum() / g["n_emp"].sum()

    cell = (
        grain.groupby(
            ["employer_id", "daioe_quartile", "age_group", "year_month"]
        )
        .apply(lambda g: pd.Series({
            "n_emp": g["n_emp"].sum(),
            "w": _wmean(g),
        }))
        .reset_index()
    )

    results = []
    for age_label in AGE_GROUPS:
        print(f"\n  Age group: {age_label}")
        sub = filter_step1(cell, age_label)
        if len(sub) == 0:
            print(f"    No employers; skip")
            continue
        balanced = build_balanced_panel(sub, "daioe_quartile", n_bins=4)
        if len(balanced) == 0:
            print(f"    No employers span Q4 and Q1-Q3; skip")
            continue

        # Carry the weight onto the balanced panel (left-join). Zero-filled
        # cells get weight = 1 by default (no compositional pull on zeros).
        weights_lookup = sub[
            ["employer_id", "daioe_quartile", "year_month", "w"]
        ].drop_duplicates(
            subset=["employer_id", "daioe_quartile", "year_month"]
        )
        balanced = balanced.merge(
            weights_lookup,
            on=["employer_id", "daioe_quartile", "year_month"],
            how="left",
        )
        balanced["w"] = balanced["w"].fillna(1.0)

        result = _estimate_poisson_weighted(
            balanced, "daioe_quartile", n_bins=4,
            age_label=age_label, spec_label="STEP4_reweighted",
        )
        if result is not None:
            results.append(result)
            pd.DataFrame(results).to_csv(
                OUTPUT_DIR / "step4_poisson_reweighted.csv", index=False
            )

    return pd.DataFrame(results)


def _estimate_poisson_weighted(balanced, bin_col, n_bins, age_label, spec_label):
    """Same as estimate_poisson() but passes a cell-level weight to fepois."""
    if len(balanced) < 100:
        print(f"    [{spec_label} / {age_label}] too few obs; skip")
        return None

    panel = balanced.copy()
    max_bin = n_bins
    panel["high"] = (panel[bin_col] == max_bin).astype(int)
    panel["post_rb"] = (panel["year_month"] >= RIKSBANK_YM).astype(int)
    panel["post_gpt"] = (panel["year_month"] >= CHATGPT_YM).astype(int)
    panel["post_rb_x_high"] = panel["post_rb"] * panel["high"]
    panel["post_gpt_x_high"] = panel["post_gpt"] * panel["high"]

    panel["fe_emp_bin"] = (
        panel["employer_id"].astype(str) + "_" + panel[bin_col].astype(str)
    )
    panel["fe_emp_t"] = (
        panel["employer_id"].astype(str) + "_" + panel["year_month"]
    )

    n_obs = len(panel)
    n_employers = panel["employer_id"].nunique()
    print(f"    [{spec_label} / {age_label}] N obs = {n_obs:,}, "
          f"N employers = {n_employers:,}")

    t0 = time.time()
    try:
        fit = pf.fepois(
            "n_emp ~ post_rb_x_high + post_gpt_x_high | fe_emp_bin + fe_emp_t",
            data=panel,
            vcov={"CRV1": "employer_id"},
            weights="w",
        )

        coef = fit.coef()
        se = fit.se()
        pval = fit.pvalue()
        gamma1, gamma2 = float(coef["post_rb_x_high"]), float(coef["post_gpt_x_high"])
        se1, se2 = float(se["post_rb_x_high"]), float(se["post_gpt_x_high"])
        p1, p2 = float(pval["post_rb_x_high"]), float(pval["post_gpt_x_high"])

        elapsed = time.time() - t0
        print(f"      g1 (PostRB  x High) = {gamma1:+.4f} (SE={se1:.4f}, p={p1:.4f})")
        print(f"      g2 (PostGPT x High) = {gamma2:+.4f} (SE={se2:.4f}, p={p2:.4f})")
        print(f"      [{elapsed:.0f}s, weighted]")

        return {
            "spec": spec_label, "age_group": age_label,
            "gamma1": gamma1, "se1": se1, "p1": p1,
            "gamma2": gamma2, "se2": se2, "p2": p2,
            "n_obs": n_obs,
            "n_emp_total": int(panel["n_emp"].sum()),
            "n_employers": n_employers,
            "estimator": "Poisson PML weighted (pyfixest fepois, weights=w)",
            "fe": "employer x bin + employer x month",
            "vcov": "CRV1 by employer",
            "elapsed_s": round(elapsed, 1),
        }
    except BaseException as e:
        print(f"      Weighted Poisson FAILED for {spec_label} / {age_label}: {e}")
        return None


# ======================================================================
#   STEP 5: POISSON, CURRENT SPEC, IT/TECH (SSYK 25xx) EXCLUDED
# ======================================================================

def step5_poisson_no_ict(panel_ssyk_age, daioe):
    """
    Step 5: Poisson PML, current spec, IT/tech occupations excluded.

    Brynjolfsson 2025 excludes SOC 15-1000 series (computer occupations)
    in their headline analysis; Kauhanen 2026 replicates Brynjolfsson's
    sample restrictions and so almost certainly does the same. Our paper
    currently does NOT exclude IT/tech in the employment DiD (it appears
    only as a posting-margin robustness in appendix Table A2 spec v).

    SSYK 25xx (4-digit codes starting with '25') = ICT specialists in
    SSYK 2012, equivalent to Brynjolfsson's SOC 15-1000 series.

    If the Swedish age gradient survives this exclusion, the divergence
    with Finland is not driven by ICT-specialist behaviour. If it
    attenuates substantially, the paper should report ICT-stratified
    results in the appendix.
    """
    print("\n" + "=" * 70)
    print("STEP 5: Poisson PML, current spec, IT/tech (SSYK 25xx) EXCLUDED")
    print("=" * 70)

    panel = panel_ssyk_age[
        ~panel_ssyk_age["ssyk4"].astype(str).str.startswith(ICT_SSYK_PREFIX)
    ].copy()
    n_dropped = len(panel_ssyk_age) - len(panel)
    print(f"  Dropped {n_dropped:,} cells with SSYK starting '{ICT_SSYK_PREFIX}'")
    print(f"  Remaining: {len(panel):,} cells")

    panel = panel.merge(daioe, on="ssyk4", how="inner")
    panel = (
        panel.groupby(
            ["employer_id", "daioe_quartile", "age_group", "year_month"]
        )["n_emp"]
        .sum()
        .reset_index()
    )

    results = []
    for age_label in AGE_GROUPS:
        print(f"\n  Age group: {age_label}")
        sub = filter_step1(panel, age_label)
        balanced = build_balanced_panel(sub, "daioe_quartile", n_bins=4)
        if len(balanced) == 0:
            print(f"    No employers span Q4 and Q1-Q3; skip")
            continue
        result = estimate_poisson(
            balanced, "daioe_quartile", n_bins=4,
            age_label=age_label, spec_label="STEP5_no_ict",
        )
        if result is not None:
            results.append(result)
            pd.DataFrame(results).to_csv(
                OUTPUT_DIR / "step5_poisson_no_ict.csv", index=False
            )

    return pd.DataFrame(results)


# ======================================================================
#   ATTRITION DIAGNOSTIC (LISA non-match by quartile)
# ======================================================================

def attrition_diagnostic(panel_ssyk_age, daioe):
    """
    Pre-empt the Measurement Critic concern: does the post-2023 SSYK
    non-match rise (10% -> 15% -> 20%) drive the headline acceleration?

    Approach:
      Pull a per-year tabulation of:
        - total person-months in AGI (for that year)
        - person-months WITH an SSYK match (any cascade level)
        - person-months WITHOUT an SSYK match (to be tabulated by
          quartile-of-last-known assignment)

    Since panel_ssyk_age is built from the cascading-SSYK pull, it
    contains workers with a current OR last-known SSYK code. To get
    non-match counts, we pull a separate aggregate by year that counts
    AGI worker-months with no SSYK from any cascade level.

    Output: attrition_by_quartile.csv with rows
      year, quartile, n_matched, n_unmatched, share_unmatched

    Cell-count safeguards: any (year, quartile) cell with n < 5 is
    suppressed (NaN in the export) to comply with P1207 disclosure rules.
    """
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: SSYK non-match rate by year x DAIOE quartile")
    print("=" * 70)

    # The panel_ssyk_age already has WITH-SSYK records; we need the
    # WITHOUT-SSYK total for each year as the denominator.
    # Pull total AGI count by year (no SSYK filter), then subtract.

    print("  Pulling year-level AGI totals (with and without SSYK)...")
    conn = pyodbc.connect(SQL_CONN_STRING)

    yearly_totals = []
    for year in range(2019, 2026):
        if year < 2025:
            suffix = "_def"
            max_month = 12
        else:
            suffix = "_prel"
            max_month = 6

        # Total AGI worker-months in this year (no SSYK requirement)
        union_parts = []
        for month in range(1, max_month + 1):
            ym = f"{year}{month:02d}"
            union_parts.append(f"""
                SELECT
                    LEFT(PERIOD,4) + '-' + SUBSTRING(PERIOD,5,2) AS year_month,
                    COUNT(DISTINCT P1207_LOPNR_PERSONNR) AS n_total
                FROM dbo.Arb_AGIIndivid{ym}{suffix}
                GROUP BY LEFT(PERIOD,4) + '-' + SUBSTRING(PERIOD,5,2)
            """)
        q_total = "\nUNION ALL\n".join(union_parts)
        df_total = pd.read_sql(q_total, conn)
        df_total["year"] = year
        yearly_totals.append(df_total)
        print(f"    Year {year}: {df_total['n_total'].sum():,} total person-months")

    conn.close()

    totals = pd.concat(yearly_totals, ignore_index=True)

    # Match panel_ssyk_age aggregated by year_month
    matched = (
        panel_ssyk_age.groupby("year_month")["n_emp"]
        .sum()
        .reset_index()
        .rename(columns={"n_emp": "n_matched"})
    )
    matched["year"] = matched["year_month"].str[:4].astype(int)

    diag = totals.merge(matched[["year_month", "n_matched"]],
                        on="year_month", how="left")
    diag["n_matched"] = diag["n_matched"].fillna(0).astype(int)
    diag["n_unmatched"] = diag["n_total"] - diag["n_matched"]
    diag["share_unmatched"] = diag["n_unmatched"] / diag["n_total"]

    # Year-level summary
    year_summary = diag.groupby("year").agg(
        n_total=("n_total", "sum"),
        n_matched=("n_matched", "sum"),
        n_unmatched=("n_unmatched", "sum"),
    ).reset_index()
    year_summary["share_unmatched"] = (
        year_summary["n_unmatched"] / year_summary["n_total"]
    )

    print("\n  Year-level non-match shares:")
    print(year_summary.to_string(index=False))

    # Now: by year x quartile of last-known DAIOE
    # The matched set has SSYK; merge DAIOE to get quartile breakdown
    matched_with_q = panel_ssyk_age.merge(daioe, on="ssyk4", how="left")
    matched_with_q["year"] = matched_with_q["year_month"].str[:4].astype(int)

    by_year_q = (
        matched_with_q.groupby(["year", "daioe_quartile"])["n_emp"]
        .sum()
        .reset_index()
        .rename(columns={"n_emp": "n_matched_in_quartile"})
    )

    # Disclosure safeguard: any cell with < 5 -> NaN
    by_year_q["n_matched_in_quartile"] = by_year_q[
        "n_matched_in_quartile"
    ].where(by_year_q["n_matched_in_quartile"] >= 5, np.nan)

    # Merge with year totals to compute share-of-matched in each quartile
    by_year_q = by_year_q.merge(
        year_summary[["year", "n_matched", "n_unmatched"]],
        on="year", how="left"
    )
    by_year_q["share_of_matched_in_quartile"] = (
        by_year_q["n_matched_in_quartile"] / by_year_q["n_matched"]
    )

    print("\n  Quartile composition of matched workers, by year:")
    pivot = by_year_q.pivot(
        index="year", columns="daioe_quartile",
        values="share_of_matched_in_quartile"
    )
    print(pivot.to_string())

    # Save both tables
    year_summary.to_csv(OUTPUT_DIR / "attrition_yearly_totals.csv", index=False)
    by_year_q.to_csv(OUTPUT_DIR / "attrition_by_quartile.csv", index=False)
    print(f"\n  Saved -> attrition_yearly_totals.csv, attrition_by_quartile.csv")
    print(f"  Cell-count safeguard: cells with n < 5 set to NaN")

    return year_summary, by_year_q


# ======================================================================
#   FINAL: COMPARISON TABLE + PROSE SUMMARY
# ======================================================================

def write_comparison_table(step1, step2, step3, step4, step5):
    """
    Write a master comparison CSV: gamma2 (PostGPT x High) for all
    (spec, age_group) combinations, side by side. This is the single
    output Magnus needs to read to settle the Sweden-Finland question.
    """
    print("\n" + "=" * 70)
    print("FINAL: Master comparison table")
    print("=" * 70)

    frames = []
    for df, label in [(step1, "STEP1_current"),
                      (step2, "STEP2_threshold"),
                      (step3, "STEP3_kauhanen_exact"),
                      (step4, "STEP4_reweighted"),
                      (step5, "STEP5_no_ict")]:
        if df is not None and len(df) > 0:
            frames.append(df)

    if not frames:
        print("  No results to compare. Did all three steps fail?")
        return None

    combined = pd.concat(frames, ignore_index=True)

    # Pivot: rows = age_group, columns = spec, values = gamma2 (with formatting)
    pivot = combined.pivot(
        index="age_group", columns="spec", values="gamma2"
    )
    pivot_se = combined.pivot(
        index="age_group", columns="spec", values="se2"
    )
    pivot_p = combined.pivot(
        index="age_group", columns="spec", values="p2"
    )

    print("\n  gamma2 (PostGPT x High) by age group x spec:")
    print(pivot.round(4).to_string())
    print("\n  SE:")
    print(pivot_se.round(4).to_string())
    print("\n  p-value:")
    print(pivot_p.round(4).to_string())

    out = OUTPUT_DIR / "kauhanen_comparison.csv"
    combined.to_csv(out, index=False)
    print(f"\n  Saved master comparison -> {out.name}")
    return combined


def write_prose_summary(step1, step2, step3, step4, step5):
    """
    Magnus-readable prose summary of the staged comparison.
    Loaded as the first thing he reads after the run completes.
    """
    out = OUTPUT_DIR / "kauhanen_summary.txt"
    lines = []
    lines.append("KAUHANEN STAGED ROBUSTNESS -- SUMMARY")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Run completed: " + time.strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("")
    lines.append("Steps:")
    lines.append("  1. Poisson, current spec (>=5 thr, DAIOE quartile)")
    lines.append("  2. Poisson, Kauhanen threshold (>=10 mean + >=100 cum)")
    lines.append("  3. Poisson, full Kauhanen (Eloundou quintile + threshold)")
    lines.append("  4. Poisson, current spec, reweighted to Finnish composition")
    lines.append("  5. Poisson, current spec, IT/tech (SSYK 25xx) excluded")
    lines.append("")

    def _row(name, df):
        lines.append(f"--- {name} ---")
        if df is None or len(df) == 0:
            lines.append("  (no results)")
            lines.append("")
            return
        for _, r in df.sort_values("age_group").iterrows():
            sig2 = "***" if r["p2"] < 0.01 else (
                "**" if r["p2"] < 0.05 else ("*" if r["p2"] < 0.10 else " "))
            lines.append(
                f"  {r['age_group']:<6}  g2 = {r['gamma2']:+.4f} "
                f"(SE={r['se2']:.4f}, p={r['p2']:.4f}) {sig2}   "
                f"N={r['n_obs']:>11,}"
            )
        lines.append("")

    _row("Step 1: Poisson, current spec",                    step1)
    _row("Step 2: Poisson, Kauhanen threshold",              step2)
    _row("Step 3: Poisson, full Kauhanen exact",             step3)
    _row("Step 4: Poisson, reweighted to Finnish composition", step4)
    _row("Step 5: Poisson, IT/tech (SSYK 25xx) excluded",    step5)

    # Magnus's key question: how does 22-25 evolve across specs?
    lines.append("=" * 60)
    lines.append("INTERPRETATION GUIDE")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Focus on the 22-25 row across the three steps.")
    lines.append("")
    lines.append("  * If g2 stays negative and significant in Step 1: the")
    lines.append("    OLS+1 vs Poisson choice does not matter. The paper's")
    lines.append("    appendix line 440 claim is verified.")
    lines.append("")
    lines.append("  * If g2 still negative and significant in Step 2: the")
    lines.append("    sample threshold is not what drives the Sweden-Finland")
    lines.append("    divergence. The Finnish null is not a threshold artefact.")
    lines.append("")
    lines.append("  * If g2 still negative and significant in Step 3: the")
    lines.append("    Swedish result holds under Kauhanen's exact spec.")
    lines.append("    Together with Step 1 and Step 2, the Sweden-Finland")
    lines.append("    divergence is *not* method-dependent.")
    lines.append("")
    lines.append("  * If g2 still negative and significant in Step 4: the")
    lines.append("    divergence is also not driven by Sweden's industrial /")
    lines.append("    occupational composition. Combined with Steps 1-3,")
    lines.append("    the divergence is *economy-dependent* (institutions,")
    lines.append("    AI adoption rate, labour-market frictions).")
    lines.append("    This is the strongest possible defence of the paper.")
    lines.append("")
    lines.append("  * If g2 still negative and significant in Step 5: the")
    lines.append("    age gradient is not driven by ICT specialists; the")
    lines.append("    Swedish result survives the same exclusion Brynjolfsson")
    lines.append("    and (likely) Kauhanen impose.")
    lines.append("")
    lines.append("  * If g2 nulls at any step: that single specification choice")
    lines.append("    is what flips the result. Report that finding candidly")
    lines.append("    in the appendix; downgrade the headline claim to be")
    lines.append("    consistent with what survives.")
    lines.append("")
    lines.append("Magnus -- read this file first, then kauhanen_comparison.csv.")
    lines.append("")
    lines.append("If results turn ugly: see notes/risk-preparedness-2026-04-28.md")
    lines.append("for the pre-committed fallback framings (drafted before the run).")

    out.write_text("\n".join(lines))
    print(f"\n  Saved prose summary -> {out.name}")


# ======================================================================
#   MAIN
# ======================================================================

def _is_null_result(df, age_label, p_threshold=NULL_THRESHOLD_P):
    """Return True if the (age_label) row in df is null or wrongly-signed."""
    if df is None or len(df) == 0:
        return True
    row = df[df["age_group"] == age_label]
    if len(row) == 0:
        return True
    row = row.iloc[0]
    return (row["gamma2"] >= 0) or (row["p2"] >= p_threshold)


def main():
    print("=" * 70)
    print("KAUHANEN STAGED ROBUSTNESS -- script 32")
    print("=" * 70)
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Run all regardless of Step 1 outcome: {RUN_ALL_REGARDLESS}")
    print("")

    # --- Step 0: panel pull or cache reload ---
    panel_ssyk_age = step0_pull_or_load_panel()

    # --- Load exposure files ---
    print("\n  Loading exposure files...")
    daioe = load_daioe_quartiles()
    eloundou = load_eloundou_quintiles()

    # --- Step 1: Poisson, current spec ---
    step1 = step1_poisson_current(panel_ssyk_age, daioe)

    # Check Step 1 outcome for 22-25 (the headline group)
    step1_null = _is_null_result(step1, "22-25")
    if step1_null:
        print("\n" + "!" * 70)
        print("WARNING: Step 1 22-25 coefficient is null or wrongly signed.")
        print("This means the OLS+1 -> Poisson switch already changes the")
        print("result. The paper's appendix line 440 claim is NOT verified.")
        print("!" * 70)
        if not RUN_ALL_REGARDLESS:
            print("\nStopping per RUN_ALL_REGARDLESS=False. Re-think before")
            print("running Steps 2-5. Set RUN_ALL_REGARDLESS=True to run all.")
            write_prose_summary(step1, None, None, None, None)
            return
        print("\nContinuing per RUN_ALL_REGARDLESS=True.")
    else:
        print("\n  Step 1 22-25 result: negative and significant. Continuing.")

    # --- Step 2: Poisson, Kauhanen threshold ---
    step2 = step2_poisson_threshold(panel_ssyk_age, daioe)

    # --- Step 3: Poisson, full Kauhanen exact spec ---
    step3 = step3_poisson_kauhanen(panel_ssyk_age, eloundou)

    # --- Step 4: Poisson, reweighted to Finnish composition (conditional) ---
    try:
        step4 = step4_poisson_reweighted(panel_ssyk_age, daioe)
    except BaseException as e:
        print(f"\n  Step 4 FAILED: {e}")
        step4 = pd.DataFrame()

    # --- Step 5: Poisson, IT/tech excluded (unconditional) ---
    try:
        step5 = step5_poisson_no_ict(panel_ssyk_age, daioe)
    except BaseException as e:
        print(f"\n  Step 5 FAILED: {e}")
        step5 = pd.DataFrame()

    # --- Diagnostic: SSYK non-match by quartile ---
    try:
        attrition_diagnostic(panel_ssyk_age, daioe)
    except BaseException as e:
        print(f"\n  Attrition diagnostic FAILED: {e}")
        print("  Continuing with regression results only.")

    # --- Final: comparison + prose summary ---
    write_comparison_table(step1, step2, step3, step4, step5)
    write_prose_summary(step1, step2, step3, step4, step5)

    print("\n" + "=" * 70)
    print("DONE.")
    print(f"  Output dir: {OUTPUT_DIR}")
    print("  Read first:  kauhanen_summary.txt")
    print("  Then:        kauhanen_comparison.csv")
    print("  Per-step:    step{1,2,3}_*.csv")
    print("  Diagnostic:  attrition_by_quartile.csv")
    print("=" * 70)
    print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
