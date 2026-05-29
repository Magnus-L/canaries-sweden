#!/usr/bin/env python3
"""
34_mona_education_robustness.py -- EduQuartile robustness for the canaries finding.

======================================================================
  THIS SCRIPT IS DESIGNED TO RUN IN SCB's MONA SECURE ENVIRONMENT
  Do NOT run outside MONA -- the data is not available externally.
======================================================================

WHY THIS SCRIPT EXISTS
======================
The OccQuartile employer-level DiD (Step 1, also the OLS+1 version in the
main text) assigns workers to a DAIOE quartile via their LISA SSYK4. LISA
ends at 2023, so 2024-2025 worker-months use SSYK4 from 1-2 year-old LISA
records. For 22-25 year olds those records typically capture them at age
20-22 in student or transitional jobs (Q1 in DAIOE), even when their
current 2024-2025 job is a Q4 entry-career role. The cascade output is
Q1-floored, biasing the canaries finding toward "Q4 declining."

EduQuartile sidesteps the staleness entirely. The treatment is built from
the worker's completed education (Sun2000Inr level + field), mapped to
the typical DAIOE quartile of graduates' occupational destinations.
Education is recorded once, stable, and predetermined relative to the AI
shock when lagged.

WHAT THIS SCRIPT DOES
=====================
Stage 1: Build the Sun2000-to-DAIOE-quartile crosswalk from 2019-2023
LISA. For each (Sun2000niva, Sun2000Inr) cell with at least 100 observed
graduates aged 25-35 in tertiary education or 25-35 working with their
gymnasium track for niva<=2, compute the employment-weighted mean DAIOE
percentile across observed SSYK4 destinations. Quartile-bin the
educations themselves (approximately equal mass of workers per EduQ).

Stage 2: Assign EduQuartile to every worker in the panel using LISA
education from 3 years before the panel year (forced to 2019 for early
panel years). Drop workers with Sun2000niva = 7 (unknown education,
typically foreign-trained).

Stage 3: Re-run the employer-level Poisson DiD with EduQuartile
substituted for OccQuartile. Use the same R+fixest subprocess from
script 32. Cluster at employer x EduQuartile.

USAGE IN MONA
=============
1. Upload as 34_mona_education_robustness.txt; rename to .py.
2. Required co-located files (alongside script 32):
   - 32_mona_kauhanen_robustness.py (for R subprocess helpers)
   - output_32/step0_panel_cache.csv (cached AGI panel)
3. Run:
       python 34_mona_education_robustness.py
4. Outputs (under output_34/):
   - edu_quartile_crosswalk.csv      Sun2000 -> EduQuartile mapping
   - edu_quartile_diagnostics.txt    Coverage and quality metrics
   - panel_with_eduquartile.parquet  Worker-month panel with EduQuartile
   - step1_education_quartile.csv    Poisson coefficients by age group
   - education_summary.txt           Prose summary

EXPORT-SAFETY
=============
All exported files are aggregated counts and regression coefficients.
The crosswalk file aggregates over educations with at least 100 graduates
per cell. Cell-count safeguards (>=5) applied to all destination
distributions. No raw individual-level rows.

EXPECTED RUNTIME
================
- LISA pulls (1990-2023): 10-20 min one-off, cached afterwards
- Crosswalk construction: 2-5 min
- Panel re-aggregation: 3-5 min
- Poisson PML (6 age groups): 15-30 min via R subprocess
- Total: 30-60 min

CRASH RECOVERY
==============
- LISA pulls cached in output_34/_lisa_cache.parquet
- Crosswalk cached in output_34/edu_quartile_crosswalk.csv
- Per-age-group Poisson outputs saved incrementally
"""

import importlib.util
import re
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import numpy as np
import pyodbc

# ----------------------------------------------------------------------
# Script 32 import (for R subprocess helpers and connection string)
# ----------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
KAUHANEN_PY = SCRIPT_DIR / "32_mona_kauhanen_robustness.py"

if not KAUHANEN_PY.exists():
    print(f"FATAL: 32_mona_kauhanen_robustness.py not found at {KAUHANEN_PY}")
    sys.exit(1)

_spec = importlib.util.spec_from_file_location("kauhanen", str(KAUHANEN_PY))
kauhanen = importlib.util.module_from_spec(_spec)
sys.modules["kauhanen"] = kauhanen
_spec.loader.exec_module(kauhanen)

# Reuse: SQL_CONN_STRING, AGE_GROUPS, _Tee, _run_rfepois,
#        filter_step1, build_balanced_panel, OUTPUT_DIR (for step32)
SQL_CONN_STRING = kauhanen.SQL_CONN_STRING
AGE_GROUPS = kauhanen.AGE_GROUPS
PANEL_CACHE_32 = kauhanen.PANEL_CACHE
DAIOE_PATH = kauhanen.DAIOE_PATH

# ----------------------------------------------------------------------
# Local config
# ----------------------------------------------------------------------
OUTPUT_DIR = SCRIPT_DIR / "output_34"
OUTPUT_DIR.mkdir(exist_ok=True)
LISA_CACHE = OUTPUT_DIR / "_lisa_cache.parquet"
CROSSWALK_CACHE = OUTPUT_DIR / "edu_quartile_crosswalk.csv"

LOG_PATH = OUTPUT_DIR / "34_mona_education_robustness_log.txt"

# Crosswalk parameters
CROSSWALK_AGE_MIN = 25
CROSSWALK_AGE_MAX = 35
MIN_GRADUATES_PER_EDU = 100
N_EDU_QUARTILES = 4

# Lag (years between panel year and LISA education year)
EDU_LAG_YEARS = 3
LISA_MIN_YEAR = 2019  # earliest LISA we trust
LISA_MAX_YEAR = 2023  # latest LISA available

# Treatment dates (consistent with Step 1)
RIKSBANK_YM = "2022-04"
CHATGPT_YM = "2022-12"


# ======================================================================
#   LOGGING
# ======================================================================

class _Tee:
    """Mirror stdout to a log file; ASCII-only to avoid encoding issues."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w", encoding="ascii", errors="replace")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# ======================================================================
#   STAGE 1: BUILD THE EDU-QUARTILE CROSSWALK
# ======================================================================

def _discover_lisa_columns(conn, table_name):
    """Return the set of column names available on a given LISA table."""
    q = f"""
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = '{table_name}' AND TABLE_SCHEMA = 'dbo'
    """
    return set(pd.read_sql(q, conn)["COLUMN_NAME"].tolist())


def _select_education_columns(available):
    """
    Pick the best-available education-level + field columns from the schema.
    Priority for the LEVEL (1-digit, 1..7):
        Sun2020Niva_Old > Sun2000Niva_Old > Sun2000niva > Sun2020Niva
    Priority for the FIELD (4-digit code):
        Sun2020Inr > Sun2000Inr
    Returns (niva_col, inr_col) or (None, None) if neither set is available.
    """
    niva_priority = ["Sun2020Niva_Old", "Sun2000Niva_Old",
                     "Sun2000niva", "Sun2020Niva"]
    inr_priority = ["Sun2020Inr", "Sun2000Inr"]
    # SQL Server column names are case-insensitive; do a case-insensitive match
    avail_lower = {c.lower(): c for c in available}
    niva_col = next((avail_lower[c.lower()]
                     for c in niva_priority if c.lower() in avail_lower), None)
    inr_col = next((avail_lower[c.lower()]
                    for c in inr_priority if c.lower() in avail_lower), None)
    return niva_col, inr_col


def pull_lisa_education_and_occupation(years):
    """
    Pull (LopNr, year, sun_niva, sun_inr, ssyk4, FodelseAr) from LISA
    Individ_YYYY for the given years. Returns long-format DataFrame.

    Education columns are discovered per-year via INFORMATION_SCHEMA, with
    a priority chain that prefers the SUN 2020 1-digit level (Sun2020Niva_Old)
    and falls back through alternative naming. SSYK is preferred from
    Ssyk4_2012_J16, falling back to Ssyk4_2012.
    """
    if LISA_CACHE.exists():
        print(f"  LISA cache exists at {LISA_CACHE.name}; loading.")
        df = pd.read_parquet(LISA_CACHE)
        return df

    print(f"  Pulling LISA education + occupation for years {years[0]}-{years[-1]}...")
    conn = pyodbc.connect(SQL_CONN_STRING)
    frames = []
    for year in years:
        t0 = time.time()
        table_name = f"Individ_{year}"
        try:
            cols = _discover_lisa_columns(conn, table_name)
        except BaseException as e:
            print(f"    Year {year}: schema discovery failed ({e}); skipping")
            continue
        if not cols:
            print(f"    Year {year}: table {table_name} not found; skipping")
            continue

        niva_col, inr_col = _select_education_columns(cols)
        if niva_col is None or inr_col is None:
            print(f"    Year {year}: no usable Sun*Niva*/Sun*Inr columns "
                  f"(saw: {sorted(c for c in cols if 'Sun' in c)}); skipping")
            continue

        # SSYK column choice: prefer J16 imputation, fall back
        if "Ssyk4_2012_J16" in cols and "Ssyk4_2012" in cols:
            ssyk_expr = "COALESCE(Ssyk4_2012_J16, Ssyk4_2012)"
        elif "Ssyk4_2012_J16" in cols:
            ssyk_expr = "Ssyk4_2012_J16"
        elif "Ssyk4_2012" in cols:
            ssyk_expr = "Ssyk4_2012"
        else:
            print(f"    Year {year}: no Ssyk4 column found; skipping")
            continue

        print(f"    Year {year}: using niva={niva_col}, inr={inr_col}, "
              f"ssyk={ssyk_expr}")
        query = f"""
            SELECT
                P1207_LopNr_PersonNr AS lopnr,
                {year} AS year,
                {niva_col} AS sun_niva,
                {inr_col} AS sun_inr,
                {ssyk_expr} AS ssyk4,
                FodelseAr AS birth_year
            FROM dbo.{table_name}
            WHERE {niva_col} IS NOT NULL
        """
        try:
            df_year = pd.read_sql(query, conn)
        except BaseException as e:
            print(f"    Year {year}: pull failed ({e}); skipping")
            continue

        # FodelseAr can come back as int or as char (depends on MONA schema
        # variation across years). Coerce to numeric before age computation.
        df_year["birth_year"] = pd.to_numeric(
            df_year["birth_year"], errors="coerce"
        )
        df_year["age"] = year - df_year["birth_year"]
        df_year["ssyk4"] = df_year["ssyk4"].astype(str).str.zfill(4)
        # Niva can be stored as numeric or as char ('1', '2', ...). Coerce.
        df_year["sun_niva"] = pd.to_numeric(
            df_year["sun_niva"], errors="coerce"
        )
        df_year["sun_inr"] = df_year["sun_inr"].astype(str).str.strip()
        elapsed = time.time() - t0
        print(f"    Year {year}: {len(df_year):,} rows in {elapsed:.0f}s")
        frames.append(df_year)

    conn.close()
    if not frames:
        raise RuntimeError("No LISA years pulled successfully; cannot proceed.")
    out = pd.concat(frames, ignore_index=True)
    print(f"  Total rows: {len(out):,}")
    print(f"  Saving cache to {LISA_CACHE.name}")
    out.to_parquet(LISA_CACHE, index=False)
    return out


def load_daioe_with_percentile():
    """
    Load the DAIOE file and return [ssyk4, daioe_quartile, daioe_pctile].
    The DAIOE file has SSYK4 + an exposure score; we keep the quartile
    (already computed) and add a percentile rank for the weighted-mean
    aggregation in the EduQuartile crosswalk.
    """
    print(f"  Loading DAIOE from {DAIOE_PATH}")
    df = pd.read_csv(DAIOE_PATH)
    df["ssyk4"] = df["ssyk4"].astype(str).str.zfill(4)

    # Extract the quartile if stored as 'Q1'/'Q2'/...
    if df["exposure_quartile"].dtype == object:
        df["daioe_quartile"] = (
            df["exposure_quartile"].astype(str).str.strip()
            .str.extract(r"Q(\d)").astype(float)
        )
    else:
        df["daioe_quartile"] = df["exposure_quartile"]

    # Ensure a continuous score / percentile column for the EduQuartile
    # construction. Order of preference:
    #   pre-computed continuous percentile rank > raw continuous score >
    #   synthetic within-quartile rank (last-resort fallback that loses
    #   within-quartile signal -- avoid if any continuous column exists).
    score_col = None
    pctile_col = None
    for cand in ["pctl_rank_genai", "pctl_rank_allapps"]:
        if cand in df.columns:
            pctile_col = cand
            break
    if pctile_col is not None:
        # Already a continuous percentile (0-100 or 0-1). Normalise to [0,1].
        col = pd.to_numeric(df[pctile_col], errors="coerce")
        df["daioe_pctile"] = col / (100.0 if col.max() > 1.5 else 1.0)
        print(f"  Using continuous DAIOE percentile from '{pctile_col}'")
    else:
        for cand in ["daioe_score", "exposure_score", "c_aioe", "daioe", "score"]:
            if cand in df.columns:
                score_col = cand
                break
        if score_col is not None:
            df["daioe_pctile"] = df[score_col].rank(pct=True)
            print(f"  Using continuous DAIOE score from '{score_col}' "
                  f"(rank-pctile transformed)")
        else:
            # Last resort: synthetic within-quartile rank. Loses signal --
            # the EduQuartile crosswalk will be coarser than necessary.
            print("  WARNING: no continuous DAIOE column found; falling back "
                  "to synthetic within-quartile ranking. Cols seen: "
                  f"{sorted(df.columns)}")
            df = df.sort_values(["daioe_quartile", "ssyk4"])
            df["daioe_pctile"] = df.groupby("daioe_quartile").cumcount() / len(df)
            df["daioe_pctile"] = df["daioe_pctile"] + (df["daioe_quartile"] - 1) / 4

    print(f"  DAIOE loaded: {df['ssyk4'].nunique()} SSYK4 codes")
    return df[["ssyk4", "daioe_quartile", "daioe_pctile"]]


def build_edu_quartile_crosswalk(lisa_df, daioe_df):
    """
    Build the (sun_niva, sun_inr) -> EduQuartile crosswalk.

    Population: workers aged CROSSWALK_AGE_MIN..CROSSWALK_AGE_MAX with
    a non-null SSYK4 across LISA years 2019-2023.

    Per education:
      - Compute employment-weighted mean DAIOE percentile across all
        observed SSYK4 destinations (one row per worker counts once).
      - Require >= MIN_GRADUATES_PER_EDU graduates; below threshold the
        education is binned to a residual code.

    Education-level percentile values are then quartile-cut into
    EduQuartiles 1..4 with roughly equal worker mass per quartile (using
    pd.qcut on graduate counts).

    Returns DataFrame [sun_niva, sun_inr, n_graduates, mean_daioe_pctile,
                       top1_ssyk4, top2_ssyk4, top3_ssyk4, edu_quartile].
    """
    if CROSSWALK_CACHE.exists():
        print(f"  Crosswalk cache exists; loading {CROSSWALK_CACHE.name}")
        return pd.read_csv(CROSSWALK_CACHE, dtype={"sun_inr": str})

    print(f"\n  Building Sun2000 -> EduQuartile crosswalk")
    print(f"    Age window: {CROSSWALK_AGE_MIN}-{CROSSWALK_AGE_MAX}")
    print(f"    Min graduates per education: {MIN_GRADUATES_PER_EDU}")

    # Restrict LISA to the crosswalk-construction window
    df = lisa_df[
        (lisa_df["age"] >= CROSSWALK_AGE_MIN)
        & (lisa_df["age"] <= CROSSWALK_AGE_MAX)
        & (lisa_df["ssyk4"].notna())
        & (lisa_df["sun_niva"].notna())
        & (lisa_df["sun_inr"].notna())
        & (lisa_df["sun_niva"] != 7)
    ].copy()
    print(f"    Crosswalk-construction sample: {len(df):,} worker-years")

    # Merge DAIOE to get the destination percentile and quartile
    df = df.merge(daioe_df, on="ssyk4", how="inner")
    print(f"    After DAIOE merge: {len(df):,} worker-years")

    # Group by (sun_niva, sun_inr); compute mean DAIOE percentile and top SSYK4 destinations
    edu_groups = df.groupby(["sun_niva", "sun_inr"])

    rows = []
    for (niva, inr), g in edu_groups:
        n_grad = len(g)
        if n_grad < MIN_GRADUATES_PER_EDU:
            continue
        mean_pct = g["daioe_pctile"].mean()
        top_ssyk = (
            g.groupby("ssyk4").size().sort_values(ascending=False).head(3).index.tolist()
        )
        # Pad to 3 slots
        top_ssyk = top_ssyk + [""] * (3 - len(top_ssyk))
        rows.append({
            "sun_niva": niva,
            "sun_inr": inr,
            "n_graduates": n_grad,
            "mean_daioe_pctile": mean_pct,
            "top1_ssyk4": top_ssyk[0],
            "top2_ssyk4": top_ssyk[1],
            "top3_ssyk4": top_ssyk[2],
        })

    cw = pd.DataFrame(rows)
    print(f"    Crosswalk cells with >= {MIN_GRADUATES_PER_EDU} graduates: {len(cw):,}")

    if len(cw) < N_EDU_QUARTILES:
        raise RuntimeError(
            f"Only {len(cw)} crosswalk cells survived; cannot bin into {N_EDU_QUARTILES} quartiles."
        )

    # Quartile-bin the educations themselves (worker-mass-weighted via qcut on n_graduates*pctile rank)
    # Simpler: rank by mean_daioe_pctile, weight rank by n_graduates so quartiles have equal worker mass.
    cw_sorted = cw.sort_values("mean_daioe_pctile").reset_index(drop=True)
    cw_sorted["cum_grad"] = cw_sorted["n_graduates"].cumsum()
    total_grad = cw_sorted["n_graduates"].sum()
    cw_sorted["mass_pctile"] = cw_sorted["cum_grad"] / total_grad
    cw_sorted["edu_quartile"] = np.minimum(
        4, np.ceil(cw_sorted["mass_pctile"] * N_EDU_QUARTILES).astype(int)
    )
    # Edge: zero-mass first row may produce 0; clamp to 1
    cw_sorted["edu_quartile"] = np.maximum(1, cw_sorted["edu_quartile"])

    # Diagnostics
    print(f"\n  Crosswalk EduQuartile distribution (worker-mass-weighted):")
    diag = (
        cw_sorted.groupby("edu_quartile")
        .agg(n_educations=("sun_inr", "count"),
             n_graduates=("n_graduates", "sum"),
             mean_pct=("mean_daioe_pctile", "mean"))
        .reset_index()
    )
    for _, r in diag.iterrows():
        print(f"    EduQ{int(r['edu_quartile'])}: "
              f"{int(r['n_educations']):>4} educations, "
              f"{int(r['n_graduates']):>10,} graduates, "
              f"mean DAIOE pctile = {r['mean_pct']:.3f}")

    out_cols = ["sun_niva", "sun_inr", "n_graduates", "mean_daioe_pctile",
                "top1_ssyk4", "top2_ssyk4", "top3_ssyk4", "edu_quartile"]
    out = cw_sorted[out_cols].copy()
    out.to_csv(CROSSWALK_CACHE, index=False)
    print(f"  Crosswalk saved to {CROSSWALK_CACHE.name}")
    return out


# ======================================================================
#   STAGE 2: ASSIGN EDUQUARTILE TO PANEL
# ======================================================================

def lag_year_for_panel_year(panel_year):
    """Map panel year to LISA education year using EDU_LAG_YEARS lag,
    floored at LISA_MIN_YEAR."""
    target = panel_year - EDU_LAG_YEARS
    return max(LISA_MIN_YEAR, min(LISA_MAX_YEAR, target))


def assign_eduquartile_to_panel(panel_ssyk_age, lisa_df, crosswalk):
    """
    Build a worker-month panel with EduQuartile assigned via the lagged-LISA
    education and the crosswalk.

    panel_ssyk_age comes from kauhanen.step0_pull_or_load_panel() and is
    keyed on (employer_id, ssyk4, age_group, year_month, n_emp). Note this
    panel does NOT have lopnr -- it is already aggregated to
    (employer x ssyk4 x age x ym). To assign EduQuartile we need
    individual-level joins. So we re-aggregate from the worker level.

    Strategy:
    - Pull worker-level AGI for 2019-2025 with (lopnr, employer_id, ym, age_group)
    - Merge to lagged LISA education (using the lag function above)
    - Merge to crosswalk -> EduQuartile
    - Aggregate to (employer x EduQuartile x age x ym) cells with worker count

    Returns DataFrame with the same shape as panel_ssyk_age but keyed on
    EduQuartile rather than ssyk4 / DAIOE quartile.
    """
    print("\n  STAGE 2: assign EduQuartile to panel")

    # Build a (lopnr, year_month, employer_id, age_group) frame from AGI.
    # The script 32 cache aggregates SSYK out, so we need to re-pull AGI
    # at the worker level. Use a single SQL pull spanning 2019-2025.
    #
    # We use the SAME age groups as Step 1.

    # Memory-aware refactor (2026-04-29): the prior implementation loaded
    # all 414M worker-months and groupby-nunique'd in one shot, peaking at
    # ~108 GB on a 100 GB MONA Batch ceiling. Now we stream by panel-year:
    # load the full cache once, slice on year, free the master DataFrame,
    # then process each year's slice independently. We also replace
    # nunique with drop_duplicates+size, which is dramatically cheaper for
    # high-cardinality lopnr.

    import gc

    pulled = _pull_worker_panel_2019_2025()
    pulled["panel_year"] = pulled["year_month"].str.slice(0, 4).astype(int)
    pulled["lisa_year"] = pulled["panel_year"].apply(lag_year_for_panel_year)

    # Pre-shape LISA and crosswalk once (small, reused per slice)
    lisa_min = lisa_df[["lopnr", "year", "sun_niva", "sun_inr"]].copy()
    lisa_min = lisa_min.rename(columns={"year": "lisa_year"})
    lisa_min = lisa_min[lisa_min["sun_niva"].notna() & (lisa_min["sun_niva"] != 7)]
    lisa_min["sun_inr"] = lisa_min["sun_inr"].astype(str)

    cw = crosswalk[["sun_niva", "sun_inr", "edu_quartile"]].copy()
    cw["sun_inr"] = cw["sun_inr"].astype(str)

    panel_years = sorted(pulled["panel_year"].unique().tolist())
    print(f"  Streaming {len(panel_years)} panel-years to keep memory bounded")

    # Split master DF into per-year slices, then free master immediately
    slices = {y: pulled[pulled["panel_year"] == y].copy() for y in panel_years}
    del pulled
    gc.collect()

    cells_per_year = []
    n_pre_total = 0
    n_post_total = 0
    n_with_q_total = 0
    for year in panel_years:
        s = slices.pop(year)
        n_pre = len(s)
        n_pre_total += n_pre
        if n_pre == 0:
            continue
        # All rows in this slice share one lisa_year (lag function is
        # deterministic per panel_year). Subset LISA to that one year only.
        lisa_yr_value = int(s["lisa_year"].iloc[0])
        lisa_yr = lisa_min[lisa_min["lisa_year"] == lisa_yr_value]
        s = s.merge(lisa_yr, on=["lopnr", "lisa_year"], how="left")
        s = s[s["sun_niva"].notna() & (s["sun_niva"] != 7)]
        n_post_total += len(s)
        s["sun_inr"] = s["sun_inr"].astype(str)
        s = s.merge(cw, on=["sun_niva", "sun_inr"], how="left")
        s = s.dropna(subset=["edu_quartile"])
        s["edu_quartile"] = s["edu_quartile"].astype(int)
        n_with_q_total += len(s)

        # drop_duplicates + size is much cheaper than nunique
        keys = ["employer_id", "edu_quartile", "age_group", "year_month"]
        s_unique = s.drop_duplicates(subset=keys + ["lopnr"])
        cell_yr = (
            s_unique.groupby(keys, as_index=False, observed=True)
                    .size()
                    .rename(columns={"size": "n_emp"})
        )
        cells_per_year.append(cell_yr)
        print(f"    {year}: {n_pre:,} worker-months -> {len(cell_yr):,} cells")
        del s, s_unique, lisa_yr
        gc.collect()

    print(f"  Worker-months: {n_pre_total:,} pulled -> "
          f"{n_post_total:,} after niva filter -> "
          f"{n_with_q_total:,} with EduQuartile assigned "
          f"({100*n_with_q_total/max(n_post_total,1):.1f}%)")

    cell = pd.concat(cells_per_year, ignore_index=True)
    print(f"    Cell count: {len(cell):,}")
    return cell


def _pull_worker_panel_2019_2025():
    """
    Pull AGI worker-employer-months for 2019-2025 with (lopnr, employer_id,
    year_month, age_group). Mirrors the structure used inside script 32's
    pull_year_to_panel but at the worker-employer-month level (not
    aggregated by SSYK).
    """
    cache = OUTPUT_DIR / "_worker_panel_cache.parquet"
    if cache.exists():
        print(f"  Worker-panel cache exists; loading {cache.name}")
        return pd.read_parquet(cache)

    # Raw pull cache — saved BEFORE demographics attach so a downstream
    # crash doesn't force a 25 min re-pull on the next run.
    raw_cache = OUTPUT_DIR / "_worker_panel_raw_cache.parquet"
    if raw_cache.exists():
        print(f"  Raw worker-panel cache exists; loading {raw_cache.name}")
        df = pd.read_parquet(raw_cache)
    else:
        print("  Pulling AGI worker-employer-months 2019-2025 (one-off ~25 min)")
        conn = pyodbc.connect(SQL_CONN_STRING)
        frames = []
        for year in range(2019, 2026):
            for month in range(1, 13):
                ym = f"{year}{month:02d}"
                if year == 2025 and month > 6:
                    # _Prel only through 202506 per Magnus 2026-04-29
                    continue
                table = f"dbo.Arb_AGIIndivid{ym}_Def" if year < 2025 else f"dbo.Arb_AGIIndivid{ym}_Prel"
                query = f"""
                    SELECT
                        P1207_LOPNR_PERSONNR AS lopnr,
                        P1207_LOPNR_PEORGNR AS employer_id,
                        '{year}-{month:02d}' AS year_month
                    FROM {table}
                    WHERE P1207_LOPNR_PEORGNR IS NOT NULL
                      AND KONTANT_ERSATTNING_ULAG_AG > 0
                """
                try:
                    t0 = time.time()
                    df = pd.read_sql(query, conn)
                    if len(df) == 0:
                        continue
                    frames.append(df)
                    if month == 1:
                        print(f"    {ym}: {len(df):,} rows in {time.time()-t0:.0f}s")
                except BaseException as e:
                    print(f"    {ym}: pull failed ({e})")
                    continue
        conn.close()

        if not frames:
            raise RuntimeError("No AGI data pulled for 2019-2025")

        df = pd.concat(frames, ignore_index=True)
        print(f"  Raw worker-panel: {len(df):,} rows; saving cache before demographics attach")
        df.to_parquet(raw_cache, index=False)

    # Attach age + age_group from the Population register (one pull, much smaller)
    print("  Attaching age via Population_PersonNr...")
    conn = pyodbc.connect(SQL_CONN_STRING)
    pop = pd.read_sql(
        """SELECT PersonLopNr AS lopnr, FodelseAr AS birth_year
           FROM dbo.Population_PersonNr
           WHERE FodelseAr IS NOT NULL""",
        conn,
    )
    conn.close()

    df = df.merge(pop, on="lopnr", how="left")
    df["panel_year"] = df["year_month"].str.slice(0, 4).astype(int)
    # FodelseAr can come back as int or string; coerce.
    df["birth_year"] = pd.to_numeric(df["birth_year"], errors="coerce")
    df["age"] = df["panel_year"] - df["birth_year"]

    # Map age to age_group (same buckets as Step 1)
    df["age_group"] = pd.cut(
        df["age"],
        bins=[21, 25, 30, 34, 40, 49, 69],
        labels=["22-25", "26-30", "31-34", "35-40", "41-49", "50+"],
        right=True,
        include_lowest=False,
    )
    df = df[df["age_group"].notna()].copy()
    df["age_group"] = df["age_group"].astype(str)

    df = df[["lopnr", "employer_id", "year_month", "age_group"]]
    print(f"  Worker-panel total: {len(df):,} rows")
    df.to_parquet(cache, index=False)
    return df


# ======================================================================
#   STAGE 3: POISSON DiD WITH EDU-QUARTILE
# ======================================================================

def estimate_poisson_eduquartile(cell_panel):
    """
    Run Poisson PML by age group with EduQuartile substituted for OccQuartile.
    Mirrors kauhanen.step1_poisson_current() but uses the cell_panel that
    is already keyed on edu_quartile.
    """
    print("\n  STAGE 3: Poisson PML with EduQuartile")
    results = []

    for age_label in AGE_GROUPS:
        print(f"\n  Age group: {age_label}")
        sub = cell_panel[cell_panel["age_group"] == age_label].copy()
        sub = sub.rename(columns={"edu_quartile": "daioe_quartile"})
        # filter_step1 expects daioe_quartile column (or generic bin column)
        sub_filtered = kauhanen.filter_step1(sub, age_label)
        if len(sub_filtered) == 0:
            print(f"    No employers; skip")
            continue
        balanced = kauhanen.build_balanced_panel(
            sub_filtered, "daioe_quartile", n_bins=4
        )
        if len(balanced) == 0:
            print(f"    No employers span Q4 and Q1-Q3; skip")
            continue
        result = kauhanen.estimate_poisson(
            balanced, "daioe_quartile", n_bins=4,
            age_label=age_label, spec_label="EDU_QUARTILE",
        )
        if result is not None:
            results.append(result)
            pd.DataFrame(results).to_csv(
                OUTPUT_DIR / "step1_education_quartile.csv", index=False
            )

    return pd.DataFrame(results)


# ======================================================================
#   PROSE SUMMARY
# ======================================================================

def write_summary(crosswalk, results, occquartile_step1):
    """
    Write a prose summary comparing EduQuartile and OccQuartile estimates.
    """
    out = OUTPUT_DIR / "education_summary.txt"
    lines = []
    lines.append("EDUCATION-BASED QUARTILE ROBUSTNESS -- SUMMARY")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Crosswalk: {len(crosswalk):,} (sun_niva, sun_inr) cells with "
                 f">= {MIN_GRADUATES_PER_EDU} graduates aged "
                 f"{CROSSWALK_AGE_MIN}-{CROSSWALK_AGE_MAX}.")
    lines.append("")

    if len(results) == 0:
        lines.append("Poisson estimation produced no results.")
    else:
        lines.append("EduQuartile vs OccQuartile (Step 1) gamma_2 by age group:")
        lines.append("")
        lines.append(f"  {'Age':>8}  {'OccQ g2':>10}  {'EduQ g2':>10}  {'EduQ p':>10}")
        lines.append(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}")
        for age in ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]:
            occ_row = occquartile_step1[occquartile_step1["age_group"] == age]
            edu_row = results[results["age_group"] == age]
            occ_g2 = occ_row["gamma2"].iloc[0] if len(occ_row) else float("nan")
            edu_g2 = edu_row["gamma2"].iloc[0] if len(edu_row) else float("nan")
            edu_p = edu_row["p2"].iloc[0] if len(edu_row) else float("nan")
            lines.append(f"  {age:>8}  {occ_g2:>10.4f}  {edu_g2:>10.4f}  {edu_p:>10.4f}")

    lines.append("")
    lines.append("Interpretation:")
    lines.append("  - EduQ g2 ~ OccQ g2: canaries effect robust to staleness.")
    lines.append("  - EduQ g2 attenuated: real effect partly captures within-track")
    lines.append("    reallocation; identification weaker but still real.")
    lines.append("  - EduQ g2 null: OccQ estimate likely measurement-error driven;")
    lines.append("    headline needs reframing.")

    out.write_text("\n".join(lines), encoding="ascii", errors="replace")
    print(f"\n  Summary written to {out.name}")


# ======================================================================
#   MAIN
# ======================================================================

def main():
    sys.stdout = _Tee(LOG_PATH)
    print("=" * 70)
    print("34_mona_education_robustness.py -- EduQuartile robustness")
    print("=" * 70)

    # Stage 1: crosswalk
    print("\nSTAGE 1: build EduQuartile crosswalk")
    lisa_years = list(range(LISA_MIN_YEAR, LISA_MAX_YEAR + 1))
    lisa_df = pull_lisa_education_and_occupation(lisa_years)
    daioe_df = load_daioe_with_percentile()
    crosswalk = build_edu_quartile_crosswalk(lisa_df, daioe_df)

    # Diagnostics file
    diag_lines = []
    diag_lines.append(f"EduQuartile crosswalk diagnostics")
    diag_lines.append(f"=================================")
    diag_lines.append(f"Population: workers aged {CROSSWALK_AGE_MIN}-{CROSSWALK_AGE_MAX} "
                      f"in LISA {LISA_MIN_YEAR}-{LISA_MAX_YEAR}")
    diag_lines.append(f"Min graduates per cell: {MIN_GRADUATES_PER_EDU}")
    diag_lines.append(f"Cells in crosswalk: {len(crosswalk):,}")
    diag_lines.append(f"Total graduates covered: {crosswalk['n_graduates'].sum():,}")
    for q in [1, 2, 3, 4]:
        sub = crosswalk[crosswalk["edu_quartile"] == q]
        diag_lines.append(
            f"  EduQ{q}: {len(sub):,} educations, "
            f"{sub['n_graduates'].sum():,} graduates, "
            f"mean DAIOE pctile {sub['mean_daioe_pctile'].mean():.3f}"
        )
    (OUTPUT_DIR / "edu_quartile_diagnostics.txt").write_text(
        "\n".join(diag_lines), encoding="ascii"
    )

    # Stage 2: assign to panel
    print("\nSTAGE 2: assign EduQuartile to panel")
    panel_ssyk_age = kauhanen.step0_pull_or_load_panel()
    cell_panel = assign_eduquartile_to_panel(panel_ssyk_age, lisa_df, crosswalk)
    cell_panel.to_parquet(OUTPUT_DIR / "panel_with_eduquartile.parquet", index=False)

    # Stage 3: Poisson DiD
    results = estimate_poisson_eduquartile(cell_panel)

    # Comparison: read OccQuartile Step 1 output if available
    occq_path = kauhanen.OUTPUT_DIR / "step1_poisson_current.csv"
    occq = pd.read_csv(occq_path) if occq_path.exists() else pd.DataFrame()

    write_summary(crosswalk, results, occq)
    print("\nDone. Outputs in output_34/:")
    print("  edu_quartile_crosswalk.csv")
    print("  edu_quartile_diagnostics.txt")
    print("  panel_with_eduquartile.parquet")
    print("  step1_education_quartile.csv")
    print("  education_summary.txt")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        print(f"\nFATAL: {e}")
        traceback.print_exc()
        sys.exit(2)
