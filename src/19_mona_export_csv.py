#!/usr/bin/env python3
"""
19_mona_export_csv.py — Export descriptive time series for Figure 3.

======================================================================
  FOR LYDIA — RUN IN MONA
  This is a LIGHTWEIGHT data export only (no regressions).
  Runtime: ~5-10 minutes (year-by-year SQL queries).
======================================================================

Exports monthly employment counts by:
  - age_group: "22-25" vs "26+"
  - high_ai: 1 (top quartile DAIOE genAI) vs 0 (rest)
  - year_month: "YYYY-MM"

Output: fig3_canaries_timeseries.csv
  Columns: year_month, age_group, high_ai, n_employed

This CSV is all we need to recreate Figure 3 outside MONA.
No individual-level data leaves MONA — only aggregated counts.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time

# ── MONA SQL connection (same as scripts 14-18) ──
import pyodbc
conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=monasql.micro.intra;"
    "DATABASE=P1207;"
    "Trusted_Connection=yes;"
)

# ── CONFIGURATION ──
DAIOE_PATH = r"\\micro.intra\Projekt\P1207$\P1207_Gem\Lydia P1207\daioe_quartiles.csv"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ── YEAR-BY-YEAR SQL WITH SSYK CASCADE ──

def pull_year(year, conn):
    """
    Pull one year of AGI data with cascading SSYK lookup (2023 -> 2022 -> 2021).
    Same approach as scripts 14-18 for data consistency.
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

    # Aggregate in SQL: only need person_id, ssyk4, birth_year, period
    # Age computed in SQL for filtering; individual rows returned for
    # downstream COUNT(DISTINCT person_id) aggregation
    query = f"""
    WITH base AS (
        {union_query}
    )
    SELECT
        employer_id,
        period,
        RIGHT('0000'+CAST(ssyk4 AS VARCHAR(4)),4) AS ssyk4,
        birth_year,
        person_id
    FROM base
    WHERE birth_year IS NOT NULL
      AND ssyk4 IS NOT NULL
    """
    return pd.read_sql(query, conn)


# ── LOAD DATA ──
print("Loading AGI data year by year...")
t0 = time.time()
frames = []
for year in range(2019, 2026):
    frames.append(pull_year(year, conn))
df = pd.concat(frames, ignore_index=True)
print(f"  Loaded {len(df):,} records in {time.time()-t0:.0f}s")

# Parse year-month from PERIOD (format: YYYYMM)
df["year_month"] = (
    df["period"].astype(str).str[:4] + "-" +
    df["period"].astype(str).str[4:6]
)

# Compute age
df["age"] = df["year_month"].str[:4].astype(int) - df["birth_year"].astype(int)
df["ssyk4"] = df["ssyk4"].astype(str).str.zfill(4)

# Filter to working age
df = df[(df["age"] >= 22) & (df["age"] <= 69)].copy()

# Age groups: 22-25 vs 26+
df["age_group"] = np.where(df["age"] <= 25, "22-25", "26+")

# ── MERGE WITH DAIOE ──
print("Merging with DAIOE...")
daioe = pd.read_csv(DAIOE_PATH)
daioe["ssyk4"] = daioe["ssyk4"].astype(str).str.zfill(4)
df = df.merge(daioe[["ssyk4", "high_exposure"]], on="ssyk4", how="inner")

# ── AGGREGATE ──
print("Aggregating...")
agg = (
    df.groupby(["year_month", "age_group", "high_exposure"])["person_id"]
    .nunique()
    .reset_index()
    .rename(columns={"person_id": "n_employed", "high_exposure": "high_ai"})
)

# ── SAVE ──
out_path = OUTPUT_DIR / "fig3_canaries_timeseries.csv"
agg.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"Rows: {len(agg)}")
print(f"Period: {agg['year_month'].min()} to {agg['year_month'].max()}")
print(f"\nSample:")
print(agg.head(8).to_string(index=False))
print("\nDone! Copy fig3_canaries_timeseries.csv out of MONA.")
