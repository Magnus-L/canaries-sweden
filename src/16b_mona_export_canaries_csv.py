#!/usr/bin/env python3
"""
16b_mona_export_canaries_csv.py — Export descriptive time series for Figure 3.

╔══════════════════════════════════════════════════════════════════════╗
║  FOR LYDIA — RUN IN MONA                                            ║
║  This is a LIGHTWEIGHT data export only (no regressions).            ║
║  Runtime: < 1 minute.                                                ║
╚══════════════════════════════════════════════════════════════════════╝

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

# ── CONFIGURATION (same as script 09) ──
INPUT_PATH = Path("agi_monthly_extract.parquet")
DAIOE_PATH = Path("daioe_quartiles.csv")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

AGI_COLUMNS = {
    "person_id": "LopNr",
    "year_month": "Period",
    "ssyk4": "SSYK4",
    "birth_year": "FodelseAr",
}

# ── LOAD DATA ──
print("Loading AGI data...")
suffix = INPUT_PATH.suffix.lower()
if suffix == ".parquet":
    df = pd.read_parquet(INPUT_PATH)
elif suffix == ".csv":
    df = pd.read_csv(INPUT_PATH)
else:
    df = pd.read_sas(INPUT_PATH)

rename_map = {v: k for k, v in AGI_COLUMNS.items() if v in df.columns}
df = df.rename(columns=rename_map)

df["year_month"] = df["year_month"].astype(str).str[:7]
df["date"] = pd.to_datetime(df["year_month"] + "-01")
df["age"] = df["date"].dt.year - df["birth_year"].astype(int)
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
