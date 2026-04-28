#!/usr/bin/env python3
"""
build_finland_marginals.py -- Fetch Eurostat LFS marginals for Finland.

Runs LOCALLY (not in MONA). Builds the input file
  empirical_data/finland_marginals_2022.txt
that script 32 reads inside MONA to compute compositional reweighting
factors (Step 4 of the Kauhanen staged robustness).

WHY THIS SCRIPT EXISTS:
  Step 4 of the Kauhanen staged robustness asks: "If Sweden had Finland's
  occupational and industrial composition, would the canaries effect
  still appear?" To answer, we need Finland's marginal distribution of
  employment by occupation (ISCO-1) and industry (NACE-2). These are
  publicly available from Eurostat LFS. We fetch them once, locally,
  and write a small file that Magnus uploads to MONA's empirical_data/
  folder.

DATASETS (Eurostat REST API):
  1. lfsa_egais     -- Employed persons by professional status and
                       occupation (ISCO-08, 1-digit). Source for ISCO marginal.
  2. lfsa_egan22d   -- Employed persons by detailed economic activity
                       (NACE Rev. 2, 2-digit, 88 divisions). Source for NACE.

QUERY PARAMETERS:
  geo=FI, sex=T (total), age=Y15-64, time=2022, freq=A, unit=THS_PER (thousands)

OUTPUT:
  empirical_data/finland_marginals_2022.txt
  CSV-formatted (.txt extension per MONA upload convention).
  Columns: dimension, code, label, share_fin

USAGE:
  python3 src/build_finland_marginals.py

  Output is small (~100 rows). Verify with `head` after running.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

# ---- Configuration ----

EUROSTAT_BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
YEAR = "2022"
COUNTRY = "FI"
AGE_BAND = "Y15-64"  # Eurostat label for ages 15-64 (closest to our 22-69 band)
SEX = "T"            # Total

OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "empirical_data"
    / f"finland_marginals_{YEAR}.txt"
)


# ---- JSON-stat parser ----

def parse_jsonstat(payload: dict) -> pd.DataFrame:
    """
    Parse a Eurostat JSON-stat 2.0 response into a long-format DataFrame.

    The values dict maps serial_index (str) to value (float). The serial
    index decomposes into a tuple of category positions across the
    'id' dimensions, with the last dimension cycling fastest. We invert
    the mapping to recover the (dimension, category) tuple for each value.
    """
    dim_ids = payload["id"]
    dim_sizes = payload["size"]
    dimensions = payload["dimension"]

    # For each dimension, build a position -> code list
    dim_codes = {}
    dim_labels = {}
    for dim in dim_ids:
        cat = dimensions[dim]["category"]
        idx_map = cat["index"]
        label_map = cat.get("label", {})
        # Sort by position
        sorted_codes = sorted(idx_map.items(), key=lambda x: x[1])
        codes = [c for c, _ in sorted_codes]
        dim_codes[dim] = codes
        dim_labels[dim] = label_map

    # Mixed-radix decode
    def decode(serial: int) -> dict:
        coords = {}
        s = serial
        for dim, size in zip(reversed(dim_ids), reversed(dim_sizes)):
            coords[dim] = dim_codes[dim][s % size]
            s //= size
        return coords

    rows = []
    for serial_str, value in payload["value"].items():
        coords = decode(int(serial_str))
        row = dict(coords)
        row["value"] = value
        rows.append(row)

    df = pd.DataFrame(rows)

    # Attach human-readable labels for the leaf dimensions we care about
    for dim in df.columns:
        if dim in dim_labels and dim_labels[dim]:
            df[f"{dim}_label"] = df[dim].map(dim_labels[dim])

    return df


def fetch_eurostat(dataset: str, params: dict) -> pd.DataFrame:
    """Fetch + parse one Eurostat dataset."""
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{EUROSTAT_BASE}/{dataset}?{qs}"
    print(f"  GET {url}")
    with urlopen(url, timeout=60) as r:
        payload = json.loads(r.read().decode("utf-8"))
    df = parse_jsonstat(payload)
    print(f"    parsed {len(df)} rows; "
          f"non-null values: {df['value'].notna().sum()}")
    return df


# ---- ISCO-1 marginal ----

def fetch_isco1_marginal(country: str) -> pd.DataFrame:
    """
    Fetch ISCO-08 1-digit employed-persons distribution for one country.

    Dataset lfsa_egais provides 'Employed persons by professional status
    and occupation'. We aggregate across professional status (wstatus)
    by summing values for wstatus = 'TOTAL' or, if not in the response,
    by summing across the wstatus dimension after parsing.
    """
    print(f"\n[ISCO-1] {country}")
    df = fetch_eurostat(
        dataset="lfsa_egais",
        params={
            "format": "JSON",
            "lang": "EN",
            "freq": "A",
            "time": YEAR,
            "geo": country,
            "sex": SEX,
            "age": AGE_BAND,
            "unit": "THS_PER",
        },
    )

    df = df.dropna(subset=["value"])

    # wstatus = 'EMP' is the total "Employed persons" category. The other
    # wstatus codes are subsets (CFAM = Contributing family workers,
    # NCFAM = Employed except CFAM, NRP = no response). Summing across
    # them double-counts; we restrict to EMP for the canonical total.
    if "wstatus" in df.columns:
        df = df[df["wstatus"] == "EMP"]

    # Eurostat ISCO codes: 'OC1'..'OC9', 'OC0'. Drop TOTAL and NRP aggregates.
    df = df[df["isco08"].astype(str).str.match(r"^OC[0-9]$")]

    isco_label_map = {
        "OC0": "Armed forces occupations",
        "OC1": "Managers",
        "OC2": "Professionals",
        "OC3": "Technicians and associate professionals",
        "OC4": "Clerical support workers",
        "OC5": "Service and sales workers",
        "OC6": "Skilled agricultural, forestry and fishery workers",
        "OC7": "Craft and related trades workers",
        "OC8": "Plant and machine operators, and assemblers",
        "OC9": "Elementary occupations",
    }

    out = df.groupby("isco08", as_index=False)["value"].sum()
    out["label"] = out["isco08"].map(isco_label_map).fillna("Unknown")
    total = out["value"].sum()
    out["share"] = out["value"] / total
    out["dimension"] = "ISCO1"
    # Strip the "OC" prefix to leave just the digit, easier to merge in MONA
    out["code"] = out["isco08"].str.replace("OC", "", regex=False)

    print(f"  ISCO-1 categories: {len(out)}, total = {total:.1f} thousand")
    return out[["dimension", "code", "label", "share"]]


# ---- NACE-2 marginal ----

def fetch_nace2_marginal(country: str) -> pd.DataFrame:
    """
    Fetch NACE Rev. 2 2-digit employed-persons distribution for one country.

    Dataset lfsa_egan22d provides 'Employed persons by detailed economic
    activity (NACE Rev. 2 two-digit level)'. We sum across age/sex.
    """
    print(f"\n[NACE-2] {country}")
    df = fetch_eurostat(
        dataset="lfsa_egan22d",
        params={
            "format": "JSON",
            "lang": "EN",
            "freq": "A",
            "time": YEAR,
            "geo": country,
            "sex": SEX,
            "age": AGE_BAND,
            "unit": "THS_PER",
        },
    )

    df = df.dropna(subset=["value"])

    # Same wstatus filter as the ISCO function (apply only if dimension exists)
    if "wstatus" in df.columns:
        df = df[df["wstatus"] == "EMP"]

    # Eurostat NACE-2 codes are letter+2digits ('A01'..'U99'). Drop aggregates
    # (TOTAL, single letters A..U, and other coarser aggregates).
    df = df[df["nace_r2"].astype(str).str.match(r"^[A-Z][0-9]{2}$")]

    out = df.groupby("nace_r2", as_index=False)["value"].sum()
    if "nace_r2_label" in df.columns:
        labels = df.groupby("nace_r2", as_index=False)["nace_r2_label"].first()
        out = out.merge(labels, on="nace_r2", how="left")
        out = out.rename(columns={"nace_r2_label": "label"})
    else:
        out["label"] = ""

    total = out["value"].sum()
    out["share"] = out["value"] / total
    out["dimension"] = "NACE2"
    # Keep the 2-digit NACE division (drop the section letter); the
    # division is the unique identifier and matches Företagsdatabasen
    # SNI codes when zero-padded to 2 digits.
    out["code"] = out["nace_r2"].str[1:].str.zfill(2)

    print(f"  NACE-2 categories: {len(out)}, total = {total:.1f} thousand")
    return out[["dimension", "code", "label", "share"]]


# ---- Main ----

def main() -> int:
    print(f"Building Finland marginals for year {YEAR}, "
          f"age {AGE_BAND}, sex {SEX}")
    print(f"Output: {OUTPUT_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Fetch ISCO-1 and NACE-2 marginals from Eurostat
    isco_fi = fetch_isco1_marginal("FI")
    nace_fi = fetch_nace2_marginal("FI")

    out = pd.concat([isco_fi, nace_fi], ignore_index=True)
    out["share"] = out["share"].round(6)
    out.to_csv(OUTPUT_PATH, index=False)

    print("\n=== SUMMARY ===")
    print(f"ISCO-1 categories: {(out['dimension']=='ISCO1').sum()}")
    print(f"NACE-2 categories: {(out['dimension']=='NACE2').sum()}")
    print(f"Total rows: {len(out)}")
    print(f"\nFile saved to: {OUTPUT_PATH}")
    print("\nFirst 10 rows:")
    print(out.head(10).to_string(index=False))
    print("\nNext step: upload this file to the MONA empirical_data/ folder.")
    print("Script 32 expects path:")
    print(r"  \\micro.intra\Projekt\P1207$\P1207_Gem\Lydia P1207\empirical_data\finland_marginals_2022.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
