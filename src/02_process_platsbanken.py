#!/usr/bin/env python3
"""
02_process_platsbanken.py — Parse Platsbanken JSONL → SSYK4 × month aggregates.

Reads the downloaded JSONL zip files (one JSON object per line, per ad),
extracts the fields we need, and aggregates to occupation × year-month.

Key design choices:
  - Memory-efficient: streams line by line (zipfile + line iteration), never
    loads an entire year's data into memory at once.
  - Skips ads without a valid SSYK 4-digit code.
  - Deduplicates on ad ID (original_id) across all years.
  - Outputs two files:
    1. postings_ssyk4_monthly.csv — SSYK4 × year-month counts and vacancies
    2. postings_monthly_total.csv — total monthly postings (for scary chart)

Data schema (from jobtechdev.se documentation):
  - occupation_group.legacy_ams_taxonomy_id → SSYK 4-digit code
  - publication_date → ISO datetime
  - number_of_vacancies → integer (can be null)
  - original_id → unique ad identifier
  - workplace_address.municipality_code → kommun code
  - employer.name → employer name
  - source_type → data source
"""

import sys
import json
import zipfile
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import RAW, PROCESSED, PLATSBANKEN_YEARS

import pandas as pd


def extract_ad_fields(ad: dict) -> dict | None:
    """
    Extract the fields we need from a single ad JSON object.

    Returns None if the ad lacks a valid SSYK code (our primary key for
    matching with DAIOE exposure data).
    """
    # Extract SSYK 4-digit code from occupation_group.
    # Raw format: occupation_group is a dict with legacy_ams_taxonomy_id.
    # Enriched format: occupation_group is a list of dicts.
    occ_group = ad.get("occupation_group")
    if not occ_group:
        return None

    if isinstance(occ_group, list):
        # Enriched format — take the first entry
        if len(occ_group) == 0:
            return None
        ssyk_code = occ_group[0].get("legacy_ams_taxonomy_id")
    elif isinstance(occ_group, dict):
        # Raw format — direct access
        ssyk_code = occ_group.get("legacy_ams_taxonomy_id")
    else:
        return None

    if not ssyk_code:
        return None

    # Validate: must be a 4-digit numeric string
    ssyk_code = str(ssyk_code).strip()
    if not ssyk_code.isdigit() or len(ssyk_code) != 4:
        return None

    # Extract publication date
    pub_date = ad.get("publication_date")
    if not pub_date:
        return None

    # Parse to year-month (first 7 chars: "2024-03")
    ym = str(pub_date)[:7]
    if len(ym) != 7 or ym[4] != "-":
        return None

    # Number of vacancies (default to 1 if missing)
    n_vac = ad.get("number_of_vacancies")
    if n_vac is None or n_vac < 1:
        n_vac = 1

    # Municipality code (optional, for geographic analysis)
    addr = ad.get("workplace_address", {}) or {}
    muni = addr.get("municipality_code", "")

    return {
        "ad_id": ad.get("original_id") or ad.get("id", ""),
        "ssyk4": ssyk_code,
        "year_month": ym,
        "n_vacancies": int(n_vac),
        "municipality_code": str(muni),
        "source_type": ad.get("source_type", ""),
    }


def process_year(year: int, seen_ids: set) -> list[dict]:
    """
    Process one year's JSONL zip file, streaming line by line.

    Returns a list of extracted ad records. Updates seen_ids in place
    for deduplication across years.

    Looks for files in priority order:
      1. {year}.jsonl.zip          (full raw data)
      2. {year}_sample.jsonl.zip   (1% enriched sample, for testing)

    Why line-by-line: The 2022 file decompresses to several GB. Parsing
    one line at a time keeps memory usage constant regardless of file size.
    """
    # Try full file first, then sample
    zip_path = RAW / f"{year}.jsonl.zip"
    if not zip_path.exists():
        zip_path = RAW / f"{year}_sample.jsonl.zip"
    if not zip_path.exists():
        print(f"  WARNING: No data file found for {year} — skipping")
        return []

    records = []
    n_total = 0
    n_no_ssyk = 0
    n_duplicate = 0
    n_parse_error = 0

    print(f"  Processing {year}...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Find the JSONL file inside the zip
        jsonl_names = [n for n in zf.namelist() if n.endswith(".jsonl")]
        if not jsonl_names:
            print(f"    WARNING: No .jsonl file found in {zip_path.name}")
            return []

        for jsonl_name in jsonl_names:
            with zf.open(jsonl_name) as f:
                for line in f:
                    n_total += 1

                    try:
                        ad = json.loads(line)
                    except json.JSONDecodeError:
                        n_parse_error += 1
                        continue

                    result = extract_ad_fields(ad)
                    if result is None:
                        n_no_ssyk += 1
                        continue

                    # Deduplicate
                    ad_id = result["ad_id"]
                    if ad_id and ad_id in seen_ids:
                        n_duplicate += 1
                        continue
                    if ad_id:
                        seen_ids.add(ad_id)

                    records.append(result)

                    # Progress indicator every 500k ads
                    if n_total % 500_000 == 0:
                        print(f"    ... {n_total:,} ads processed, {len(records):,} kept")

    print(f"    {year}: {n_total:,} total | {len(records):,} kept | "
          f"{n_no_ssyk:,} no SSYK | {n_duplicate:,} duplicates | "
          f"{n_parse_error:,} parse errors")

    return records


def process_jobstream(seen_ids: set) -> list[dict]:
    """
    Process the JobStream snapshot file (if it exists).

    This supplements historical data with the very latest ads.
    Same deduplication logic as historical files.
    """
    snapshot = RAW / "jobstream_snapshot.jsonl"
    if not snapshot.exists():
        print("  No JobStream snapshot found — skipping")
        return []

    print("  Processing JobStream snapshot...")
    records = []
    n_total = 0

    with open(snapshot, "r", encoding="utf-8") as f:
        for line in f:
            n_total += 1
            try:
                ad = json.loads(line)
            except json.JSONDecodeError:
                continue

            result = extract_ad_fields(ad)
            if result is None:
                continue

            ad_id = result["ad_id"]
            if ad_id and ad_id in seen_ids:
                continue
            if ad_id:
                seen_ids.add(ad_id)

            records.append(result)

    print(f"    JobStream: {n_total:,} total | {len(records):,} new ads added")
    return records


def aggregate_to_monthly(records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate ad-level records to SSYK4 × year-month.

    Returns two DataFrames:
    1. ssyk4_monthly — occupation-level monthly counts and total vacancies
    2. total_monthly — aggregate monthly totals (for scary chart)
    """
    df = pd.DataFrame(records)

    # SSYK4 × month aggregation
    ssyk4_monthly = (
        df.groupby(["ssyk4", "year_month"])
        .agg(
            n_ads=("ad_id", "count"),
            n_vacancies=("n_vacancies", "sum"),
        )
        .reset_index()
    )

    # Total monthly aggregation
    total_monthly = (
        df.groupby("year_month")
        .agg(
            n_ads=("ad_id", "count"),
            n_vacancies=("n_vacancies", "sum"),
            n_occupations=("ssyk4", "nunique"),
        )
        .reset_index()
    )

    # Parse year_month to proper dates for time series
    total_monthly["date"] = pd.to_datetime(total_monthly["year_month"] + "-01")
    total_monthly = total_monthly.sort_values("date").reset_index(drop=True)

    return ssyk4_monthly, total_monthly


def main():
    """
    Parse all Platsbanken JSONL files and produce monthly aggregates.

    Processes 2020–2025 historical data plus any JobStream snapshot.
    Deduplicates across all sources on ad ID.
    """
    print("=" * 70)
    print("STEP 2: Process Platsbanken JSONL → SSYK4 × month aggregates")
    print("=" * 70)

    seen_ids: set = set()
    all_records: list[dict] = []

    # Process each year
    for year in PLATSBANKEN_YEARS:
        records = process_year(year, seen_ids)
        all_records.extend(records)

    # Process JobStream snapshot
    records = process_jobstream(seen_ids)
    all_records.extend(records)

    print(f"\nTotal records (all years, deduplicated): {len(all_records):,}")

    if not all_records:
        print("ERROR: No records extracted. Check that JSONL files are downloaded.")
        sys.exit(1)

    # Aggregate
    ssyk4_monthly, total_monthly = aggregate_to_monthly(all_records)

    # Save
    out1 = PROCESSED / "postings_ssyk4_monthly.csv"
    ssyk4_monthly.to_csv(out1, index=False)
    print(f"\nSaved: {out1.name}")
    print(f"  {len(ssyk4_monthly):,} rows (SSYK4 × month)")
    print(f"  {ssyk4_monthly['ssyk4'].nunique()} unique SSYK codes")

    out2 = PROCESSED / "postings_monthly_total.csv"
    total_monthly.to_csv(out2, index=False)
    print(f"\nSaved: {out2.name}")
    print(f"  {len(total_monthly)} months")
    print(f"  Date range: {total_monthly['year_month'].min()} to {total_monthly['year_month'].max()}")
    print(f"  Total ads: {total_monthly['n_ads'].sum():,}")

    # Print monthly summary
    print("\nMonthly ad counts (sample):")
    for _, row in total_monthly.head(12).iterrows():
        print(f"  {row['year_month']}: {row['n_ads']:>8,} ads, "
              f"{row['n_vacancies']:>9,} vacancies, "
              f"{row['n_occupations']:>3} occupations")

    print("\nDone. Run 03_fetch_auxiliary.py next.")


if __name__ == "__main__":
    main()
