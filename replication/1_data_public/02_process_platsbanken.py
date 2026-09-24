#!/usr/bin/env python3
"""
02_process_platsbanken.py: advertisements to occupation-by-month counts.

WHAT IT BUILDS
Streams every archive line by line (never a whole year in memory), keeps an
advertisement when it carries a four-digit SSYK 2012 occupation code
(occupation_group.legacy_ams_taxonomy_id) and a publication date in 2006 to
2026, removes repeats of the advertisement identifier (original_id) across all
files, and counts advertisements and vacancies (number_of_vacancies, one when
missing) by occupation and month. The annual archives 2020 to 2025 are read
first, then the closed quarters of 2026.

It then compares the result with the frozen file every posting estimate starts
from (config.POSTINGS_SSYK4, built on 24 February 2026 from the same annual
archives plus months from the live feed, whose extraction was not kept) and
prints where the two differ.

INPUTS   config.JOBADS_DIR/<year>.jsonl.zip and <2026-Qn>.jsonl.zip
OUTPUTS  data/processed/postings_ssyk4_monthly.csv (occupation x month:
         n_ads, n_vacancies), postings_monthly_total.csv (month totals),
         output/results/postings_ssyk4_rebuild_vs_frozen.csv
SERVES   the input of every posting estimate (Section 3; Online Appendix II)
RUNTIME  about 25 minutes for the eight archives
"""

import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

import pandas as pd  # noqa: E402


def extract_ad_fields(ad: dict) -> dict | None:
    """The fields the analysis needs, or None when the occupation code or the
    publication date is unusable. The order of the checks defines the sample
    and is the same as in 2_postings/01_postings_accounting.py."""
    occ_group = ad.get("occupation_group")
    if not occ_group:
        return None
    if isinstance(occ_group, list):
        if len(occ_group) == 0:
            return None
        ssyk_code = occ_group[0].get("legacy_ams_taxonomy_id")
    elif isinstance(occ_group, dict):
        ssyk_code = occ_group.get("legacy_ams_taxonomy_id")
    else:
        return None
    if not ssyk_code:
        return None
    ssyk_code = str(ssyk_code).strip()
    if not ssyk_code.isdigit() or len(ssyk_code) != 4:
        return None

    pub_date = ad.get("publication_date")
    if not pub_date:
        return None
    ym = str(pub_date)[:7]
    if len(ym) != 7 or ym[4] != "-":
        return None
    try:
        year_int = int(ym[:4])
    except ValueError:
        return None
    if year_int < 2006 or year_int > 2026:
        return None

    n_vac = ad.get("number_of_vacancies")
    if n_vac is None or n_vac < 1:
        n_vac = 1
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


def process_file(stem, seen_ids: set) -> list[dict]:
    """One archive, streamed; updates seen_ids so repeats across files drop."""
    zip_path = config.platsbanken_zip(stem)
    if not zip_path.exists():
        zip_path = config.JOBADS_DIR / f"{stem}_sample.jsonl.zip"
    if not zip_path.exists():
        print(f"  WARNING: no archive for {stem}, skipped")
        return []
    records, n_total, n_no_ssyk, n_duplicate, n_parse_error = [], 0, 0, 0, 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for jsonl_name in (n for n in zf.namelist() if n.endswith(".jsonl")):
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
                    ad_id = result["ad_id"]
                    if ad_id and ad_id in seen_ids:
                        n_duplicate += 1
                        continue
                    if ad_id:
                        seen_ids.add(ad_id)
                    records.append(result)
    print(f"  {stem}: {n_total:,} read | {len(records):,} kept | "
          f"{n_no_ssyk:,} no usable code or date | {n_duplicate:,} repeats | "
          f"{n_parse_error:,} parse errors")
    return records


def main():
    print("Platsbanken archives to occupation-by-month counts")
    seen: set = set()
    records: list[dict] = []
    for stem in list(config.PLATSBANKEN_YEARS) + list(config.PLATSBANKEN_QUARTERS):
        records.extend(process_file(stem, seen))
    if not records:
        sys.exit("no records: download the archives first (01_download_platsbanken.py)")

    df = pd.DataFrame(records)
    ssyk4_monthly = (df.groupby(["ssyk4", "year_month"])
                     .agg(n_ads=("ad_id", "count"), n_vacancies=("n_vacancies", "sum"))
                     .reset_index())
    total = (df.groupby("year_month")
             .agg(n_ads=("ad_id", "count"), n_vacancies=("n_vacancies", "sum"),
                  n_occupations=("ssyk4", "nunique"))
             .reset_index())
    total["date"] = pd.to_datetime(total["year_month"] + "-01")
    total = total.sort_values("date").reset_index(drop=True)

    ssyk4_monthly.to_csv(config.PROCESSED / "postings_ssyk4_monthly.csv", index=False)
    total.to_csv(config.PROCESSED / "postings_monthly_total.csv", index=False)
    print(f"  wrote postings_ssyk4_monthly.csv: {len(ssyk4_monthly):,} cells, "
          f"{ssyk4_monthly['ssyk4'].nunique()} codes, "
          f"{total['year_month'].min()} to {total['year_month'].max()}")

    # The rebuild against the frozen file, month by month, 2020 to 2025
    # (the months of the frozen file that the estimates use).
    fz = pd.read_csv(config.POSTINGS_SSYK4, dtype={"ssyk4": str})
    rb = ssyk4_monthly.copy()
    for d in (fz, rb):
        d["ssyk4"] = d["ssyk4"].astype(str).str.zfill(4)
    win = lambda d: d[(d["year_month"] >= "2020-01") & (d["year_month"] <= "2025-12")]
    cmp = (win(fz)[["ssyk4", "year_month", "n_ads"]]
           .merge(win(rb)[["ssyk4", "year_month", "n_ads"]],
                  on=["ssyk4", "year_month"], how="outer",
                  suffixes=("_frozen", "_rebuilt")).fillna(0))
    cmp["diff"] = cmp["n_ads_rebuilt"] - cmp["n_ads_frozen"]
    bym = (cmp.groupby("year_month")[["n_ads_frozen", "n_ads_rebuilt", "diff"]]
           .sum().reset_index())
    bym.to_csv(config.RESULTS / "postings_ssyk4_rebuild_vs_frozen.csv", index=False)
    ndiff = int((cmp["diff"] != 0).sum())
    print(f"  against the frozen file, January 2020 to December 2025: "
          f"{ndiff:,} of {len(cmp):,} cells differ, "
          f"{int(cmp['n_ads_frozen'].sum()):,} frozen against "
          f"{int(cmp['n_ads_rebuilt'].sum()):,} rebuilt advertisements")
    for _, r in bym[bym["diff"] != 0].iterrows():
        print(f"    {r['year_month']}: {int(r['diff']):+,}")


if __name__ == "__main__":
    main()
