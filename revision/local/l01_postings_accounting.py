#!/usr/bin/env python3
"""
l01_postings_accounting.py -- Ed.1 / plan T1: the posting-sample accounting.

The editor: "report the number of raw advertisements, the number removed
because of missing or invalid occupation codes, and the number removed as
duplicates." v1's script 02 computed these but only PRINTED them (defect D3),
and lumped every non-kept ad into one bucket. This script re-reads the raw
JSONL zips and counts every drop reason separately, per year and in total:

    n_raw          lines read
    n_parse_error  invalid JSON
    n_no_occfield  occupation_group absent or empty
    n_bad_code     occupation code present but not a 4-digit numeric SSYK
    n_no_date      publication_date absent or unparseable
    n_out_of_range publication year outside 2006-2026
    n_duplicate    ad id already seen (across years, same order as v1)
    n_kept         rows entering the analysis file

The drop order matches v1's extract_ad_fields exactly, so n_kept reproduces
the v1 sample by construction; a mismatch is a red flag, not a rounding issue.

Output:  tables/postings_accounting.csv  (one row per year + TOTAL)
         tables/postings_accounting.tex  (paper-ready)
Runtime: ~15-25 min on the full files (streamed, constant memory).
Test:    --sample uses the 1% files (~1 min).
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW, V2_TAB, PLATSBANKEN_YEARS


def classify_ad(ad: dict) -> tuple[str, dict | None]:
    """
    Mirror of v1 extract_ad_fields (src/02), but returning the DROP REASON
    instead of None. Order of checks is identical to v1 so the kept sample
    is identical by construction.
    """
    occ_group = ad.get("occupation_group")
    if not occ_group:
        return "no_occfield", None
    if isinstance(occ_group, list):
        if len(occ_group) == 0:
            return "no_occfield", None
        code = occ_group[0].get("legacy_ams_taxonomy_id")
    elif isinstance(occ_group, dict):
        code = occ_group.get("legacy_ams_taxonomy_id")
    else:
        return "no_occfield", None
    if not code:
        return "no_occfield", None

    code = str(code).strip()
    if not code.isdigit() or len(code) != 4:
        return "bad_code", None

    pub = ad.get("publication_date")
    if not pub:
        return "no_date", None
    ym = str(pub)[:7]
    if len(ym) != 7 or ym[4] != "-":
        return "no_date", None
    try:
        y = int(ym[:4])
    except ValueError:
        return "no_date", None
    if y < 2006 or y > 2026:
        return "out_of_range", None

    return "ok", {"ad_id": ad.get("original_id") or ad.get("id", ""),
                  "ssyk4": code, "year_month": ym}


def account_year(year: int, seen_ids: set, sample: bool, monthly: dict) -> dict:
    zpath = RAW / (f"{year}_sample.jsonl.zip" if sample else f"{year}.jsonl.zip")
    if not zpath.exists():
        zpath = RAW / f"{year}_sample.jsonl.zip"
    if not zpath.exists():
        print(f"  {year}: no file, skipping")
        return {}

    c = dict(year=year, n_raw=0, n_parse_error=0, n_no_occfield=0,
             n_bad_code=0, n_no_date=0, n_out_of_range=0,
             n_duplicate=0, n_kept=0)

    def bump(ym: str, source: str, field: str):
        key = (ym, source or "(none)")
        d = monthly.setdefault(key, {"n_ads": 0, "n_valid_code": 0})
        d[field] += 1

    with zipfile.ZipFile(zpath) as zf:
        for name in (n for n in zf.namelist() if n.endswith(".jsonl")):
            with zf.open(name) as f:
                for line in f:
                    c["n_raw"] += 1
                    try:
                        ad = json.loads(line)
                    except json.JSONDecodeError:
                        c["n_parse_error"] += 1
                        continue
                    reason, rec = classify_ad(ad)
                    # month + source for the monthly coverage series (Ed.2):
                    # the month is knowable even when the occupation code is
                    # not, as long as the ad carries a date.
                    _ym = str(ad.get("publication_date") or "")[:7]
                    _src = str(ad.get("source_type") or "")
                    _has_month = len(_ym) == 7 and _ym[4] == "-"
                    if _has_month:
                        bump(_ym, _src, "n_ads")
                    if reason != "ok":
                        c[f"n_{reason}"] += 1
                        continue
                    if _has_month:
                        bump(_ym, _src, "n_valid_code")
                    ad_id = rec["ad_id"]
                    if ad_id and ad_id in seen_ids:
                        c["n_duplicate"] += 1
                        continue
                    if ad_id:
                        seen_ids.add(ad_id)
                    c["n_kept"] += 1
    print(f"  {year}: raw {c['n_raw']:,} | kept {c['n_kept']:,} | "
          f"no-field {c['n_no_occfield']:,} | bad-code {c['n_bad_code']:,} | "
          f"no-date {c['n_no_date']:,} | out-of-range {c['n_out_of_range']:,} | "
          f"dupes {c['n_duplicate']:,} | parse {c['n_parse_error']:,}")
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="1%% files, for testing")
    args = ap.parse_args()

    print("L1: postings accounting (Ed.1)" + (" [SAMPLE]" if args.sample else ""))
    seen: set = set()
    monthly: dict = {}
    rows = [r for y in PLATSBANKEN_YEARS
            if (r := account_year(y, seen, args.sample, monthly))]
    df = pd.DataFrame(rows)

    total = df.drop(columns="year").sum().to_dict()
    total["year"] = "TOTAL"
    df = pd.concat([df, pd.DataFrame([total])], ignore_index=True)

    # Consistency: raw = sum of parts
    parts = ["n_parse_error", "n_no_occfield", "n_bad_code", "n_no_date",
             "n_out_of_range", "n_duplicate", "n_kept"]
    df["check_sum"] = df[parts].sum(axis=1)
    assert (df["check_sum"] == df["n_raw"]).all(), "accounting does not sum to n_raw"
    df = df.drop(columns="check_sum")

    suffix = "_sample" if args.sample else ""
    out = V2_TAB / f"postings_accounting{suffix}.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")

    # Monthly valid-code share by source (Ed.2). Duplicates are counted here
    # (deduplication is an analysis choice, coverage is a data property).
    mrows = [{"year_month": ym, "source_type": src, **v}
             for (ym, src), v in sorted(monthly.items())]
    mdf = pd.DataFrame(mrows)
    mdf["valid_share"] = mdf["n_valid_code"] / mdf["n_ads"]
    mout = V2_TAB / f"postings_coverage_monthly{suffix}.csv"
    mdf.to_csv(mout, index=False)
    print(f"Saved {mout}")

    # Paper-ready LaTeX (full run only)
    if not args.sample:
        cols = ["year", "n_raw", "n_no_occfield", "n_bad_code", "n_no_date",
                "n_out_of_range", "n_duplicate", "n_parse_error", "n_kept"]
        hdr = ["Year", "Raw ads", "No occ.\\ field", "Invalid code", "No date",
               "Out of range", "Duplicate", "Parse error", "Kept"]
        lines = [r"\begin{tabular}{l" + "r" * (len(cols) - 1) + "}",
                 r"\hline\hline", " & ".join(hdr) + r" \\", r"\hline"]
        for _, r in df.iterrows():
            vals = [str(r["year"])] + [f"{int(r[c]):,}" for c in cols[1:]]
            lines.append(" & ".join(vals) + r" \\")
        lines += [r"\hline\hline", r"\end{tabular}"]
        (V2_TAB / "postings_accounting.tex").write_text("\n".join(lines))
        print(f"Saved {V2_TAB / 'postings_accounting.tex'}")


if __name__ == "__main__":
    main()
