#!/usr/bin/env python3
"""
01_published_age_gap.py: the age pattern in Statistics Sweden's published
occupational employment, in which the agency codes the worker.

WHAT IT BUILDS
Statistics Sweden publishes employment by four-digit occupation (SSYK 2012),
age band and year (table YREG54BAS of the occupational register). The
occupations are matched to the DAIOE quartiles, and for each published age
band the log employment gap between top-quartile occupations and the rest is
computed each year, with its change since 2022, the last year before the
launch. Three caveats go with it: the register behind the series moves from
RAMS to BAS at reference year 2022, on the treatment boundary; the published
series lags the monthly declarations, so the check reaches 2024 and not 2025;
and the published bands (16 to 24, 25 to 29 and so on) are not the paper's.

INPUTS   data/raw/scb_yreg54bas.json (the saved response of the Statistics
         Sweden API, fetched on 24 February 2026; deleted, it is fetched
         again); data/processed/daioe_quartiles.csv
OUTPUTS  output/results/public_yreg_check.csv, public_yreg_summary.txt;
         output/tables/public_yreg.tex
SERVES   Online Appendix III.6, Table A24 (16 to 24 is the band whose gap
         falls most by 2024, by 0.006)
RUNTIME  seconds
"""

import json
import sys

from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def download_scb_employment():
    """
    Download employment by SSYK4 × age group × year from SCB's PxWeb API.

    We omit the industry (SNI2007) and sex (Kon) dimensions so they are
    automatically totalled. Result: ~21,500 cells (430 occ × 10 age × 5 yr).
    """
    url = "https://api.scb.se/OV0104/v1/doris/en/ssd/AM/AM0208/AM0208E/YREG54BAS"

    # POST query: all occupations × all age groups × all years
    # Omitting SNI2007 and Kon → totalled across industry and sex
    query = {
        "query": [
            {
                "code": "Yrke2012",
                "selection": {"filter": "all", "values": ["*"]},
            },
            {
                "code": "Alder",
                "selection": {"filter": "all", "values": ["*"]},
            },
            {
                "code": "ContentsCode",
                "selection": {"filter": "item", "values": ["000006Y1"]},
            },
            {
                "code": "Tid",
                "selection": {"filter": "all", "values": ["*"]},
            },
        ],
        "response": {"format": "json-stat2"},
    }

    out_path = config.RAW / "scb_yreg54bas.json"

    # Check if already downloaded
    if out_path.exists():
        print(f"  Already downloaded → {out_path.name}")
        with open(out_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print("  Downloading YREG54BAS from SCB API...")
    resp = requests.post(url, json=query, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"  Saved → {out_path.name}")

    return data


def parse_jsonstat2(data):
    """
    Parse JSON-stat2 format into a tidy pandas DataFrame.

    JSON-stat2 stores data as a flat array with dimensions defined by
    the 'dimension' and 'size' fields. We reconstruct the multi-index
    from the Cartesian product of dimension categories.

    We keep both the raw code and the human-readable label for each
    dimension, so we can extract SSYK codes from the code field.
    """
    dims = list(data["id"])
    sizes = data["size"]
    values = data["value"]

    # Build category codes and labels for each dimension
    dim_codes = {}
    dim_labels = {}
    for dim_name in dims:
        dim_info = data["dimension"][dim_name]
        cat = dim_info["category"]
        index_map = cat["index"]
        if isinstance(index_map, dict):
            sorted_codes = sorted(index_map.items(), key=lambda x: x[1])
            codes = [c[0] for c in sorted_codes]
        else:
            codes = list(index_map)
        labels = cat.get("label", {})
        dim_codes[dim_name] = codes
        dim_labels[dim_name] = [labels.get(c, c) for c in codes]

    # Build Cartesian product (rightmost dimension varies fastest)
    from itertools import product as cart_product

    code_rows = list(cart_product(*[dim_codes[d] for d in dims]))
    label_rows = list(cart_product(*[dim_labels[d] for d in dims]))

    df = pd.DataFrame(code_rows, columns=[f"{d}_code" for d in dims])
    for i, d in enumerate(dims):
        df[f"{d}_label"] = [r[i] for r in label_rows]
    df["value"] = values

    return df


def process_employment(data):
    """Parse SCB data and create clean employment DataFrame."""
    print("  Parsing JSON-stat2...")
    df = parse_jsonstat2(data)

    # The parser creates code and label columns for each dimension
    # Yrke2012_code = "1120", Yrke2012_label = "Senior officials..."
    # Alder_label = "16-24 years", Tid_code = "2020"

    # Extract SSYK4 code directly from the code column
    df["ssyk4"] = df["Yrke2012_code"].astype(str)
    df["ssyk_label"] = df["Yrke2012_label"]
    df["age_group"] = df["Alder_label"]
    df["year"] = df["Tid_code"].astype(int)
    df["n_employed"] = pd.to_numeric(df["value"], errors="coerce")

    # Keep only 4-digit SSYK codes (filter out aggregates like "0002")
    df = df[df["ssyk4"].str.match(r"^\d{4}$")]

    # Drop missing/suppressed cells
    df = df.dropna(subset=["n_employed"])
    df = df[df["n_employed"] > 0].copy()
    df["n_employed"] = df["n_employed"].astype(int)

    # Keep only needed columns
    df = df[["ssyk4", "ssyk_label", "age_group", "year", "n_employed"]].copy()

    print(f"  Parsed: {len(df):,} cells, "
          f"{df['ssyk4'].nunique()} occupations, "
          f"{df['age_group'].nunique()} age groups, "
          f"{df['year'].nunique()} years")

    return df



def t_public_yreg():
    """SCB's published aggregates: coded by the agency, not by us."""
    d = pd.read_csv(config.RESULTS / "public_yreg_check.csv")
    piv = d.pivot(index="age_group", columns="year", values="dd_vs_2022")
    years = [c for c in piv.columns if int(c) >= 2023]
    piv = piv.dropna(how="all", subset=years)
    rows = [f"{a.replace(' years', '')} & "
            + " & ".join(f"{piv.loc[a, y]:+.3f}" for y in years) + r" \\"
            for a in piv.index]
    (config.TABLES / "public_yreg.tex").write_text("\n".join([
        r"\begin{tabular}{l" + "c" * len(years) + "}", r"\hline\hline",
        "Age band & " + " & ".join(str(y) for y in years) + r" \\",
        r"\hline", *rows, r"\hline\hline", r"\end{tabular}"]))
    print("  wrote public_yreg.tex")



def main():
    print("Published employment by occupation and age: the top-quartile gap")
    data = download_scb_employment()
    emp = process_employment(data)
    years = sorted(emp["year"].unique())
    print(f"  published years available: {years}")

    daioe = pd.read_csv(config.PROCESSED / "daioe_quartiles.csv",
                        dtype={"ssyk4": str})
    daioe["ssyk4"] = daioe["ssyk4"].str.zfill(4)
    emp["ssyk4"] = emp["ssyk4"].astype(str).str.zfill(4)
    m = emp.merge(daioe[["ssyk4", "exposure_quartile"]], on="ssyk4",
                  how="inner")
    # exposure_quartile is stored as strings ("Q4 (highest)") in the
    # processed file, and newer pandas reads them as the Arrow-backed
    # 'str' dtype, which is not object. Match on the string content,
    # not on the dtype.
    m["high"] = (m["exposure_quartile"].astype(str)
                 .str.startswith("Q4").astype(int))

    # Young = the youngest published band containing 22-25 (SCB publishes
    # "16-24 years" / "25-34 years" bands); report both for transparency.
    bands = sorted(m["age_group"].unique())
    print(f"  published age bands: {bands}")

    grp = (m.groupby(["year", "age_group", "high"], observed=True)
           ["n_employed"].sum().reset_index())
    grp["ln_emp"] = np.log(grp["n_employed"])
    # Q4-vs-rest log gap per band-year, then its change from the last
    # pre-ChatGPT reference year (2022)
    wide = grp.pivot_table(index=["year", "age_group"], columns="high",
                           values="ln_emp").reset_index()
    wide["gap"] = wide[1] - wide[0]
    ref = wide[wide["year"] == 2022][["age_group", "gap"]].rename(
        columns={"gap": "gap_2022"})
    wide = wide.merge(ref, on="age_group", how="left")
    wide["dd_vs_2022"] = wide["gap"] - wide["gap_2022"]
    out = wide[["year", "age_group", "gap", "dd_vs_2022"]]
    out.to_csv(config.RESULTS / "public_yreg_check.csv", index=False)

    lines = ["PUBLISHED OCCUPATIONAL EMPLOYMENT (coded by Statistics Sweden)",
             "=" * 56,
             f"years: {years}",
             "",
             "Q4-vs-rest log employment gap, change from 2022, by band:"]
    for band in bands:
        sub = out[(out["age_group"] == band) & (out["year"] > 2022)]
        for _, r in sub.iterrows():
            lines.append(f"  {band:>14} {int(r['year'])}: "
                         f"{r['dd_vs_2022']:+.4f}")
    lines += ["",
              "Caveats:",
              " - RAMS->BAS register switch at reference year 2022 sits on",
              "   the treatment boundary; SCB flags it as a series break.",
              " - Publication lag: the newest published year lags AGI; the",
              "   2025 occupation register will not exist before ~2027, so",
              "   this check covers the first problem year (2024), not 2025.",
              " - Annual frequency and published age bands (not 22-25)."]
    (config.RESULTS / "public_yreg_summary.txt").write_text("\n".join(lines))
    t_public_yreg()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
