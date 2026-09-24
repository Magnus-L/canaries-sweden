#!/usr/bin/env python3
"""
10_summary_statistics.py: summary statistics of the posting sample, January
2020 to June 2026.

THE STATISTICS
Pooled occupation-by-month cells with at least one advertisement, by DAIOE
generative-AI exposure quartile and over all cells: distinct occupations,
cells, the mean, standard deviation and median of advertisements per cell,
the mean and standard deviation of vacancies per cell, and the mean exposure
percentile. The same method reproduces the submitted version's table on its
window (October 2019 to February 2026) cell for cell, which is checked first.

VACANCIES FOR 2026
The panel of 03 carries advertisements only. Vacancies for 2020 to 2025 come
from config.POSTINGS_SSYK4; for January to June 2026 the two quarter archives
are streamed again with the classification of 01 and the de-duplication of
03, and vacancies are summed (number_of_vacancies, one when missing or below
one). A second check requires the recounted advertisements to equal 03's
2026 counts cell for cell.

INPUTS   data/processed/postings_daioe_merged.csv (check 1),
         postings_daioe_merged_extended.csv (03); config.POSTINGS_SSYK4;
         output/results/postings_ssyk4_monthly_2026H1.csv (03, check 2);
         config.JOBADS_DIR/2026-Q1.jsonl.zip, 2026-Q2.jsonl.zip
OUTPUTS  output/results/postings_sumstats_v3.csv;
         output/tables/tableI2b_sumstats_postings_v3.tex
SERVES   Online Appendix I.2, Table A3
RUNTIME  about 3 minutes
"""
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import config, sibling  # noqa: E402

acc = sibling("01_postings_accounting")
PROC = config.PROCESSED
QORDER = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]

# The table of the submitted version (October 2019 to February 2026).
OLD = {
    "occupations": [103, 99, 76, 91, 369],
    "cells": [7239, 7234, 5524, 6675, 26672],
    "ads_mean": [99.5, 197.6, 199.7, 181.3, 167.3],
    "ads_sd": [213.4, 566.2, 525.7, 367.6, 438.3],
    "ads_median": [22, 53, 49, 63, 43],
    "vac_mean": [256.6, 428.7, 308.3, 258.9, 314.6],
    "vac_sd": [687.3, 1230.2, 812.8, 714.9, 899.2],
    "pctl_mean": [12.4, 37.4, 61.9, 88.2, 48.4],
}


def stats(d):
    """The table's rows, by quartile and for all cells."""
    out = {}
    for q in QORDER + ["All"]:
        s = d if q == "All" else d[d["exposure_quartile"] == q]
        out[q] = {"occupations": s["ssyk4"].nunique(), "cells": len(s),
                  "ads_mean": s["n_ads"].mean(), "ads_sd": s["n_ads"].std(),
                  "ads_median": s["n_ads"].median(),
                  "vac_mean": s["n_vacancies"].mean(),
                  "vac_sd": s["n_vacancies"].std(),
                  "pctl_mean": s["pctl_rank_genai"].mean()}
    return pd.DataFrame(out)


def recount_2026():
    """Ads and vacancies for January to June 2026: the pass of 03 with
    vacancies summed as 1_data_public/02 sums them."""
    seen, counts = set(), {}
    for stem in config.PLATSBANKEN_QUARTERS:
        with zipfile.ZipFile(config.platsbanken_zip(stem)) as zf:
            for name in (n for n in zf.namelist() if n.endswith(".jsonl")):
                with zf.open(name) as f:
                    for line in f:
                        try:
                            ad = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        reason, rec = acc.classify_ad(ad)
                        if reason != "ok":
                            continue
                        if rec["ad_id"] and rec["ad_id"] in seen:
                            continue
                        if rec["ad_id"]:
                            seen.add(rec["ad_id"])
                        nv = ad.get("number_of_vacancies")
                        if nv is None or nv < 1:
                            nv = 1
                        k = (rec["ssyk4"], rec["year_month"])
                        a, v = counts.get(k, (0, 0))
                        counts[k] = (a + 1, v + int(nv))
    df = pd.DataFrame([{"ssyk4": k[0].zfill(4), "year_month": k[1],
                        "n_ads": a, "n_vacancies": v}
                       for k, (a, v) in counts.items()])
    return df[(df["year_month"] >= "2026-01") & (df["year_month"] <= "2026-06")]


def fmt(x, dec):
    return f"{x:,.{dec}f}"


def main():
    print("Summary statistics of the posting sample, January 2020 to June 2026")
    # Check 1: the method reproduces the table of the submitted version.
    old = pd.read_csv(PROC / "postings_daioe_merged.csv", dtype={"ssyk4": str})
    got = stats(old)
    for row, vals in OLD.items():
        for q, v in zip(QORDER + ["All"], vals):
            dec = 0 if row in ("occupations", "cells", "ads_median") else 1
            assert round(got.loc[row, q], dec) == v, (row, q, got.loc[row, q], v)
    print("  check 1 passed: the submitted version's table is reproduced cell for cell")

    # Vacancies on the new window.
    base = pd.read_csv(config.POSTINGS_SSYK4, dtype={"ssyk4": str})
    base["ssyk4"] = base["ssyk4"].str.zfill(4)
    base = base[(base["year_month"] >= "2020-01") & (base["year_month"] <= "2025-12")]
    r26 = recount_2026()
    c03 = pd.read_csv(config.RESULTS / "postings_ssyk4_monthly_2026H1.csv",
                      dtype={"ssyk4": str})
    c03["ssyk4"] = c03["ssyk4"].str.zfill(4)
    chk = c03.merge(r26, on=["ssyk4", "year_month"], how="outer",
                    suffixes=("_03", "_new"), indicator=True)
    assert (chk["_merge"] == "both").all() and \
        (chk["n_ads_03"] == chk["n_ads_new"]).all(), "2026 recount differs from 03"
    print(f"  check 2 passed: the 2026 recount equals 03 in all {len(chk):,} cells")
    vac = pd.concat([base[["ssyk4", "year_month", "n_ads", "n_vacancies"]], r26],
                    ignore_index=True)

    ext = pd.read_csv(PROC / "postings_daioe_merged_extended.csv",
                      dtype={"ssyk4": str})
    ext["ssyk4"] = ext["ssyk4"].str.zfill(4)
    ext = ext[(ext["year_month"] >= "2020-01") & (ext["year_month"] <= "2026-06")]
    m = ext.merge(vac, on=["ssyk4", "year_month"], how="left",
                  suffixes=("", "_v"), validate="one_to_one")
    assert m["n_vacancies"].notna().all()
    assert (m["n_ads"] == m["n_ads_v"]).all()
    assert len(m) == 28084 and (m["n_ads"] > 0).all(), len(m)
    new = stats(m)
    new.to_csv(config.RESULTS / "postings_sumstats_v3.csv")
    print(new.round(1).to_string())

    q = QORDER + ["All"]
    r = lambda key, dec: " & ".join(fmt(new.loc[key, c], dec) for c in q)
    tex = rf"""\begin{{table}}[ht!]
\centering
\footnotesize
\caption{{The posting sample: occupation $\times$ month, 2020:01--2026:06.}}
\label{{tab:sumstats_postings_v3}}
\begin{{tabular}}{{@{{}}lrrrrr@{{}}}}
\toprule
 & Q1 & Q2 & Q3 & Q4 & All \\
\midrule
Occupations (SSYK4) & {r("occupations", 0)} \\
Occupation$\times$month cells & {r("cells", 0)} \\[4pt]
Ads per month (mean) & {r("ads_mean", 1)} \\
Ads per month (SD) & {r("ads_sd", 1)} \\
Ads per month (median) & {r("ads_median", 0)} \\[4pt]
Vacancies per month (mean) & {r("vac_mean", 1)} \\
Vacancies per month (SD) & {r("vac_sd", 1)} \\[4pt]
DAIOE GAI percentile (mean) & {r("pctl_mean", 1)} \\
\bottomrule
\end{{tabular}}
\begin{{minipage}}{{0.95\textwidth}}\footnotesize\vspace{{4pt}}Q1 is the least and Q4 the most exposed quartile of the DAIOE generative-AI percentile ranking. Statistics are over occupation-by-month cells with at least one advertisement, the sample of Online Appendix~II.2 and II.6.
\end{{minipage}}
\end{{table}}
"""
    out = config.TABLES / "tableI2b_sumstats_postings_v3.tex"
    out.write_text(tex, encoding="utf-8")
    print(f"  wrote {out.name}")


if __name__ == "__main__":
    main()
