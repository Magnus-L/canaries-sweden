#!/usr/bin/env python3
"""
l48_tab_sumstats_postings_v3.py: summary statistics for the posting
sample, on the window the paper's posting estimates use.

WHY THIS EXISTS
Online Appendix I.2 said the posting summary statistics were "on the
October 2019 to February 2026 window and are available on request". The
posting estimates of II.2 and II.6 run on January 2020 to June 2026
(28,084 cells, l08). The as-submitted table (offline appendix,
tab:sumstats_postings) was typed into the appendix in March 2026 and no
script produced it. This script writes down the method it used, proves it
by reproducing that table cell for cell on the old file, and applies the
same method to the II.6 panel.

THE METHOD (reproduced exactly, see the gate)
Pooled occupation-by-month cells, grouped by DAIOE generative-AI
exposure quartile and over all cells: distinct occupations, cells, the
mean, SD and median of ads per cell, the mean and SD of vacancies per
cell, and the mean exposure percentile across cells.

VACANCIES FOR 2026
postings_daioe_merged_extended.csv carries ads only. Vacancies for 2020
to 2025 come from data/processed/postings_ssyk4_monthly.csv (src/02).
For January to June 2026 the quarterly cache files l08 read are streamed
again with l01's classifier and l08's de-duplication, and vacancies are
summed as src/02 sums them (number_of_vacancies, 1 if missing or below
1). A second gate requires the recounted ads to equal l08's cached
2026-H1 counts cell for cell, so the vacancy column rests on exactly the
ads that the estimates use.

    python3 revision/local/l48_tab_sumstats_postings_v3.py

INPUTS  data/processed/postings_daioe_merged.csv (gate),
        data/processed/postings_daioe_merged_extended.csv (l08),
        data/processed/postings_ssyk4_monthly.csv (src/02),
        revision/output/postings_ssyk4_monthly_2026H1.csv (l08, gate),
        ~/.cache/aiel-jobads/2026-Q1.jsonl.zip, 2026-Q2.jsonl.zip
OUTPUTS revision/tables/postings_sumstats_v3.csv,
        revision/tables/tableI2b_sumstats_postings_v3.tex, copied to
        canaries-sweden-paper/tables/.
IN THE PAPER Online Appendix I.2, Table tab:sumstats_postings_v3.
"""
import importlib.util
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd

REV = Path(__file__).resolve().parents[1]
ROOT = REV.parent
_cfg_spec = importlib.util.spec_from_file_location("v2config", REV / "config.py")
_cfg = importlib.util.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg)
_l01_spec = importlib.util.spec_from_file_location(
    "l01", REV / "local" / "l01_postings_accounting.py")
l01 = importlib.util.module_from_spec(_l01_spec)
_l01_spec.loader.exec_module(l01)

PROC = ROOT / "data" / "processed"
CACHE = Path.home() / ".cache" / "aiel-jobads"
Q_FILES = ["2026-Q1.jsonl.zip", "2026-Q2.jsonl.zip"]
PAPER_TAB = ROOT.parent / "canaries-sweden-paper" / "tables"
QORDER = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]

# The as-submitted table (offline appendix, tab:sumstats_postings).
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
    """Ads and vacancies for 2026-H1, l08's pass with vacancies summed."""
    seen, counts = set(), {}
    for fname in Q_FILES:
        with zipfile.ZipFile(CACHE / fname) as zf:
            for name in (n for n in zf.namelist() if n.endswith(".jsonl")):
                with zf.open(name) as f:
                    for line in f:
                        try:
                            ad = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        reason, rec = l01.classify_ad(ad)
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
    print("L48: posting summary statistics on the II.6 window")
    # Gate 1: the method reproduces the as-submitted table.
    old = pd.read_csv(PROC / "postings_daioe_merged.csv", dtype={"ssyk4": str})
    got = stats(old)
    for row, vals in OLD.items():
        for q, v in zip(QORDER + ["All"], vals):
            dec = 0 if row in ("occupations", "cells", "ads_median") else 1
            assert round(got.loc[row, q], dec) == v, (row, q, got.loc[row, q], v)
    print("  gate 1 passed: the as-submitted table is reproduced cell for cell")

    # Vacancies on the new window.
    base = pd.read_csv(PROC / "postings_ssyk4_monthly.csv", dtype={"ssyk4": str})
    base["ssyk4"] = base["ssyk4"].str.zfill(4)
    base = base[(base["year_month"] >= "2020-01") & (base["year_month"] <= "2025-12")]
    r26 = recount_2026()
    l08 = pd.read_csv(_cfg.V2_OUT / "postings_ssyk4_monthly_2026H1.csv",
                      dtype={"ssyk4": str})
    l08["ssyk4"] = l08["ssyk4"].str.zfill(4)
    chk = l08.merge(r26, on=["ssyk4", "year_month"], how="outer",
                    suffixes=("_l08", "_new"), indicator=True)
    assert (chk["_merge"] == "both").all() and \
        (chk["n_ads_l08"] == chk["n_ads_new"]).all(), "2026 recount differs from l08"
    print(f"  gate 2 passed: 2026-H1 recount equals l08 in all {len(chk):,} cells")
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
    new.to_csv(_cfg.V2_TAB / "postings_sumstats_v3.csv")
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
    out = _cfg.V2_TAB / "tableI2b_sumstats_postings_v3.tex"
    out.write_text(tex, encoding="utf-8")
    shutil.copy(out, PAPER_TAB / out.name)
    print(f"  wrote {out.name} and copied it to the paper repo")


if __name__ == "__main__":
    main()
