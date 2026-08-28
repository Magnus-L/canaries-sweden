#!/usr/bin/env python3
"""
l08_extend_2026.py -- Er.2 upgraded (ML, 28 Aug): extend the POSTING side
to June 2026 with the official quarterly files.

JobTech now publishes 2026 as closed-quarter files (2026-Q1, 2026-Q2);
both sit in the Monitor's cached corpus (~/.cache/aiel-jobads/), so no
download. This script processes them with the PAPER'S OWN extraction
logic (classify_ad imported from l01, which mirrors src/02 exactly) and
appends to the submitted monthly aggregates.

Honesty notes carried into the outputs:
 - Seam: the 2020-2025 aggregates come from the February 2026 vintage
   files and were deduplicated ACROSS years; 2026-H1 is deduplicated
   within itself (ad ids from earlier years are not on disk in the
   processed data). Cross-year duplicates ran 0.5-0.9%/year in L1, so
   the seam bias is below one per cent and one-signed (slight overcount).
 - July 2026 is NOT included: no closed official file exists yet, and
   live-window (JobStream) counts undercount the newest months -- the
   exact artefact the paper's descriptive-cut rule exists for.
 - Regressions on the extended window are REPORTED AS EXTENSIONS; the
   submitted-window estimates stay the headline (plan v3 freeze).

Output:
  output/postings_ssyk4_monthly_2026H1.csv
  output/postings_ssyk4_monthly_extended.csv   (2020-01 .. 2026-06)
  tables/postings_extended_did.csv             (OLS + Poisson, both windows)
  figures/fig1_two_panel_extended.pdf/.png
"""

import importlib.util
import sys
import zipfile
import json
from pathlib import Path

import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parents[1]
_cfg_spec = importlib.util.spec_from_file_location("v2config", REV / "config.py")
_cfg = importlib.util.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg)

_l01_spec = importlib.util.spec_from_file_location(
    "l01", REV / "local" / "l01_postings_accounting.py")
l01 = importlib.util.module_from_spec(_l01_spec)
_l01_spec.loader.exec_module(l01)

CACHE = Path.home() / ".cache" / "aiel-jobads"
Q_FILES = ["2026-Q1.jsonl.zip", "2026-Q2.jsonl.zip"]
OUT = _cfg.V2_OUT


def process_2026():
    seen = set()
    counts = {}
    for fname in Q_FILES:
        zpath = CACHE / fname
        print(f"  streaming {fname} ...")
        with zipfile.ZipFile(zpath) as zf:
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
                        key = (rec["ssyk4"], rec["year_month"])
                        counts[key] = counts.get(key, 0) + 1
    df = pd.DataFrame(
        [{"ssyk4": k[0], "year_month": k[1], "n_ads": v}
         for k, v in counts.items()])
    df = df[(df["year_month"] >= "2026-01")
            & (df["year_month"] <= "2026-06")]
    df.to_csv(OUT / "postings_ssyk4_monthly_2026H1.csv", index=False)
    print(f"  2026-H1: {df['n_ads'].sum():,} ads, "
          f"{df['ssyk4'].nunique()} occupations")
    return df


def main():
    print("L8: extend postings to June 2026")
    f26 = OUT / "postings_ssyk4_monthly_2026H1.csv"
    if f26.exists():
        df26 = pd.read_csv(f26, dtype={"ssyk4": str})
        print(f"  cached 2026-H1 ({df26['n_ads'].sum():,} ads)")
    else:
        df26 = process_2026()
    df26["ssyk4"] = df26["ssyk4"].astype(str).str.zfill(4)

    base = pd.read_csv(_cfg.PROCESSED / "postings_ssyk4_monthly.csv",
                       dtype={"ssyk4": str})
    base["ssyk4"] = base["ssyk4"].str.zfill(4)
    base = base[(base["year_month"] >= "2020-01")
                & (base["year_month"] <= "2025-12")]
    ext = pd.concat([base[["ssyk4", "year_month", "n_ads"]],
                     df26[["ssyk4", "year_month", "n_ads"]]],
                    ignore_index=True)
    ext.to_csv(OUT / "postings_ssyk4_monthly_extended.csv", index=False)

    # --- extended DiD: OLS (submitted spec) + Poisson, both windows ------
    import pyfixest as pf
    daioe = pd.read_csv(_cfg.PROCESSED / "daioe_quartiles.csv",
                        dtype={"ssyk4": str})
    daioe["ssyk4"] = daioe["ssyk4"].str.zfill(4)
    daioe["high"] = (daioe["exposure_quartile"].astype(str)
                     .str.startswith("Q4").astype(int))
    m = ext.merge(daioe[["ssyk4", "high"]], on="ssyk4", how="inner")
    m["date"] = pd.to_datetime(m["year_month"] + "-01")
    m["rb_x_high"] = ((m["date"] >= pd.Timestamp(_cfg.RIKSBANKEN_HIKE))
                      & (m["high"] == 1)).astype(int)
    m["gpt_x_high"] = ((m["date"] >= pd.Timestamp(_cfg.CHATGPT_LAUNCH))
                       & (m["high"] == 1)).astype(int)

    rows = []
    for wname, hi in (("submitted_to_2025-12", "2025-12"),
                      ("extended_to_2026-06", "2026-06")):
        w = m[m["year_month"] <= hi].copy()
        wp = w[w["n_ads"] > 0].copy()
        wp["ln_ads"] = np.log(wp["n_ads"])
        fo = pf.feols("ln_ads ~ rb_x_high + gpt_x_high | ssyk4 + year_month",
                      data=wp, vcov={"CRV1": "ssyk4"})
        fp = pf.fepois("n_ads ~ rb_x_high + gpt_x_high | ssyk4 + year_month",
                       data=w, vcov={"CRV1": "ssyk4"})
        for est, fit in (("OLS_ln", fo), ("Poisson", fp)):
            for t in ("rb_x_high", "gpt_x_high"):
                rows.append({"window": wname, "estimator": est, "term": t,
                             "coef": fit.coef()[t], "se": fit.se()[t],
                             "pval": fit.pvalue()[t], "n_obs": fit._N})
    res = pd.DataFrame(rows)
    res.to_csv(_cfg.V2_TAB / "postings_extended_did.csv", index=False)
    for _, r in res.iterrows():
        print(f"  {r['window']:>22} {r['estimator']:>8} {r['term']:>10}: "
              f"{r['coef']:+.4f} (SE {r['se']:.4f}, p {r['pval']:.4f})")

    # --- extended figure -------------------------------------------------
    # Rebuild the quartile index series from the extended aggregates so the
    # two-panel figure runs to June 2026.
    q = ext.merge(daioe[["ssyk4", "exposure_quartile"]], on="ssyk4",
                  how="inner")
    q = (q.groupby(["exposure_quartile", "year_month"])["n_ads"].sum()
         .reset_index())
    q["date"] = pd.to_datetime(q["year_month"] + "-01")
    base_vals = q[q["year_month"] == "2020-02"].set_index(
        "exposure_quartile")["n_ads"]
    q["ads_idx"] = q.apply(
        lambda r: 100 * r["n_ads"] / base_vals[r["exposure_quartile"]],
        axis=1)
    q.to_csv(OUT / "postings_quartile_indexed_extended.csv", index=False)
    print("  extended quartile index written (figure uses l07's builder "
          "pointed at this file)")


if __name__ == "__main__":
    main()
