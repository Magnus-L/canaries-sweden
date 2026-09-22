#!/usr/bin/env python3
"""
l08_extend_2026.py: the posting window extended to June 2026, and the
posting difference-in-differences on both windows.

QUESTION
Does the posting conclusion survive current data? JobTech publishes 2026
as closed-quarter files, so the posting series can be extended to the end
of the second quarter with the paper's own extraction logic.

WHAT IT BUILDS
The two 2026 quarter files are streamed with the same classification as
the accounting script (l01's classify_ad), deduplicated within the half
year, and aggregated to occupation by month; the 2020 to 2025 aggregates
of the processed data are appended to them. The 2020 to 2025 rows were
deduplicated across years and the 2026 rows within the half year, so the
seam carries a slight overcount, below one per cent on the cross-year
duplicate rates of script l01. July 2026 is not included: no closed file
exists and the live feed undercounts the newest months. The merged,
exposure-tagged panel on the full window is written so that every posting
regression (scripts l03, l05, l06) runs on the same months.

Equation (1) is then estimated on two windows, January 2020 to December
2025 and January 2020 to June 2026, by OLS on ln(postings) with the zero
cells dropped and by Poisson with them kept: PostRB x High and PostGPT x
High, occupation and month effects, standard errors clustered by
occupation (pyfixest). The quartile index series to June 2026 is
rebuilt for Figure 1.

INPUTS AND OUTPUTS
Reads the cached corpus files 2026-Q1.jsonl.zip and 2026-Q2.jsonl.zip
under ~/.cache/aiel-jobads/, data/processed/postings_ssyk4_monthly.csv
and daioe_quartiles.csv. Writes revision/output/
postings_ssyk4_monthly_2026H1.csv, postings_ssyk4_monthly_extended.csv and
postings_quartile_indexed_extended.csv;
data/processed/postings_daioe_merged_extended.csv; and
revision/tables/postings_extended_did.csv.

IN THE PAPER
Section 2 (289,601 advertisements for the 2026 half year) and Section 3
(beta_1 = -0.127 and beta_2 = -0.059 on the window to June 2026); Online
Appendix II.13, Table tab:extended (built by script l10); Figure 1
through script l07.
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

    # The merged, exposure-tagged panel on the full window, so that every
    # posting regression (l03, l05, l06) runs on the same months. The
    # processed file postings_daioe_merged.csv stops at February 2026 and
    # its last two months are the live-feed artefact.
    _d = pd.read_csv(_cfg.PROCESSED / "daioe_quartiles.csv",
                     dtype={"ssyk4": str})
    _d["ssyk4"] = _d["ssyk4"].str.zfill(4)
    _merged = ext.merge(_d, on="ssyk4", how="inner")
    _merged["high_exposure"] = (_merged["exposure_quartile"].astype(str)
                                .str.startswith("Q4").astype(int))
    _mp = _cfg.PROCESSED / "postings_daioe_merged_extended.csv"
    _merged.to_csv(_mp, index=False)
    print(f"  wrote {_mp.name}: {_merged['year_month'].min()} to "
          f"{_merged['year_month'].max()}, {len(_merged):,} occupation-months")

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
