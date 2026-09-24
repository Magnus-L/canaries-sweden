#!/usr/bin/env python3
"""
03_extend_2026_and_did.py: the posting window to June 2026, and Equation (1)
on the windows to December 2025 and to June 2026.

WHAT IT BUILDS
The two closed-quarter archives of 2026 are streamed with the classification
of 01 (classify_ad), repeats of an advertisement identifier within the half
year are removed, and the advertisements dated January to June 2026 are
counted by occupation and month and appended to the counts for 2020 to 2025.
The 2020 to 2025 counts had repeats removed across years and the 2026 counts
within the half year, so the seam carries a slight overcount, which 17
measures (under one per cent). No month is taken from the live feed. The
exposure-tagged panel on the full window is written so that every posting
regression runs on the same months.

Equation (1) is then estimated on January 2020 to December 2025 and on January
2020 to June 2026: ln(postings) on PostRB x High (from April 2022) and
PostGPT x High (from December 2022) with occupation and month fixed effects,
by OLS on the positive cells and by Poisson pseudo-maximum likelihood on the
same cells, standard errors clustered by occupation (pyfixest). The quartile
index series to June 2026 is rebuilt for Figure 1.

INPUTS   config.JOBADS_DIR/2026-Q1.jsonl.zip, 2026-Q2.jsonl.zip;
         config.POSTINGS_SSYK4; data/processed/daioe_quartiles.csv
OUTPUTS  output/results/postings_ssyk4_monthly_2026H1.csv,
         postings_ssyk4_monthly_extended.csv, postings_quartile_indexed_extended.csv,
         postings_extended_did.csv; data/processed/postings_daioe_merged_extended.csv
SERVES   Section 2 (289,601 advertisements for 2026), Section 3 and Equation
         (1) (beta_1 = -0.127, beta_2 = -0.059), Online Appendix II.6
         (Table A6, through 16) and Figure 1 (through 18)
RUNTIME  about 5 minutes (the 2026 counts are cached in
         postings_ssyk4_monthly_2026H1.csv; delete it to recount)
"""

import sys
import zipfile
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import config, sibling  # noqa: E402

acc = sibling("01_postings_accounting")
OUT = config.RESULTS


def process_2026():
    seen = set()
    counts = {}
    for stem in config.PLATSBANKEN_QUARTERS:
        zpath = config.platsbanken_zip(stem)
        print(f"  streaming {zpath.name} ...")
        with zipfile.ZipFile(zpath) as zf:
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
    print("Postings to June 2026, and Equation (1) on both windows")
    f26 = OUT / "postings_ssyk4_monthly_2026H1.csv"
    if f26.exists():
        df26 = pd.read_csv(f26, dtype={"ssyk4": str})
        print(f"  cached 2026-H1 ({df26['n_ads'].sum():,} ads)")
    else:
        df26 = process_2026()
    df26["ssyk4"] = df26["ssyk4"].astype(str).str.zfill(4)

    base = pd.read_csv(config.POSTINGS_SSYK4, dtype={"ssyk4": str})
    base["ssyk4"] = base["ssyk4"].str.zfill(4)
    base = base[(base["year_month"] >= "2020-01")
                & (base["year_month"] <= "2025-12")]
    ext = pd.concat([base[["ssyk4", "year_month", "n_ads"]],
                     df26[["ssyk4", "year_month", "n_ads"]]],
                    ignore_index=True)
    ext.to_csv(OUT / "postings_ssyk4_monthly_extended.csv", index=False)

    # The merged, exposure-tagged panel on the full window, so that every
    # posting regression (04 to 07, 10) runs on the same months. The file
    # postings_daioe_merged.csv of the submitted version runs to February
    # 2026 and its last two months come from the live feed.
    _d = pd.read_csv(config.PROCESSED / "daioe_quartiles.csv",
                     dtype={"ssyk4": str})
    _d["ssyk4"] = _d["ssyk4"].str.zfill(4)
    _merged = ext.merge(_d, on="ssyk4", how="inner")
    _merged["high_exposure"] = (_merged["exposure_quartile"].astype(str)
                                .str.startswith("Q4").astype(int))
    _mp = config.PROCESSED / "postings_daioe_merged_extended.csv"
    _merged.to_csv(_mp, index=False)
    print(f"  wrote {_mp.name}: {_merged['year_month'].min()} to "
          f"{_merged['year_month'].max()}, {len(_merged):,} occupation-months")

    # --- Equation (1): OLS on ln(ads) and Poisson, both windows ------------
    import pyfixest as pf
    daioe = pd.read_csv(config.PROCESSED / "daioe_quartiles.csv",
                        dtype={"ssyk4": str})
    daioe["ssyk4"] = daioe["ssyk4"].str.zfill(4)
    daioe["high"] = (daioe["exposure_quartile"].astype(str)
                     .str.startswith("Q4").astype(int))
    m = ext.merge(daioe[["ssyk4", "high"]], on="ssyk4", how="inner")
    m["date"] = pd.to_datetime(m["year_month"] + "-01")
    m["rb_x_high"] = ((m["date"] >= pd.Timestamp(config.RIKSBANKEN_HIKE))
                      & (m["high"] == 1)).astype(int)
    m["gpt_x_high"] = ((m["date"] >= pd.Timestamp(config.CHATGPT_LAUNCH))
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
    res.to_csv(config.RESULTS / "postings_extended_did.csv", index=False)
    for _, r in res.iterrows():
        print(f"  {r['window']:>22} {r['estimator']:>8} {r['term']:>10}: "
              f"{r['coef']:+.4f} (SE {r['se']:.4f}, p {r['pval']:.4f})")

    # --- the quartile index series to June 2026, for Figure 1 ------------
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
    print("  quartile index to June 2026 written (drawn by 18_figures.py)")


if __name__ == "__main__":
    main()
