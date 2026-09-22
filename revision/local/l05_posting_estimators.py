#!/usr/bin/env python3
"""
l05_posting_estimators.py: the posting difference-in-differences by
Poisson, on the panel with its zero cells.

QUESTION
The posting regression of Equation (1) is estimated by OLS on the log of
the count, which drops occupation-months with no advertisement. Does the
result depend on that? This script estimates the same interactions by
Poisson pseudo-maximum likelihood on the balanced occupation-by-month
panel with the zeros kept.

DESIGN
Occupation-by-month cells from January 2020 to the last closed quarter
(June 2026), balanced over the DAIOE-matched occupations with zero counts
filled in. Terms: PostRB x High (from April 2022) and PostGPT x High (from
December 2022), High the top exposure quartile. P1: occupation and month
effects. P2: occupation and one-digit occupation group by month effects.
P3: the decile interactions of script l03 in Poisson form, the fifth
decile the reference. Standard errors clustered by occupation (pyfixest).

INPUTS AND OUTPUTS
Reads data/processed/postings_daioe_merged_extended.csv (script l08), or
postings_daioe_merged.csv if absent. Writes
revision/tables/postings_poisson.csv and postings_poisson_deciles.csv.

IN THE PAPER
Section 3, the statement that neither posting coefficient moves with the
estimator; Online Appendix II.4. The extended-window Poisson estimates of
Table tab:extended come from script l08.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REV))
from config import (PROCESSED, V2_TAB, RIKSBANKEN_HIKE,  # noqa: E402
                    CHATGPT_LAUNCH, POSTINGS_REGRESSION_END)


def load_panel():
    df = pd.read_csv(PROCESSED / ("postings_daioe_merged_extended.csv"
                     if (PROCESSED / "postings_daioe_merged_extended.csv").exists()
                     else "postings_daioe_merged.csv"),
                     dtype={"ssyk4": str})
    df["ssyk4"] = df["ssyk4"].str.zfill(4)
    df = df[(df["year_month"] >= "2020-01")
            & (df["year_month"] <= POSTINGS_REGRESSION_END)].copy()

    # Balanced occupation x month panel with explicit zeros: Poisson wants
    # the zero cells the OLS-on-ln(n) spec had to drop.
    occs = df[["ssyk4", "pctl_rank_genai", "exposure_quartile",
               "high_exposure"]].drop_duplicates("ssyk4")
    months = sorted(df["year_month"].unique())
    full = (occs.assign(_k=1)
            .merge(pd.DataFrame({"year_month": months, "_k": 1}), on="_k")
            .drop(columns="_k")
            .merge(df[["ssyk4", "year_month", "n_ads"]],
                   on=["ssyk4", "year_month"], how="left"))
    full["n_ads"] = full["n_ads"].fillna(0).astype(int)

    full["date"] = pd.to_datetime(full["year_month"] + "-01")
    full["post_rb"] = (full["date"] >= pd.Timestamp(RIKSBANKEN_HIKE)).astype(int)
    full["post_gpt"] = (full["date"] >= pd.Timestamp(CHATGPT_LAUNCH)).astype(int)
    full["high"] = full["high_exposure"].astype(int)
    full["rb_x_high"] = full["post_rb"] * full["high"]
    full["gpt_x_high"] = full["post_gpt"] * full["high"]
    full["ssyk1_month"] = full["ssyk4"].str[0] + "_" + full["year_month"]
    return full


def tidy(fit, spec, terms):
    rows = []
    for t in terms:
        rows.append({"spec": spec, "term": t,
                     "coef": fit.coef()[t], "se": fit.se()[t],
                     "pval": fit.pvalue()[t], "n_obs": fit._N})
    return rows


def main():
    import pyfixest as pf
    print("L5: posting Poisson estimators (E6/R1.8)")
    full = load_panel()
    print(f"  balanced panel: {len(full):,} cells "
          f"({(full['n_ads'] == 0).mean():.1%} zeros)")

    rows = []
    # P1: occupation + month FE
    f1 = pf.fepois("n_ads ~ rb_x_high + gpt_x_high | ssyk4 + year_month",
                   data=full, vcov={"CRV1": "ssyk4"})
    rows += tidy(f1, "P1_occ_month", ["rb_x_high", "gpt_x_high"])
    # P2: occupation + SSYK1 x month FE (spec-4 analogue)
    f2 = pf.fepois("n_ads ~ rb_x_high + gpt_x_high | ssyk4 + ssyk1_month",
                   data=full, vcov={"CRV1": "ssyk4"})
    rows += tidy(f2, "P2_occ_groupmonth", ["rb_x_high", "gpt_x_high"])
    out = pd.DataFrame(rows)
    out.to_csv(V2_TAB / "postings_poisson.csv", index=False)
    for _, r in out.iterrows():
        print(f"  {r['spec']:>18} {r['term']:>10}: {r['coef']:+.4f} "
              f"(SE {r['se']:.4f}, p {r['pval']:.4f})")

    # P3: decile interactions, Poisson
    occ = full[["ssyk4", "pctl_rank_genai"]].drop_duplicates("ssyk4")
    occ["decile"] = pd.qcut(occ["pctl_rank_genai"], 10, labels=False) + 1
    full = full.merge(occ[["ssyk4", "decile"]], on="ssyk4")
    terms = []
    DECS = [d for d in range(1, 11) if d != 5]   # median decile = reference
    for d in DECS:
        isd = (full["decile"] == d).astype(int)
        full[f"rb_d{d}"] = full["post_rb"] * isd
        full[f"gpt_d{d}"] = full["post_gpt"] * isd
        terms += [f"rb_d{d}", f"gpt_d{d}"]
    f3 = pf.fepois(f"n_ads ~ {' + '.join(terms)} | ssyk4 + year_month",
                   data=full, vcov={"CRV1": "ssyk4"})
    dec = pd.DataFrame(tidy(f3, "P3_deciles", terms))
    dec["period"] = np.where(dec["term"].str.startswith("rb"),
                             "post_riksbank", "post_chatgpt")
    dec["decile"] = dec["term"].str.extract(r"d(\d+)").astype(int)
    dec.to_csv(V2_TAB / "postings_poisson_deciles.csv", index=False)
    g = dec[dec["period"] == "post_chatgpt"].sort_values("decile")
    print("  Poisson decile gradient (post-ChatGPT): "
          + ", ".join(f"d{int(r.decile)}={r.coef:+.3f}"
                      for r in g.itertuples()))


if __name__ == "__main__":
    main()
