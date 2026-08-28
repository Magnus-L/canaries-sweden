#!/usr/bin/env python3
"""
l06_seasonality_variant.py -- T14/R1.9: the seasonality rider.

R1 (minor): "There is currently a lot of seasonality left in the event-
study plots. Instead of month fixed effects, it might therefore be
better to use industry-by-month fixed effects."

Platsbanken carries no industry, so the defensible analogue is the
occupation GROUP: SSYK 1-digit x CALENDAR month FE (seasonality patterns
allowed to differ by broad occupation group), added to the year-month FE
the baseline carries. Estimated as the pooled DiD in three nested
variants on ln(n_ads), plus a Poisson companion:

  S0  ssyk4 + year_month                       (baseline, as submitted)
  S1  ssyk4 + year_month + ssyk1 x calmonth     (the referee's fix)
  S2  ssyk4 + ssyk1 x year_month                (spec 4; nests S1)

Output: tables/postings_seasonality.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REV))
from config import (PROCESSED, V2_TAB, RIKSBANKEN_HIKE,  # noqa: E402
                    CHATGPT_LAUNCH)


def main():
    import pyfixest as pf
    print("L6: seasonality variants (R1.9)")
    df = pd.read_csv(PROCESSED / "postings_daioe_merged.csv",
                     dtype={"ssyk4": str})
    df["ssyk4"] = df["ssyk4"].str.zfill(4)
    df = df[(df["year_month"] >= "2020-01")
            & (df["year_month"] <= "2025-12")].copy()
    df = df[df["n_ads"] > 0].copy()
    df["ln_ads"] = np.log(df["n_ads"])
    df["date"] = pd.to_datetime(df["year_month"] + "-01")
    df["rb_x_high"] = ((df["date"] >= pd.Timestamp(RIKSBANKEN_HIKE))
                       & (df["high_exposure"] == 1)).astype(int)
    df["gpt_x_high"] = ((df["date"] >= pd.Timestamp(CHATGPT_LAUNCH))
                        & (df["high_exposure"] == 1)).astype(int)
    df["ssyk1"] = df["ssyk4"].str[0]
    df["calmonth"] = df["year_month"].str[5:7]
    df["ssyk1_cal"] = df["ssyk1"] + "_" + df["calmonth"]
    df["ssyk1_ym"] = df["ssyk1"] + "_" + df["year_month"]

    specs = {
        "S0_baseline": "ln_ads ~ rb_x_high + gpt_x_high | ssyk4 + year_month",
        "S1_groupseason": ("ln_ads ~ rb_x_high + gpt_x_high "
                           "| ssyk4 + year_month + ssyk1_cal"),
        "S2_groupmonth": ("ln_ads ~ rb_x_high + gpt_x_high "
                          "| ssyk4 + ssyk1_ym"),
        "S1_poisson": None,  # Poisson companion of S1, on counts
    }
    rows = []
    for name, fml in specs.items():
        if name == "S1_poisson":
            fit = pf.fepois("n_ads ~ rb_x_high + gpt_x_high "
                            "| ssyk4 + year_month + ssyk1_cal",
                            data=df, vcov={"CRV1": "ssyk4"})
        else:
            fit = pf.feols(fml, data=df, vcov={"CRV1": "ssyk4"})
        for t in ("rb_x_high", "gpt_x_high"):
            rows.append({"spec": name, "term": t, "coef": fit.coef()[t],
                         "se": fit.se()[t], "pval": fit.pvalue()[t],
                         "n_obs": fit._N})
            print(f"  {name:>15} {t:>10}: {fit.coef()[t]:+.4f} "
                  f"(SE {fit.se()[t]:.4f}, p {fit.pvalue()[t]:.4f})")
    pd.DataFrame(rows).to_csv(V2_TAB / "postings_seasonality.csv",
                              index=False)
    print("Saved postings_seasonality.csv")


if __name__ == "__main__":
    main()
