#!/usr/bin/env python3
"""
06_seasonality.py: Equation (1) with group-specific seasonal effects.

WHAT IT ESTIMATES
Occupation-by-month cells from January 2020 to June 2026 with a positive
count, ln(postings), PostRB x High and PostGPT x High as in Equation (1).
S0: occupation and month effects (the baseline). S1: adds one-digit
occupation group (the first digit of SSYK 2012) by calendar-month effects,
so that seasonal patterns may differ by broad occupation group. S2:
occupation and one-digit group by month-of-sample effects, which nests S1.
S1 is also fitted by Poisson on the counts. Standard errors clustered by
occupation (pyfixest).

INPUTS   data/processed/postings_daioe_merged_extended.csv (03)
OUTPUTS  output/results/postings_seasonality.csv
SERVES   Section 3 and Online Appendix II.7, Table A8 (through 16)
RUNTIME  under a minute
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (PROCESSED, RESULTS, RIKSBANKEN_HIKE,  # noqa: E402
                    CHATGPT_LAUNCH, POSTINGS_REGRESSION_END)


def main():
    import pyfixest as pf
    print("Equation (1) with group-specific seasonal effects")
    df = pd.read_csv(PROCESSED / ("postings_daioe_merged_extended.csv"
                     if (PROCESSED / "postings_daioe_merged_extended.csv").exists()
                     else "postings_daioe_merged.csv"),
                     dtype={"ssyk4": str})
    df["ssyk4"] = df["ssyk4"].str.zfill(4)
    df = df[(df["year_month"] >= "2020-01")
            & (df["year_month"] <= POSTINGS_REGRESSION_END)].copy()
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
    pd.DataFrame(rows).to_csv(RESULTS / "postings_seasonality.csv",
                              index=False)
    print("Saved postings_seasonality.csv")


if __name__ == "__main__":
    main()
