#!/usr/bin/env python3
"""
l02_coverage_diagnostics.py: coverage of the occupation field on the
processed posting data.

QUESTION
Could a drift in how advertisers code occupations produce the posting
results? This script reports, on the processed occupation-by-month
aggregates, how many occupations post in a month, how the share of
zero-posting cells moves by exposure quartile, whether the exposed
quartile's share of coded advertisements moves after the launch, and why
369 of the 400 four-digit occupations enter the regression sample.

WHAT IT BUILDS
On the window January 2020 to December 2025: (a) the number of SSYK 2012
codes with at least one advertisement, per month; (b) on the balanced
occupation-by-month grid of DAIOE-matched occupations, the share of
zero cells per month and exposure quartile; (c) each quartile's share of
coded advertisements per month, with the mean before and after December
2022; (d) the count of distinct occupation codes at each stage (the
postings file, matched to DAIOE, with a positive count in the regression
panel) and the codes lost at each step.

INPUTS AND OUTPUTS
Reads data/processed/postings_ssyk4_monthly.csv and daioe_quartiles.csv.
Writes revision/tables/coverage_active_occupations.csv,
coverage_zero_cells.csv, coverage_quartile_shares.csv,
occupation_reconciliation.csv and occupation_reconciliation_lists.txt.

IN THE PAPER
Section 2: 369 of 400 occupations enter the sample; Online Appendix II.12.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROCESSED, V2_TAB


def main():
    print("L2: coverage diagnostics (Ed.2)")
    post = pd.read_csv(PROCESSED / "postings_ssyk4_monthly.csv",
                       dtype={"ssyk4": str})
    post["ssyk4"] = post["ssyk4"].str.zfill(4)
    daioe = pd.read_csv(PROCESSED / "daioe_quartiles.csv", dtype={"ssyk4": str})
    daioe["ssyk4"] = daioe["ssyk4"].str.zfill(4)

    # Core window, matching the paper's regressions
    post = post[(post["year_month"] >= "2020-01")
                & (post["year_month"] <= "2025-12")].copy()
    months = sorted(post["year_month"].unique())

    # (a) active occupations per month ---------------------------------------
    active = (post[post["n_ads"] > 0]
              .groupby("year_month")["ssyk4"].nunique()
              .rename("n_active_occupations").reset_index())
    active.to_csv(V2_TAB / "coverage_active_occupations.csv", index=False)
    print(f"  (a) active occupations: {active['n_active_occupations'].min()}"
          f"-{active['n_active_occupations'].max()} per month")

    # (b) zero cells over time, by quartile ----------------------------------
    merged = post.merge(daioe[["ssyk4", "exposure_quartile"]],
                        on="ssyk4", how="inner")
    occs = merged[["ssyk4", "exposure_quartile"]].drop_duplicates()
    full = (occs.assign(_k=1)
            .merge(pd.DataFrame({"year_month": months, "_k": 1}), on="_k")
            .drop(columns="_k")
            .merge(merged[["ssyk4", "year_month", "n_ads"]],
                   on=["ssyk4", "year_month"], how="left"))
    full["n_ads"] = full["n_ads"].fillna(0)
    zero = (full.assign(zero=lambda d: (d["n_ads"] == 0).astype(int))
            .groupby(["year_month", "exposure_quartile"])
            .agg(n_occ=("ssyk4", "nunique"), n_zero=("zero", "sum"))
            .reset_index())
    zero["zero_share"] = zero["n_zero"] / zero["n_occ"]
    zero.to_csv(V2_TAB / "coverage_zero_cells.csv", index=False)
    zq4 = zero[zero["exposure_quartile"] == "Q4 (highest)"]
    print(f"  (b) Q4 zero-cell share: {zq4['zero_share'].mean():.3f} mean, "
          f"{zq4['zero_share'].iloc[-1]:.3f} final month")

    # (c) quartile shares of coded ads over time -----------------------------
    qs = (merged.groupby(["year_month", "exposure_quartile"])["n_ads"].sum()
          .reset_index())
    tot = qs.groupby("year_month")["n_ads"].transform("sum")
    qs["share"] = qs["n_ads"] / tot
    qs.to_csv(V2_TAB / "coverage_quartile_shares.csv", index=False)
    q4 = qs[qs["exposure_quartile"] == "Q4 (highest)"]
    pre = q4[q4["year_month"] < "2022-12"]["share"].mean()
    post_m = q4[q4["year_month"] >= "2022-12"]["share"].mean()
    print(f"  (c) Q4 share of coded ads: {pre:.3f} pre-ChatGPT, "
          f"{post_m:.3f} post")

    # (d) 400-vs-369 reconciliation ------------------------------------------
    stage = {}
    lost = {}
    s0 = set(post["ssyk4"].unique())
    stage["1_in_postings_file_2020-2025"] = len(s0)
    s1 = set(merged["ssyk4"].unique())
    stage["2_matched_to_DAIOE"] = len(s1)
    lost["lost_at_DAIOE_match"] = sorted(s0 - s1)
    # regression panel: n_ads > 0 rows (the OLS specification drops zeros)
    s2 = set(merged.loc[merged["n_ads"] > 0, "ssyk4"].unique())
    stage["3_regression_panel_nads_gt0"] = len(s2)
    lost["lost_at_zero_drop"] = sorted(s1 - s2)

    rec = pd.DataFrame(
        [{"stage": k, "n_occupations": v} for k, v in stage.items()]
    )
    rec.to_csv(V2_TAB / "occupation_reconciliation.csv", index=False)
    with open(V2_TAB / "occupation_reconciliation_lists.txt", "w") as f:
        for k, codes in lost.items():
            f.write(f"{k} ({len(codes)}): {', '.join(codes)}\n")
    print("  (d) reconciliation:", stage)
    for k, codes in lost.items():
        print(f"      {k}: {len(codes)} codes")

    print("Done.")


if __name__ == "__main__":
    main()
