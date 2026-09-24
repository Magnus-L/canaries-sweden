#!/usr/bin/env python3
"""
02_coverage_diagnostics.py: coverage of the occupation field, January 2020 to
December 2025.

WHAT IT BUILDS
On the occupation-by-month counts: (a) the number of SSYK 2012 codes with at
least one advertisement in each month; (b) on the balanced grid of the
occupations DAIOE prices, the share of occupation-months with no
advertisement, by month and exposure quartile; (c) each quartile's share of
coded advertisements by month, with its mean before and after December 2022;
(d) the number of distinct codes at each stage (in the counts, priced by
DAIOE, with a positive count in the regression panel) and the codes lost at
each stage. Script 17 extends (a) and (b) to June 2026 after checking that it
reproduces these files.

INPUTS   config.POSTINGS_SSYK4; data/processed/daioe_quartiles.csv
OUTPUTS  output/results/coverage_active_occupations.csv, coverage_zero_cells.csv,
         coverage_quartile_shares.csv, occupation_reconciliation.csv,
         occupation_reconciliation_lists.txt
SERVES   Section 2 (369 of 400 occupations) and Online Appendix II.4 and II.5
RUNTIME  seconds
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROCESSED, POSTINGS_SSYK4, RESULTS  # noqa: E402


def main():
    print("Coverage of the occupation field on the occupation-by-month counts")
    post = pd.read_csv(POSTINGS_SSYK4,
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
    active.to_csv(RESULTS / "coverage_active_occupations.csv", index=False)
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
    zero.to_csv(RESULTS / "coverage_zero_cells.csv", index=False)
    zq4 = zero[zero["exposure_quartile"] == "Q4 (highest)"]
    print(f"  (b) Q4 zero-cell share: {zq4['zero_share'].mean():.3f} mean, "
          f"{zq4['zero_share'].iloc[-1]:.3f} final month")

    # (c) quartile shares of coded ads over time -----------------------------
    qs = (merged.groupby(["year_month", "exposure_quartile"])["n_ads"].sum()
          .reset_index())
    tot = qs.groupby("year_month")["n_ads"].transform("sum")
    qs["share"] = qs["n_ads"] / tot
    qs.to_csv(RESULTS / "coverage_quartile_shares.csv", index=False)
    q4 = qs[qs["exposure_quartile"] == "Q4 (highest)"]
    pre = q4[q4["year_month"] < "2022-12"]["share"].mean()
    post_m = q4[q4["year_month"] >= "2022-12"]["share"].mean()
    print(f"  (c) Q4 share of coded ads: {pre:.3f} pre-ChatGPT, "
          f"{post_m:.3f} post")

    # (d) from 400 codes to 369 ------------------------------------------
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
    rec.to_csv(RESULTS / "occupation_reconciliation.csv", index=False)
    with open(RESULTS / "occupation_reconciliation_lists.txt", "w") as f:
        for k, codes in lost.items():
            f.write(f"{k} ({len(codes)}): {', '.join(codes)}\n")
    print("  (d) reconciliation:", stage)
    for k, codes in lost.items():
        print(f"      {k}: {len(codes)} codes")

    print("Done.")


if __name__ == "__main__":
    main()
