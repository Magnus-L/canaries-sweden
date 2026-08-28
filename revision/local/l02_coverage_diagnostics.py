#!/usr/bin/env python3
"""
l02_coverage_diagnostics.py -- Ed.2 / plan T1: posting-coverage diagnostics
that run on the PROCESSED data (no re-streaming; pair with l01, which does
the raw pass).

Answers, from postings_ssyk4_monthly.csv + daioe_quartiles.csv:
  (a) active occupations per month (how many SSYK4 codes post at all);
  (b) zero-posting occupation-month cells over time, by exposure quartile
      (relative to the balanced occupation set);
  (c) the composition of coded ads across exposure quartiles over time --
      if coding practice were drifting against exposed occupations, the Q4
      share of coded ads would fall mechanically;
  (d) the 400-vs-369 reconciliation: unique SSYK4 codes at each pipeline
      stage (raw postings file, DAIOE-matched, regression panel with
      n_ads > 0) with the named lists of codes lost at each step.

Output:
  tables/coverage_active_occupations.csv
  tables/coverage_zero_cells.csv
  tables/coverage_quartile_shares.csv
  tables/occupation_reconciliation.csv   (one row per stage + lost-code lists)
Runtime: seconds.
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
    # regression panel: n_ads > 0 rows (v1 script 05 drops zeros)
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
