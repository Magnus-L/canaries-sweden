#!/usr/bin/env python3
"""
04_merge_and_classify.py: DAIOE exposure quartiles, matched to the postings.

WHAT IT BUILDS
The exposure measure is the Dynamic AI Occupational Exposure index (DAIOE) of
Engberg et al., in its 2023 cross-section and generative-AI variant
(pctl_rank_genai, a percentile from 0 to 100). Each four-digit SSYK 2012
occupation counts once when the quartile cut points are set, so large
occupations do not dominate the distribution; Q4 is the most exposed quarter
and high_exposure marks it. The occupation-by-month posting counts are then
matched to the index on the four-digit code; codes the index does not price
(military occupations, most managerial codes and two others) drop out. The
matched panel and posting indices by quartile and in total (100 in February
2020) are written for the window of the submitted version (October 2019 to
February 2026); the window the estimates use is cut from them in 2_postings.

INPUTS   data/raw/daioe_ssyk2012.csv; config.POSTINGS_SSYK4
OUTPUTS  data/processed/daioe_quartiles.csv (423 occupations),
         postings_daioe_merged.csv, postings_quartile_indexed.csv,
         postings_total_indexed.csv
SERVES   the exposure quartile of every posting estimate (Section 2)
RUNTIME  seconds
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROCESSED = config.PROCESSED


def load_daioe() -> pd.DataFrame:
    """One row per SSYK4 occupation with its generative-AI percentile, 2023."""
    df = pd.read_csv(config.DAIOE_RAW, sep="\t")
    df["ssyk4"] = df["ssyk2012_4"].str[:4].str.strip()     # "0110 Officerare" -> "0110"
    df_ref = df[df["year"] == config.DAIOE_REF_YEAR].copy()
    daioe = (df_ref[["ssyk4", "pctl_rank_genai", "pctl_rank_allapps"]]
             .dropna(subset=["pctl_rank_genai"])
             .drop_duplicates(subset=["ssyk4"]).copy())
    print(f"  DAIOE {config.DAIOE_REF_YEAR}: {len(daioe)} occupations")
    return daioe


def compute_quartiles(daioe: pd.DataFrame) -> pd.DataFrame:
    """Quartiles of the unweighted distribution over occupations."""
    q25 = daioe["pctl_rank_genai"].quantile(0.25)
    q50 = daioe["pctl_rank_genai"].quantile(0.50)
    q75 = daioe["pctl_rank_genai"].quantile(0.75)
    print(f"  cut points: {q25:.1f}, {q50:.1f}, {q75:.1f}")

    def assign_quartile(pctl):
        if pctl <= q25:
            return "Q1 (lowest)"
        elif pctl <= q50:
            return "Q2"
        elif pctl <= q75:
            return "Q3"
        return "Q4 (highest)"

    daioe = daioe.copy()
    daioe["exposure_quartile"] = daioe["pctl_rank_genai"].apply(assign_quartile)
    daioe["high_exposure"] = (daioe["pctl_rank_genai"] > q75).astype(int)
    return daioe


def merge_postings_with_daioe(postings: pd.DataFrame, daioe: pd.DataFrame) -> pd.DataFrame:
    postings["ssyk4"] = postings["ssyk4"].astype(str).str.zfill(4)
    daioe["ssyk4"] = daioe["ssyk4"].astype(str).str.zfill(4)
    merged = postings.merge(daioe, on="ssyk4", how="left")
    matched_occ = merged.loc[merged["pctl_rank_genai"].notna(), "ssyk4"].nunique()
    print(f"  occupations matched: {matched_occ} of {postings['ssyk4'].nunique()}")
    return merged.dropna(subset=["pctl_rank_genai"])


def build_indexed_series(merged: pd.DataFrame) -> pd.DataFrame:
    """Posting counts by quartile and month, each indexed to February 2020."""
    q = (merged.groupby(["exposure_quartile", "year_month"])
         .agg(n_ads=("n_ads", "sum"), n_vacancies=("n_vacancies", "sum"))
         .reset_index())
    q["date"] = pd.to_datetime(q["year_month"] + "-01")
    base_ym = pd.Timestamp(config.BASE_MONTH).strftime("%Y-%m")
    out = []
    for name in sorted(q["exposure_quartile"].unique()):
        qdf = q[q["exposure_quartile"] == name].copy()
        base_row = qdf[qdf["year_month"] == base_ym]
        qdf["ads_idx"] = (qdf["n_ads"] / base_row["n_ads"].values[0]) * 100
        qdf["vac_idx"] = (qdf["n_vacancies"] / base_row["n_vacancies"].values[0]) * 100
        out.append(qdf)
    return pd.concat(out, ignore_index=True).sort_values(["exposure_quartile", "date"])


def build_total_index(merged: pd.DataFrame) -> pd.DataFrame:
    total = (merged.groupby("year_month")
             .agg(n_ads=("n_ads", "sum"), n_vacancies=("n_vacancies", "sum"))
             .reset_index())
    total["date"] = pd.to_datetime(total["year_month"] + "-01")
    base_ym = pd.Timestamp(config.BASE_MONTH).strftime("%Y-%m")
    base_row = total[total["year_month"] == base_ym]
    total["ads_idx"] = (total["n_ads"] / base_row["n_ads"].values[0]) * 100 \
        if len(base_row) else np.nan
    return total.sort_values("date")


def main():
    print("DAIOE quartiles, matched to the postings")
    postings = pd.read_csv(config.POSTINGS_SSYK4)
    daioe = compute_quartiles(load_daioe())
    daioe.to_csv(PROCESSED / "daioe_quartiles.csv", index=False)
    merged = merge_postings_with_daioe(postings, daioe)
    merged.to_csv(PROCESSED / "postings_daioe_merged.csv", index=False)
    build_indexed_series(merged).to_csv(PROCESSED / "postings_quartile_indexed.csv", index=False)
    build_total_index(merged).to_csv(PROCESSED / "postings_total_indexed.csv", index=False)
    print(f"  wrote daioe_quartiles.csv and postings_daioe_merged.csv "
          f"({len(merged):,} occupation-months)")


if __name__ == "__main__":
    main()
