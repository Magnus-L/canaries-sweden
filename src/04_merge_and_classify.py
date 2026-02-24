#!/usr/bin/env python3
"""
04_merge_and_classify.py — Merge Platsbanken postings with DAIOE genAI exposure.

Matches SSYK 4-digit occupation codes between Platsbanken postings and
the DAIOE index, assigns exposure quartiles, and produces the analysis-ready
dataset.

Key design choices:
  - Use DAIOE year 2023 cross-section (reference year, before any plausible
    AI effects on occupational composition).
  - Quartiles are unweighted: each SSYK4 occupation counts once, regardless
    of posting volume. This prevents large low-exposure occupations from
    dominating the distribution.
  - Match on exact 4-digit SSYK code (string match after zero-padding).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import RAW, PROCESSED, DAIOE_REF_YEAR, BASE_MONTH

import pandas as pd
import numpy as np


def load_daioe() -> pd.DataFrame:
    """
    Load DAIOE genAI exposure data for the reference year.

    Returns one row per SSYK4 occupation with the genAI percentile ranking.
    The percentile ranking ranges 0–100, where higher = more exposed to genAI.
    """
    print("Loading DAIOE...")

    daioe_file = RAW / "daioe_ssyk2012.csv"
    df = pd.read_csv(daioe_file, sep="\t")

    # Extract 4-digit SSYK code from the combined code+title column
    # Format: "0110 Officerare" → "0110"
    df["ssyk4"] = df["ssyk2012_4"].str[:4].str.strip()

    # Filter to reference year
    df_ref = df[df["year"] == DAIOE_REF_YEAR].copy()
    if len(df_ref) == 0:
        # Fallback: try nearest available year
        available = sorted(df["year"].unique())
        fallback = max(y for y in available if y <= DAIOE_REF_YEAR + 1)
        print(f"  Year {DAIOE_REF_YEAR} not found; using {fallback}")
        df_ref = df[df["year"] == fallback].copy()

    # Keep relevant columns, deduplicate
    daioe = (
        df_ref[["ssyk4", "pctl_rank_genai", "pctl_rank_allapps"]]
        .dropna(subset=["pctl_rank_genai"])
        .drop_duplicates(subset=["ssyk4"])
        .copy()
    )

    print(f"  {len(daioe)} occupations with genAI exposure data")
    print(f"  Exposure range: {daioe['pctl_rank_genai'].min():.1f} – "
          f"{daioe['pctl_rank_genai'].max():.1f}")

    return daioe


def compute_quartiles(daioe: pd.DataFrame) -> pd.DataFrame:
    """
    Assign each occupation to a genAI exposure quartile.

    Quartile boundaries are computed from the unweighted distribution
    of occupations (each SSYK4 counts once). This is the standard
    approach in the literature (Brynjolfsson et al., Kauhanen & Rouvinen).
    """
    q25 = daioe["pctl_rank_genai"].quantile(0.25)
    q50 = daioe["pctl_rank_genai"].quantile(0.50)
    q75 = daioe["pctl_rank_genai"].quantile(0.75)

    print(f"  Quartile boundaries: Q25={q25:.1f}, Q50={q50:.1f}, Q75={q75:.1f}")

    def assign_quartile(pctl):
        if pctl <= q25:
            return "Q1 (lowest)"
        elif pctl <= q50:
            return "Q2"
        elif pctl <= q75:
            return "Q3"
        else:
            return "Q4 (highest)"

    daioe = daioe.copy()
    daioe["exposure_quartile"] = daioe["pctl_rank_genai"].apply(assign_quartile)
    daioe["high_exposure"] = (daioe["pctl_rank_genai"] > q75).astype(int)

    # Report quartile sizes
    print("  Quartile distribution:")
    for q, count in daioe["exposure_quartile"].value_counts().sort_index().items():
        print(f"    {q}: {count} occupations")

    return daioe


def merge_postings_with_daioe(
    postings: pd.DataFrame, daioe: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge SSYK4 × month postings with DAIOE exposure classification.

    Reports match rate — we expect >95% of postings to match a DAIOE code.
    Unmatched occupations are typically military (01xx) or very rare codes.
    """
    print("Merging postings with DAIOE...")

    # Ensure ssyk4 is string in both
    postings["ssyk4"] = postings["ssyk4"].astype(str).str.zfill(4)
    daioe["ssyk4"] = daioe["ssyk4"].astype(str).str.zfill(4)

    # Merge
    merged = postings.merge(daioe, on="ssyk4", how="left")

    # Match statistics
    n_total = len(merged)
    n_matched = merged["pctl_rank_genai"].notna().sum()
    n_unmatched = n_total - n_matched

    # Occupation-level match rate
    all_occ = postings["ssyk4"].nunique()
    matched_occ = merged.loc[merged["pctl_rank_genai"].notna(), "ssyk4"].nunique()

    print(f"  Row-level: {n_matched:,}/{n_total:,} matched ({100*n_matched/n_total:.1f}%)")
    print(f"  Occupation-level: {matched_occ}/{all_occ} matched ({100*matched_occ/all_occ:.1f}%)")

    # Report unmatched codes
    unmatched_codes = merged.loc[merged["pctl_rank_genai"].isna(), "ssyk4"].unique()
    if len(unmatched_codes) > 0:
        unmatched_ads = merged.loc[merged["pctl_rank_genai"].isna(), "n_ads"].sum()
        print(f"  Unmatched SSYK codes ({len(unmatched_codes)}): "
              f"{sorted(unmatched_codes)[:10]}... ({unmatched_ads:,} ads)")

    # Drop unmatched
    merged = merged.dropna(subset=["pctl_rank_genai"])

    return merged


def build_indexed_series(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Create indexed monthly posting series by exposure quartile.

    Each quartile's series is indexed to 100 at the base month.
    This is the main data for Figure 1 (quartile trends) and
    Figure 2 (Q4 minus Q1 gap).
    """
    print("Building indexed monthly series by quartile...")

    # Aggregate to quartile × month
    quarterly = (
        merged.groupby(["exposure_quartile", "year_month"])
        .agg(n_ads=("n_ads", "sum"), n_vacancies=("n_vacancies", "sum"))
        .reset_index()
    )

    quarterly["date"] = pd.to_datetime(quarterly["year_month"] + "-01")

    # Index each quartile to 100 at base month
    base_ym = pd.Timestamp(BASE_MONTH).strftime("%Y-%m")

    indexed = []
    for q in sorted(quarterly["exposure_quartile"].unique()):
        qdf = quarterly[quarterly["exposure_quartile"] == q].copy()
        base_row = qdf[qdf["year_month"] == base_ym]

        if len(base_row) == 0:
            print(f"  WARNING: No data for {q} in base month {base_ym}")
            continue

        base_val = base_row["n_ads"].values[0]
        qdf["ads_idx"] = (qdf["n_ads"] / base_val) * 100
        qdf["vac_idx"] = (qdf["n_vacancies"] / base_row["n_vacancies"].values[0]) * 100
        indexed.append(qdf)

    result = pd.concat(indexed, ignore_index=True)
    result = result.sort_values(["exposure_quartile", "date"])

    return result


def build_total_index(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Create total indexed posting series (all occupations combined).

    This is for the scary chart overlay with OMXS30.
    """
    print("Building total posting index...")

    total = (
        merged.groupby("year_month")
        .agg(n_ads=("n_ads", "sum"), n_vacancies=("n_vacancies", "sum"))
        .reset_index()
    )
    total["date"] = pd.to_datetime(total["year_month"] + "-01")

    base_ym = pd.Timestamp(BASE_MONTH).strftime("%Y-%m")
    base_row = total[total["year_month"] == base_ym]
    if len(base_row) > 0:
        total["ads_idx"] = (total["n_ads"] / base_row["n_ads"].values[0]) * 100
    else:
        print("  WARNING: No base month data for total index")
        total["ads_idx"] = np.nan

    return total.sort_values("date")


def main():
    print("=" * 70)
    print("STEP 4: Merge postings with DAIOE and classify")
    print("=" * 70)

    # Load data
    postings = pd.read_csv(PROCESSED / "postings_ssyk4_monthly.csv")
    print(f"Postings: {len(postings):,} rows, {postings['ssyk4'].nunique()} occupations")

    daioe = load_daioe()
    daioe = compute_quartiles(daioe)

    # Save DAIOE with quartiles for reference
    daioe.to_csv(PROCESSED / "daioe_quartiles.csv", index=False)

    # Merge
    merged = merge_postings_with_daioe(postings, daioe)

    # Save merged data (the main analysis-ready dataset)
    out = PROCESSED / "postings_daioe_merged.csv"
    merged.to_csv(out, index=False)
    print(f"\nSaved: {out.name} ({len(merged):,} rows)")

    # Build indexed series by quartile
    indexed = build_indexed_series(merged)
    out_idx = PROCESSED / "postings_quartile_indexed.csv"
    indexed.to_csv(out_idx, index=False)
    print(f"Saved: {out_idx.name}")

    # Build total index
    total_idx = build_total_index(merged)
    out_total = PROCESSED / "postings_total_indexed.csv"
    total_idx.to_csv(out_total, index=False)
    print(f"Saved: {out_total.name}")

    # Summary
    print("\nQuartile × month summary (latest 3 months):")
    latest = indexed.nlargest(12, "date")
    for _, row in latest.iterrows():
        print(f"  {row['year_month']} | {row['exposure_quartile']}: "
              f"idx={row['ads_idx']:.1f}, {row['n_ads']:,} ads")

    print("\nDone. Run 05_analysis.py next.")


if __name__ == "__main__":
    main()
