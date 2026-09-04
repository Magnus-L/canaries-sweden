#!/usr/bin/env python3
"""
41_vintage_event_studies.py -- T2/E4 (MONA run M2): event studies by
occupation-code vintage and by hiring margin.

======================================================================
  RUNS IN SCB's MONA SECURE ENVIRONMENT ONLY.
  Requires cache/panel_vintage.parquet (script 39 writes it).
======================================================================

The editor: "report separate event studies for workers classified using
2023, 2022, and 2021 occupation codes ... These analyses should show
whether the post-2023 decline among workers aged 22-25 is concentrated
among observations whose occupation assignments are older or otherwise
less reliable."

Design. Pre-2023 rows always use own-year codes (they have no cascade).
For 2023-2025 rows, three variants restrict the CODED population by the
vintage that supplied the code:

  V2023  keep only vintage == '2023'   (freshest assignments)
  V2022  keep only vintage == '2022'
  V2021  keep only vintage == '2021'   (stalest assignments)

Each variant rebuilds the balanced panel and runs the Poisson half-year
ES for ages 22-25 (headline group; add more ages via AGES). Reading:
if the 2024-25 decline is a staleness artefact, it should be ABSENT in
V2023 and grow with staleness; if it is real, V2023 carries it.

Second margin (E4's last clause): incumbents vs NEW person-employer
matches. Year-level person x employer presence (as script 40 stage E)
defines new match = pair absent in year-1; monthly cells then split by
that year-level flag. Poisson ES per margin for 22-25.

Output (output_41/): vintage_es.csv, margin_es.csv, 41_summary.txt.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mona_common as mc

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_41"
OUT.mkdir(exist_ok=True)
CACHE = mc.PANEL_CACHE

AGES = ["22-25"]          # extend if runtime allows
RUN_MARGIN_SPLIT = True   # the person-level SQL stage
STEP1_MIN_CUMULATIVE = 5


def build_variant(panel, keep_vintage):
    """
    Coded panel where 2023+ observations must come from `keep_vintage`.
    Pre-2023 rows (vintage 'own') are common to all variants.
    """
    coded = panel[panel["ssyk4"] != "____"]
    pre = coded[coded["vintage"] == "own"]
    late = coded[(coded["vintage"] == keep_vintage)
                 & (coded["year_month"] >= "2023-01")]
    sub = pd.concat([pre, late], ignore_index=True)
    return (sub.groupby(["employer_id", "year_month", "ssyk4", "age_group"],
                        observed=True)["n_emp"].sum().reset_index())


def es_for(agg, age, tag):
    agg = mc.merge_daioe_and_filter(agg, mc.load_daioe())
    agg = mc.aggregate_to_quartile(agg)
    all_months = sorted(agg["year_month"].unique())
    sub = agg[agg["age_group"] == age]
    cum = sub.groupby("employer_id")["n_emp"].sum()
    sub = sub[sub["employer_id"].isin(cum[cum >= STEP1_MIN_CUMULATIVE].index)]
    bal = mc.add_treatment(mc.balance_panel(sub, all_months))
    bal["halfyear"] = mc.assign_halfyear(bal["year_month"])
    print(f"  [{tag}] {len(bal):,} cells, "
          f"{bal['employer_id'].nunique():,} employers")
    res = mc.run_fepois_es(bal, OUT, tag=tag)
    if not res.empty:
        res["variant"] = tag
        res["age_group"] = age
    return res


def margin_split(conn, panel):
    """
    Person x employer x year presence -> new-match flag -> two monthly
    panels (incumbent pairs / new pairs) -> ES each. Uses the production
    cascade for codes (the margin question is about hiring, not coding).
    """
    print("  margin split: person x employer x year pull...")
    frames = []
    for year in range(2019, 2026):
        suffix, max_month = ("_def", 12) if year < 2025 else ("_prel", 6)
        monthly = "\nUNION ALL\n".join(
            f"""SELECT DISTINCT agi.P1207_LOPNR_PERSONNR AS person_id,
                       agi.P1207_LOPNR_PEORGNR AS employer_id
                FROM dbo.Arb_AGIIndivid{year}{m:02d}{suffix} agi"""
            for m in range(1, max_month + 1))
        f = pd.read_sql(f"SELECT DISTINCT person_id, employer_id "
                        f"FROM ({monthly}) u", conn)
        f["year"] = year
        frames.append(f)
        print(f"    {year}: {len(f):,} pairs")
    pe = pd.concat(frames, ignore_index=True)
    prev = pe.copy()
    prev["year"] += 1
    prev["existed_prev"] = 1
    pe = pe.merge(prev, on=["person_id", "employer_id", "year"], how="left")
    pe["new_match"] = (pe["existed_prev"].isna()).astype(int)
    pe.loc[pe["year"] == 2019, "new_match"] = np.nan  # censored
    return pe[["person_id", "employer_id", "year", "new_match"]]


def main():
    mc.Tee(OUT / "41_log.txt")
    print("=" * 70)
    print("41: VINTAGE EVENT STUDIES (E4)")
    print("=" * 70)
    if not CACHE.exists():
        print("FATAL: run 39 first")
        sys.exit(1)
    panel = pd.read_parquet(CACHE)

    results = []
    for keep in ("2023", "2022", "2021"):
        print(f"\n--- variant V{keep} ---")
        agg = build_variant(panel, keep)
        for age in AGES:
            r = es_for(agg, age, tag=f"V{keep}_{age}")
            if not r.empty:
                results.append(r)
    if results:
        pd.concat(results).to_csv(OUT / "vintage_es.csv", index=False)
        print("\nSaved vintage_es.csv")

    # NOTE on the margin split: the per-month person-level panel by margin
    # requires re-aggregating AGI at the person level, which the cached
    # cell panel cannot deliver. The pull in margin_split() gives the
    # year-level flag; building the monthly margin panels needs a second
    # person-level monthly pull (heavy). Run it only if the vintage ES
    # does not settle E4's last clause on its own; the incumbents/new-
    # matches coverage TABLE (script 40 stage E) may suffice for the
    # letter, with the margin ES held for round 1.5.
    if RUN_MARGIN_SPLIT:
        print("\n--- margin split (year-level flag) ---")
        pe = margin_split(mc.connect(), panel)
        pe_agg = (pe.groupby(["year", "new_match"], dropna=True)
                  .size().rename("n_pairs").reset_index())
        pe_agg.to_csv(OUT / "margin_pair_counts.csv", index=False)
        print("  margin_pair_counts.csv (monthly margin ES deferred; "
              "see note in script)")


if __name__ == "__main__":
    main()
