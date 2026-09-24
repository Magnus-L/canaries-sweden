#!/usr/bin/env python3
"""
41_vintage_event_studies.py: the submitted event study estimated
separately on workers coded from each register vintage.

QUESTION
The question is whether the post-2023 decline among workers
aged 22-25 in the submitted design is concentrated among observations
whose occupation code is old. This script splits the coded population
from 2023 by the register vintage that supplied the code and runs the
submitted half-year event study on each subsample. Conditioning on code
vintage conditions on how recently a worker was observed, which is itself
a function of tenure, entry and mobility, so the columns are not
comparable with one another and none estimates a treatment effect; the
instability of the columns is the result.

DESIGN
From the vintage-tagged panel, three variants: rows before 2023 keep the
year's own code in every variant; rows from 2023 are kept only if the
code came from the 2023 register (V2023), the 2022 register (V2022) or
the 2021 register (V2021). Each variant is merged with the DAIOE
quartiles, restricted to employers with a cumulative count of at least
five, balanced and zero-filled over employer by exposure quartile by
month for ages 22-25 and restricted to employers in both the top quartile
and a lower one; the Poisson event study interacts High with each
half-year, reference the first half of 2022, under employer-by-quartile
and employer-by-month effects, standard errors clustered by employer. A
second pull of person by employer by year presence flags new
person-employer matches (a pair absent the year before) and counts pairs
by year and margin; the monthly event study by margin is not built, since
the cell panel cannot deliver it.

INPUTS AND OUTPUTS
Reads cache/panel_vintage.parquet (script 39) and daioe_quartiles.dta, and
for the margin count Arb_AGIIndivid for 2019 to 2025 in MONA. Writes to
output_41/: vintage_es.csv and margin_pair_counts.csv.

IN THE PAPER
Online Appendix IV.2, Table A26 (built by 4_exhibits from these exports),
which reports the three columns and states why they are not comparable.
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

    # The per-month person-level panel by margin requires re-aggregating
    # the declarations at the person level, which the cached cell panel
    # cannot deliver. The pull in margin_split() gives the year-level flag
    # and the pair counts; the monthly margin event study is not built, and
    # the coverage by worker group of script 40 answers the question in
    # its place.
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
