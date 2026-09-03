#!/usr/bin/env python3
"""
40_coverage_diagnostics.py -- T1/E3 (MONA run M1): the employment-coverage
accounting the editor's letter itemises.

======================================================================
  RUNS IN SCB's MONA SECURE ENVIRONMENT ONLY.
  Requires output_39/panel_vintage.parquet (script 39 writes it).
======================================================================

From the vintage-tagged panel (employer x ssyk4 x age x month x vintage):

  A. Match rates by month x age group: share of AGI worker-months with a
     usable SSYK code (vintage != 'none'), the editor's headline series.
  B. Vintage composition by month x age group for 2023-2025: share of
     coded workers classified with 2023 / 2022 / 2021 codes.
  C. Match rates by month x age x DAIOE quartile OF THE ASSIGNED CODE --
     the differential the bounding exercise needs. (Unmatched workers have
     no quartile by construction; their counts appear in A's denominator.)
  D. Excluded counts: number and share of employed workers with no code,
     by month x age group -- "the number and share of employed workers
     excluded because no occupation code can be assigned".

The incumbent / recent-hire / new-entrant split (E3's remaining clause)
uses a person-level first-appearance pull, shared with script 41; it runs
here as stage E if RUN_ENTRANT_SPLIT = True (needs SQL, ~15 min):
  person x year presence -> first AGI year -> entrant (first year),
  recent hire (first seen at THIS employer this year, seen in AGI before),
  incumbent (at this employer in the previous year too); match rates per
  group per year.

Output (output_40/): coverage_by_month_age.csv, vintage_composition.csv,
coverage_by_quartile.csv, excluded_counts.csv, entrant_split_coverage.csv,
coverage_summary.txt. All aggregates; export floor applied.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mona_common as mc

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_40"
OUT.mkdir(exist_ok=True)
CACHE = HERE / "output_39" / "panel_vintage.parquet"

RUN_ENTRANT_SPLIT = True


def stage_a_to_d(panel):
    """Stages A-D run on the cached panel; no SQL."""
    panel = panel.copy()
    panel["coded"] = (panel["ssyk4"] != "____").astype(int)

    # A + D: match rate and excluded counts by month x age
    by_ma = (panel.groupby(["year_month", "age_group"], observed=True)
             .apply(lambda g: pd.Series({
                 "n_workers": g["n_emp"].sum(),
                 "n_coded": g.loc[g["coded"] == 1, "n_emp"].sum()}))
             .reset_index())
    by_ma["match_rate"] = by_ma["n_coded"] / by_ma["n_workers"]
    by_ma["n_excluded"] = by_ma["n_workers"] - by_ma["n_coded"]
    by_ma["excluded_share"] = 1 - by_ma["match_rate"]
    by_ma.to_csv(OUT / "coverage_by_month_age.csv", index=False)
    by_ma[["year_month", "age_group", "n_excluded", "excluded_share"]].to_csv(
        OUT / "excluded_counts.csv", index=False)
    print("  A/D: coverage_by_month_age.csv, excluded_counts.csv")

    # B: vintage composition among coded workers, 2023+
    late = panel[(panel["year_month"] >= "2023-01") & (panel["coded"] == 1)]
    comp = (late.groupby(["year_month", "age_group", "vintage"],
                         observed=True)["n_emp"].sum().reset_index())
    tot = comp.groupby(["year_month", "age_group"],
                       observed=True)["n_emp"].transform("sum")
    comp["share"] = comp["n_emp"] / tot
    comp.to_csv(OUT / "vintage_composition.csv", index=False)
    print("  B: vintage_composition.csv")

    # C: match rate by quartile of the assigned code. The denominator here
    # is coded workers only; the DIFFERENTIAL diagnostic is the quartile
    # composition of coded workers over time plus 32's last-known-quartile
    # attrition table (cited in the response letter alongside this one).
    daioe = mc.load_daioe()
    coded = panel[panel["coded"] == 1].merge(daioe, on="ssyk4", how="inner")
    by_q = (coded.groupby(["year_month", "age_group", "exposure_quartile"],
                          observed=True)["n_emp"].sum().reset_index())
    totq = by_q.groupby(["year_month", "age_group"],
                        observed=True)["n_emp"].transform("sum")
    by_q["share_of_coded"] = by_q["n_emp"] / totq
    by_q.to_csv(OUT / "coverage_by_quartile.csv", index=False)
    print("  C: coverage_by_quartile.csv")
    return by_ma


def stage_e_entrant_split(conn):
    """
    Person-level split: entrant / recent hire / incumbent, with match
    rates per group per year. Year granularity (the editor asks for the
    groups and their coverage; monthly person-level would be very costly).
    """
    print("  E: entrant split (SQL, person x employer x year)...")
    frames = []
    for year in range(2019, 2026):
        suffix, max_month = ("_def", 12) if year < 2025 else ("_prel", 6)
        individ_year = min(year, 2023)
        if individ_year >= 2023:
            code_expr = ("COALESCE(i23.Ssyk4_2012_J16, i22.Ssyk4_2012_J16, "
                         "i21.Ssyk4_2012_J16)")
            joins = """
                LEFT JOIN dbo.Individ_2023 i23
                    ON agi.P1207_LOPNR_PERSONNR = i23.P1207_LopNr_PersonNr
                LEFT JOIN dbo.Individ_2022 i22
                    ON agi.P1207_LOPNR_PERSONNR = i22.P1207_LopNr_PersonNr
                LEFT JOIN dbo.Individ_2021 i21
                    ON agi.P1207_LOPNR_PERSONNR = i21.P1207_LopNr_PersonNr"""
        else:
            code_expr = "ind.Ssyk4_2012_J16"
            joins = f"""
                LEFT JOIN dbo.Individ_{individ_year} ind
                    ON agi.P1207_LOPNR_PERSONNR = ind.P1207_LopNr_PersonNr"""
        monthly = "\nUNION ALL\n".join(
            f"""SELECT agi.P1207_LOPNR_PERSONNR AS person_id,
                       agi.P1207_LOPNR_PEORGNR AS employer_id,
                       CASE WHEN {code_expr} IS NULL THEN 0 ELSE 1 END AS coded
                FROM dbo.Arb_AGIIndivid{year}{m:02d}{suffix} agi {joins}"""
            for m in range(1, max_month + 1))
        q = f"""
        SELECT person_id, employer_id, MAX(coded) AS coded, COUNT(*) AS n_months
        FROM ({monthly}) u
        GROUP BY person_id, employer_id
        """
        t0 = time.time()
        f = pd.read_sql(q, conn)
        f["year"] = year
        frames.append(f)
        print(f"    {year}: {len(f):,} person-employer pairs "
              f"({time.time()-t0:.0f}s)")

    pe = pd.concat(frames, ignore_index=True)
    first_year = pe.groupby("person_id")["year"].min().rename("first_agi_year")
    pe = pe.merge(first_year, on="person_id")
    prev = pe[["person_id", "employer_id", "year"]].copy()
    prev["year"] += 1
    prev["at_employer_prev_year"] = 1
    pe = pe.merge(prev, on=["person_id", "employer_id", "year"], how="left")
    pe["group"] = np.where(pe["year"] == pe["first_agi_year"], "entrant",
                  np.where(pe["at_employer_prev_year"] == 1, "incumbent",
                           "recent_hire"))

    res = (pe.groupby(["year", "group"])
           .agg(n_pairs=("person_id", "size"), n_coded=("coded", "sum"))
           .reset_index())
    res["match_rate"] = res["n_coded"] / res["n_pairs"]
    res.loc[res["year"] == 2019, "group"] = res.loc[
        res["year"] == 2019, "group"].replace(
        {"entrant": "entrant (censored: first panel year)"})
    res.to_csv(OUT / "entrant_split_coverage.csv", index=False)
    print("  E: entrant_split_coverage.csv")
    return res


def main():
    mc.Tee(OUT / "40_log.txt")
    print("=" * 70)
    print("40: EMPLOYMENT COVERAGE DIAGNOSTICS (E3)")
    print("=" * 70)
    if not CACHE.exists():
        print("FATAL: run 39 first (panel cache missing)")
        sys.exit(1)
    panel = pd.read_parquet(CACHE)
    by_ma = stage_a_to_d(panel)

    ent = None
    if RUN_ENTRANT_SPLIT:
        ent = stage_e_entrant_split(mc.connect())

    # Summary
    lines = ["COVERAGE SUMMARY", "=" * 40]
    for yr in ("2023", "2024", "2025"):
        sub = by_ma[by_ma["year_month"].str.startswith(yr)]
        y22 = sub[sub["age_group"] == "22-25"]
        lines.append(f"{yr}: overall match "
                     f"{sub['n_coded'].sum()/sub['n_workers'].sum():.3f}; "
                     f"22-25 match {y22['n_coded'].sum()/y22['n_workers'].sum():.3f}")
    if ent is not None:
        for g in ("incumbent", "recent_hire", "entrant"):
            sub = ent[(ent["group"] == g) & (ent["year"] >= 2024)]
            if len(sub):
                lines.append(f"2024-25 {g}: match "
                             f"{sub['n_coded'].sum()/sub['n_pairs'].sum():.3f}")
    (OUT / "coverage_summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
