#!/usr/bin/env python3
"""
40_coverage_diagnostics.py: what the occupation register covers, and how
old its codes are.

QUESTION
The editor asked what share of employed workers can be assigned an
occupation code, whether that share moves over the window, which register
vintage supplies the code, and how coverage differs between incumbents,
recent hires and entrants. This script answers from the vintage-tagged
panel, in which every worker-month records which Individ register
supplied its code.

WHAT IT BUILDS
From cache/panel_vintage.parquet (employer by occupation by age band by
month by vintage, written by script 39 through mona_common):
  A and D  the match rate (share of worker-months with a code from any
           vintage) and the excluded number and share, by month and age
           band;
  B        among coded worker-months from 2023, the share coded from the
           2023, 2022 and 2021 registers, by month and age band;
  C        among coded worker-months, the composition across DAIOE
           exposure quartiles of the assigned code, by month and age band;
  E        from a person by employer by year pull of the employer
           declarations for 2019 to 2025, each person-employer pair
           classified as entrant (the person's first year in the
           declarations), incumbent (at the same employer the year before)
           or recent hire (otherwise), with the share of pairs carrying a
           code per group and year. The 2019 entrant row is censored,
           since 2019 is the first panel year.

INPUTS AND OUTPUTS
Reads the panel cache and, for stage E, Arb_AGIIndivid joined to the
Individ registers in MONA. Writes to output_40/:
coverage_by_month_age.csv, excluded_counts.csv, vintage_composition.csv,
coverage_by_quartile.csv, entrant_split_coverage.csv and
coverage_summary.txt. Aggregates only, with the export floor applied.

IN THE PAPER
Online Appendix IV.1 and Table IV.1 (script l21): Panel A, the share of
employment excluded for want of a code, zero to two decimal places from
2020; Panel B, the age of the code in 2023, 2024 and 2025; Panel C, the
match rate by worker group pooled over 2020 to 2023 (incumbents 99.1 per
cent, recent hires 99.4, entrants 92.9). Script 49 reconciles the match
rate here with the non-match series of the submitted appendix.
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
CACHE = mc.PANEL_CACHE

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
