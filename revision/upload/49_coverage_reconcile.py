#!/usr/bin/env python3
"""
49_coverage_reconcile.py -- reconcile the match rate, and test coverage
attrition against a PRE-DETERMINED exposure quartile.

======================================================================
  RUNS IN SCB's MONA SECURE ENVIRONMENT ONLY. Standalone: submit this
  file itself. It opens its own SQL connection and writes output_49/.
======================================================================

WHY (18 Sep 2026). Script 40 reports a match rate of 1.000 for 2023,
2024 and 2025. The submitted online appendix tells the editor the
non-match rate is 10 per cent through 2023, 15 per cent in 2024 and 20
per cent in 2025, and the editor quotes those figures back at us. Both
cannot be right, and we cannot answer his first employment-side demand
until we know which is.

The likely explanation is that the two count different things. This
script computes all three definitions on the same rows, per year and
age group, so the answer is a table rather than an argument:

  M1 OWN-YEAR   the worker's code from that year's own Individ table.
                Undefined for 2024 and 2025, where no register exists;
                that is what makes v1's non-match rate rise.
  M2 CASCADE    the 2023, then 2022, then 2021 register, which is what
                every v2 script uses.
  M3 EVER-SEEN  any code the worker has ever had in 2019-2023, which is
                the carry-forward reading of v1's rule.

SECOND, AND THE REASON THIS SCRIPT MATTERS MORE THAN THE FIRST PART.
Script 40's coverage-by-quartile table conditions on the CURRENTLY
assigned code, so a real decline in exposed employment and a coding
artefact produce the same table. The editor's question is whether
coding attrition differs across exposure groups, and that has to be
asked of a quartile fixed BEFORE the coverage problem starts. We take
every worker with a known 2022 code, freeze their quartile there, and
report the probability of being coded in each later year by that frozen
quartile. Neutrality across quartiles closes the missingness channel and
leaves misclassification, which script 45 measures.

EXPORT: rates and counts by year, age group and quartile. No raw rows.
"""

import sys
import time
from pathlib import Path

import pandas as pd

import mona_common as mc

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_49"
OUT.mkdir(exist_ok=True)

YEARS = range(2019, 2026)
AGE_CASE = """CASE
        WHEN age BETWEEN 22 AND 25 THEN '22-25'
        WHEN age BETWEEN 26 AND 30 THEN '26-30'
        WHEN age BETWEEN 31 AND 34 THEN '31-34'
        WHEN age BETWEEN 35 AND 40 THEN '35-40'
        WHEN age BETWEEN 41 AND 49 THEN '41-49'
        WHEN age BETWEEN 50 AND 69 THEN '50+'
        ELSE NULL END"""


def persons_with_codes(year: int, conn) -> pd.DataFrame:
    """
    One row per person employed in `year`, with the three code
    definitions. Person-level, not cell-level: the match rate is a
    property of workers, and cell counts cannot express it.
    """
    suffix, max_month = mc._year_suffix(year)
    own = min(year, 2023)
    months = "\nUNION ALL\n".join(
        f"""SELECT DISTINCT P1207_LOPNR_PERSONNR AS person_id
            FROM dbo.Arb_AGIIndivid{year}{m:02d}{suffix}"""
        for m in range(1, max_month + 1))
    q = f"""
    WITH emp AS (SELECT DISTINCT person_id FROM ({months}) u),
    coded AS (
        SELECT e.person_id,
               own.Ssyk4_2012_J16  AS ssyk_own,
               i23.Ssyk4_2012_J16  AS ssyk_23,
               i22.Ssyk4_2012_J16  AS ssyk_22,
               i21.Ssyk4_2012_J16  AS ssyk_21,
               i20.Ssyk4_2012_J16  AS ssyk_20,
               i19.Ssyk4_2012_J16  AS ssyk_19,
               COALESCE(i23.FodelseAr, i22.FodelseAr, i21.FodelseAr,
                        i20.FodelseAr, i19.FodelseAr) AS birth_year
        FROM emp e
        LEFT JOIN dbo.Individ_{own} own ON e.person_id = own.P1207_LopNr_PersonNr
        LEFT JOIN dbo.Individ_2023 i23  ON e.person_id = i23.P1207_LopNr_PersonNr
        LEFT JOIN dbo.Individ_2022 i22  ON e.person_id = i22.P1207_LopNr_PersonNr
        LEFT JOIN dbo.Individ_2021 i21  ON e.person_id = i21.P1207_LopNr_PersonNr
        LEFT JOIN dbo.Individ_2020 i20  ON e.person_id = i20.P1207_LopNr_PersonNr
        LEFT JOIN dbo.Individ_2019 i19  ON e.person_id = i19.P1207_LopNr_PersonNr
    )
    SELECT {AGE_CASE} AS age_group,
           CASE WHEN ssyk_own IS NOT NULL THEN 1 ELSE 0 END AS m1_own,
           CASE WHEN COALESCE(ssyk_23, ssyk_22, ssyk_21) IS NOT NULL
                THEN 1 ELSE 0 END AS m2_cascade,
           CASE WHEN COALESCE(ssyk_23, ssyk_22, ssyk_21, ssyk_20, ssyk_19)
                     IS NOT NULL THEN 1 ELSE 0 END AS m3_ever,
           RIGHT('0000' + CAST(ssyk_22 AS VARCHAR(4)), 4) AS ssyk_2022,
           COUNT(*) AS n
    FROM (SELECT *, {year} - birth_year AS age FROM coded
          WHERE birth_year IS NOT NULL) a
    WHERE age BETWEEN 22 AND 69
    GROUP BY {AGE_CASE},
             CASE WHEN ssyk_own IS NOT NULL THEN 1 ELSE 0 END,
             CASE WHEN COALESCE(ssyk_23, ssyk_22, ssyk_21) IS NOT NULL
                  THEN 1 ELSE 0 END,
             CASE WHEN COALESCE(ssyk_23, ssyk_22, ssyk_21, ssyk_20, ssyk_19)
                       IS NOT NULL THEN 1 ELSE 0 END,
             RIGHT('0000' + CAST(ssyk_22 AS VARCHAR(4)), 4)
    """
    d = pd.read_sql(q, conn)
    d["year"] = year
    return d


def main():
    mc.Tee(OUT / "49_log.txt")
    print("=" * 70)
    print("49: RECONCILE THE MATCH RATE, AND TEST IT ON A FROZEN QUARTILE")
    print("=" * 70)
    print(mc.mem_line("  "))

    conn = mc.connect()
    frames = []
    for y in YEARS:
        t0 = time.time()
        f = persons_with_codes(y, conn)
        print(f"  {y}: {f['n'].sum():,} employed persons "
              f"({time.time() - t0:.0f}s)")
        frames.append(f)
    d = pd.concat(frames, ignore_index=True)

    # --- part 1: the three definitions side by side
    rows = []
    for (y, a), g in d.groupby(["year", "age_group"]):
        n = g["n"].sum()
        rows.append({"year": y, "age_group": a, "n_employed": n,
                     "m1_own_year": (g["m1_own"] * g["n"]).sum() / n,
                     "m2_cascade": (g["m2_cascade"] * g["n"]).sum() / n,
                     "m3_ever_seen": (g["m3_ever"] * g["n"]).sum() / n})
    rec = pd.DataFrame(rows).sort_values(["age_group", "year"])
    rec.to_csv(OUT / "match_rate_definitions.csv", index=False)
    print("\nMATCH RATE BY DEFINITION (share of employed persons)")
    print(rec.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- part 2: coding attrition against the 2022 quartile, fixed before
    #     the coverage problem begins
    daioe = mc.load_daioe()
    q22 = d[d["ssyk_2022"].notna()].merge(
        daioe, left_on="ssyk_2022", right_on="ssyk4", how="inner")
    rows = []
    for (y, a, q_), g in q22.groupby(["year", "age_group",
                                      "exposure_quartile"]):
        n = g["n"].sum()
        if n < 5:
            continue
        rows.append({"year": y, "age_group": a, "exposure_quartile_2022": q_,
                     "n_workers": n,
                     "coded_cascade": (g["m2_cascade"] * g["n"]).sum() / n})
    att = pd.DataFrame(rows).sort_values(
        ["age_group", "exposure_quartile_2022", "year"])
    att.to_csv(OUT / "attrition_by_frozen_quartile.csv", index=False)

    lines = ["CODING ATTRITION BY 2022 EXPOSURE QUARTILE (22-25)", "=" * 52,
             "Share of workers with a known 2022 code who are coded in year y.",
             "A gap that opens between Q4 and Q1 after 2023 is the editor's",
             "mechanism; a flat profile closes it.", ""]
    sub = att[att["age_group"] == "22-25"]
    piv = sub.pivot(index="year", columns="exposure_quartile_2022",
                    values="coded_cascade")
    lines.append(piv.to_string(float_format=lambda x: f"{x:.4f}"))
    if {1, 4} <= set(piv.columns):
        lines += ["", "Q4 minus Q1, by year:"]
        for y in piv.index:
            lines.append(f"  {y}: {piv.loc[y, 4] - piv.loc[y, 1]:+.4f}")
    (OUT / "49_summary.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print("\n49 done. " + mc.mem_line())


if __name__ == "__main__":
    main()
