#!/usr/bin/env python3
"""
49b_staleness_by_quartile.py -- the live channel: stale codes, not missing ones.

======================================================================
  RUNS IN SCB's MONA SECURE ENVIRONMENT ONLY. Standalone: submit this
  file itself. Writes output_49b/.
======================================================================

WHY. Script 49 established that essentially nobody is uncoded: coverage in
the estimation population is 99.1 to 99.96 per cent, and highest in 2024 and
2025. Part 2 of that script then asked the wrong question. It conditioned on
holding a 2022 code and asked whether the worker had a code in the 2023,
2022 or 2021 cascade, which such a worker has by construction, so every cell
came back 1.0000. That table says nothing and must not be quoted.

The channel that survives is STALENESS. Every worker has a code, but 99.7 to
99.9 per cent of them carry the 2023 one, so by 2025 the panel classifies
people by what they did up to two years earlier. Two quantities decide how
much that can matter, and both are measurable here:

  1. FRESHNESS BY FROZEN QUARTILE. Among workers employed in year y who held
     a 2022 code, what share are carried by the 2023 register rather than by
     an older one, split by the 2022 quartile. If exposed workers go stale
     faster, the editor's mechanism operates through misclassification even
     though nobody is missing.

  2. THE BOUNDARY-CROSSING RATE. Among workers with both a 2022 and a 2023
     code, how often does the 2023 code move them across the top-quartile
     boundary, by direction. This is the empirical counterpart of the
     misclassification axis in script 45's frontier: it replaces an assumed
     rate with a measured one for the one transition we can observe.

EXPORT: shares and counts by year, age group and quartile. No raw rows.
"""

import time
from pathlib import Path

import pandas as pd

import mona_common as mc

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_49b"
OUT.mkdir(exist_ok=True)

YEARS = range(2023, 2026)
AGE_CASE = """CASE
        WHEN age BETWEEN 22 AND 25 THEN '22-25'
        WHEN age BETWEEN 26 AND 30 THEN '26-30'
        WHEN age BETWEEN 31 AND 34 THEN '31-34'
        WHEN age BETWEEN 35 AND 40 THEN '35-40'
        WHEN age BETWEEN 41 AND 49 THEN '41-49'
        WHEN age BETWEEN 50 AND 69 THEN '50+'
        ELSE NULL END"""


def pull(year: int, conn) -> pd.DataFrame:
    """Employed persons in `year` with their 2022 and 2023 codes and the
    vintage the cascade would actually assign."""
    suffix, max_month = mc._year_suffix(year)
    months = "\nUNION ALL\n".join(
        f"""SELECT DISTINCT P1207_LOPNR_PERSONNR AS person_id
            FROM dbo.Arb_AGIIndivid{year}{m:02d}{suffix}"""
        for m in range(1, max_month + 1))
    q = f"""
    WITH emp AS (SELECT DISTINCT person_id FROM ({months}) u),
    j AS (
        SELECT e.person_id,
               RIGHT('0000' + CAST(i23.Ssyk4_2012_J16 AS VARCHAR(4)), 4) AS ssyk_23,
               RIGHT('0000' + CAST(i22.Ssyk4_2012_J16 AS VARCHAR(4)), 4) AS ssyk_22,
               CASE WHEN i23.Ssyk4_2012_J16 IS NOT NULL THEN '2023'
                    WHEN i22.Ssyk4_2012_J16 IS NOT NULL THEN '2022'
                    WHEN i21.Ssyk4_2012_J16 IS NOT NULL THEN '2021'
                    ELSE 'none' END AS vintage,
               COALESCE(i23.FodelseAr, i22.FodelseAr, i21.FodelseAr) AS birth_year
        FROM emp e
        LEFT JOIN dbo.Individ_2023 i23 ON e.person_id = i23.P1207_LopNr_PersonNr
        LEFT JOIN dbo.Individ_2022 i22 ON e.person_id = i22.P1207_LopNr_PersonNr
        LEFT JOIN dbo.Individ_2021 i21 ON e.person_id = i21.P1207_LopNr_PersonNr
    )
    SELECT {AGE_CASE} AS age_group, vintage, ssyk_22, ssyk_23, COUNT(*) AS n
    FROM (SELECT *, {year} - birth_year AS age FROM j WHERE birth_year IS NOT NULL) a
    WHERE age BETWEEN 22 AND 69
    GROUP BY {AGE_CASE}, vintage, ssyk_22, ssyk_23
    """
    d = pd.read_sql(q, conn)
    d["year"] = year
    return d


def main():
    mc.Tee(OUT / "49b_log.txt")
    print("=" * 70)
    print("49b: STALENESS AND BOUNDARY CROSSING BY FROZEN QUARTILE")
    print("=" * 70)
    print(mc.mem_line("  "))

    conn = mc.connect()
    frames = []
    for y in YEARS:
        t0 = time.time()
        f = pull(y, conn)
        print(f"  {y}: {f['n'].sum():,} employed persons ({time.time()-t0:.0f}s)")
        frames.append(f)
    d = pd.concat(frames, ignore_index=True)

    daioe = mc.load_daioe()
    q22 = daioe.rename(columns={"ssyk4": "ssyk_22",
                                "exposure_quartile": "q22"})[["ssyk_22", "q22"]]
    q23 = daioe.rename(columns={"ssyk4": "ssyk_23",
                                "exposure_quartile": "q23"})[["ssyk_23", "q23"]]
    d = d.merge(q22, on="ssyk_22", how="left").merge(q23, on="ssyk_23", how="left")

    # 1. freshness among workers with a 2022 code
    have22 = d[d["q22"].notna()].copy()
    have22["fresh"] = (have22["vintage"] == "2023").astype(int)
    rows = []
    for (y, a, q_), g in have22.groupby(["year", "age_group", "q22"]):
        n = g["n"].sum()
        if n < 5:
            continue
        rows.append({"year": y, "age_group": a, "quartile_2022": int(q_),
                     "n_workers": n,
                     "share_carried_by_2023_code": (g["fresh"] * g["n"]).sum() / n})
    fresh = pd.DataFrame(rows)
    fresh.to_csv(OUT / "freshness_by_frozen_quartile.csv", index=False)

    # 2. boundary crossing among workers with both codes
    both = d[d["q22"].notna() & d["q23"].notna()].copy()
    both["in4_22"] = (both["q22"] == 4).astype(int)
    both["in4_23"] = (both["q23"] == 4).astype(int)
    rows = []
    for (y, a), g in both.groupby(["year", "age_group"]):
        n4 = g.loc[g["in4_22"] == 1, "n"].sum()
        n0 = g.loc[g["in4_22"] == 0, "n"].sum()
        if min(n4, n0) < 5:
            continue
        out4 = g.loc[(g["in4_22"] == 1) & (g["in4_23"] == 0), "n"].sum()
        into4 = g.loc[(g["in4_22"] == 0) & (g["in4_23"] == 1), "n"].sum()
        rows.append({"year": y, "age_group": a,
                     "n_top_2022": n4, "n_rest_2022": n0,
                     "left_top_by_2023": out4 / n4,
                     "entered_top_by_2023": into4 / n0,
                     "net_flow_out_of_top": (out4 - into4) / n4})
    cross = pd.DataFrame(rows)
    cross.to_csv(OUT / "boundary_crossing.csv", index=False)

    lines = ["FRESHNESS: share carried by the 2023 code, ages 22-25", "=" * 56]
    f2 = fresh[fresh["age_group"] == "22-25"].pivot(
        index="year", columns="quartile_2022",
        values="share_carried_by_2023_code")
    lines.append(f2.to_string(float_format=lambda x: f"{x:.4f}"))
    if {1, 4} <= set(f2.columns):
        lines += ["", "Q4 minus Q1:"] + [
            f"  {y}: {f2.loc[y, 4] - f2.loc[y, 1]:+.4f}" for y in f2.index]
    lines += ["", "BOUNDARY CROSSING, ages 22-25", "=" * 56,
              cross[cross["age_group"] == "22-25"].to_string(
                  index=False, float_format=lambda x: f"{x:.4f}")]
    (OUT / "49b_summary.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print("\n49b done. " + mc.mem_line())


if __name__ == "__main__":
    main()
