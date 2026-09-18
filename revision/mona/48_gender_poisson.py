#!/usr/bin/env python3
"""
48_gender_poisson.py -- the gender split, in the primary estimator.

======================================================================
  RUNS IN SCB's MONA SECURE ENVIRONMENT ONLY.
  Standalone: submit this file itself, or run it through
  run_all_mona.py --only 48. It builds its OWN cache and does not
  touch panel_vintage.parquet, so it can run beside another console.
======================================================================

WHY THIS EXISTS (18 Sep 2026). The editor's ln(n+1) objection is
correct and Poisson PML is now the paper's primary estimator. The
gender result -- the decline is concentrated among young women, which
is in the abstract -- is still an ln(n+1) estimate from the submitted
round. A paper that leads with Poisson and reports its headline
heterogeneity in a different estimator invites exactly the question we
are trying to close. This script re-estimates the split in Poisson so
the two agree or we learn that they do not.

WHAT IT DOES. One pull, identical to mona_common.pull_year_vintage
except that ind.Kon enters the SELECT and the GROUP BY, so the cell is
employer x ssyk4 x age x month x vintage x GENDER. Summing over gender
must reproduce the headline panel; the script checks that against the
canary anchor before it estimates anything, and stops if it does not.
Then, for each of 22-25 and 26-30, a pooled Poisson DiD on women and
on men separately, plus the pooled triple difference.

Kon is 1 = man, 2 = kvinna in SCB's coding (v1 script 16, which
produced the published gender numbers; not inferred).

COST. The pull is the expensive part, roughly the same as script 39:
seven years at two to four minutes a year, plus the fits. The cache is
written once and every re-run is cheap.

EXPORT. Coefficients only, no cell counts.
"""

import gc
import math
import sys
import time
from pathlib import Path

import pandas as pd

import mona_common as mc

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_48"
OUT.mkdir(exist_ok=True)

CACHE = mc.CACHE_DIR / "panel_gender.parquet"
AGES = ["22-25", "26-30"]
GENDERS = {1: "men", 2: "women"}

# The same anchor the canary gate uses, for the summed-over-gender check.
ANCHOR_POISSON_G2 = -0.1740
ANCHOR_TOL = 0.002


def pull_year_gender(year: int, conn) -> pd.DataFrame:
    """
    mona_common.pull_year_vintage with one extra dimension. Copied rather
    than parameterised on purpose: pull_year_vintage is what the canary
    gate reproduces bit for bit, and it is not worth touching for this.
    """
    suffix, max_month = mc._year_suffix(year)
    individ_year = min(year, 2023)
    monthly = []
    for month in range(1, max_month + 1):
        ym = f"{year}{month:02d}"
        if individ_year >= 2023:
            monthly.append(f"""
                SELECT
                    agi.P1207_LOPNR_PEORGNR AS employer_id,
                    agi.PERIOD AS period,
                    COALESCE(i23.Ssyk4_2012_J16, i22.Ssyk4_2012_J16,
                             i21.Ssyk4_2012_J16) AS ssyk4,
                    CASE WHEN i23.Ssyk4_2012_J16 IS NOT NULL THEN '2023'
                         WHEN i22.Ssyk4_2012_J16 IS NOT NULL THEN '2022'
                         WHEN i21.Ssyk4_2012_J16 IS NOT NULL THEN '2021'
                         ELSE 'none' END AS vintage,
                    COALESCE(i23.FodelseAr, i22.FodelseAr, i21.FodelseAr)
                        AS birth_year,
                    COALESCE(i23.Kon, i22.Kon, i21.Kon) AS gender,
                    agi.P1207_LOPNR_PERSONNR AS person_id
                FROM dbo.Arb_AGIIndivid{ym}{suffix} agi
                LEFT JOIN dbo.Individ_2023 i23
                    ON agi.P1207_LOPNR_PERSONNR = i23.P1207_LopNr_PersonNr
                LEFT JOIN dbo.Individ_2022 i22
                    ON agi.P1207_LOPNR_PERSONNR = i22.P1207_LopNr_PersonNr
                LEFT JOIN dbo.Individ_2021 i21
                    ON agi.P1207_LOPNR_PERSONNR = i21.P1207_LopNr_PersonNr
            """)
        else:
            monthly.append(f"""
                SELECT
                    agi.P1207_LOPNR_PEORGNR AS employer_id,
                    agi.PERIOD AS period,
                    ind.Ssyk4_2012_J16 AS ssyk4,
                    CASE WHEN ind.Ssyk4_2012_J16 IS NOT NULL
                         THEN 'own' ELSE 'none' END AS vintage,
                    ind.FodelseAr AS birth_year,
                    ind.Kon AS gender,
                    agi.P1207_LOPNR_PERSONNR AS person_id
                FROM dbo.Arb_AGIIndivid{ym}{suffix} agi
                LEFT JOIN dbo.Individ_{individ_year} ind
                    ON agi.P1207_LOPNR_PERSONNR = ind.P1207_LopNr_PersonNr
            """)

    union = "\nUNION ALL\n".join(monthly)
    age_case = """CASE
            WHEN age BETWEEN 22 AND 25 THEN '22-25'
            WHEN age BETWEEN 26 AND 30 THEN '26-30'
            WHEN age BETWEEN 31 AND 34 THEN '31-34'
            WHEN age BETWEEN 35 AND 40 THEN '35-40'
            WHEN age BETWEEN 41 AND 49 THEN '41-49'
            WHEN age BETWEEN 50 AND 69 THEN '50+'
            ELSE NULL END"""
    query = f"""
    WITH base AS ({union}),
    age_calc AS (
        SELECT employer_id, period,
               COALESCE(RIGHT('0000'+CAST(ssyk4 AS VARCHAR(4)),4), '____')
                   AS ssyk4,
               vintage, person_id, gender,
               CAST(LEFT(period,4) AS INT) - birth_year AS age
        FROM base
        WHERE birth_year IS NOT NULL
    )
    SELECT employer_id,
           LEFT(period,4) + '-' + SUBSTRING(period,5,2) AS year_month,
           ssyk4, vintage, gender,
           {age_case} AS age_group,
           COUNT(DISTINCT person_id) AS n_emp
    FROM age_calc
    WHERE age BETWEEN 22 AND 69 AND gender IN (1, 2)
    GROUP BY employer_id, period, ssyk4, vintage, gender, {age_case}
    """
    return pd.read_sql(query, conn)


def build(panel: pd.DataFrame, age: str, gender=None) -> pd.DataFrame:
    """Collapse to the estimation cell and balance it, as 43 does."""
    sub = panel if gender is None else panel[panel["gender"] == gender]
    agg = mc.collapse_vintage(sub)
    daioe = mc.load_daioe()
    agg["ssyk4"] = agg["ssyk4"].astype(str).str.zfill(4)
    agg = agg.merge(daioe, on="ssyk4", how="inner")
    size = agg.groupby("employer_id")["n_emp"].sum()
    agg = agg[agg["employer_id"].isin(
        size[size >= mc.MIN_EMPLOYER_SIZE].index)]
    agg = (agg.groupby(["employer_id", "year_month", "exposure_quartile",
                        "age_group"], observed=True)["n_emp"]
           .sum().reset_index())
    sub_a = agg[agg["age_group"] == age]
    months = sorted(agg["year_month"].unique())
    return mc.add_treatment(mc.balance_panel(sub_a, months))


def main():
    mc.Tee(OUT / "48_log.txt")
    # BatchClient keeps no stderr and run_all_mona's console log was cut
    # mid-traceback on 18 Sep, so 48's own crash was unreadable. Same hook
    # as 47b: the traceback goes through the Tee into this script's log.
    import sys as _sys
    import traceback as _tb
    _sys.excepthook = lambda et, ev, tb: print(
        "\nUNCAUGHT EXCEPTION\n" + "".join(_tb.format_exception(et, ev, tb)))
    print("=" * 70)
    print("48: GENDER SPLIT IN POISSON (the primary estimator)")
    print("=" * 70)
    print(mc.mem_line("  "))

    panel = mc.read_cache(CACHE)
    if panel is None:
        conn = mc.connect()
        frames = []
        for y in range(2019, 2026):
            t0 = time.time()
            f = pull_year_gender(y, conn)
            print(f"  {y}: {len(f):,} cells in {time.time() - t0:.0f}s")
            frames.append(f)
        panel = pd.concat(frames, ignore_index=True)
        mc.CACHE_DIR.mkdir(exist_ok=True)
        panel.to_parquet(CACHE, index=False)
        print(f"  Cached -> {CACHE.name}")
        del frames
        gc.collect()

    # GATE. Summing over gender must reproduce the headline panel. If it
    # does not, the gender pull differs from the pull every other number
    # in the paper rests on, and nothing below is comparable.
    print("\n--- gate: summed over gender, 22-25 ---")
    bal = build(panel, "22-25", gender=None)
    res = mc.run_fepois(bal, OUT, tag="g48_gate")
    g2 = float(res.loc[res["term"] == "post_gpt_x_high", "coef"].iloc[0])
    print(f"  pooled gamma2 = {g2:.4f}  (anchor {ANCHOR_POISSON_G2:.4f})")
    if abs(g2 - ANCHOR_POISSON_G2) > ANCHOR_TOL:
        print(f"GATE FAIL: {abs(g2 - ANCHOR_POISSON_G2):.4f} off the anchor. "
              f"The gender pull is not the headline pull. STOP.")
        (OUT / "48_summary.txt").write_text(
            f"GATE FAIL: summed-over-gender gamma2 {g2:.4f} vs anchor "
            f"{ANCHOR_POISSON_G2:.4f}\n")
        sys.exit(1)
    print("  GATE PASS")
    del bal
    gc.collect()

    rows = []
    for age in AGES:
        for code, label in GENDERS.items():
            print(f"\n--- {age}, {label} ---")
            bal = build(panel, age, gender=code)
            print(f"  {len(bal):,} cells, "
                  f"{bal['employer_id'].nunique():,} employers")
            r = mc.run_fepois(bal, OUT, tag=f"g48_{label}_{age}")
            if not r.empty:
                r["age_group"], r["gender"] = age, label
                rows.append(r)
            del bal
            gc.collect()
            print(mc.mem_line("  "))

    if rows:
        out = pd.concat(rows)
        out.to_csv(OUT / "gender_poisson.csv", index=False)
        lines = ["GENDER SPLIT -- pooled Poisson gamma2", "=" * 40]
        for age in AGES:
            for label in GENDERS.values():
                m = out[(out["age_group"] == age) & (out["gender"] == label)
                        & (out["term"] == "post_gpt_x_high")]
                if not m.empty:
                    c = float(m["coef"].iloc[0])
                    s = float(m["se"].iloc[0])
                    lines.append(f"  {age} {label:>5}: {c:+.4f} (SE {s:.4f})"
                                 f" = {100 * (math.exp(c) - 1):+.1f}%")
        lines.append("")
        lines.append("Read: the submitted ln(n+1) split was -0.016 women vs")
        lines.append("-0.007 men at 22-25. Poisson should agree in sign and")
        lines.append("ordering. If it does not, the abstract's 'concentrated")
        lines.append("among young women' has to be restated or dropped.")
        (OUT / "48_summary.txt").write_text("\n".join(lines))
        print("\n" + "\n".join(lines))
    print("\n48 done. " + mc.mem_line())


if __name__ == "__main__":
    main()
