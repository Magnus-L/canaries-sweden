#!/usr/bin/env python3
"""
51_vintage_ai_unboxed.py -- how much of each year's occupation coding is
ACTUALLY from that year, with the clerical cut AI Unboxed depends on.

======================================================================
  RUNS IN SCB's MONA SECURE ENVIRONMENT ONLY. Standalone: submit THIS
  file. Writes output_51/. Individ aggregates only, about 6 minutes,
  no AGI, no panel, no estimation.
  Local end-to-end test: revision/local/test_51_synthetic.py
======================================================================

WHY, and this is not a canaries question. Our own note
`data-notes/occupation-missingness.md` has carried an open question since
10 August 2026 that nobody has run:

  "What share of the DAIOE panel's 2023 occupations are carried forward
   rather than assigned in 2023? Same SsykAr_J16 test, restricted to the
   panel's firms. This is the one with a live consequence: stale 2023
   codes attenuate estimates that use 2023 as an endpoint."

AI Unboxed measures clerical employment within firms and ends in 2023. If
a material share of 2023's clerical classifications are codes carried
forward from 2021 or 2022, then the measured change in clerical employment
is attenuated toward zero at exactly the endpoint the paper leans on. The
same note also records that 2020 and 2021 flows are "implausibly low ...
that pattern is what carrying a code forward looks like", so the middle of
the panel may be affected too.

Attenuation is the FAVOURABLE direction for a paper that finds an effect:
it means the true effect is at least as large. That is worth knowing
before submission rather than after a referee asks, and it is worth
knowing precisely rather than as a presumption.

WHAT IT MEASURES, per year 2019-2023:
  1. the distribution of (year - SsykAr_J16): the age of the code actually
     in force, by age band and by one-digit occupation
  2. the same restricted to CLERICAL (SSYK 4xxx), which is AI Unboxed's
     outcome, and to the neighbouring groups it is compared against
  3. SsykStatus_J16, whose value set separates the codes that agree with
     the November job from those that do not
  4. transitions into and out of clerical among people coded in BOTH
     adjacent years, split by whether each code was freshly assigned:
     if the clerical exits we measure are concentrated among people whose
     code was NOT reassigned, they are a coding artefact, not a move

ALL OUTPUT IS AGGREGATE, with every cell floored at five. No firm, no
person, no panel.

WHAT IT CANNOT DO. It does not re-estimate AI Unboxed and it does not
touch that project's data. It answers one question: how stale is the
occupation variable in the years that paper uses, and is the staleness
concentrated in clerical work. Deciding what follows is a separate
conversation.
"""

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import mona_common as mc

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_51"
OUT.mkdir(exist_ok=True)
YEARS = [2019, 2020, 2021, 2022, 2023]
FLOOR = 5

CODED = ("{a}.Ssyk4_2012_J16 IS NOT NULL AND LTRIM({a}.Ssyk4_2012_J16) <> '' "
         "AND LEFT(LTRIM({a}.Ssyk4_2012_J16), 1) <> '*'")
AGE = """CASE
        WHEN {y} - TRY_CAST({a}.FodelseAr AS INT) BETWEEN 22 AND 29 THEN '22-29'
        WHEN {y} - TRY_CAST({a}.FodelseAr AS INT) BETWEEN 30 AND 49 THEN '30-49'
        WHEN {y} - TRY_CAST({a}.FodelseAr AS INT) BETWEEN 50 AND 69 THEN '50+'
        ELSE NULL END"""


def opt(label, fn, *a, **kw):
    """capture noisily: an inessential step fails loudly and the run goes on."""
    try:
        return fn(*a, **kw)
    except BaseException as ex:
        print(f"  [optional] {label} FAILED ({type(ex).__name__}): {str(ex)[:200]}")
        return None


def export(df: pd.DataFrame, name: str, col: str = "n"):
    out = df.copy()
    small = (out[col] > 0) & (out[col] < FLOOR)
    if small.any():
        print(f"    floor: {int(small.sum()):,} cells suppressed")
        out.loc[small, col] = np.nan
    out.to_csv(OUT / name, index=False)
    print(f"  wrote {name}: {len(out):,} rows")


def q_code_age(y: int, conn) -> pd.DataFrame:
    """The age of the code in force, by one-digit occupation and age band."""
    q = f"""
    SELECT LEFT(RIGHT('0000' + CAST(a.Ssyk4_2012_J16 AS VARCHAR(4)), 4), 1) AS ssyk1,
           {AGE.format(y=y, a='a')} AS age_group,
           {y} - TRY_CAST(a.SsykAr_J16 AS INT) AS code_age,
           LTRIM(RTRIM(a.SsykStatus_J16)) AS ssyk_status,
           COUNT(*) AS n
    FROM dbo.Individ_{y} a
    WHERE {CODED.format(a='a')}
      AND {y} - TRY_CAST(a.FodelseAr AS INT) BETWEEN 22 AND 69
    GROUP BY LEFT(RIGHT('0000' + CAST(a.Ssyk4_2012_J16 AS VARCHAR(4)), 4), 1),
             {AGE.format(y=y, a='a')},
             {y} - TRY_CAST(a.SsykAr_J16 AS INT),
             LTRIM(RTRIM(a.SsykStatus_J16))
    """
    return pd.read_sql(q, conn)


def q_clerical_flows(t: int, conn) -> pd.DataFrame:
    """
    Clerical (SSYK 4xxx) in t and in t+1 for the same person, crossed with
    whether each year's code was FRESHLY assigned that year. A clerical exit
    recorded for someone whose code was never reassigned is a coding
    artefact; one recorded for someone reassigned in t+1 is a move.
    """
    q = f"""
    SELECT CASE WHEN LEFT(RIGHT('0000'+CAST(a.Ssyk4_2012_J16 AS VARCHAR(4)),4),1)='4'
                THEN 1 ELSE 0 END AS clerical_t,
           CASE WHEN LEFT(RIGHT('0000'+CAST(b.Ssyk4_2012_J16 AS VARCHAR(4)),4),1)='4'
                THEN 1 ELSE 0 END AS clerical_t1,
           CASE WHEN TRY_CAST(a.SsykAr_J16 AS INT) = {t} THEN 1 ELSE 0 END AS fresh_t,
           CASE WHEN TRY_CAST(b.SsykAr_J16 AS INT) = {t+1} THEN 1 ELSE 0 END AS fresh_t1,
           {AGE.format(y=t, a='a')} AS age_group,
           COUNT(*) AS n
    FROM dbo.Individ_{t} a
    JOIN dbo.Individ_{t+1} b ON a.P1207_LopNr_PersonNr = b.P1207_LopNr_PersonNr
    WHERE {CODED.format(a='a')} AND {CODED.format(a='b')}
      AND {t} - TRY_CAST(a.FodelseAr AS INT) BETWEEN 22 AND 69
    GROUP BY CASE WHEN LEFT(RIGHT('0000'+CAST(a.Ssyk4_2012_J16 AS VARCHAR(4)),4),1)='4'
                  THEN 1 ELSE 0 END,
             CASE WHEN LEFT(RIGHT('0000'+CAST(b.Ssyk4_2012_J16 AS VARCHAR(4)),4),1)='4'
                  THEN 1 ELSE 0 END,
             CASE WHEN TRY_CAST(a.SsykAr_J16 AS INT) = {t} THEN 1 ELSE 0 END,
             CASE WHEN TRY_CAST(b.SsykAr_J16 AS INT) = {t+1} THEN 1 ELSE 0 END,
             {AGE.format(y=t, a='a')}
    """
    return pd.read_sql(q, conn)


def headline(codeage: pd.DataFrame) -> list:
    """The two numbers this script exists to produce."""
    lines = []
    d = codeage.dropna(subset=["n"])
    for y in sorted(d["year"].unique()):
        s = d[d["year"] == y]
        tot = s["n"].sum()
        fresh = s.loc[s["code_age"] == 0, "n"].sum()
        cler = s[s["ssyk1"] == "4"]
        ctot, cfresh = cler["n"].sum(), cler.loc[cler["code_age"] == 0, "n"].sum()
        lines.append(f"  {int(y)}: {fresh/max(tot,1):.1%} of codes assigned that "
                     f"year, all occupations; clerical {cfresh/max(ctot,1):.1%}")
    return lines


def main():
    mc.Tee(OUT / "51_log.txt")
    sys.excepthook = lambda et, ev, tb: print(
        "\nUNCAUGHT EXCEPTION\n" + "".join(traceback.format_exception(et, ev, tb)))
    t0 = time.time()
    print("=" * 70)
    print("51: HOW STALE IS THE OCCUPATION VARIABLE, AND IS IT STALE FOR CLERKS")
    print("    (the open question in occupation-missingness.md, for AI Unboxed)")
    print("=" * 70)
    print(mc.mem_line("  "))
    conn = mc.connect()

    frames = []
    for y in YEARS:
        t = time.time()
        try:
            f = q_code_age(y, conn).assign(year=y)
            frames.append(f)
            print(f"  code age {y}: {len(f):,} cells ({time.time()-t:.0f}s)")
        except Exception as ex:
            print(f"  code age {y} FAILED ({type(ex).__name__}): {str(ex)[:200]}")
    if not frames:
        raise RuntimeError("no code-age year succeeded; nothing to report")
    codeage = pd.concat(frames, ignore_index=True)
    export(codeage, "code_age_by_occupation.csv")

    flows = []
    for t in YEARS[:-1]:
        try:
            flows.append(q_clerical_flows(t, conn).assign(year=t))
            print(f"  clerical flows {t}->{t+1}: ok")
        except Exception as ex:
            print(f"  clerical flows {t} FAILED ({type(ex).__name__}): {str(ex)[:200]}")
    if flows:
        opt("clerical flow export", export,
            pd.concat(flows, ignore_index=True), "clerical_flows_by_freshness.csv")

    lines = ["OCCUPATION CODE VINTAGE, AND THE CLERICAL CUT", "=" * 58,
             "Share of coded people whose code was assigned in the year itself:",
             ""] + headline(codeage) + [""]
    if flows:
        fl = pd.concat(flows, ignore_index=True).dropna(subset=["n"])
        ex = fl[(fl.clerical_t == 1)]
        if len(ex) and ex["n"].sum() > 0:
            left = ex[ex.clerical_t1 == 0]
            byfresh = left.groupby("fresh_t1")["n"].sum()
            tot = ex["n"].sum()
            lines += ["Clerical exits, by whether the NEXT year's code was freshly",
                      "assigned (an exit recorded without a reassignment is a coding",
                      "artefact, not a move):"]
            for k, v in byfresh.items():
                lab = "reassigned that year" if k == 1 else "NOT reassigned"
                lines.append(f"  {lab:<24} {v/max(tot,1):.2%} of clerical stock")
            lines.append("")
    lines += ["READ THIS AS A MEASUREMENT FACT, NOT A RESULT.",
              "  It does not re-estimate AI Unboxed and does not touch its data.",
              "  Stale codes at an endpoint attenuate a measured change toward",
              "  zero, so for a paper that FINDS an effect the direction is",
              "  favourable: the true effect is at least as large. The number",
              "  matters for how the paper describes its measure, not for",
              "  whether its finding survives.",
              f"Runtime {(time.time()-t0)/60:.1f} min. " + mc.mem_line()]
    (OUT / "51_summary.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
