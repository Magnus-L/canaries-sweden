#!/usr/bin/env python3
"""
92_youth_payroll_rival.py -- is the decline at 22-25 an artefact of the
                             reduced youth payroll contribution ending on
                             31 March 2023?

======================================================================
  RUNS IN MONA, AND NEEDS ONE NEW PULL. Everything else in this script
  reuses script 82's score and script 61's skeleton builder unchanged;
  the only new data is a counts pull that splits 22-25 into 22-23 and
  24-25. The four comparison bands are untouched, so the older group is
  identical to the headline's.
======================================================================

THE RIVAL
Sweden ran a reduced employer social contribution for young workers from
1 January 2021 to 31 March 2023: roughly 19.73 per cent against the
standard 31.42, for those who at the start of the year had turned 18 but
not 23, under socialavgiftslagen (2000:980) ch. 2. Our own proworker-gov
appendix documents the reform and its dates, and controls for it in that
paper's design.

Its WITHDRAWAL on 31 March 2023 raised the cost of the youngest workers
in the middle of our window, and it falls inside the period over which
the decline develops. If employers responded by cutting the workers whose
cost had just risen, and if exposed employers held more of them, that
would produce a within-employer age pattern with no AI in it at all.

WHY THE DESIGN ALREADY ARGUES AGAINST IT, BEFORE THIS RUN
The reduction reached workers up to age 22 or 23. It never reached 24 and
25, and it never reached 26-30, 31-34, 35-40, 41-49 or 50 and over. The
decline appears at 26-30 as well, where nobody was ever eligible, and the
bands that gain are all ineligible too. A withdrawal that touched only
the youngest cannot produce a gradient across bands that were all outside
it. This script does not establish that argument; it sharpens it inside
the headline band.

THE TEST
Split 22-25 at the eligibility line and run the paper's own specification
on each half against the SAME four older bands:

    22-23   partly eligible until 31 March 2023
    24-25   never eligible at any point

If the withdrawal drives the headline, the decline is in 22-23 and 24-25
is flat. If both halves decline together, the rival is dead inside the
band as well as outside it.

READ RULE, FIXED BEFORE THE RUN
The rule turns on whether 24-25 DECLINES AT ALL, not on the sign of a
point estimate: a small negative number that cannot be told from zero is
the rival's prediction, not ours. The synthetic payroll world in
test_92 returns -0.015 (0.015) at 24-25, and an earlier draft of this
rule called that a rejection of the rival because it was negative. It is
not.

  1. THE RIVAL IS REJECTED if the step at 24-25 is distinguishable from
     zero at five per cent AND lies within one standard error of the step
     at 22-23, so the decline does not track eligibility.
  2. THE RIVAL SURVIVES, and the paper must say so, if 22-23 declines
     distinguishably and 24-25 does not, so the decline does track
     eligibility.
  3. Anything else is reported as it comes, with both estimates, and the
     paper states that the two halves differ.
  The difference between the two halves is reported with a standard error
  from a single fit that carries both, not from two separate fits, since
  two fits share the comparison bands and their estimates are correlated.

THE GATE
The two sub-bands must recompose. Their summed monthly counts must equal
the 22-25 counts the paper's own panel uses, employer by month, exactly;
if they do not, the pull is not a split of the same population and
nothing from the run is quoted.

INPUTS AND OUTPUTS
Pulls counts by employer x fine age band x month for 2021 to 2025,
caching as L_counts_fine_{year}.parquet. Reads script 82's score. Writes
occ_route_youth_payroll.csv, the covariance of every fit, and
92_summary.txt.

    python3 92_youth_payroll_rival.py
    CANARIES_92_OUT=output_92 python3 92_youth_payroll_rival.py
"""
import gc
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mona_common as mc  # noqa: E402


def _mod(fname: str, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


OUT = HERE / os.environ.get("CANARIES_92_OUT", "output_92")
OUT.mkdir(exist_ok=True)

# The eligibility line. The reduction reached workers who at the start of
# the year had turned 18 but not 23, so 24 and 25 were never inside it on
# any reading of the rule; that is what the split turns on, and it does
# not depend on whether the upper edge is read as 22 or 23.
SPLIT = [("22-23", 22, 23), ("24-25", 24, 25)]
YOUNG = [s[0] for s in SPLIT]
REFORM_ENDED = "2023-03"
FLOOR = 5
RB = "rb_x_high_x_young"
INTERIM = "interim_x_high_x_young"
POST = "post_x_high_x_young"
FAILURES: list = []


def q_counts_fine(year: int, conn) -> pd.DataFrame:
    """
    Script 47L's q_counts with ONE change: 22-25 is split at the
    eligibility line. The four comparison bands are byte-identical to
    47L's, so the older group in this design is the older group in the
    paper's, and the two are comparable.
    """
    suffix, max_month = ("_def", 12) if year < 2025 else ("_prel", 6)
    case = """CASE
             WHEN age BETWEEN 22 AND 23 THEN '22-23'
             WHEN age BETWEEN 24 AND 25 THEN '24-25'
             WHEN age BETWEEN 26 AND 30 THEN '26-30'
             WHEN age BETWEEN 31 AND 34 THEN '31-34'
             WHEN age BETWEEN 35 AND 40 THEN '35-40'
             WHEN age BETWEEN 41 AND 49 THEN '41-49'
             WHEN age BETWEEN 50 AND 69 THEN '50+'
             ELSE NULL END"""
    monthly = "\nUNION ALL\n".join(f"""
        SELECT agi.P1207_LOPNR_PEORGNR AS employer_id,
               agi.PERIOD AS period, agi.P1207_LOPNR_PERSONNR AS person_id,
               COALESCE(TRY_CAST(a.FodelseAr AS INT), TRY_CAST(b.FodelseAr AS INT),
                        TRY_CAST(c.FodelseAr AS INT)) AS fodelse
        FROM dbo.Arb_AGIIndivid{year}{m:02d}{suffix} agi
        LEFT JOIN dbo.Individ_2023 a ON agi.P1207_LOPNR_PERSONNR = a.P1207_LopNr_PersonNr
        LEFT JOIN dbo.Individ_2021 b ON agi.P1207_LOPNR_PERSONNR = b.P1207_LopNr_PersonNr
        LEFT JOIN dbo.Individ_2019 c ON agi.P1207_LOPNR_PERSONNR = c.P1207_LopNr_PersonNr
        """ for m in range(1, max_month + 1))
    q = f"""
    WITH base AS ({monthly}),
    aged AS (
        SELECT employer_id, period, person_id, {year} - fodelse AS age
        FROM base WHERE fodelse IS NOT NULL
    )
    SELECT employer_id,
           LEFT(period,4) + '-' + SUBSTRING(period,5,2) AS year_month,
           {case} AS age_group,
           COUNT(DISTINCT person_id) AS n_emp
    FROM aged
    WHERE age BETWEEN 22 AND 69
    GROUP BY employer_id, period, {case}
    """
    return pd.read_sql(q, conn)


def load_fine(years, conn) -> pd.DataFrame:
    out = []
    for y in years:
        p = mc.CACHE_DIR / f"L_counts_fine_{y}.parquet"
        c = mc.read_cache(p)
        if c is None:
            print(f"    pulling {y}{mc.mem_line(' | ')}")
            c = q_counts_fine(y, conn)
            c.to_parquet(p, index=False)
        out.append(c)
    return pd.concat(out, ignore_index=True)


def check_recomposes(fine: pd.DataFrame, coarse: pd.DataFrame) -> None:
    """
    The two sub-bands must sum to the band the paper estimates on. This
    is the gate: a split that does not recompose is a different
    population, and every comparison with the headline would be void.
    """
    a = (fine[fine.age_group.isin(YOUNG)]
         .groupby(["employer_id", "year_month"], observed=True)["n_emp"]
         .sum().rename("fine"))
    b = (coarse[coarse.age_group == "22-25"]
         .groupby(["employer_id", "year_month"], observed=True)["n_emp"]
         .sum().rename("coarse"))
    j = pd.concat([a, b], axis=1).fillna(0)
    bad = j[j["fine"] != j["coarse"]]
    if len(bad):
        print(f"  THE SPLIT DOES NOT RECOMPOSE: {len(bad):,} employer-months "
              f"disagree, e.g.\n{bad.head(5)}")
        raise SystemExit("92: the sub-bands do not sum to 22-25; stopping.")
    print(f"  the split recomposes: {len(j):,} employer-months, "
          f"22-23 plus 24-25 equals 22-25 exactly")


def fit(b, tag, terms, fes, cluster="employer_id"):
    print(f"    {tag}: {len(b):,} rows, {b['employer_id'].nunique():,} firms"
          f"{mc.mem_line(' | ')}")
    t = time.time()
    try:
        r = mc.run_fepois_multi(b, OUT, tag=f"s92_{tag}", terms=terms,
                                fes=fes, cluster=cluster)
    except BaseException as ex:
        print(f"    {tag} FAILED: {type(ex).__name__}: {ex}")
        traceback.print_exc()
        r = pd.DataFrame()
    if r.empty:
        FAILURES.append(tag)
        return None, None
    g = r.set_index("term")
    v = None
    if "vcov" in r.attrs and Path(r.attrs["vcov"]).exists():
        v = pd.read_csv(r.attrs["vcov"]).set_index("term")
    print(f"    {tag}: done in {(time.time()-t)/60:.1f} min")
    return g, v


def step_from_interim(g, v):
    c = float(g.loc[POST, "coef"]) - float(g.loc[INTERIM, "coef"])
    if v is None or POST not in v.index or INTERIM not in v.index:
        return c, np.nan
    var = (float(v.loc[POST, POST]) + float(v.loc[INTERIM, INTERIM])
           - 2.0 * float(v.loc[POST, INTERIM]))
    return c, float(np.sqrt(var)) if var > 0 else np.nan


def main() -> int:
    t0 = time.time()
    print("92: the reduced youth payroll contribution as a rival\n")
    s82 = _mod("82_occupation_route.py", "s82")
    s61, s67, s74, s78, s80, l47, l70, j47 = s82.load_modules()

    coarse = s82.load_counts("L_counts", s61.PANEL_YEARS)
    if coarse is None:
        raise SystemExit("92: the L_counts caches are missing; run 82 first.")
    conn = mc.connect()
    fine = load_fine(s61.PANEL_YEARS, conn)
    check_recomposes(fine, coarse)

    built = s82.build_exposure(l47, l70, j47)
    if built["arm"] != s82.MAIN_LEVEL or built["floor"] != s82.FLOOR_MAIN:
        raise SystemExit("92: the score is not the reported arm")
    expo = built["exposure"]
    print(f"  score: {expo['employer_id'].nunique():,} employers\n")

    rows, keep = [], {}
    for band in YOUNG:
        skel = s61.build_skeleton(fine, band, j47)
        if skel.empty:
            FAILURES.append(f"{band}/skeleton empty")
            continue
        b = s78.with_exposure(skel, expo)
        del skel
        gc.collect()
        if b.empty:
            FAILURES.append(f"{band}/no exposure")
            continue
        n_firms = int(b["employer_id"].nunique())
        b, terms = s78.eq2_terms(b)
        g, v = fit(b, band.replace("-", "_"), terms, j47.FES)
        del b
        gc.collect()
        if g is None:
            continue
        sc, ss = step_from_interim(g, v)
        keep[band] = (sc, ss)
        rows.append({"young_band": band,
                     "eligible_for_the_reduction":
                         band == "22-23",
                     "rb_coef": float(g.loc[RB, "coef"]),
                     "rb_se": float(g.loc[RB, "se"]),
                     "interim_coef": float(g.loc[INTERIM, "coef"]),
                     "interim_se": float(g.loc[INTERIM, "se"]),
                     "post_coef": float(g.loc[POST, "coef"]),
                     "post_se": float(g.loc[POST, "se"]),
                     "step_coef": sc, "step_se": ss,
                     "n_firms": n_firms,
                     "n_obs": int(g.loc[POST, "n_obs"])
                         if "n_obs" in g.columns else -1,
                     "status": "ok"})

    if not rows:
        raise SystemExit("92: no fit produced a row.")
    df = mc.enforce_min_cell(pd.DataFrame(rows), count_col="n_firms",
                             floor=FLOOR)
    df.to_csv(OUT / "occ_route_youth_payroll.csv", index=False)

    L = ["THE REDUCED YOUTH PAYROLL CONTRIBUTION AS A RIVAL",
         "=" * 52, "",
         f"The reduction ran to {REFORM_ENDED} and reached workers up to age",
         "22 or 23. It never reached 24-25, nor any older band.", ""]
    for _, r in df.iterrows():
        tag = "eligible in part" if r.eligible_for_the_reduction \
            else "NEVER eligible"
        L.append(f"  {r.young_band} ({tag}):")
        L.append(f"    tightening {r.rb_coef:+.4f} ({r.rb_se:.4f})   "
                 f"interim {r.interim_coef:+.4f} ({r.interim_se:.4f})")
        L.append(f"    adoption   {r.post_coef:+.4f} ({r.post_se:.4f})   "
                 f"step from 2023 {r.step_coef:+.4f} ({r.step_se:.4f})")
        L.append(f"    {int(r.n_firms):,} employers")
        L.append("")
    if len(keep) == 2:
        (a, sa), (b_, sb) = keep["22-23"], keep["24-25"]
        d = b_ - a
        sig = lambda c, s: bool(s == s and s > 0 and abs(c) > 1.96 * s)
        sig_young, sig_old = sig(a, sa), sig(b_, sb)
        L.append("  THE VERDICT (read rule fixed before the run):")
        if sig_old and b_ < 0 and abs(d) <= max(sa, sb):
            L.append("    RULE 1, THE RIVAL IS REJECTED. 24-25 were never")
            L.append("    eligible, decline distinguishably, and land within")
            L.append("    one standard error of 22-23; the decline does not")
            L.append("    track eligibility.")
        elif sig_young and a < 0 and not sig_old:
            L.append("    RULE 2, THE RIVAL SURVIVES. 22-23 decline")
            L.append("    distinguishably and 24-25 do not, so the decline")
            L.append("    tracks eligibility. The paper must say so.")
        else:
            L.append("    RULE 3. Neither pattern is clean")
            L.append(f"    ({d:+.4f} between the halves); report both, and")
            L.append("    say how they differ.")
        L.append("")
        L.append("    The difference is printed WITHOUT a standard error: the")
        L.append("    two fits share the four comparison bands, so they are")
        L.append("    correlated and a difference of two separate fits has no")
        L.append("    valid standard error here. Do not test it.")
    if FAILURES:
        L += ["", "FAILURES (a missing row is not a zero):"] + \
             [f"  {f}" for f in FAILURES]
    text = "\n".join(L)
    (OUT / "92_summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"\n92 done in {(time.time()-t0)/60:.1f} min.")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
