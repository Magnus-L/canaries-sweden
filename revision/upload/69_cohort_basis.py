#!/usr/bin/env python3
"""
69_cohort_basis.py -- is the seasonal mechanical? Fixed birth cohorts
                      against moving age bands, same firms, same months.

======================================================================
  RUNS IN MONA. One SQL pull (cohort counts), cached as L_cohort_YYYY.
  Reuses 47h's 2019 frame for exposure and 61's design. Writes output_69/.
======================================================================

THE PROBLEM THIS EXISTS TO SETTLE.

47L computes age as `{year} - FodelseAr`, calendar year minus birth year,
because FodelseAr is the only age variable in the P1207 delivery: there is
no birth month, no birth date and no Alder. Every worker therefore ages on
1 January, and the age bands are moving windows rather than cohorts.

The consequence is that on 1 January each year the 22-25 band loses an
entire birth cohort to 26-30 and gains an entire new one, mechanically,
with nobody changing job. Swedish cohorts are not the same size (1999:
88,173 births; 2003: 99,157), so the size of that lump differs by year.
If exposed firms sit at a different point inside the band than unexposed
firms, which is likely where professional services recruit graduates at
23-25 rather than at 22, then the lump differs by exposure as well.

Month-by-age fixed effects absorb the part of this that is common across
firms. They cannot absorb the exposure-differential part. That is exactly
the pattern script 64 found and failed its pre-committed pre-period test
on: the exposure-differential young-to-older ratio is positive in Q4 every
year (+0.076, +0.092, +0.062, +0.025) and negative in Q1 every year.

Script 68 removes that pattern with exposure-differential calendar terms,
which treats it as seasonality. If it is instead discrete cohort turnover
in a moving window then calendar dummies are the wrong instrument, because
the size of the January lump moves with cohort size and a fixed
calendar-month effect cannot absorb a jump that changes every year.

WHAT THIS SCRIPT DOES INSTEAD.

It rebuilds the same panel on FIXED BIRTH COHORTS. A worker's group is set
by birth year once, from their age in the reference year 2022, and never
changes. Nobody crosses a group boundary, ever, so the mechanical January
turnover is removed by construction rather than modelled.

Note this is strictly better than the exact age the delivery does not
carry. With true ages people still cross boundaries; the crossings merely
spread across twelve months instead of landing in January. With fixed
cohorts there are no crossings at all.

The cost is real and is stated rather than hidden: the young cohort ages
over the panel, from 21-24 in 2021 to 25-28 in 2025. This design therefore
follows one group of people as they age, and cannot speak about whoever is
22-25 at each date. That is the trade, and it is the right one for a
diagnostic about whether the seasonal is mechanical.

THE READ RULE, FIXED BEFORE THE RUN.

Both bases are estimated on the same firms and the same months, and both
report (a) a pooled headline and (b) calendar-quarter interactions fitted
on the PRE-ChatGPT window only, which is where a seasonal must show up if
it is a seasonal at all.

  1. SEASONAL MECHANICAL if the largest absolute calendar-quarter
     coefficient falls by at least half on the cohort basis. Then 68's
     calendar controls are the wrong instrument and the paper should
     report the cohort basis.

  2. SEASONAL REAL if that largest coefficient is within 25 per cent of
     the age-band figure. Then the cycle is not an artefact of the moving
     window, 68 is the right response, and it should run as written.

  3. AMBIGUOUS in between. Report both, quote neither as settled, and say
     in the paper that the two bases disagree.

  4. HEADLINE ROBUST if the pooled cohort coefficient sits inside the
     age-band coefficient's 95 per cent interval. If it does not, the
     headline depends on the window moving, which is itself a finding and
     must be said.

Nothing below chooses a threshold after seeing a number. If the outcome is
AMBIGUOUS it stays ambiguous.

Output (output_69/):
  cohort_pooled.csv    pooled coefficients, both bases
  cohort_season.csv    pre-period calendar-quarter coefficients, both bases
  69_summary.txt       the read rule evaluated, in words
"""

import gc
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mona_common as mc

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_69"
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR

# The reference year that fixes the cohorts. 2022 and not 2019: the claim
# is about who was young WHEN CHATGPT ARRIVED, so a worker aged 22-25 in
# 2022 is the object of interest. Defining cohorts on 2019 ages would
# instead follow people who were already 25-28 at the launch.
REF_YEAR = 2022

# Read straight off 47L's band definitions so the two bases cannot drift
# apart through a transcription error. (lo, hi) are ages in REF_YEAR.
BAND_AGES = {"22-25": (22, 25), "26-30": (26, 30), "31-34": (31, 34),
             "35-40": (35, 40), "41-49": (41, 49), "50+": (50, 69)}
YOUNG_BANDS = ["22-25", "26-30"]
INCUMBENT_BANDS = ["31-34", "35-40", "41-49", "50+"]

PANEL_FROM = "2021-01"
PANEL_YEARS = list(range(2021, 2026))
POOLED_FROM = "2024-01"          # 61's adoption dating, unchanged
PRE_TO = "2022-11"               # last month before the ChatGPT launch
TRUNC = 2021
DESIGN = "OL_daioe"              # one design; this is a diagnostic
ARM = "true"

# Read rule thresholds, fixed here before the run.
MECHANICAL_DROP = 0.50           # cohort seasonal <= half of age-band
REAL_BAND = 0.75                 # cohort seasonal >= 75% of age-band
FAILURES = []


def edu_exposure(j47, design: str, arm: str):
    """
    47j's incumbent education exposure, built exactly as 61 builds it.

    Reproduced rather than imported because 61 does this inside its main().
    The steps and their order matter: the ScoreBook needs every weight year,
    not only the base year, and the spec must be built into the book before
    incumbent_exposure is asked for anything.
    """
    h47 = j47._h47()
    counts = {}
    for y in h47.WEIGHT_YEARS:
        w = mc.read_cache(CACHE / f"edu_hr_weights_{y}.parquet",
                          require=h47.WEIGHT_COLS)
        if w is None:
            raise SystemExit(f"edu_hr_weights_{y}.parquet missing: run 47h "
                             f"first. This script performs no education SQL.")
        counts[y] = w
    book = h47.ScoreBook(counts, h47.load_key(), h47.load_scores())
    spec = dict(h47.DESIGNS[design])
    book.build(design, spec)
    frame19 = mc.read_cache(CACHE / f"edu_hr_{j47.BASE_YEAR}.parquet",
                            require=h47.YEAR_COLS + ["n_emp"])
    if frame19 is None:
        raise SystemExit(f"edu_hr_{j47.BASE_YEAR}.parquet missing: run 47h.")
    expo, _ = j47.incumbent_exposure(frame19, book, design, spec, arm, TRUNC)
    del frame19
    gc.collect()
    return expo


def cohort_years(band: str) -> tuple:
    """Birth years of the workers who are in `band` in REF_YEAR."""
    lo, hi = BAND_AGES[band]
    return REF_YEAR - hi, REF_YEAR - lo


def opt(label, fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except BaseException as ex:
        print(f"  [optional] {label} FAILED ({type(ex).__name__}: {ex})")
        traceback.print_exc()
        return None


def _mod(fname: str, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def q_cohort_counts(year: int, conn) -> pd.DataFrame:
    """
    Employment counts by employer x FIXED BIRTH COHORT x month.

    Identical to 47L's q_counts in every respect except the CASE: the band
    is assigned from birth year alone, so it does not depend on `year` and
    a worker never moves between groups. Birth year is still the only
    worker attribute used; no occupation, no education.
    """
    suffix, max_month = ("_def", 12) if year < 2025 else ("_prel", 6)
    arms = []
    for band in YOUNG_BANDS + INCUMBENT_BANDS:
        y0, y1 = cohort_years(band)
        arms.append(f"WHEN fodelse BETWEEN {y0} AND {y1} THEN '{band}'")
    band_case = "CASE\n        " + "\n        ".join(arms) + "\n        ELSE NULL END"
    lo = min(cohort_years(b)[0] for b in YOUNG_BANDS + INCUMBENT_BANDS)
    hi = max(cohort_years(b)[1] for b in YOUNG_BANDS + INCUMBENT_BANDS)
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
    kept AS (
        SELECT employer_id, period, person_id, fodelse
        FROM base WHERE fodelse IS NOT NULL
          AND fodelse BETWEEN {lo} AND {hi}
    )
    SELECT employer_id,
           LEFT(period,4) + '-' + SUBSTRING(period,5,2) AS year_month,
           {band_case} AS age_group,
           COUNT(DISTINCT person_id) AS n_emp
    FROM kept
    GROUP BY employer_id, period, {band_case}
    HAVING {band_case} IS NOT NULL
    """
    return pd.read_sql(q, conn)


def build_skeleton(counts: pd.DataFrame, young: str, j47) -> pd.DataFrame:
    """
    61's skeleton, unchanged. Taking it from 61 rather than reimplementing
    guarantees the two bases differ ONLY in how `age_group` was assigned,
    which is the whole point of the comparison.
    """
    l61 = _mod("61_redated_triple.py", "l61")
    return l61.build_skeleton(counts, young, j47)


def season_terms(bal: pd.DataFrame) -> tuple:
    """
    Calendar-quarter interactions on the PRE-ChatGPT window only.

    A seasonal is a recurring within-year pattern, so it has to be visible
    before the treatment or it is not a seasonal. Fitting it on the pre
    window alone also keeps this diagnostic entirely separate from the
    estimate: nothing here can move the headline.

    Q3 is omitted, matching 64's reference quarter, so the coefficients are
    each quarter's exposure-differential young-to-older gap relative to Q3.
    """
    b = bal[bal["year_month"].astype(str) <= PRE_TO].copy()
    if b.empty:
        return b, []
    q = pd.to_datetime(b["year_month"] + "-01").dt.quarter
    hy = b["high"] * b["young"]
    terms = []
    for qq in (1, 2, 4):
        c = f"hy_q{qq}"
        b[c] = hy * (q == qq).astype(int)
        terms.append(c)
    return b, terms


def pooled_terms(bal: pd.DataFrame) -> tuple:
    """61's pooled interaction plus its Riksbank control, nothing else."""
    b = bal
    ym = b["year_month"].astype(str)
    hy = b["high"] * b["young"]
    b["post_rb_x_high_x_young"] = (ym >= mc.RIKSBANK_YM).astype(int) * hy
    b["post_x_high_x_young"] = (ym >= POOLED_FROM).astype(int) * hy
    return b, ["post_rb_x_high_x_young", "post_x_high_x_young"]


def run_basis(counts: pd.DataFrame, basis: str, young: str, expo, j47,
              sink: list, season_sink: list):
    """Both fits for one basis and one young band."""
    skel = build_skeleton(counts, young, j47)
    if skel.empty:
        print(f"  {basis} {young}: skeleton empty, skipped")
        return
    b = skel.merge(expo[["employer_id", "fq"]], on="employer_id", how="inner")
    del skel
    gc.collect()
    if b.empty:
        print(f"  {basis} {young}: no firms matched exposure, skipped")
        return
    b["high"] = (b["fq"] == 4).astype(int)
    print(f"  {basis} {young}: panel {len(b):,} rows, "
          f"{b['employer_id'].nunique():,} firms{mc.mem_line(' | ')}")

    bs, sterms = season_terms(b)
    if sterms and len(bs):
        r = mc.run_fepois_multi(bs, OUT, tag=f"r69_{basis}_{young}_season",
                                terms=sterms, fes=j47.FES)
        if r.empty:
            FAILURES.append(f"{basis}/{young}/season")
        else:
            for _, row in r.iterrows():
                season_sink.append({"basis": basis, "young_band": young,
                                    "term": row["term"], "coef": row["coef"],
                                    "se": row["se"]})
    del bs
    gc.collect()

    b, pterms = pooled_terms(b)
    r = mc.run_fepois_multi(b, OUT, tag=f"r69_{basis}_{young}_pooled",
                            terms=pterms, fes=j47.FES)
    if r.empty:
        FAILURES.append(f"{basis}/{young}/pooled")
    else:
        for _, row in r.iterrows():
            if row["term"] != "post_x_high_x_young":
                continue
            sink.append({"basis": basis, "young_band": young,
                         "coef": row["coef"], "se": row["se"],
                         "n_firms": int(b["employer_id"].nunique())})
    del b
    gc.collect()


def verdict(pooled: pd.DataFrame, season: pd.DataFrame) -> list:
    """The pre-committed read rule, evaluated. No thresholds are set here."""
    out = []
    for young in YOUNG_BANDS:
        s = season[season.young_band == young]
        ab = s[s.basis == "ageband"]["coef"].abs().max() if len(
            s[s.basis == "ageband"]) else np.nan
        co = s[s.basis == "cohort"]["coef"].abs().max() if len(
            s[s.basis == "cohort"]) else np.nan
        if not (np.isfinite(ab) and np.isfinite(co)) or ab <= 0:
            out.append(f"{young}: seasonal verdict UNAVAILABLE "
                       f"(ageband {ab}, cohort {co})")
        else:
            ratio = co / ab
            if ratio <= MECHANICAL_DROP:
                v = ("SEASONAL MECHANICAL. The cycle is cohort turnover in a "
                     "moving window. 68's calendar controls are the wrong "
                     "instrument; report the cohort basis.")
            elif ratio >= REAL_BAND:
                v = ("SEASONAL REAL. Not an artefact of the moving window. "
                     "68 is the right response and should run as written.")
            else:
                v = ("AMBIGUOUS. Report both bases, settle neither, and say "
                     "in the paper that they disagree.")
            out.append(f"{young}: largest pre-period calendar coefficient "
                       f"{ab:.4f} on age bands, {co:.4f} on cohorts, "
                       f"ratio {ratio:.2f}. {v}")

        p = pooled[pooled.young_band == young]
        pa = p[p.basis == "ageband"]
        pc = p[p.basis == "cohort"]
        if len(pa) and len(pc):
            a, sa = float(pa.iloc[0]["coef"]), float(pa.iloc[0]["se"])
            c = float(pc.iloc[0]["coef"])
            lo, hi = a - 1.96 * sa, a + 1.96 * sa
            inside = lo <= c <= hi
            out.append(
                f"{young}: pooled {a:+.4f} ({sa:.4f}) on age bands, "
                f"{c:+.4f} on cohorts. "
                + ("HEADLINE ROBUST, the cohort estimate is inside the "
                   "age-band interval."
                   if inside else
                   "HEADLINE MOVES: the cohort estimate is outside the "
                   "age-band 95 per cent interval, so the result depends "
                   "on the window moving. This is itself a finding."))
        else:
            out.append(f"{young}: pooled verdict UNAVAILABLE")
    return out


def main():
    mc.Tee(OUT / "69_log.txt")
    t0 = time.time()
    print("=" * 70)
    print("69 cohort basis: fixed birth cohorts vs moving age bands")
    print(f"cohorts fixed on ages in {REF_YEAR}:")
    for b in YOUNG_BANDS + INCUMBENT_BANDS:
        y0, y1 = cohort_years(b)
        print(f"    {b:<7} born {y0}-{y1}")
    print("=" * 70)

    j47 = _mod("47j_within_employer_triple.py", "j47")
    l47 = _mod("47L_age_baseline_exposure.py", "l47")

    # The connection is opened only if a cohort pull is actually needed.
    # With every year cached this script touches no database at all, which
    # is also what lets it run under the local dry-run harness.
    conn = {"h": None}

    def db():
        if conn["h"] is None:
            conn["h"] = mc.connect()
        return conn["h"]

    # exposure: 47j's incumbent education mix, exactly as 61 uses it
    expo = edu_exposure(j47, DESIGN, ARM)
    print(f"  exposure: {len(expo):,} firms classified")

    # counts on both bases
    ab_counts, co_counts = [], []
    for y in PANEL_YEARS:
        # READ ONLY. 70 runs in a parallel MONA slot off the same cache,
        # and two jobs writing one parquet is a race we do not need: these
        # were built by 47L and consumed by 61, 63 and 67, so they exist.
        cf = CACHE / f"L_counts_{y}.parquet"
        c = mc.read_cache(cf)
        if c is None:
            raise SystemExit(
                f"L_counts_{y}.parquet is not cached. 69 does not build it, "
                f"because 70 reads the same file from another slot. Run 47L "
                f"first, alone, then resubmit this lane.")
        print(f"  ageband counts {y}: cached ({len(c):,} cells)")
        ab_counts.append(c)

        cf2 = CACHE / f"L_cohort_{y}.parquet"
        c2 = mc.read_cache(cf2)
        if c2 is None:
            t = time.time()
            c2 = q_cohort_counts(y, db())
            mc.write_cache(c2, cf2)
            print(f"  cohort  counts {y}: {len(c2):,} cells "
                  f"({time.time()-t:.0f}s)")
        else:
            print(f"  cohort  counts {y}: cached ({len(c2):,} cells)")
        co_counts.append(c2)

    ab = pd.concat(ab_counts, ignore_index=True)
    co = pd.concat(co_counts, ignore_index=True)
    del ab_counts, co_counts
    gc.collect()

    last = max(str(ab["year_month"].max()), str(co["year_month"].max()))
    if last < POOLED_FROM:
        raise SystemExit(
            f"counts end at {last}, before the adoption window opens at "
            f"{POOLED_FROM}: the pooled coefficient would be estimated on "
            f"no data. Refusing to run.")

    sink, season_sink = [], []
    for young in YOUNG_BANDS:
        for basis, counts in (("ageband", ab), ("cohort", co)):
            opt(f"{basis}/{young}", run_basis, counts, basis, young, expo,
                j47, sink, season_sink)

    pooled = pd.DataFrame(sink)
    season = pd.DataFrame(season_sink)
    if len(pooled):
        pooled.to_csv(OUT / "cohort_pooled.csv", index=False)
    if len(season):
        season.to_csv(OUT / "cohort_season.csv", index=False)

    lines = ["69 cohort basis", "=" * 70, "",
             f"cohorts fixed on ages in {REF_YEAR}; panel {PANEL_FROM} to "
             f"{last}; pooled from {POOLED_FROM}; seasonal fitted on the "
             f"pre-launch window to {PRE_TO}.", ""]
    if len(pooled) and len(season):
        lines += verdict(pooled, season)
    else:
        lines.append("NO VERDICT: one or both bases produced no fit.")
    if FAILURES:
        lines += ["", "FAILED FITS:"] + [f"  {f}" for f in FAILURES]
    lines += ["", "Cost of this design, stated rather than hidden: the "
              "cohort follows one group of people as they age, from 21-24 "
              f"in 2021 to 25-28 in 2025. It cannot speak about whoever is "
              "22-25 at each date. It is a diagnostic about whether the "
              "seasonal is mechanical, and a robustness check on the "
              "headline, not a replacement estimand."]
    (OUT / "69_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    mc.runlog("69_cohort_basis", 0, (time.time() - t0) / 60)
    print(f"\ndone in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
