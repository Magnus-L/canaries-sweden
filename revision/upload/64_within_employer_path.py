#!/usr/bin/env python3
"""
64_within_employer_path.py -- show the path, do not argue about it.

======================================================================
  RUNS IN MONA. No SQL. Reads 47h's cached 2019 frame and 47L's cached
  monthly counts, exactly as 61 does. Writes output_64/.
======================================================================

WHY THIS IS THE LAST THING THE REVISION NEEDS.

Script 61 dated the treatment where SCB's adoption data put it and found
-0.0509 at 22-25 and -0.0437 at 26-30, with a register artefact of two
thirds of a percentage point. That is the paper's register evidence.

It also found a positive coefficient in the two months before January
2024 and a sharply negative one after. The boundary was pre-specified
from script 60 and the adoption window is measured against the
pre-period, so the reversal cannot mechanically produce the result, and
the same shape appears in both age bands and under both scoring rules.
All true, and all of it is reasoning about a path rather than a picture
of one. A referee will ask for the picture.

So this replaces the three windows with one coefficient per quarter. The
design, the exposure and the fixed effects are 61's, unchanged; only the
treatment's time shape changes, from a step function to a full set of
quarter interactions. That nests everything 61 estimated, and it answers
two questions a step function cannot:

  IS THE PRE-PERIOD FLAT? Six quarters run before the launch. If they
  drift, the design's parallel-trends assumption is in trouble and the
  headline goes with it. This is the first thing to read and it is a
  test the paper has not yet passed in this design.

  IS THE DECLINE A STEP OR A TREND? If the coefficients fall through
  2024 and 2025 the reading is a diffusing treatment. If everything
  happens in one quarter at the boundary we chose, it is a level shift
  at a date we selected, and the paper must say so.

THE READ RULE, FIXED BEFORE THE RESULTS EXIST.

  1. The pre-period is 2021Q1 to 2022Q3, with 2022Q3 as the omitted
     reference. It fails on either of two counts: any single quarter
     beyond 2.5 of its own standard errors, or a drift across the six
     whose slope is beyond 2 of its standard errors. Both criteria are
     in standard errors rather than in log points, because a fixed
     threshold in log points is a different test at every sample size.
     The two are complementary rather than nested, and neither
     dominates. Against a clean ramp over six quarters the
     single-quarter test is in fact the sharper of the two, since the
     endpoint of a ramp carries a larger t than its slope does. What
     the drift test adds is robustness in the other direction: one
     noisy quarter can breach a single-quarter threshold without any
     trend being present, and it moves the slope very little. We
     therefore require both, and report both, rather than choosing.
  2. 2023Q4 contains the two months 61 reported as positive and did not
     interpret. It is reported here and not interpreted either.
  3. A decline present in at least three of the five quarters from
     2024Q1 onward is a path. One quarter is not.
  4. The as-of arm is run for the youngest band, so the artefact is
     measured on the path and not assumed from the pooled fit.

HOW IT IS FAST. The skeleton is 61's and is built once per band, then the
quartile is merged in, so two panel builds serve every arm. Three fits.

Output (output_64/):
  path.csv        design x arm x band x quarter, coefficient and SE
  64_summary.txt
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
OUT = HERE / "output_64"
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR

REF_Q = "2022Q3"          # the quarter before the launch quarter
DESIGN = "OL_daioe"
JOBS = (("OL_daioe", "true", "22-25"),
        ("OL_daioe", "asof", "22-25"),
        ("OL_daioe", "true", "26-30"))
TRUNC = 2021
PRE_FLAT_T = 2.5          # read rule 1a, any single quarter, in its own SEs
PRE_SLOPE_T = 2.0         # read rule 1b, the drift across the six
FAILURES = []


def _mod(name, alias):
    import importlib.util
    spec = importlib.util.spec_from_file_location(alias, HERE / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def pre_slope(pre: pd.DataFrame) -> tuple:
    """
    Slope of the pre-period coefficients on quarter index, weighted by
    their own precision, with its t.

    The coefficients share an omitted reference quarter and are therefore
    correlated, which this ignores. That makes the standard error an
    approximation, and it is the right direction of approximation for a
    guard: the neglected covariance is positive, so the true slope is
    estimated at least as precisely as this reports and a drift we
    declare is not an artefact of the shortcut.
    """
    if len(pre) < 3:
        return float("nan"), float("nan")
    x = np.arange(len(pre), dtype=float)
    y = pre["coef"].to_numpy(dtype=float)
    w = 1.0 / np.clip(pre["se"].to_numpy(dtype=float), 1e-12, None) ** 2
    xb = np.average(x, weights=w)
    sxx = float(np.sum(w * (x - xb) ** 2))
    if sxx <= 0:
        return float("nan"), float("nan")
    b = float(np.sum(w * (x - xb) * y) / sxx)
    se = float(np.sqrt(1.0 / sxx))
    return b, b / se if se > 0 else float("nan")


def quarter(ym: pd.Series) -> pd.Series:
    """2024-02 -> 2024Q1. Kept as a string because it becomes a term name."""
    y = ym.str.slice(0, 4)
    m = ym.str.slice(5, 7).astype(int)
    return y + "Q" + (((m - 1) // 3) + 1).astype(str)


def add_quarter_terms(bal: pd.DataFrame) -> tuple:
    """
    One triple interaction per quarter, with REF_Q omitted.

    The Riksbank control that 61 carries is dropped here rather than
    forgotten: a full set of quarter interactions already spans it, and
    including both would be collinear.
    """
    q = quarter(bal["year_month"].astype(str))
    hy = bal["high"] * bal["young"]
    terms = []
    for qq in sorted(q.unique()):
        if qq == REF_Q:
            continue
        col = "q_" + qq
        bal[col] = (q == qq).astype(int) * hy
        terms.append(col)
    return bal, terms


def main():
    mc.Tee(OUT / "64_log.txt")
    t0 = time.time()
    print("=" * 70)
    print("64: THE WITHIN-EMPLOYER PATH, ONE COEFFICIENT PER QUARTER")
    print("=" * 70)
    print(f"  reference quarter {REF_Q}; the design, exposure and fixed")
    print("  effects are 61's and only the time shape changes.")
    print(f"  READ RULE 1: if any pre-period coefficient exceeds {PRE_FLAT_T}"
          " of its own standard errors, or the six trend monotonically,")
    print("  this does not support the headline.")
    print(mc.mem_line("  "))

    s61 = _mod("61_redated_triple.py", "s61")
    j47 = s61._j47()
    h47 = j47._h47()

    counts = {}
    for y in (2019, 2020, 2021):
        w = mc.read_cache(CACHE / f"edu_hr_weights_{y}.parquet",
                          require=h47.WEIGHT_COLS)
        if w is None:
            raise RuntimeError(f"edu_hr_weights_{y}.parquet missing: run 47h "
                               f"first. This script performs no SQL.")
        counts[y] = w
    book = h47.ScoreBook(counts, h47.load_key(), h47.load_scores())
    spec = dict(h47.DESIGNS[DESIGN])
    book.build(DESIGN, spec)

    frame19 = mc.read_cache(CACHE / "edu_hr_2019.parquet",
                            require=h47.YEAR_COLS + ["n_emp"])
    if frame19 is None:
        raise RuntimeError("edu_hr_2019.parquet missing: run 47h first.")

    cnt = []
    for y in s61.PANEL_YEARS:
        c = mc.read_cache(CACHE / f"L_counts_{y}.parquet")
        if c is None:
            raise RuntimeError(f"L_counts_{y}.parquet missing: run 47L first.")
        cnt.append(c)
    panel_counts = pd.concat(cnt, ignore_index=True)
    del cnt
    gc.collect()
    last = str(panel_counts["year_month"].max())
    print(f"  counts: {len(panel_counts):,} employer-age-months, ending {last}")
    if last < s61.POOLED_FROM:
        raise RuntimeError(
            f"the counts end at {last}, before the adoption window opens at "
            f"{s61.POOLED_FROM}. Refusing to run.")

    expos = {}
    for arm in sorted({a for _, a, _ in JOBS}):
        e, _ = j47.incumbent_exposure(frame19, book, DESIGN, spec, arm, TRUNC)
        expos[arm] = e
        print(f"  exposure {DESIGN} {arm}: {len(e):,} firms")
    del frame19
    gc.collect()

    rows = []
    for band in sorted({b for _, _, b in JOBS}):
        t1 = time.time()
        skel = s61.build_skeleton(panel_counts, band, j47)
        if skel.empty:
            print(f"  {band}: empty panel, skipped")
            continue
        print(f"\n  skeleton {band}: {len(skel):,} rows "
              f"({time.time()-t1:.0f}s, built once for every arm)")
        for nm, arm, b in JOBS:
            if b != band:
                continue
            t2 = time.time()
            bal = skel.merge(expos[arm][["employer_id", "fq"]],
                             on="employer_id", how="inner")
            if bal.empty:
                print(f"  {nm} {arm} {band}: no firms matched, skipped")
                continue
            bal["high"] = (bal["fq"] == 4).astype(int)
            bal, terms = add_quarter_terms(bal)
            r = mc.run_fepois_multi(bal, OUT, tag=f"p64_{arm}_{band.replace('-','_')}",
                                    terms=terms, fes=j47.FES)
            del bal
            gc.collect()
            if r.empty:
                FAILURES.append(f"{nm}/{arm}/{band}")
                print(f"  {nm} {arm} {band}: FAILED, recorded and skipped")
                continue
            for _, x in r.iterrows():
                rows.append({"design": nm, "arm": arm, "young_band": band,
                             "quarter": str(x["term"])[2:],
                             "coef": float(x["coef"]), "se": float(x["se"]),
                             "n_obs": int(x["n_obs"]),
                             "status": str(x.get("status", "ok"))})
            pd.DataFrame(rows).to_csv(OUT / "path.csv", index=False)
            print(f"  {nm} {arm} {band} [{(time.time()-t2)/60:.1f} min]")
            for h in sorted(set(d["quarter"] for d in rows
                                if d["arm"] == arm and d["young_band"] == band)):
                d = [x for x in rows if x["arm"] == arm
                     and x["young_band"] == band and x["quarter"] == h][0]
                t = d["coef"] / max(d["se"], 1e-12)
                print(f"    {h}  {d['coef']:+.4f} (SE {d['se']:.4f}) t {t:+.2f}")
        del skel
        gc.collect()

    P = pd.DataFrame(rows)
    lines = ["THE WITHIN-EMPLOYER PATH, QUARTER BY QUARTER", "=" * 52, "",
             "Young against older workers inside one employer in one month,",
             "exposure from the education mix of incumbents aged 31+ in 2019,",
             f"one coefficient per quarter with {REF_Q} omitted.", ""]
    for band in sorted(P["young_band"].unique()) if not P.empty else []:
        for arm in sorted(P[P.young_band == band]["arm"].unique()):
            d = P[(P.young_band == band) & (P.arm == arm)].sort_values("quarter")
            lines.append(f"{DESIGN}, {arm} codes, young = {band}:")
            for _, x in d.iterrows():
                star = "" if abs(x["coef"]) < 2 * x["se"] else "  *"
                lines.append(f"  {x['quarter']:<8} {x['coef']:+.4f} "
                             f"({x['se']:.4f}){star}")
            pre = d[d["quarter"] < REF_Q]
            post = d[d["quarter"] >= "2024Q1"]
            if not pre.empty:
                tt = (pre["coef"] / pre["se"].clip(lower=1e-12)).abs()
                worst = pre.loc[tt.idxmax()]
                b, bt = pre_slope(pre.sort_values("quarter"))
                bad = tt.max() > PRE_FLAT_T or abs(bt) > PRE_SLOPE_T
                lines.append(f"  pre-period {'NOT FLAT' if bad else 'FLAT'}: "
                             f"largest single quarter {worst['coef']:+.4f} "
                             f"({worst['se']:.4f}) at {worst['quarter']}, "
                             f"t {tt.max():.2f}; drift {b:+.4f} per quarter, "
                             f"t {bt:+.2f}")
            if not post.empty:
                neg = int((post["coef"] < 0).sum())
                sig = int((post["coef"] < -2 * post["se"]).sum())
                lines.append(f"  from 2024Q1: {neg} of {len(post)} quarters "
                             f"negative, {sig} of them beyond two standard "
                             f"errors")
            lines.append("")
    if not P.empty and {"true", "asof"} <= set(P["arm"]):
        t = P[(P.arm == "true") & (P.young_band == "22-25")].set_index("quarter")["coef"]
        a = P[(P.arm == "asof") & (P.young_band == "22-25")].set_index("quarter")["coef"]
        k = t.index.intersection(a.index)
        if len(k):
            gap = (a[k] - t[k]).abs().max()
            lines += [f"ARTEFACT ON THE PATH: the largest quarterly gap between",
                      f"the as-of and true arms at 22-25 is {gap:.4f}.", ""]
    if FAILURES:
        lines += ["FITS THAT FAILED: " + "; ".join(FAILURES),
                  "A missing quarter is a missing fit, never a zero.", ""]
    lines += [
        "READ THIS BEFORE QUOTING ANY OF IT:",
        "  1. The pre-period test comes first. A drifting pre-period ends",
        f"     the headline. Both thresholds, {PRE_FLAT_T} standard errors",
        f"     on any single quarter and {PRE_SLOPE_T} on the drift across",
        "     them, were fixed before these numbers existed.",
        "  2. 2023Q4 contains the two months script 61 reported as positive",
        "     and declined to interpret. The same applies here.",
        "  3. A decline in one quarter is not a path. Three of the five",
        "     quarters from 2024Q1 is.",
        "  4. Quarters are not independent draws, so counting stars across",
        "     them is not a test. The shape is the evidence.",
        "  5. 2025 is the preliminary AGI file and stops in June, so 2025Q2",
        "     is a two-month quarter.",
        "  6. Every coefficient is the exposed quartile MINUS the rest,",
        "     not what an exposed firm experienced. To split it, note that",
        "     the national average is the employment-weighted mean of the",
        "     two, so with the exposed quartile at a quarter of employment",
        "     an exposed firm sits three quarters of the coefficient below",
        "     that average and the rest sit one quarter above it. The",
        "     average itself is absorbed by the month-by-age effect and has",
        "     to come from the descriptive series, not from here.",
        "", f"Runtime {(time.time()-t0)/60:.1f} min. " + mc.mem_line()]
    (OUT / "64_summary.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print("\n64 done.")


if __name__ == "__main__":
    main()
