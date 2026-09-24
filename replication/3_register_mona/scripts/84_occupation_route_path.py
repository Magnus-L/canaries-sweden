#!/usr/bin/env python3
"""
84_occupation_route_path.py: the quarterly path of Figure 3, on script
82's occupation-route score.

======================================================================
  RUNS IN MONA. Shapes are chosen with the environment variable
  CANARIES_84_SHAPES (default QM: Q the quarterly path, M the monthly
  diagnostic) and the folder with CANARIES_84_OUT (default output_84);
  master.py sets both. No database connection is needed once script 82
  has cached the occupation cascade and the 2019 counts.
======================================================================

QUESTION
When did the young-to-older ratio inside exposed employers move? The
pooled estimates of script 82 average over the adoption window; this
script decomposes the same fit into one coefficient per calendar quarter,
on the same occupation-route score, so that the timing evidence and
Table 1 rest on one definition of the treatment.

THE SCORE IS SCRIPT 82'S AND IS NOT REBUILT HERE
The quartile comes from 82_occupation_route.build_exposure(), the
primary arm: uniform3, the backward cascade, a floor of five incumbent
person-months. One definition of the treatment variable, in one place.
The cascade pull behind it is cached, so this script reads it rather
than pulling again.

THE TERMS ARE SCRIPT 68'S AND ARE NOT REBUILT EITHER
add_seasonal_terms(b, "quarter") and (b, "month") are called on 68's own
module. Copying the term list here would let the path drift away from
the specification the pooled estimates come from, and the point of the
figure is that it decomposes the same fit. The Riksbank interaction
stays on in every path fit, so each path coefficient is a step from the
level of the tightening months; the figure's caption states that
reading.

WHAT IS ESTIMATED
  Q  the quarterly path at 22-25 and at 26-30, every calendar quarter
     from 2022Q4 onward, with the three quarter-of-year terms removing
     the cycle and the fourth quarter omitted. Two fits. This is the
     figure, and the source of the fixed contrasts of Online Appendix
     Table A12 (calendar 2024 against calendar 2023, and the quarters on
     either side of the boundary).
  M  the monthly path at both bands, every month from December 2022
     onward, with eleven month-of-year terms and December omitted. Two
     fits. This is the diagnostic of Online Appendix III.2 (Figure A6)
     and it carries NO verdict: the calendar terms are at quarter
     frequency in the reported specification, so a monthly coefficient
     retains whatever separates the month from its own quarter's mean,
     and at 22-25 that residual is large.

Script 68 also fits a yearly path. No exhibit draws it and the paper
quotes no number from it, so it is not run here.

READ RULES, fixed before the run and printed at the start and in the
summary. There is NO coefficient gate: this is a decomposition of a fit
that has already been made and reported, not a new estimate of it.

  1. THE DATING REPRODUCES at 22-25 if no quarter before 2024 carries a
     negative coefficient distinguishable from zero at five per cent,
     AND at least one quarter from 2024 onward does. The rule is written
     on the year and not on the quarter, because a rule that named a
     particular quarter would be read off the figure it is supposed to
     judge.
  2. THE LAG REPRODUCES if the first quarter meeting that description at
     26-30 falls LATER than the first at 22-25.
  3. The monthly path carries NO verdict and is exported for the
     appendix diagnostic alone.
  Neither rule is a condition for quoting anything else: script 82's
  pooled estimates stand whatever the path does.

A CONSISTENCY LINE, REPORTED AND NOT GATED
The unweighted mean of the quarter coefficients from 2024 onward is
printed beside script 82's pooled adoption step of -0.0578. The two are
not the same statistic: the pooled term weights employer-months and the
mean does not, and the path's later quarters rest on fewer months. A
difference of a few thousandths is expected; a difference of the order
of the estimate itself would mean the two fits are not on one panel, and
the summary says so in those words if it appears.

INPUTS AND OUTPUTS
Reads, through the modules it imports: L_baseline_2019_cascade and
L_baseline_2019 (script 82), L_counts_2019 (script 47L, cached by script
82) and L_counts_2021 to 2025 (script 47L). Performs no SQL of its own;
if the cascade is absent it is pulled through 82's own query.

Writes to output_84/: occ_route_path.csv (young_band, shape, period,
coef, se, t, n_firms, n_obs, status), the vcov_s84_*.csv files and
84_summary.txt.

IN THE PAPER
Figure 3 of the paper; Online Appendix III.2, Table A12 and Figure A6.
"""

import gc
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mona_common as mc

HERE = Path(__file__).resolve().parent
OUT = HERE / os.environ.get("CANARIES_84_OUT", "output_84")
SHAPES_ARG = os.environ.get("CANARIES_84_SHAPES", "QM").upper()
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR
# 82 is imported for its score builder; its own OUT is pointed here so a
# run of this script leaves no stray output_82 folder behind.
os.environ.setdefault("CANARIES_82_OUT", str(OUT))

POST_FROM = "2024-01"            # adoption, as in 68, 75, 78, 80 and 83
BANDS = ["22-25", "26-30"]       # the two young bands the paper reports
FLOOR = 5                        # the export floor, as in mona_common
SIG5 = 1.959963984540054         # two-sided five per cent
DATING_YEAR = "2024"             # rule 1: no significant fall before this
SHAPES = [s for s, f in (("quarter", "Q"), ("month", "M")) if f in SHAPES_ARG]

# Script 82's pooled adoption step at 22-25 on this same score, for the
# consistency line. Not a gate.
POOLED_22 = -0.0578
# The education route's own dating, as the paper states it. The path
# numbers behind it are in script 68's export and are not copied here;
# what the verdicts are read against is the description, which is what
# the paper's sentences actually claim.
EDU_DATING = ("flat through 2023Q3, a dip in 2023Q4, and the shortfall "
              "opening in the second half of 2024, reaching 26-30 a year "
              "later")

NOTES, FAILURES = [], []

READ_RULES = [
    "READ RULES, FIXED BEFORE THE RUN:",
    "  There is NO coefficient gate. This is a decomposition of a fit",
    "  that has already been made and reported, not a new estimate of",
    "  it, and lane 28's pooled estimates stand whatever the path does.",
    "  What these rules decide is which sentences about TIMING survive.",
    "    1. THE DATING REPRODUCES at 22-25 if no quarter before",
    f"       {DATING_YEAR} carries a negative coefficient distinguishable",
    "       from zero at five per cent, AND at least one quarter from",
    f"       {DATING_YEAR} onward does. The rule is written on the year",
    "       and not on the quarter, because a rule naming 2024Q3 would",
    "       be read off the figure it is meant to judge.",
    "    2. THE LAG REPRODUCES if the first quarter meeting that",
    "       description at 26-30 falls LATER than the first at 22-25.",
    "       The paper's sentence is that the shortfall reaches the older",
    "       of the two young bands a year later; if they open together,",
    "       or the older one first, that sentence goes.",
    "    3. The monthly path carries NO verdict. The reported",
    "       specification removes the cycle at quarter frequency, so a",
    "       monthly coefficient retains whatever separates the month",
    "       from its own quarter's mean, and at 22-25 that residual is",
    "       large. It is exported for the appendix diagnostic alone.",
    f"  The education route reports {EDU_DATING}.",
    f"  Employer counts below {FLOOR} are suppressed before anything",
    "  leaves MONA.",
]


def opt(label, fn, *a, **kw):
    """Run one arm. An arm that dies is recorded and the others still
    run; nothing partial is silently treated as a result."""
    try:
        return fn(*a, **kw)
    except BaseException as ex:
        print(f"  [optional] {label} FAILED ({type(ex).__name__}: {ex})")
        traceback.print_exc()
        FAILURES.append(label)
        return None


def _mod(fname: str, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def cnt(v) -> str:
    """A count as the summary may print it: the export floor applies to
    the text that leaves MONA as much as to the CSV beside it."""
    if v is None or v != v:
        return "(suppressed)"
    v = int(v)
    return "(suppressed)" if 0 < v < FLOOR else f"{v:,}"


def save(rows, name: str, count_col: str = "n_firms") -> pd.DataFrame:
    """One export, with the suppression applied in the one place every
    table passes through."""
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if not df.empty and count_col in df.columns:
        df = mc.enforce_min_cell(df, count_col=count_col, floor=FLOOR)
    df.to_csv(OUT / name, index=False)
    return df


def tstat(coef, se) -> float:
    return float(coef) / se if se and se == se and se > 0 else np.nan


def fit(b: pd.DataFrame, tag: str, terms: list, fes: tuple,
        cluster: str = "employer_id"):
    """
    One Poisson fit. Returns the coefficient table indexed by term, or
    None. A failure is recorded and returns None, because a missing path
    point must never be drawn as a zero.
    """
    print(f"    {tag}: {len(b):,} rows, {b['employer_id'].nunique():,} firms"
          f"{mc.mem_line(' | ')}")
    t = time.time()
    try:
        r = mc.run_fepois_multi(b, OUT, tag=f"s84_{tag}", terms=terms,
                                fes=fes, cluster=cluster)
    except BaseException as ex:
        print(f"    {tag} FAILED: {type(ex).__name__}: {ex}")
        traceback.print_exc()
        r = pd.DataFrame()
    if r.empty:
        FAILURES.append(tag)
        print(f"    {tag}: FAILED, recorded and skipped")
        return None
    print(f"    {tag}: done in {(time.time()-t)/60:.1f} min")
    return r.set_index("term")


def load_counts(prefix: str, years, require=None):
    out = []
    for y in years:
        c = mc.read_cache(CACHE / f"{prefix}_{y}.parquet", require=require)
        if c is None:
            return None
        out.append(c)
    return pd.concat(out, ignore_index=True) if out else None


def drain(mod, tag: str) -> None:
    """Move an imported script's own notes and failures into ours."""
    for n in list(getattr(mod, "NOTES", [])):
        NOTES.append(f"{tag}: {n}")
    for f in list(getattr(mod, "FAILURES", [])):
        FAILURES.append(f"{tag}/{f}")


def load_modules():
    """
    The scripts this one reuses rather than reimplements: 82 for the
    score, 68 for the path terms, 61 for the balanced skeleton, 78 for
    the exposure merge, 47j for the fixed effects and the incumbent
    bands, 47L and 70 because 82's builder needs them.

    Three guards. Each is a place where this script's docstring would
    otherwise describe a specification it is not fitting.
    """
    s82 = _mod("82_occupation_route.py", "s82")
    s82.OUT = OUT
    s61, s67, s74, s78, s80, l47, l70, j47 = s82.load_modules()
    s68 = _mod("68_seasonal_control.py", "s68")
    for m_ in (s68,):
        m_.OUT, m_.CACHE = OUT, CACHE
    if s68.POST_FROM != POST_FROM:
        raise RuntimeError(f"68 dates adoption at {s68.POST_FROM} and this "
                           f"script at {POST_FROM}; refusing to run.")
    if s68.REF_QUARTER != 4:
        raise RuntimeError("68 omits a quarter other than the fourth; the "
                           "figure's reading would change.")
    if s82.MAIN_LEVEL != "uniform3" or s82.MAIN_ARM != "backward" \
            or s82.FLOOR_MAIN != FLOOR:
        raise RuntimeError("82's primary arm is not the one the paper "
                           "reports; refusing to run.")
    return s82, s61, s68, s78, l47, l70, j47


def path_rows(g, terms, shape: str, band: str, n_firms: int) -> list:
    """
    One fit's path coefficients, labelled as the figure reads them.

    The label is stripped exactly as script 68 strips it, so a row from
    this export and a row from the education route's carry the same
    period string and the two can be drawn on one pair of axes without a
    lookup table between them.
    """
    pref = {"quarter": "pq_", "month": "pm_"}[shape]
    out = []
    for t_ in terms:
        if not t_.startswith(pref) or t_ not in g.index:
            continue
        lab = (t_.split("_x_high")[0]
               .removeprefix("pq_").removeprefix("pm_"))
        c, se = float(g.loc[t_, "coef"]), float(g.loc[t_, "se"])
        out.append({"young_band": band, "shape": shape,
                    "period": lab.replace("_", "-"),
                    "coef": c, "se": se, "t": tstat(c, se),
                    "n_firms": n_firms,
                    "n_obs": int(g.loc[t_, "n_obs"]),
                    "status": str(g.loc[t_].get("status", "ok"))})
    return out


def first_fall(rows: list, band: str) -> str | None:
    """
    The first quarter whose coefficient is negative and distinguishable
    from zero at five per cent, or None. Quarters are compared as
    strings, which orders them correctly because the label is
    YYYYQn.
    """
    q = sorted((r for r in rows
                if r["young_band"] == band and r["shape"] == "quarter"
                and r["status"] == "ok"),
               key=lambda r: r["period"])
    for r in q:
        if r["coef"] < 0 and abs(r["t"]) >= SIG5:
            return r["period"]
    return None


def verdicts(rows: list) -> tuple:
    """The two dating rules, read on the quarterly path alone."""
    lines, got = [], {}
    if not any(r["shape"] == "quarter" for r in rows):
        return "NO QUARTERLY PATH", ["  the quarterly path produced nothing"]
    for band in BANDS:
        got[band] = first_fall(rows, band)
    early = [r["period"] for r in rows
             if r["young_band"] == BANDS[0] and r["shape"] == "quarter"
             and r["period"][:4] < DATING_YEAR and r["coef"] < 0
             and abs(r["t"]) >= SIG5]
    v1 = ("THE DATING REPRODUCES"
          if not early and got[BANDS[0]] is not None
          else "THE DATING DOES NOT REPRODUCE")
    lines.append(f"  1 the dating at {BANDS[0]}    {v1}")
    lines.append(f"     first quarter negative and distinguishable from "
                 f"zero: {got[BANDS[0]] or 'none'}")
    if early:
        lines.append(f"     quarters before {DATING_YEAR} that are already "
                     f"negative and significant: {', '.join(early)}")
    a, b = got[BANDS[0]], got[BANDS[1]]
    if a is None or b is None:
        v2 = "NO VERDICT, one band has no such quarter"
    elif b > a:
        v2 = "THE LAG REPRODUCES"
    else:
        v2 = "THE LAG DOES NOT REPRODUCE"
    lines.append(f"  2 the lag to {BANDS[1]}       {v2}")
    lines.append(f"     {BANDS[0]} opens at {a or 'none'}, "
                 f"{BANDS[1]} at {b or 'none'}")
    return v1, lines


def consistency(rows: list) -> str:
    """The unweighted mean of the 2024-onward quarters at 22-25, beside
    script 82's pooled step. Reported, never gated."""
    q = [r["coef"] for r in rows
         if r["young_band"] == BANDS[0] and r["shape"] == "quarter"
         and r["status"] == "ok" and r["period"][:4] >= DATING_YEAR]
    if not q:
        return "  no post-adoption quarters to average"
    m = float(np.mean(q))
    line = (f"  the unweighted mean of the {len(q)} quarters from "
            f"{DATING_YEAR} at {BANDS[0]} is {m:+.4f}, against lane 28b's "
            f"pooled adoption step of {POOLED_22:+.4f}")
    if abs(m - POOLED_22) > abs(POOLED_22):
        line += ("\n  THE TWO DIFFER BY MORE THAN THE ESTIMATE ITSELF, "
                 "which means the path and the pooled fit are not on one "
                 "panel; do not draw the figure until that is explained")
    else:
        line += (" (different statistics: the pooled term weights "
                 "employer-months and the mean does not)")
    return line


def main():
    mc.Tee(OUT / "84_log.txt")
    t0 = time.time()
    print("=" * 70)
    print(f"84: THE PATH OF FIGURE 3 ON THE OCCUPATION ROUTE   "
          f"shapes {','.join(SHAPES) or 'none'}")
    print("=" * 70)
    print("\n".join(READ_RULES))
    print(mc.mem_line("  "))
    if not SHAPES:
        raise RuntimeError("CANARIES_84_SHAPES selected no shape.")

    s82, s61, s68, s78, l47, l70, j47 = load_modules()

    # THE SCORE. Script 82's, built in script 82's own function, not rebuilt.
    built = s82.build_exposure(l47, l70, j47)
    occ = built["exposure"]
    drain(s82, "82")
    print(f"  the score: {len(occ):,} employers on the {built['arm']} arm "
          f"at a floor of {built['floor']} {built['basis']}")
    NOTES.append(f"the score is 82's build_exposure(): {len(occ):,} "
                 f"employers, arm {built['arm']}, floor {built['floor']} "
                 f"{built['basis']}")

    counts = load_counts("L_counts", s61.PANEL_YEARS)
    if counts is None:
        raise RuntimeError("L_counts_* missing: run 47L first.")
    last = str(counts["year_month"].max())
    if last < POST_FROM:
        raise RuntimeError(f"the counts end at {last} and the adoption "
                           f"window opens at {POST_FROM}; refusing to run.")
    print(f"  counts: {len(counts):,} employer-age-months, ending {last}")

    rows = []
    for band in BANDS:
        skel = s61.build_skeleton(counts, band, j47)
        if skel.empty:
            FAILURES.append(f"{band}/empty skeleton")
            continue
        b = s78.with_exposure(skel, occ)
        del skel
        gc.collect()
        if b.empty:
            FAILURES.append(f"{band}/no exposure")
            continue
        n_firms = int(b["employer_id"].nunique())
        print(f"  {band}: {n_firms:,} employers, {len(b):,} cells")
        for shape in SHAPES:
            # 68's own term builder, on 68's own module. The columns of
            # one shape do not disturb another: the fit reads the term
            # list, not every column in the frame.
            b, terms = s68.add_seasonal_terms(b, shape)
            g = fit(b, f"path_{shape}_{band.replace('-', '_')}", terms,
                    j47.FES)
            if g is None:
                continue
            got = path_rows(g, terms, shape, band, n_firms)
            if not got:
                FAILURES.append(f"{band}/{shape}/no path terms")
                continue
            rows += got
            save(rows, "occ_route_path.csv")
            print(f"    {shape} path, cycle removed:")
            for r in got:
                star = "*" if abs(r["t"]) >= SIG5 else " "
                print(f"          {r['period']:<8} {r['coef']:+.4f} "
                      f"({r['se']:.4f}) {star}")
        del b
        gc.collect()

    v1, vlines = verdicts(rows)
    cons = consistency(rows)
    print("\nTHE DATING:")
    print("\n".join(vlines))
    print(cons)

    L = ["THE PATH OF FIGURE 3, ON THE OCCUPATION ROUTE",
         "=" * 52, "",
         "Lanes 28 and 29 moved every register estimate in the paper onto",
         "a firm score built from the employer's own 2019 occupation mix.",
         "Figure 3, the quarterly path that dates the step, was the one",
         "exhibit still drawn on the education route. This is that figure's",
         "own fit, on the same score as Table 1, with script 68's terms and",
         "script 61's panel: the Riksbank interaction stays on, so every",
         "path coefficient is a step from the level of the tightening",
         "months, which is the reading the caption already states.", ""]
    for band in BANDS:
        q = [r for r in rows if r["young_band"] == band
             and r["shape"] == "quarter"]
        if not q:
            continue
        L.append(f"THE QUARTERLY PATH AT {band} "
                 f"({cnt(q[0]['n_firms'])} employers):")
        for r in sorted(q, key=lambda r: r["period"]):
            star = " *" if abs(r["t"]) >= SIG5 else ""
            L.append(f"  {r['period']:<8} {r['coef']:+.4f} "
                     f"({r['se']:.4f}){star}")
        L.append("")
    m = [r for r in rows if r["shape"] == "month"]
    if m:
        L.append(f"THE MONTHLY PATH: {len(m)} coefficients exported for the "
                 f"appendix diagnostic; it carries no verdict, because the")
        L.append("  reported specification removes the cycle at quarter "
                 "frequency and a month keeps what separates it from its")
        L.append("  own quarter's mean.")
        L.append("")
    L.append("THE DATING:")
    L += vlines
    L.append("")
    L.append("CONSISTENCY WITH THE POOLED ESTIMATE, reported and not gated:")
    L.append(cons)
    L.append("")
    if FAILURES:
        L.append(f"WHAT FAILED: {', '.join(FAILURES)}")
        L.append("A missing row is a missing fit, never a zero, and the "
                 "figure must not be drawn through it.")
        L.append("")
    if NOTES:
        L.append("NOTES:")
        L += [f"  {n}" for n in NOTES]
        L.append("")
    L += READ_RULES
    L.append("")
    L.append(f"Runtime {(time.time()-t0)/60:.1f} min. "
             f"{mc.mem_line('')}")
    (OUT / "84_summary.txt").write_text("\n".join(L) + "\n",
                                        encoding="utf-8")
    print(f"\n  wrote {OUT / '84_summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
