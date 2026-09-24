#!/usr/bin/env python3
"""
85_occupation_route_plain_profile.py: the age profile with and without
the calendar terms, the descriptive counterpart, and the oldest band
split at 65, all on script 82's occupation-route score.

======================================================================
  RUNS IN MONA. Parts are chosen with CANARIES_85_PARTS (default PDS: P
  the profile arms, D the descriptive counterpart, S the oldest band
  split at 65) and the output folder with CANARIES_85_OUT (default
  output_85); master.py sets both. No database connection is needed once
  script 82 has cached the occupation cascade and the 2019 counts, and
  script 78 the counts with the oldest band split.
======================================================================

QUESTION
Part P. Figure 2 of the paper draws each age band against 41-49 twice:
on the paper's specification, with three quarter-of-year interactions per
band, and without them. The pair shows what the calendar terms remove.
Both arms are fitted here on one panel and one score.

Part D. The descriptive counterpart of the design: total headcount and
the mean headcount per populated employer-band-month, by exposure
quartile, age band and window (January 2022 to December 2023 against
January 2024 onward), with no controls of any kind.

Part S. Is the gain of the oldest band a pension-age effect? Script 78's
Part E on this score: the seven-band profile with 50 and over split into
50-64 and 65-69, each against 41-49, calendar terms in.

BOTH ARMS OF PART P ARE FITTED HERE, AND THAT IS THE POINT OF THE GATE
The arm with the calendar terms already exists in script 82's
occ_route_profile.csv. It is refitted anyway: the two series of a figure
must sit on one panel, and the only way to know that is to fit them in
one job on one frame. The refitted seasonal arm is then checked against
script 82's exported coefficients to four decimals. IF THE CHECK FAILS,
THE PANEL IS NOT THE ONE THE PAPER REPORTS AND NEITHER ARM IS QUOTED; the
summary says so at the top in those words.

THE SCORE IS SCRIPT 82'S AND IS NOT REBUILT
The quartile comes from 82_occupation_route.build_exposure(), the
primary arm: uniform3, the backward cascade, a floor of five incumbent
person-months. The profile terms come from 74's own build_terms, called
with seasonal False and True rather than copied, so the plain arm is the
paper's specification minus the calendar terms and nothing else. The
descriptive cells come from 66's describe() and the split from 78's
part_e.

READ RULES, fixed before the run and printed at the start and in the
summary.

  1. THE GATE. The seasonal arm must reproduce script 82's profile to
     four decimals at every band. A moved coefficient means a moved
     panel, and nothing from either arm is quoted.
  2. THE PLAIN ARM CARRIES NO VERDICT. It is not a rival estimate of the
     profile. The calendar terms are in the reported specification
     because the exposure-differential ratio has a seasonal cycle
     present before any treatment; an arm without them inherits that
     cycle.
  3. The difference between the arms is reported at every band,
     whichever way it falls, and the summary names any band where the
     two disagree in sign.
  4. The descriptive counterpart carries no verdict.
  5. THE PENSION-AGE RIVAL IS DISMISSED if the 50-64 half gains against
     41-49 and is distinguishable from zero at five per cent.

INPUTS AND OUTPUTS
Reads, through the modules it imports: L_baseline_2019_cascade and
L_baseline_2019 (script 82), L_counts_2019 (script 47L, cached by script
82), L_counts_2021 to 2025 (script 47L) and L_counts_split_2021 to 2025
(script 78). Reads script 82's occ_route_profile.csv for the gate if it
is on the share; if it is not, the gate cannot run and the summary says
so rather than passing. Performs no SQL of its own: a missing split-count
cache skips Part S rather than starting a pull.

Writes to output_85/: occ_route_profile_arms.csv (arm, band, coef, se,
t, n_firms, n_obs, status), occ_route_descriptive_full.csv,
occ_route_split65.csv, the vcov_s85_*.csv and vcov_s78_prof_split.csv
files and 85_summary.txt.

IN THE PAPER
Figure 2, both series (Part P); Online Appendix III.1, Table A9 (Part D);
Online Appendix III.2, Table A13, second column, and the pension-age
paragraph of Section 3 (Part S).
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
OUT = HERE / os.environ.get("CANARIES_85_OUT", "output_85")
# P the profile arms, D the descriptive counterpart, S the oldest band
# split at 65. Each is independent; a part that dies costs that part.
PARTS = os.environ.get("CANARIES_85_PARTS", "PDS").upper()
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR
os.environ.setdefault("CANARIES_82_OUT", str(OUT))

FLOOR = 5                        # the export floor, as in mona_common
MATCH_DP = 4                     # the gate, decimals, as in 80
PROFILE_REF = "41-49"            # the omitted band
# Where script 82's profile may be found, for the gate. The first that
# exists is used; a run that finds none reports NO GATE.
PRIOR = ("output_82b", "output_84a", "output_84", ".")
PRIOR_FILE = "occ_route_profile.csv"
# The education route's own pair is printed in the summary for
# comparison and enters no export (EDU_ARMS below).
# The descriptive windows, as script 66 fixes them.
PRE = ("2022-01", "2023-12")
POST_FROM = "2024-01"
# The two windows are of unequal length, which is why nothing compares
# their sums without dividing first: 2022-01..2023-12 and 2024-01..2025-06.
MONTHS_PRE, MONTHS_POST = 24, 18
# The education route's own split at 65, for the summary alone.
EDU_SPLIT = {"50-64": (+0.0343, 0.0038), "65-69": (+0.1672, 0.0313)}
EDU_ARMS = {"22-25": (-0.0288, 0.0130, -0.0099, 0.0121),
            "26-30": (-0.0146, 0.0086, -0.0096, 0.0084),
            "50+": (+0.0616, 0.0065, +0.0589, 0.0062)}

NOTES, FAILURES = [], []

READ_RULES = [
    "READ RULES, FIXED BEFORE THE RUN:",
    "  1. THE GATE. The arm WITH the calendar terms must reproduce lane",
    f"     28b's profile to {MATCH_DP} decimals at every band. A moved",
    "     coefficient means a moved panel, and then NEITHER ARM IS",
    "     QUOTED and the figure is not drawn.",
    "  2. THE PLAIN ARM CARRIES NO VERDICT. It is not a rival estimate",
    "     of the profile. The calendar terms are in the reported",
    "     specification because the exposure-differential ratio has a",
    "     seasonal cycle present before any treatment, and an arm",
    "     without them inherits it. The arm exists to show the reader",
    "     how much of the profile the control removes.",
    "  3. The difference between the arms is reported at every band",
    "     whichever way it falls, and any band where the two disagree",
    "     in SIGN is named.",
    "  4. THE DESCRIPTIVE COUNTERPART CARRIES NO VERDICT. It is raw",
    "     totals and raw per-cell means by quartile and age band, with",
     "     composition, firm size and the business cycle inside them.",
    "     It exists so the appendix table can be built on the score the",
    "     paper reports instead of on the education route.",
    "  5. THE PENSION-AGE RIVAL IS DISMISSED if, with the oldest band",
    "     split at 65, the 50-64 half gains against 41-49 and is",
    "     distinguishable from zero at five per cent: the half the",
    "     retirement-age reforms do not reach then gains on its own.",
    "     The education route gives +0.0343 (0.0038) at 50-64 and",
    "     +0.1672 (0.0313) at 65-69.",
    "  On the education route the pair ran -0.0288 (0.0130) plain and",
    "  -0.0099 (0.0121) with the cycle removed at 22-25.",
    f"  Employer counts below {FLOOR} are suppressed before anything",
    "  leaves MONA.",
]


def _mod(fname: str, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def cnt(v) -> str:
    if v is None or v != v:
        return "(suppressed)"
    v = int(v)
    return "(suppressed)" if 0 < v < FLOOR else f"{v:,}"


def tstat(c, s) -> float:
    return float(c) / s if s and s == s and s > 0 else np.nan


def save(rows, name: str, count_col: str = "n_firms") -> pd.DataFrame:
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if not df.empty and count_col in df.columns:
        df = mc.enforce_min_cell(df, count_col=count_col, floor=FLOOR)
    df.to_csv(OUT / name, index=False)
    return df


def fit(b: pd.DataFrame, tag: str, terms: list, fes: tuple):
    print(f"    {tag}: {len(b):,} rows, {b['employer_id'].nunique():,} firms"
          f"{mc.mem_line(' | ')}")
    t = time.time()
    try:
        r = mc.run_fepois_multi(b, OUT, tag=f"s85_{tag}", terms=terms,
                                fes=fes, cluster="employer_id")
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


def load_counts(prefix: str, years):
    out = []
    for y in years:
        c = mc.read_cache(CACHE / f"{prefix}_{y}.parquet")
        if c is None:
            return None
        out.append(c)
    return pd.concat(out, ignore_index=True) if out else None


def prior_profile() -> pd.DataFrame:
    """Script 82's profile, for the gate."""
    for d in PRIOR:
        p = HERE / d / PRIOR_FILE
        if p.exists():
            print(f"  the gate reads {p}")
            NOTES.append(f"the gate read lane 28b's profile from {d}")
            return pd.read_csv(p)
    FAILURES.append("no prior profile for the gate")
    print("  NO PRIOR PROFILE FOUND: the gate cannot run")
    NOTES.append("lane 28b's occ_route_profile.csv was not on the share, so "
                 "the four-decimal gate could not run; the seasonal arm here "
                 "is unchecked against it")
    return pd.DataFrame()


def load_modules():
    s82 = _mod("82_occupation_route.py", "s82")
    s82.OUT = OUT
    s61, s67, s74, s78, s80, l47, l70, j47 = s82.load_modules()
    s66 = _mod("66_plain_magnitudes.py", "s66")
    for m_ in (s66, s78):
        m_.OUT, m_.CACHE = OUT, CACHE
    if tuple(s66.PRE) != tuple(PRE) or s66.POST_FROM != POST_FROM:
        raise RuntimeError(f"66's descriptive windows are {s66.PRE} and "
                           f"{s66.POST_FROM}, not {PRE} and {POST_FROM}; "
                           f"refusing to run.")
    if s74.POOLED_FROM != l70.POOLED_FROM:
        raise RuntimeError("74 and 70 disagree on when the post period "
                           "opens; refusing to run.")
    if l70.REF_BAND != PROFILE_REF:
        raise RuntimeError(f"70's reference band is {l70.REF_BAND} and this "
                           f"script's is {PROFILE_REF}; refusing to run.")
    if s82.MAIN_LEVEL != "uniform3" or s82.MAIN_ARM != "backward" \
            or s82.FLOOR_MAIN != FLOOR:
        raise RuntimeError("82's primary arm is not the one the paper "
                           "reports; refusing to run.")
    return s82, s61, s66, s74, s78, l47, l70, j47


def part_descriptive(counts, occ, s66, j47) -> pd.DataFrame:
    """
    The descriptive counterpart of the design, on the reported score.

    Script 66's own describe(), which returns the total and the per-cell
    mean by quartile, age band and window. Script 83's Part A exports
    only the means; this writes the whole frame, totals included, which
    Panel A of the appendix table needs.

    No fit, no verdict: raw totals and raw means, with composition, firm
    size and the business cycle inside them.
    """
    g = s66.describe(counts, occ, "n_emp", j47.YOUNG_BANDS,
                     j47.INCUMBENT_BANDS)
    if g is None or g.empty:
        FAILURES.append("D/empty")
        return pd.DataFrame()
    save(g, "occ_route_descriptive_full.csv")
    piv = g.pivot_table(index=["fq", "age_group"], columns="period",
                        values="total")
    if {"pre", "post"} <= set(piv.columns):
        # The totals are WINDOW SUMS: 24 months before and 18 after, so a
        # ratio of them carries a mechanical -25 per cent. The appendix
        # table divides by the month counts before it computes anything;
        # this line does the same rather than printing a change that has
        # to be corrected by whoever reads the log.
        piv["change"] = (piv["post"] / MONTHS_POST) / \
                        (piv["pre"] / MONTHS_PRE) - 1.0
        print("  D: headcount per month, change from the pre window:")
        for (fq, band), r in piv.iterrows():
            print(f"      Q{int(fq)} {band:6s} {float(r['change']):+.1%}")
    return g


def part_split65(occ, s61, s78, j47) -> list:
    """
    The oldest band split at 65, on the reported score.

    Script 78's own part_e: the seven-band panel, the reference 41-49,
    74's seasonal terms. The counts with the oldest band split are 78's
    own pull and are cached from script 78's run, so this is one fit and
    no SQL. What it settles is whether the gain of the oldest band is a
    retirement-age effect: if the 50-64 half, which the 2020 and 2023
    reforms do not reach, gains on its own, it is not.

    78 writes its rows to prof_split.csv under its own OUT. That name
    belongs to the education route's version, so the rows are written
    here under a name of their own and the stray file is removed: two
    exposure routes must never share an export name.
    """
    # 78's part_e pulls the split counts if they are not cached, which is
    # a full read of every monthly declaration. This script performs no SQL,
    # so the caches are checked first and the arm is skipped rather than
    # allowed to start a pull nobody scheduled.
    missing = [y for y in s61.PANEL_YEARS
               if not (CACHE / f"L_counts_split_{y}.parquet").exists()]
    if missing:
        FAILURES.append(f"S/split counts not cached for {missing}")
        print(f"  S: L_counts_split_* missing for {missing}; the split at 65 "
              f"is skipped rather than pulled")
        NOTES.append("the oldest band split at 65 was skipped: its counts are "
                     "not on the share and this lane does not pull")
        return []
    rows = s78.part_e(occ, s61, j47)
    for n in list(getattr(s78, "FAILURES", [])):
        FAILURES.append(f"78/{n}")
    stray = OUT / "prof_split.csv"
    if stray.exists():
        stray.unlink()
    if not rows:
        FAILURES.append("S/no rows")
        return []
    for r in rows:
        r["t"] = tstat(r.get("coef"), r.get("se"))
    save(rows, "occ_route_split65.csv")
    print("  S: the oldest band split at 65:")
    for r in rows:
        if r.get("se", 0) > 0:
            print(f"      {r['band']:6s} {r['coef']:+.4f} ({r['se']:.4f}) "
                  f"t {r['t']:+.2f}")
    return rows


def split_verdict(rows: list) -> tuple:
    """Rule 5: the half the reforms do not reach must gain on its own."""
    r = [x for x in rows if x.get("band") == "50-64" and x.get("se", 0) > 0]
    if not r:
        return "NO VERDICT, the 50-64 band did not fit", []
    r = r[0]
    ok = r["coef"] > 0 and abs(r["t"]) >= SIG5
    v = ("THE PENSION-AGE RIVAL IS DISMISSED" if ok
         else "THE PENSION-AGE RIVAL IS NOT DISMISSED")
    lines = [f"  5 the split at 65          {v}",
             f"     50-64 {r['coef']:+.4f} ({r['se']:.4f}) t {r['t']:+.2f}; "
             f"the education route gives {EDU_SPLIT['50-64'][0]:+.4f} "
             f"({EDU_SPLIT['50-64'][1]:.4f})"]
    o = [x for x in rows if x.get("band") == "65-69" and x.get("se", 0) > 0]
    if o:
        lines.append(f"     65-69 {o[0]['coef']:+.4f} ({o[0]['se']:.4f}); "
                     f"the education route gives "
                     f"{EDU_SPLIT['65-69'][0]:+.4f} "
                     f"({EDU_SPLIT['65-69'][1]:.4f})")
    return v, lines


def main():
    mc.Tee(OUT / "85_log.txt")
    t0 = time.time()
    print("=" * 70)
    print(f"85: THE PROFILE ARMS, THE DESCRIPTIVE AND THE SPLIT AT 65"
          f"   parts {PARTS}")
    print("=" * 70)
    print("\n".join(READ_RULES))
    print(mc.mem_line("  "))

    if not set(PARTS) & set("PDS"):
        raise RuntimeError(f"CANARIES_85_PARTS={PARTS} selects no part.")
    s82, s61, s66, s74, s78, l47, l70, j47 = load_modules()
    built = s82.build_exposure(l47, l70, j47)
    occ = built["exposure"]
    for n in list(getattr(s82, "NOTES", [])):
        NOTES.append(f"82: {n}")
    print(f"  the score: {len(occ):,} employers on the {built['arm']} arm "
          f"at a floor of {built['floor']} {built['basis']}")

    counts = load_counts("L_counts", s61.PANEL_YEARS)
    if counts is None:
        raise RuntimeError("L_counts_* missing: run 47L first.")
    print(f"  counts: {len(counts):,} employer-age-months")

    b0, n_firms = pd.DataFrame(), 0
    if "P" in PARTS:
        skel = l70.all_band_skeleton(counts)
        if skel.empty:
            raise RuntimeError("the six-band skeleton is empty; refusing to "
                               "run.")
        b0 = s78.with_exposure(skel, occ)
        del skel
        gc.collect()
        if b0.empty:
            raise RuntimeError("no employer on the six-band panel carries the "
                               "exposure quartile; refusing to run.")
        n_firms = int(b0["employer_id"].nunique())
        print(f"  the six-band panel: {n_firms:,} employers, "
              f"{len(b0):,} cells")

    dsc = part_descriptive(counts, occ, s66, j47) if "D" in PARTS \
        else pd.DataFrame()
    split = part_split65(occ, s61, s78, j47) if "S" in PARTS else []

    rows = []
    for arm, seasonal in (("seasonal", True), ("plain", False)) \
            if "P" in PARTS else ():
        b, terms = s74.build_terms(b0, l70, seasonal=seasonal)
        g = fit(b, f"profile_{arm}", terms, j47.FES)
        if g is None:
            continue
        for band in s74.BANDS:
            if band == PROFILE_REF:
                rows.append({"arm": arm, "band": band, "coef": 0.0,
                             "se": 0.0, "t": np.nan, "n_firms": n_firms,
                             "n_obs": int(g["n_obs"].max()),
                             "status": "reference"})
                continue
            t_ = l70.band_col("gpt_x_high", band)
            if t_ not in g.index:
                FAILURES.append(f"{arm}/{band}/no term")
                continue
            c, se = float(g.loc[t_, "coef"]), float(g.loc[t_, "se"])
            rows.append({"arm": arm, "band": band, "coef": c, "se": se,
                         "t": tstat(c, se), "n_firms": n_firms,
                         "n_obs": int(g.loc[t_, "n_obs"]),
                         "status": str(g.loc[t_].get("status", "ok"))})
        save(rows, "occ_route_profile_arms.csv")
        print(f"    {arm}:")
        for r in [x for x in rows if x["arm"] == arm and x["se"] > 0]:
            print(f"      {r['band']:6s} {r['coef']:+.4f} ({r['se']:.4f}) "
                  f"t {r['t']:+.2f}")

    # ---- the gate ----------------------------------------------------
    prior = prior_profile()
    gate, gate_lines = "NO GATE", []
    if not prior.empty:
        moved = []
        for r in [x for x in rows if x["arm"] == "seasonal"
                  and x["status"] == "ok"]:
            p = prior[prior["band"] == r["band"]]
            if not len(p):
                moved.append(f"{r['band']} absent from lane 28b")
                continue
            d = abs(float(p["coef"].iloc[0]) - r["coef"])
            if d >= 10 ** (-MATCH_DP) / 2:
                moved.append(f"{r['band']} {float(p['coef'].iloc[0]):+.4f} "
                             f"against {r['coef']:+.4f}")
        gate = "THE PANEL IS THE ONE LANE 28b FITTED" if not moved \
            else "THE PANEL HAS MOVED: NEITHER ARM IS QUOTED"
        gate_lines = [f"  {gate}"] + [f"    {m}" for m in moved]
        if moved:
            FAILURES.append("gate")

    # ---- what the calendar terms do ----------------------------------
    diff_lines, flips = [], []
    for band in s74.BANDS:
        if band == PROFILE_REF:
            continue
        p = [x for x in rows if x["arm"] == "plain" and x["band"] == band]
        s = [x for x in rows if x["arm"] == "seasonal" and x["band"] == band]
        if not p or not s:
            continue
        p, s = p[0], s[0]
        if p["coef"] * s["coef"] < 0:
            flips.append(band)
        diff_lines.append(
            f"  {band:6s} plain {p['coef']:+.4f} ({p['se']:.4f}) t "
            f"{p['t']:+.2f}   cycle removed {s['coef']:+.4f} ({s['se']:.4f}) "
            f"t {s['t']:+.2f}   the terms move it {s['coef']-p['coef']:+.4f}")

    if split:
        sv, slines = split_verdict(split)
        print("\nTHE OLDEST BAND SPLIT AT 65:")
        print("\n".join(slines))
    print("\nTHE GATE:")
    print("\n".join(gate_lines) or "  not run")
    print("\nWHAT THE CALENDAR TERMS DO:")
    print("\n".join(diff_lines))
    if flips:
        print(f"  THE TWO ARMS DISAGREE IN SIGN AT: {', '.join(flips)}")

    L = ["THE AGE PROFILE WITH AND WITHOUT THE CALENDAR TERMS",
         "=" * 52, "",
         "Figure 2 draws each band against 41-49 twice, on the paper's",
         "specification and without its three quarter-of-year terms per",
         "band. Lane 28 fitted only the first. This is the second, on the",
         "same score and in one job with a refit of the first, so that both",
         "series of the figure sit on one panel and are known to.", "",
         f"The six-band panel: {cnt(n_firms)} employers.", "",
         "THE GATE:"]
    L += gate_lines or ["  not run"]
    L += ["", "WHAT THE CALENDAR TERMS DO:"] + diff_lines
    if len(dsc):
        L += ["", "THE DESCRIPTIVE COUNTERPART: "
              f"{len(dsc):,} quartile-band-window cells exported with the "
              "total and the per-cell mean, so the appendix table can be "
              "built on this score. No verdict: raw numbers, with "
              "composition, firm size and the cycle inside them."]
    if split:
        sv, slines = split_verdict(split)
        L += ["", "THE OLDEST BAND SPLIT AT 65:"] + slines
    if flips:
        L.append(f"  THE TWO ARMS DISAGREE IN SIGN AT: {', '.join(flips)}")
    L += ["", "THE EDUCATION ROUTE'S OWN PAIR, for reference and not for "
          "export:"]
    for band, (pc, ps, sc, ss) in EDU_ARMS.items():
        L.append(f"  {band:6s} plain {pc:+.4f} ({ps:.4f})   cycle removed "
                 f"{sc:+.4f} ({ss:.4f})")
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
    L.append(f"Runtime {(time.time()-t0)/60:.1f} min. {mc.mem_line('')}")
    (OUT / "85_summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n  wrote {OUT / '85_summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
