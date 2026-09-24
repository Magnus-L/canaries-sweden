#!/usr/bin/env python3
"""
86_occupation_route_prepath.py: the plain quarterly path from 2019 and the
pre-launch drift test, on script 82's occupation-route score.

======================================================================
  RUNS IN MONA. The output folder is CANARIES_86_OUT (default
  output_86); master.py sets it. No database connection is needed once
  script 82 has cached the occupation cascade and script 47L the counts
  for 2019 and 2020.
======================================================================

QUESTION
Online Appendix III.2 draws the quarterly path of the design on the plain
specification, 2019Q1 to 2025Q2 with 2022Q1 omitted and no calendar
terms (Figure A7 and Table A19), and reports the drift test on the
pre-launch months beneath it. Script 78's part A produces both; this
script runs it unchanged on the occupation-route score, so that the
figure, its table and the test beneath them rest on the score the paper
reports.

THE DRIFT REFIT IS THE CHECK
The drift test Table 1 prints is script 83's (occ_rest_drift.csv).
Running part A produces it again, because part A fits the path and the
drift together on one frame. Part A builds that frame from 2019-01 when
the 2019 and 2020 counts are cached and slices the pre-launch months out
of it, so its employer set is every employer appearing from 2019, a
superset of the 2021-start set behind script 83's fit, and the two cannot
agree to the fourth decimal. The check is therefore substantive
agreement: each trend within one standard error of script 83's and the
same flat verdict at both bands. The cell counts of the two frames are
printed side by side.

THE SCORE IS SCRIPT 82'S AND IS NOT REBUILT
The quartile comes from 82_occupation_route.build_exposure(), the primary
arm: uniform3, the backward cascade, a floor of five incumbent
person-months.

WHAT IS NOT CLAIMED
The path is a picture and not an estimate. Its quarters carry no
calendar terms, so each of them holds whatever separates it from its own
quarter of the year; that is the point of drawing it beside the
specification the paper reports, and it is why the paper reads the drift
test rather than any quarter of this path.

READ RULES, fixed before the run and printed at the start and in the
summary.

  1. THE CHECK. Each refitted trend within one standard error of script
     83's and the same flat verdict at both bands; otherwise the path is
     not drawn.
  2. THE PATH CARRIES NO VERDICT. No quarter of it is quoted in the
     paper.
  3. THE WINDOW IS WHATEVER THE COUNTS ALLOW. If L_counts_2019 and
     L_counts_2020 are on the share the path runs from 2019Q1, the window
     the appendix figure draws; if they are not it runs from 2021-01 and
     the summary says so. This script does NOT pull to extend it.
  4. The drift rule is 78's: the pre-period is flat if the trend lies
     within two standard errors of zero. Script 83 gives +0.000383
     (0.000773) at 22-25, flat, and +0.001608 (0.000436) at 26-30, not
     flat.

INPUTS AND OUTPUTS
Reads, through the modules it imports: L_baseline_2019_cascade and
L_baseline_2019 (script 82), and L_counts_2019 to L_counts_2025 (47L).
Reads script 83's occ_rest_drift.csv for the check if it is on the
share; if it is not, the check cannot run and the summary says so rather
than passing. Performs no SQL of its own.

Writes to output_86/: occ_route_prepath.csv (young_band, quarter, coef,
se, n_obs, status), occ_route_predrift.csv (the refit), the vcov files
and 86_summary.txt. Part A is 78's, so its fits are tagged s78_prepath_*
and s78_predrift_* and the vcov files carry those names; the two files
78 writes under its own export names are renamed here, because two
exposure routes must never share an export name.

IN THE PAPER
Online Appendix III.2, Figure A7 and Table A19, and the drift paragraph
beneath them.
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
OUT = HERE / os.environ.get("CANARIES_86_OUT", "output_86")
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR
os.environ.setdefault("CANARIES_82_OUT", str(OUT))

FLOOR = 5                        # the export floor, as in mona_common
REF_QUARTER = "2022Q1"           # the omitted quarter, as in 78
EXTENDED_FROM = "2019-01"        # the window the appendix figure draws
# Where script 83's drift may be found, for the check. The first that
# exists is used; a run that finds none reports NO GATE.
PRIOR = ("output_83b", "output_83", "output_86", ".")
PRIOR_FILE = "occ_rest_drift.csv"
TREND = "trend_x_high_x_young"
# Script 83's own numbers, for the summary alone. They enter no export.
PRIOR_TREND = {"22-25": (0.000383, 0.000773),
               "26-30": (0.001608, 0.000436)}

NOTES, FAILURES = [], []

READ_RULES = [
    "READ RULES, FIXED BEFORE THE RUN:",
    "  1. THE GATE. 78's part_a builds ONE frame for both halves and",
    "     slices the drift window out of it, so a path that runs from",
    "     2019 gives the drift a 2019-start employer set that CANNOT",
    "     reproduce lane 29b's 2021-start fit to four decimals. What is",
    "     required is substantive agreement: each trend within one",
    "     standard error of lane 29b's and the same flat verdict at both",
    "     bands. The two frames' cell counts are printed beside each",
    "     other. A trend that moves by more than a standard error, or a",
    "     verdict that flips, means THE PATH IS NOT DRAWN.",
    "  2. THE PATH CARRIES NO VERDICT. No quarter of it is quoted in the",
    "     paper. It exists so that a reader can see the series the",
    "     calendar terms are removed from.",
    "  3. THE WINDOW IS WHATEVER THE COUNTS ALLOW. With L_counts_2019 and",
    "     L_counts_2020 on the share the path runs from 2019Q1, the",
    "     window the appendix figure draws; without them it runs from",
    "     2021-01 and the summary says so. This lane does NOT pull to",
    "     extend it.",
    "  4. The drift rule is 78's: the pre-period is flat if the trend",
    "     lies within two standard errors of zero. Lane 29b gives",
    "     +0.000383 (0.000773) at 22-25, flat, and +0.001608 (0.000436)",
    "     at 26-30, not flat.",
    "  NOTHING HERE NEEDS SUPPRESSING. Neither export carries a count",
    "  of employers: a path row holds a coefficient and the number of",
    "  CELLS behind it, which is millions, and the drift rows the same.",
    "  The floor is still checked against 82's, because the score the",
    "  path is drawn on is built under it.",
]


def _mod(fname: str, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def drain(mod, tag: str) -> None:
    """Move an imported script's own notes and failures into ours."""
    for n in list(getattr(mod, "NOTES", [])):
        NOTES.append(f"{tag}: {n}")
    for f in list(getattr(mod, "FAILURES", [])):
        FAILURES.append(f"{tag}/{f}")
    if hasattr(mod, "NOTES"):
        mod.NOTES.clear()
    if hasattr(mod, "FAILURES"):
        mod.FAILURES.clear()


def load_counts(prefix: str, years):
    out = []
    for y in years:
        c = mc.read_cache(CACHE / f"{prefix}_{y}.parquet")
        if c is None:
            return None
        out.append(c)
    return pd.concat(out, ignore_index=True) if out else None


def load_modules():
    s82 = _mod("82_occupation_route.py", "s82")
    s82.OUT = OUT
    s61, s67, s74, s78, s80, l47, l70, j47 = s82.load_modules()
    # 82 already points 78's OUT and CACHE here; say it again rather than
    # rely on it, since part A writes two files of its own.
    s78.OUT, s78.CACHE = OUT, CACHE
    if s78.REF_QUARTER != REF_QUARTER:
        raise RuntimeError(f"78 omits {s78.REF_QUARTER} and this script "
                           f"names {REF_QUARTER}; refusing to run.")
    if s78.EXTENDED_FROM != EXTENDED_FROM:
        raise RuntimeError(f"78 extends from {s78.EXTENDED_FROM} and this "
                           f"script names {EXTENDED_FROM}; refusing to run.")
    if s82.MAIN_LEVEL != "uniform3" or s82.MAIN_ARM != "backward" \
            or s82.FLOOR_MAIN != FLOOR:
        raise RuntimeError("82's primary arm is not the one the paper "
                           "reports; refusing to run.")
    return s82, s61, s78, l47, l70, j47


def prior_drift() -> pd.DataFrame:
    """Script 83's drift, for the check."""
    for d in PRIOR:
        p = HERE / d / PRIOR_FILE
        if p.exists():
            print(f"  the gate reads {p}")
            NOTES.append(f"the gate read lane 29b's drift from {d}")
            return pd.read_csv(p)
    FAILURES.append("no prior drift for the gate")
    print("  NO PRIOR DRIFT FOUND: the gate cannot run")
    NOTES.append("lane 29b's occ_rest_drift.csv was not on the share, so the "
                 "four-decimal gate could not run and the refitted drift "
                 "here is unchecked against it")
    return pd.DataFrame()


def flat(coef: float, se: float) -> bool:
    """78's rule: the pre-period is flat within two standard errors."""
    return abs(coef) < 2 * se


def run_gate(drift_rows: list, prior: pd.DataFrame) -> tuple:
    """
    Rule 1, substantive agreement on the trend, and NOT a four-decimal
    reproduction.

    78's part_a builds one frame for the path and the drift together, so a
    path that runs from 2019 hands the drift an employer set script 83's
    2021-start fit does not share: about eight per cent more cells. A
    four-decimal reproduction is therefore not attainable by construction;
    what matters is whether the wider frame tells the same story.
    """
    if prior.empty or not drift_rows:
        return "NO GATE", []
    d = pd.DataFrame(drift_rows)
    bad, lines, checked = [], [], 0
    for band in sorted(d["young_band"].unique()):
        r = d[(d["young_band"] == band) & (d["term"] == TREND)]
        p = prior[(prior["young_band"] == band) & (prior["term"] == TREND)]
        if not len(r) or not len(p):
            bad.append(f"{band}: no trend term on one side")
            continue
        checked += 1
        c, se = float(r["coef"].iloc[0]), float(r["se"].iloc[0])
        pc, pse = float(p["coef"].iloc[0]), float(p["se"].iloc[0])
        gap = abs(c - pc)
        n_here = int(r["n_obs"].iloc[0])
        n_there = int(p["n_obs"].iloc[0]) if "n_obs" in p.columns else 0
        lines.append(f"    {band}: trend {c:+.5f} ({se:.5f}) against lane "
                     f"29b's {pc:+.5f} ({pse:.5f}), a gap of "
                     f"{gap / se:.2f} standard errors; "
                     f"{'FLAT' if flat(c, se) else 'NOT FLAT'} against "
                     f"{'FLAT' if flat(pc, pse) else 'NOT FLAT'}")
        lines.append(f"      cells {n_here:,} here against {n_there:,} "
                     f"there, the 2019-start frame against the 2021-start "
                     f"one, which is expected")
        if gap > se:
            bad.append(f"{band}: the trend moves {gap / se:.2f} standard "
                       f"errors")
        if flat(c, se) != flat(pc, pse):
            bad.append(f"{band}: the flat verdict flips")
    gate = ("THE WIDER FRAME TELLS THE SAME STORY" if not bad
            else "THE DRIFT DISAGREES: THE PATH IS NOT DRAWN")
    lines = [f"  {gate}",
             f"    {checked} band(s) checked against lane 29b"] + lines \
        + [f"    {m}" for m in bad]
    if bad:
        FAILURES.append("gate")
    return gate, lines


def rename_78_exports() -> None:
    """
    78 writes prepath_plain.csv and predrift.csv under its own export
    names. Those names belong to the education route's export of script 78,
    so the rows are kept here under names of this route's own and the
    originals are removed. Two exposure routes must never share an
    export name.
    """
    for src, dst in (("prepath_plain.csv", "occ_route_prepath.csv"),
                     ("predrift.csv", "occ_route_predrift.csv")):
        s, d = OUT / src, OUT / dst
        if s.exists():
            d.write_bytes(s.read_bytes())
            s.unlink()
            print(f"    {src} -> {dst}")
        else:
            FAILURES.append(f"78 wrote no {src}")


def main():
    mc.Tee(OUT / "86_log.txt")
    t0 = time.time()
    print("=" * 70)
    print("86: THE PLAIN PATH FROM 2019 AND THE DRIFT TEST, "
          "OCCUPATION ROUTE")
    print("=" * 70)
    print("\n".join(READ_RULES))
    print(mc.mem_line("  "))

    s82, s61, s78, l47, l70, j47 = load_modules()
    built = s82.build_exposure(l47, l70, j47)
    occ = built["exposure"]
    drain(s82, "82")
    print(f"  the score: {len(occ):,} employers on the {built['arm']} arm "
          f"at a floor of {built['floor']} {built['basis']}")
    NOTES.append(f"the score is 82's build_exposure(): {len(occ):,} "
                 f"employers, {built['arm']}, floor {built['floor']}")

    counts = load_counts("L_counts", s61.PANEL_YEARS)
    if counts is None:
        raise RuntimeError("L_counts_* missing: run 47L first.")
    print(f"  counts: {len(counts):,} employer-age-months")

    # Rule 3: extend the window only from what is already cached.
    early = load_counts("L_counts", [2019, 2020])
    if early is None:
        extended = None
        msg = ("L_counts_2019 and L_counts_2020 are NOT on the share: the "
               "path runs from 2021-01 and not from 2019Q1, so it is not "
               "the window the appendix figure draws")
        print(f"  {msg}")
        NOTES.append(msg)
    else:
        extended = pd.concat([early, counts], ignore_index=True)
        print(f"  the path runs from {EXTENDED_FROM}: "
              f"{len(extended):,} employer-age-months with 2019 and 2020")
        del early
        gc.collect()

    try:
        path_rows, drift_rows = s78.part_a(counts, occ, s61, j47, extended)
    except BaseException as ex:
        print(f"  Part A FAILED: {type(ex).__name__}: {ex}")
        traceback.print_exc()
        FAILURES.append("A")
        path_rows, drift_rows = [], []
    finally:
        drain(s78, "78")
        del extended, counts
        gc.collect()

    rename_78_exports()

    gate, gate_lines = run_gate(drift_rows, prior_drift())

    # ---- what the run says -------------------------------------------
    path_lines = []
    if path_rows:
        P = pd.DataFrame(path_rows)
        for band in sorted(P["young_band"].unique()):
            rs = P[P["young_band"] == band].sort_values("quarter")
            path_lines.append(f"  {band}: {len(rs)} quarters, "
                              f"{rs['quarter'].min()} to {rs['quarter'].max()}"
                              f", {REF_QUARTER} the reference")
            for _, d in rs.iterrows():
                if d["status"] == "reference":
                    path_lines.append(f"    {d['quarter']:<8} reference")
                else:
                    star = "" if abs(d["coef"]) < 2 * d["se"] else "  *"
                    path_lines.append(f"    {d['quarter']:<8} "
                                      f"{d['coef']:+.4f} ({d['se']:.4f}){star}")
    else:
        FAILURES.append("no path rows")

    drift_lines = []
    if drift_rows:
        D = pd.DataFrame(drift_rows)
        for band in sorted(D["young_band"].unique()):
            r = D[(D["young_band"] == band) & (D["term"] == TREND)]
            if not len(r):
                drift_lines.append(f"  {band}: no trend term")
                continue
            c, se = float(r["coef"].iloc[0]), float(r["se"].iloc[0])
            flat = "FLAT" if abs(c) < 2 * se else "NOT FLAT"
            pc, ps = PRIOR_TREND.get(band, (np.nan, np.nan))
            drift_lines.append(
                f"  {band}: trend {c:+.5f} ({se:.5f}) t "
                f"{(c / se if se else np.nan):+.2f}   {flat}   "
                f"lane 29b gave {pc:+.5f} ({ps:.5f})")
    else:
        FAILURES.append("no drift rows")

    print("\nTHE GATE:")
    print("\n".join(gate_lines) or "  not run")
    print("\nTHE DRIFT TEST, REFITTED:")
    print("\n".join(drift_lines) or "  none")
    print("\nTHE PLAIN PATH:")
    print("\n".join(path_lines) or "  none")

    L = ["THE PLAIN QUARTERLY PATH AND THE DRIFT TEST, OCCUPATION ROUTE",
         "=" * 58, "",
         "Script 78's part A, unchanged, on the score the paper reports:",
         "an employer ranked by the DAIOE generative-AI percentile of the",
         "2019 occupations of its own incumbents aged 31 to 69. The path",
         "carries no calendar terms and omits " + REF_QUARTER + "; the drift",
         "test beneath it is the row Table 1 of the paper prints, refitted",
         "here so that the gate can prove the two sit on one panel.", "",
         "THE GATE:"]
    L += gate_lines or ["  not run"]
    L += ["", "THE DRIFT TEST, REFITTED:"] + (drift_lines or ["  none"])
    L += ["", "THE PLAIN PATH:"] + (path_lines or ["  none"])
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
    (OUT / "86_summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n  wrote {OUT / '86_summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
