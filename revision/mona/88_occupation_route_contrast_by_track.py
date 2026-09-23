#!/usr/bin/env python3
"""
88_occupation_route_contrast_by_track.py -- the young against 41-49 by
                                            education track, on lane 28's
                                            occupation-route score.

======================================================================
  RUNS IN MONA. The output folder is CANARIES_88_OUT (default
  output_88). No database connection is needed, and this script refuses
  to open one: the counts by sex and education are 76's own pull and are
  cached from lane 21-22.
======================================================================

QUESTION
Script 77 cuts the three-band contrast by broad education track. It is
the exhibit the paper credits \\citet{nordstromskans2026structural} for,
in Online Appendix III.2 and in one sentence of Section 3, and it scores
an employer by its 2019 EDUCATION mix. Lane 33's other script, 87, moves
the female split off that score; this one moves the track contrast, and
between them nothing in the paper measures exposure with the education
register any more.

WHAT DOES NOT CHANGE, AND WHY: THE CUT STAYS EDUCATION
The reported design classifies NO young worker by occupation after 2019,
which is the point of freezing exposure on the employer's incumbents
aged 31 to 69, and the as-of backtest of Part IV is what closed the
designs that did classify them. Education is the only register that can
cut the young here. What changes is the FIRM's score, and nothing else:
the terms are 77's own contrast_terms, the panel is 77's three-band
skeleton, and the tracks are 76's.

THERE IS NO FOUR-DECIMAL GATE HERE, AND THE SUMMARY SAYS SO
77's gate was script 74's three-band contrast, -0.0153, which is the
EDUCATION route's. This route has no three-band all-worker contrast yet:
the six-band profile gives -0.0192 (0.0125) at 22-25, but that is a
different panel and the two are not required to agree, exactly as they
did not on the education route (-0.0099 six-band against -0.0153
three-band). So the all-worker fit run here IS the base, every track
cell is read against it, and no external reproduction is claimed. What
is checked instead is structural: 82's primary arm, or the script
refuses to start.

READ RULES, fixed before the run and printed at the start and in the
summary.

  1. THE BASE IS FITTED HERE. The all-worker three-band contrast on this
     score is the figure every track cell is read against. It is NOT
     required to reproduce the six-band profile, which is another panel,
     and it is NOT compared with the education route's -0.0153 except
     for reference.
  2. NO TRACK IS PROMOTED ABOVE THE POOLED PROFILE. Every track is
     reported whichever way it falls, and a track that does not fit is
     named, never read as a zero.
  3. THE CUT STAYS EDUCATION AND MUST, because the design classifies no
     young worker by occupation after 2019. Only the FIRM's score
     changes.
  4. NO SQL. If L_counts_sex_edu_* are not cached the lane reports that
     and stops rather than starting a full read.
  5. The education record ends in 2023 and 2024-25 carry it forward,
     which is why the tracks are broad. Unchanged.

INPUTS AND OUTPUTS
Reads L_baseline_2019_cascade and L_baseline_2019 (82's pull, cached by
lane 28a) and L_counts_sex_edu_2021 to 2025 (76's pull, cached by lane
21-22). Performs no SQL. Writes to output_88/:
occ_route_contrast_by_track.csv, the vcov files and 88_summary.txt. 77's
export name is not reused, because two exposure routes must never share
an export name.

IN THE PAPER
Online Appendix III.2, "The young against 41--49, by track", with
tableA_contrast_by_track; Section 3, the sentence on education tracks.
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
OUT = HERE / os.environ.get("CANARIES_88_OUT", "output_88")
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR
os.environ.setdefault("CANARIES_82_OUT", str(OUT))

FLOOR = 5
# For the summary alone; neither enters an export nor gates anything.
EDU_THREE_BAND = (-0.0153, 0.0126)
OCC_SIX_BAND = (-0.0192, 0.0125)

NOTES, FAILURES = [], []

READ_RULES = [
    "READ RULES, FIXED BEFORE THE RUN:",
    "  1. THE BASE IS FITTED HERE. The all-worker three-band contrast on",
    "     this score is the figure every track cell is read against. It",
    "     is NOT required to reproduce the six-band profile, which is",
    "     another panel, and it is NOT compared with the education",
    "     route's -0.0153 except for reference.",
    "  2. NO TRACK IS PROMOTED ABOVE THE POOLED PROFILE. Every track is",
    "     reported whichever way it falls, and a track that does not fit",
    "     is named, never read as a zero.",
    "  3. THE CUT STAYS EDUCATION AND MUST. The reported design",
    "     classifies no young worker by occupation after 2019, so",
    "     occupation cannot cut the young. Only the FIRM's score",
    "     changes, from the 2019 education mix to the 2019 occupation",
    "     mix of its incumbents aged 31 to 69.",
    "  4. NO SQL. If L_counts_sex_edu_* are not cached the lane reports",
    "     that and stops rather than starting a full read.",
    "  5. The education record ends in 2023 and 2024-25 carry it",
    "     forward, which is why the tracks are broad.",
    f"  Employer counts below {FLOOR} are suppressed before anything",
    "  leaves MONA.",
]


def _mod(fname: str, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def drain(mod, tag: str) -> None:
    for n in list(getattr(mod, "NOTES", [])):
        NOTES.append(f"{tag}: {n}")
    for f in list(getattr(mod, "FAILURES", [])):
        FAILURES.append(f"{tag}/{f}")
    if hasattr(mod, "NOTES"):
        mod.NOTES.clear()
    if hasattr(mod, "FAILURES"):
        mod.FAILURES.clear()


def load_modules():
    s82 = _mod("82_occupation_route.py", "s82")
    s82.OUT = OUT
    s61, s67, s74, s78, s80, l47, l70, j47 = s82.load_modules()
    s76 = _mod("76_gender_decomposition.py", "s76")
    s77 = _mod("77_contrast_by_track.py", "s77")
    for m_ in (s76, s77, s78):
        m_.OUT, m_.CACHE = OUT, CACHE
    if s77.POST_FROM != s78.POST_FROM:
        raise RuntimeError(f"77 and 78 disagree on the post month "
                           f"({s77.POST_FROM} against {s78.POST_FROM}); "
                           f"refusing to run.")
    if s77.REF_BAND != "41-49":
        raise RuntimeError(f"77's reference band is {s77.REF_BAND}; "
                           f"refusing to run.")
    if s82.MAIN_LEVEL != "uniform3" or s82.MAIN_ARM != "backward" \
            or s82.FLOOR_MAIN != FLOOR:
        raise RuntimeError("82's primary arm is not the one the paper "
                           "reports; refusing to run.")
    return s82, s61, s76, s77, l47, l70, j47


def save(rows, name: str, count_col: str = "n_firms") -> pd.DataFrame:
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if not df.empty and count_col in df.columns:
        df = mc.enforce_min_cell(df, count_col=count_col, floor=FLOOR)
    df.to_csv(OUT / name, index=False)
    return df


def main():
    mc.Tee(OUT / "88_log.txt")
    t0 = time.time()
    print("=" * 70)
    print("88: THE YOUNG AGAINST 41-49 BY TRACK, ON THE OCCUPATION ROUTE")
    print("=" * 70)
    print("\n".join(READ_RULES))
    print(mc.mem_line("  "))

    s82, s61, s76, s77, l47, l70, j47 = load_modules()
    h47 = j47._h47()

    missing = [y for y in s76.YEARS
               if not (CACHE / f"L_counts_sex_edu_{y}.parquet").exists()]
    if missing:
        raise RuntimeError(
            f"L_counts_sex_edu_* not cached for {missing}. This lane does "
            f"no SQL; run 76 in a lane that may pull, or restore the cache.")

    built = s82.build_exposure(l47, l70, j47)
    occ = built["exposure"]
    drain(s82, "82")
    print(f"  the score: {len(occ):,} employers on the {built['arm']} arm "
          f"at a floor of {built['floor']} {built['basis']}")
    NOTES.append(f"the score is 82's build_exposure(): {len(occ):,} "
                 f"employers, {built['arm']}, floor {built['floor']}")

    frames = []
    for y in s76.YEARS:
        c = mc.read_cache(CACHE / f"L_counts_sex_edu_{y}.parquet",
                          require=s76.EDU_COLS + ["n_emp"])
        if c is None:
            raise RuntimeError(f"L_counts_sex_edu_{y}.parquet unreadable.")
        print(f"  counts by sex and education {y}: cached ({len(c):,} cells)")
        frames.append(s76.tag_frame(c, h47))
        del c
        gc.collect()
    allf = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    rows = []

    def record(track, res):
        for band in s77.YOUNG_BANDS:
            if band in res:
                c, se, st = res[band]
                rows.append({"track": track, "band_vs_ref": band, "coef": c,
                             "se": se, "t": (c / se if se else np.nan),
                             "n_firms": res["n_firms"], "status": st})
        save(rows, "occ_route_contrast_by_track.csv")

    def fit(frame, tag):
        try:
            return s77.fit_contrast(s76.collapse(frame), occ, j47,
                                    f"s88_{tag}")
        except BaseException as ex:
            print(f"  {tag} FAILED: {type(ex).__name__}: {ex}")
            traceback.print_exc()
            return None

    base = fit(allf, "all")
    if base is None:
        FAILURES.append("contrast/all")
    else:
        record("all", base)
        for band in s77.YOUNG_BANDS:
            if band in base:
                c, se, _ = base[band]
                print(f"  base {band} vs 41-49: {c:+.4f} ({se:.4f})")

    for g in s76.TRACK_ORDER:
        sub = allf[allf["track"] == g]
        if sub.empty:
            print(f"  track {g}: no workers")
            FAILURES.append(f"contrast/{g}/empty")
            continue
        res = fit(sub, g)
        del sub
        gc.collect()
        if res is None:
            FAILURES.append(f"contrast/{g}")
            continue
        record(g, res)
        for band in s77.YOUNG_BANDS:
            if band in res:
                c, se, _ = res[band]
                print(f"    {g:<24} {band} {c:+.4f} ({se:.4f}) t "
                      f"{(c / se if se else np.nan):+.2f}")
    del allf
    gc.collect()
    drain(s77, "77")

    L = ["THE YOUNG AGAINST 41-49 BY EDUCATION TRACK, OCCUPATION ROUTE",
         "=" * 58, "",
         "77's three-band panel and its own seasonal terms, with the",
         "employer scored by the 2019 occupations of its own incumbents",
         "aged 31 to 69. The cut is the worker's own SUN 2020 record,",
         "because the design classifies no young worker by occupation",
         "after 2019; Individ ends at 2023 and later years carry it",
         "forward.", ""]
    if base:
        L += ["THE BASE, fitted here and the figure every track is read "
              "against:"]
        for band in s77.YOUNG_BANDS:
            if band in base:
                c, se, _ = base[band]
                L.append(f"  all workers, {band} vs 41-49  {c:+.4f} "
                         f"({se:.4f}) t {(c / se if se else np.nan):+.2f}"
                         f"   firms {base['n_firms']:,}")
        L += ["  For reference only, and not a gate: the education route's "
              f"three-band contrast at 22-25 was {EDU_THREE_BAND[0]:+.4f} "
              f"({EDU_THREE_BAND[1]:.4f}), and this route's SIX-band "
              f"profile gives {OCC_SIX_BAND[0]:+.4f} ({OCC_SIX_BAND[1]:.4f}) "
              "on another panel.", ""]
    if rows:
        L += ["BY TRACK:"]
        for r in rows:
            if r["track"] == "all":
                continue
            L.append(f"  {r['track']:<24} {r['band_vs_ref']}  "
                     f"{r['coef']:+.4f} ({r['se']:.4f}) t {r['t']:+.2f}"
                     f"   firms {r['n_firms']:,}")
        L.append("")
    if FAILURES:
        L.append(f"WHAT FAILED: {', '.join(FAILURES)}")
        L.append("A missing row is a missing fit, never a zero.")
        L.append("")
    if NOTES:
        L.append("NOTES:")
        L += [f"  {n}" for n in NOTES]
        L.append("")
    L += READ_RULES
    L.append("")
    L.append(f"Runtime {(time.time()-t0)/60:.1f} min. {mc.mem_line('')}")
    (OUT / "88_summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n  wrote {OUT / '88_summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
