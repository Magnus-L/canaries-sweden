#!/usr/bin/env python3
"""
87_occupation_route_gender_split.py -- the female differential split into
                                       composition and within, on lane
                                       28's occupation-route score.

======================================================================
  RUNS IN MONA. The output folder is CANARIES_87_OUT (default
  output_87). No database connection is needed, and this script refuses
  to open one: the counts by sex and education are 76's own pull and are
  cached from lane 21-22.
======================================================================

QUESTION
Table 1 of the paper carries one row the rest of the table does not
share: the part of the female differential that survives within broad
education tracks, $-0.0508$. It comes from script 76, which scores an
employer by its 2019 EDUCATION mix, and it is the last estimate anywhere
in the paper still on that score. Printed under the occupation route's
$-0.0858$ it invites a division that is not meaningful: its own pooled
counterpart on 76's panel and specification is $-0.0659$, which is where
the paper's "three quarters" comes from. This script re-runs 76 on the
score the paper reports, so that the row and the row above it are one
measure and the ratio is the one a reader will take.

WHAT DOES NOT CHANGE, AND WHY: THE CUT STAYS EDUCATION
The split is cut by the young worker's own education, and it has to be.
The reported design classifies NO young worker by occupation after 2019,
which is the whole point of freezing exposure on the employer's
incumbents aged 31 to 69; the as-of backtest of Part IV is what closed
the designs that did classify them. So occupation cannot cut the young
here and education is the only register that can. That was true when 76
was written and it is true now.

What changes is one thing: the FIRM's exposure score. It is no longer
the 2019 education mix but 82's occupation-route score, the mean DAIOE
generative-AI percentile of the 2019 occupations of the employer's own
incumbents aged 31 to 69. No young worker's own record of any kind
enters their own treatment, before or after this change.

TWO CONSEQUENCES THAT ARE EASY TO MISS
  1. The specification is upgraded to Equation (2) itself, through 78's
     gender_eq2_terms, so the pooled differential this produces is the
     one Table 1 prints and the within-track row is comparable with it.
     76 used 68's shorter term set, which is why its pooled was -0.0659
     and not -0.0858.
  2. THE WEIGHTS MOVE. The split weights young women's track shares in
     EXPOSED employers, and which employers are exposed is exactly what
     this change redefines. So the descriptive composition is rebuilt on
     this route too, not carried over; carrying it would weight this
     route's track differentials by the other route's exposed firms.

READ RULES, fixed before the run and printed at the start and in the
summary.

  1. THE GATE. The all-track differential must reproduce Table 1's
     occupation-route figure, -0.0858, to four decimals. It is fitted on
     the education frame collapsed over education, so a match is the
     evidence that this frame is the one Table 1 sits on. A MISS MEANS
     NOTHING HERE IS QUOTED, and the summary says so at the top.
  2. The split's rule is 76's, unchanged: composition if within is at or
     below half the pooled differential in absolute value, being
     affected more in the same work if at or above three quarters,
     ambiguous between.
  3. No track is promoted above the pooled profile, and every track is
     reported whichever way it falls.
  4. NO SQL. If L_counts_sex_edu_* are not cached the lane reports that
     and stops, rather than starting a full read of the monthly
     declarations.
  5. The education record ends in 2023 and 2024-25 carry it forward,
     which is why the tracks are broad. That is unchanged and is
     reported again here.

INPUTS AND OUTPUTS
Reads, through the modules it imports: L_baseline_2019_cascade and
L_baseline_2019 (82's pull, cached by lane 28a), 47h's caches for the
education score book that prices education groups in the descriptive
table, and L_counts_sex_edu_2021 to 2025 (76's pull, cached by lane
21-22). Performs no SQL.

Writes to output_87/: occ_route_education_mix_by_sex.csv,
occ_route_gender_by_track.csv, occ_route_gender_split.csv, the vcov
files and 87_summary.txt. 76's own export names are not reused, because
two exposure routes must never share an export name.

IN THE PAPER
Table 1, the within-track row; Section 3, the share that survives within
tracks; Online Appendix III.2, "The female differential, split", with
tableA_gender_split and tableA_education_mix.
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
OUT = HERE / os.environ.get("CANARIES_87_OUT", "output_87")
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR
os.environ.setdefault("CANARIES_82_OUT", str(OUT))

FLOOR = 5
MATCH_DP = 4
# Table 1's own figure, the gate. Lane 28c/29 fitted it on 38,398,914
# cells and 104,217 employers.
TABLE1_DIFF = -0.085825
DIFF_TERM = "post_x_high_x_young_x_female"
MALE_TERM = "post_x_high_x_young"

NOTES, FAILURES = [], []
GATE = {"ok": None, "detail": ""}

READ_RULES = [
    "READ RULES, FIXED BEFORE THE RUN:",
    "  1. THE GATE. The all-track differential must reproduce Table 1's",
    f"     occupation-route figure, {TABLE1_DIFF:+.4f}, to {MATCH_DP} decimals.",
    "     It is fitted on the education frame collapsed over education,",
    "     so a match is the evidence that this frame is the one Table 1",
    "     sits on. A MISS MEANS NOTHING HERE IS QUOTED.",
    "  2. The split's rule is 76's, unchanged: composition if within is",
    "     at or below half the pooled differential in absolute value,",
    "     being affected more in the same work if at or above three",
    "     quarters, ambiguous between.",
    "  3. No track is promoted above the pooled profile, and every track",
    "     is reported whichever way it falls.",
    "  4. NO SQL. If L_counts_sex_edu_* are not cached the lane reports",
    "     that and stops rather than starting a full read.",
    "  5. THE CUT STAYS EDUCATION AND MUST. The reported design",
    "     classifies no young worker by occupation after 2019, so",
    "     occupation cannot cut the young. What changed is the FIRM's",
    "     score, from the 2019 education mix to the 2019 occupation mix",
    "     of its incumbents aged 31 to 69.",
    "  6. THE WEIGHTS ARE REBUILT ON THIS ROUTE. They are young women's",
    "     track shares in EXPOSED employers, and which employers are",
    "     exposed is what this change redefines.",
    "  On the education route 76 gave pooled -0.0659 (0.0131), within",
    "  -0.0508 (0.0118), ratio 0.772.",
    f"  Cells below {FLOOR} are suppressed before anything leaves MONA.",
]
EDU_ROUTE = {"pooled": -0.065869, "within": -0.050827, "ratio": 0.77164}


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
    for m_ in (s76, s78):
        m_.OUT, m_.CACHE = OUT, CACHE
    if s76.YOUNG != "22-25" or s76.POST_FROM != s78.POST_FROM:
        raise RuntimeError(f"76 and 78 disagree on the band or the post "
                           f"month ({s76.YOUNG}, {s76.POST_FROM} against "
                           f"{s78.POST_FROM}); refusing to run.")
    if s82.MAIN_LEVEL != "uniform3" or s82.MAIN_ARM != "backward" \
            or s82.FLOOR_MAIN != FLOOR:
        raise RuntimeError("82's primary arm is not the one the paper "
                           "reports; refusing to run.")
    return s82, s61, s67, s76, s78, l47, l70, j47


def fit_eq2(frame: pd.DataFrame, occ: pd.DataFrame, s67, s76, s78, j47,
            tag: str) -> dict | None:
    """76's fit, with Equation (2)'s term set and this route's score."""
    skel = s67.build_skeleton_sex(s76.collapse(frame), s76.YOUNG, j47, "n_emp")
    if skel.empty:
        print(f"  {tag}: empty skeleton")
        return None
    b = s78.with_exposure(skel, occ)
    del skel
    gc.collect()
    if b.empty:
        print(f"  {tag}: no firm on this panel carries the quartile")
        return None
    b, terms = s78.gender_eq2_terms(b)
    n_firms = int(b["employer_id"].nunique())
    print(f"  {tag}: {len(b):,} rows, {n_firms:,} firms{mc.mem_line(' | ')}")
    t = time.time()
    try:
        r = mc.run_fepois_multi(b, OUT, tag=f"s87_{tag}", terms=terms,
                                fes=j47.FES, cluster="employer_id")
    except BaseException as ex:
        print(f"    {tag} FAILED: {type(ex).__name__}: {ex}")
        traceback.print_exc()
        r = pd.DataFrame()
    del b
    gc.collect()
    if r.empty:
        return None
    g = r.set_index("term")
    if any(t_ not in g.index for t_ in (MALE_TERM, DIFF_TERM)):
        return None
    print(f"    {tag}: done in {(time.time()-t)/60:.1f} min")
    return {"male": float(g.loc[MALE_TERM, "coef"]),
            "male_se": float(g.loc[MALE_TERM, "se"]),
            "diff": float(g.loc[DIFF_TERM, "coef"]),
            "diff_se": float(g.loc[DIFF_TERM, "se"]),
            "n_obs": int(g.loc[DIFF_TERM, "n_obs"]), "n_firms": n_firms,
            "status": str(g.loc[DIFF_TERM].get("status", "ok"))}


def save(rows, name: str, count_col: str = "n_firms") -> pd.DataFrame:
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if not df.empty and count_col in df.columns:
        df = mc.enforce_min_cell(df, count_col=count_col, floor=FLOOR)
    df.to_csv(OUT / name, index=False)
    return df


def main():
    mc.Tee(OUT / "87_log.txt")
    t0 = time.time()
    print("=" * 70)
    print("87: THE FEMALE DIFFERENTIAL SPLIT, ON THE OCCUPATION ROUTE")
    print("=" * 70)
    print("\n".join(READ_RULES))
    print(mc.mem_line("  "))

    s82, s61, s67, s76, s78, l47, l70, j47 = load_modules()
    h47 = j47._h47()

    # Rule 4: the counts are 76's pull and must already be on the share.
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

    # The education score book prices the education GROUPS in the
    # descriptive table. It is not the firm's exposure and never was.
    wt = {}
    for y in (2019, 2020, 2021):
        w = mc.read_cache(CACHE / f"edu_hr_weights_{y}.parquet",
                          require=h47.WEIGHT_COLS)
        if w is None:
            raise RuntimeError(f"edu_hr_weights_{y}.parquet missing: run 47h.")
        wt[y] = w
    book = h47.ScoreBook(wt, h47.load_key(), h47.load_scores())
    spec = dict(h47.DESIGNS["OL_daioe"])
    book.build("OL_daioe", spec)

    frames = {}
    for y in s76.YEARS:
        c = mc.read_cache(CACHE / f"L_counts_sex_edu_{y}.parquet",
                          require=s76.EDU_COLS + ["n_emp"])
        if c is None:
            raise RuntimeError(f"L_counts_sex_edu_{y}.parquet unreadable.")
        print(f"  counts by sex and education {y}: cached ({len(c):,} cells)")
        frames[y] = s76.tag_frame(c, h47)
        del c
        gc.collect()

    # ---- 1. composition, on THIS route's exposed firms ---------------
    mix, weights = None, {}
    try:
        mix = s76.education_mix(frames[s76.MIX_YEAR], occ, book, spec, h47)
    except BaseException as ex:
        print(f"  education mix FAILED: {type(ex).__name__}: {ex}")
        traceback.print_exc()
        FAILURES.append("mix")
    if mix is not None and len(mix):
        save(mix, "occ_route_education_mix_by_sex.csv", count_col="persons_avg")
        w = mix[(mix["dimension"] == "track") & (mix["exposed"] == 1)
                & (mix["gender"] == "women")]
        weights = dict(zip(w["cell"], w["share"]))
        print("  young women's track shares in exposed firms, "
              f"{s76.MIX_YEAR}: "
              + ", ".join(f"{k} {v:.3f}" for k, v in weights.items()))
        NOTES.append("the split weights are this route's exposed employers, "
                     "not the education route's")
    cf = s76.carry_forward_share(frames)

    # ---- 2. the pooled fit, which is the gate, then the tracks -------
    allf = pd.concat(frames.values(), ignore_index=True)
    del frames
    gc.collect()
    rows, by_track = [], {}
    pooled = fit_eq2(allf, occ, s67, s76, s78, j47, "all")
    if pooled is None:
        FAILURES.append("gender/all")
    else:
        rows.append(dict(track="all", **pooled))
        d, se = pooled["diff"], pooled["diff_se"]
        GATE["ok"] = abs(d - TABLE1_DIFF) < 10 ** (-MATCH_DP) / 2
        GATE["detail"] = (f"all-track differential {d:+.5f} ({se:.5f}) "
                          f"against Table 1's {TABLE1_DIFF:+.5f}")
        print(f"  GATE {'PASS' if GATE['ok'] else 'FAIL'}: {GATE['detail']}")
        if not GATE["ok"]:
            FAILURES.append("gate")

    for g in s76.TRACK_ORDER:
        sub = allf[allf["track"] == g]
        if sub.empty:
            print(f"  track {g}: no workers")
            continue
        res = fit_eq2(sub, occ, s67, s76, s78, j47, g)
        del sub
        gc.collect()
        if res is None:
            FAILURES.append(f"gender/{g}")
            continue
        by_track[g] = res
        rows.append(dict(track=g, **res))
        save(rows, "occ_route_gender_by_track.csv")
        print(f"    {g:<24} men {res['male']:+.4f} ({res['male_se']:.4f})  "
              f"differential {res['diff']:+.4f} ({res['diff_se']:.4f}) "
              f"t {res['diff']/max(res['diff_se'], 1e-12):+.2f}")
    del allf
    gc.collect()
    save(rows, "occ_route_gender_by_track.csv")

    # ---- 3. the split ------------------------------------------------
    sp = s76.split(pooled, by_track, weights) if pooled and weights else {}
    if sp:
        save(pd.DataFrame([sp]), "occ_route_gender_split.csv",
             count_col="n_firms")
    else:
        FAILURES.append("split")

    L = ["THE FEMALE DIFFERENTIAL AT 22-25, SPLIT, ON THE OCCUPATION ROUTE",
         "=" * 58, "",
         "Equation (2)'s sex specification, run on all workers and then",
         "within each broad education track, with the employer scored by",
         "the 2019 occupations of its own incumbents aged 31 to 69. The cut",
         "is the worker's own SUN 2020 record, because the design",
         "classifies no young worker by occupation after 2019; Individ ends",
         "at 2023 and later years carry it forward.", ""]
    if GATE["ok"] is not None:
        L += [f"THE GATE: {'PASS' if GATE['ok'] else 'FAIL'}. {GATE['detail']}",
              "  A FAIL means this frame is not the one Table 1 sits on and "
              "nothing here is quoted.", ""]
    if rows:
        L += ["MALE EFFECT AND FEMALE DIFFERENTIAL BY TRACK:"]
        for r in rows:
            L.append(f"  {r['track']:<24} men {r['male']:+.4f} "
                     f"({r['male_se']:.4f})   women minus men "
                     f"{r['diff']:+.4f} ({r['diff_se']:.4f}) t "
                     f"{r['diff']/max(r['diff_se'], 1e-12):+.2f}   firms "
                     f"{r['n_firms']:,}")
        L.append("")
    if weights:
        L += ["YOUNG WOMEN'S TRACK SHARES IN THIS ROUTE'S EXPOSED FIRMS "
              f"({s76.MIX_YEAR}), the weights:",
              "  " + ", ".join(f"{k} {v:.3f}" for k, v in weights.items()), ""]
    if sp:
        L += ["THE SPLIT:",
              f"  pooled differential      {sp['pooled']:+.4f} "
              f"({sp['pooled_se']:.4f})",
              f"  within tracks            {sp['within']:+.4f} "
              f"({sp['within_se']:.4f})",
              f"  composition              {sp['composition']:+.4f}",
              f"  ratio within / pooled    {sp['ratio_within']:.3f}",
              f"  verdict                  {sp['verdict']}", "",
              "THE EDUCATION ROUTE'S OWN SPLIT, for reference and not for "
              "export:",
              f"  pooled {EDU_ROUTE['pooled']:+.4f}   within "
              f"{EDU_ROUTE['within']:+.4f}   ratio "
              f"{EDU_ROUTE['ratio']:.3f}", ""]
    if cf:
        L += ["THE EDUCATION RECORD:"]
        for y, d in sorted(cf.items()):
            L.append(f"  {y}: {d['person_months']:,} young person-months, "
                     f"{d['no_record_share']:.3%} with no record"
                     + ("  (2023 carried forward)" if d["carried_forward"]
                        else ""))
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
    (OUT / "87_summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n  wrote {OUT / '87_summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
