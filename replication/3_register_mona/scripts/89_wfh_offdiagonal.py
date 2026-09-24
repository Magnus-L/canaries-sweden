#!/usr/bin/env python3
"""
89_wfh_offdiagonal.py: separating AI exposure from working from home
without asking the data to separate two collinear scores.

======================================================================
  RUNS IN MONA. Output folder CANARIES_89_OUT (default output_89).
  No database connection: every input is cached by scripts 82 and 76.
======================================================================

QUESTION
\\citet{lambert2026brokenladder} show that generative-AI exposure is
strongly correlated with the post-pandemic shift to working from home,
and that entering the two together leaves the remote-work term standing
while the AI term attenuates towards zero. The same question applies to
this paper. Teleworkability and DAIOE correlate at +0.75 across
occupations here, and a FIRM-level average of each correlates higher
still, because averaging over a firm's incumbents strips the
idiosyncratic occupation variation and leaves the shared component.

WHY THIS IS NOT A HORSE RACE
A joint regression on two firm scores that correlate at 0.85 gives wide
intervals on both, and an imprecise AI coefficient reads as a failed
defence rather than as an underpowered test. Script 46 ran that race on
the withdrawn design and is not the answer. The discriminating variation
is the OFF-DIAGONAL: firms exposed but not teleworkable, and teleworkable
but not exposed. Part A reads it directly, by splitting the sample rather
than by adding a regressor.

PART A. THE OFF-DIAGONAL, FOUR FITS
  1. the AI step among firms in the LOW-teleworkability half
  2. the AI step among firms in the HIGH-teleworkability half
  3. the teleworkability step among firms in the LOW-AI half
  4. the teleworkability step among firms in the HIGH-AI half
Fits 1 and 3 are the discriminating ones. This is the employment-margin
analogue of Online Appendix II.3, which splits occupations at the median
Dingel and Neiman score and finds the posting decline entirely in the
NON-teleworkable half, -0.233 against -0.005.

PART B. TIMING, AND WHY IT IS NOT "WFH WAS A 2020-21 SHOCK"
Remote work cannot be treated as a spent shock of 2020-21 whose effects
ended before the adoption window: the return to the office means
teleworkability is still moving, and a design that fixed the WFH effect
after 2021 would assume its own answer. So Part B asks whether the two
gradients SEPARATE in timing, not whether one is historical. Both scores,
standardised and continuous, are interacted with the full quarterly path
from 2019Q1 with 2022Q1 omitted and no calendar terms, at 22-25. Three
shapes are reportable and none is a failure: they move together and
cannot be told apart; the AI gradient opens in 2024 while the WFH
gradient does not; or the WFH gradient opens later, which would be the
backlash and a finding of its own.

THE TWO SCORES
Built by the same function on the same employers, freeze year, incumbent
restriction (aged 31 to 69 in 2019) and floor. 82's build_exposure()
takes the occupation score book as an argument, so the only difference is
the book: the DAIOE generative-AI percentile for one, the Dingel and
Neiman teleworkable share for the other. Nothing about the panel, the
cell or the freeze changes.

INPUTS AND OUTPUTS
Reads, through the modules it imports, L_baseline_2019_cascade and
L_baseline_2019 (script 82) and L_counts_sex_edu_2021 to 2025
(script 76), plus
dingel_neiman_ssyk4 from the share. Performs no SQL. Writes to
output_89/: wfh_offdiagonal.csv, wfh_scores_overlap.csv,
wfh_timing_path.csv, the vcov files and 89_summary.txt.

IN THE PAPER
Online Appendix II.3 and Section 3: averaged over an employer's
incumbents the two scores correlate at 0.88 (Pearson) and only 14 per
cent of employers lie off the diagonal of a median cut on both
(wfh_scores_overlap.csv), which is why the comparison is drawn across
occupations. The fits of Parts A and B are not quoted.
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
OUT = HERE / os.environ.get("CANARIES_89_OUT", "output_89")
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR
os.environ.setdefault("CANARIES_82_OUT", str(OUT))

FLOOR = 5
YOUNG_BAND = "22-25"
TERM = "post_x_high_x_young"
WFH_TERM = "post_x_highwfh_x_young"
# The reported headline on this panel, for reference in the summary. It
# is NOT a gate: Part A's fits are on halves of it and cannot reproduce
# it, and Part B omits the calendar terms.
HEADLINE_22_25 = (-0.0578, 0.0155)

NOTES, FAILURES = [], []

READ_RULES = [
    "READ RULES, FIXED BEFORE THE RUN:",
    "  1. WHAT IS FIXED IN ADVANCE IS THE READING, NOT THE REPORTING.",
    "     Every one of the four fits runs and every coefficient is",
    "     exported and printed below, whichever way it falls; nothing",
    "     here is conditional on any rule. The rule fixed before",
    "     the run is only this: AI is called the operative score if the",
    "     young decline appears among firms in the LOW-teleworkability",
    "     half and not among firms in the LOW-AI half. It constrains the",
    "     sentence the paper may write, not what we get to look at.",
    "  2. NO CELL IS READ AS A ZERO. Each of the four fits is roughly a",
    "     half of the panel, so its standard error will be larger than",
    "     Table 1's. A cell that is imprecise is reported as imprecise",
    "     and not as an absence of effect; the summary prints every",
    "     standard error beside its coefficient for that reason.",
    "  3. PART B IS DESCRIPTIVE AND NOTHING IS PRE-COMMITTED. All three",
    "     shapes are reported as they come, including a WFH gradient",
    "     that opens LATER than the AI one, which would be the",
    "     return-to-office backlash rather than a defect.",
    "  4. THE 50-AND-OVER GAIN IS NOT CLAIMED. Online Appendix III.2",
    "     already reports that on the continuous route the oldest band's",
    "     gain loads as strongly on teleworkability and is not specific",
    "     to AI exposure. This lane concedes that band and keeps the",
    "     young-worker margin. Fixed before the run, not after.",
    "  5. NO SQL. If the caches are missing the lane says so and stops,",
    "     rather than starting a full read.",
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
    for m_ in (s76, s78):
        m_.OUT, m_.CACHE = OUT, CACHE
    if s82.MAIN_LEVEL != "uniform3" or s82.MAIN_ARM != "backward" \
            or s82.FLOOR_MAIN != FLOOR:
        raise RuntimeError("82's primary arm is not the one the paper "
                           "reports; refusing to run.")
    return s82, s61, s67, s76, s78, l47, l70, j47


def wfh_book() -> pd.DataFrame:
    """
    The teleworkability score in the shape build_exposure() expects:
    ssyk4 and `score`, exactly as l70.daioe_scores() returns.

    Dingel and Neiman's share runs 0 to 1 where DAIOE's percentile runs
    0 to 100. The scale does not matter: build_exposure() takes an
    employment-weighted mean and then a quartile, and Part B
    standardises. It is NOT rescaled here, because a rescaling would be
    one more place the two scores could silently diverge.
    """
    for name in ("dingel_neiman_ssyk4.dta", "dingel_neiman_ssyk4.csv",
                 "dingel_neiman_ssyk4.txt"):
        p = Path(mc.SHARE) / name
        if not p.exists():
            continue
        d = pd.read_stata(str(p)) if p.suffix == ".dta" else pd.read_csv(p)
        col = next((c for c in d.columns
                    if c.lower() in ("teleworkable", "telework", "wfh",
                                     "teleworkable_share")), None)
        if col is None:
            raise RuntimeError(f"{name} carries no teleworkable column; "
                               f"found {list(d.columns)}")
        d["ssyk4"] = d["ssyk4"].astype(str).str.zfill(4)
        return d.rename(columns={col: "score"})[["ssyk4", "score"]]
    raise RuntimeError("dingel_neiman_ssyk4 not found on the share.")


def save(rows, name: str, count_col: str = "n_firms") -> pd.DataFrame:
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if not df.empty and count_col in df.columns:
        df = mc.enforce_min_cell(df, count_col=count_col, floor=FLOOR)
    df.to_csv(OUT / name, index=False)
    return df


def build_both(s82, l47, l70, j47) -> tuple:
    """
    The two firm scores, from one builder and one incumbent frame.

    The AI score is rebuilt rather than read from script 82's export, so
    that both come out of the same call on the same day with the same
    audit; a score read from one place and a score built in another is
    how two definitions of a treatment start to drift.
    """
    ai = s82.build_exposure(l47, l70, j47)
    drain(s82, "82/ai")
    wfh = s82.build_exposure(l47, l70, j47, daioe=wfh_book(), audit=False)
    drain(s82, "82/wfh")
    a, w = ai["exposure"], wfh["exposure"]
    print(f"  AI score:  {len(a):,} employers, {ai['arm']} arm, "
          f"floor {ai['floor']} {ai['basis']}")
    print(f"  WFH score: {len(w):,} employers, same builder, same floor")
    return ai, wfh


def overlap(a: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    """
    How far apart the two scores are, reported BEFORE any fit.

    Read rule 2 exists because the four Part A fits are halves of the
    panel. This table is what says whether the halves are thin: the
    cross-tabulation of the two quartiles, the rank correlation, and the
    counts in the four median cells. If a discriminating cell is small,
    the summary says so and the reader discounts that fit rather than
    reading its width as evidence.
    """
    # The firm-level mean is `mix`, not `score`: `score` is the
    # OCCUPATION-level column the book carries, and occ_route_exposure
    # returns employer_id, fq, mix, n, n_coded, n_nov, coverage,
    # share_not_2019 and share_three_digit.
    m = a[["employer_id", "mix", "fq"]].merge(
        w[["employer_id", "mix", "fq"]], on="employer_id",
        suffixes=("_ai", "_wfh"))
    if m.empty:
        raise RuntimeError("the two scores share no employer.")
    rho = m["mix_ai"].corr(m["mix_wfh"], method="spearman")
    pear = m["mix_ai"].corr(m["mix_wfh"])
    med_a, med_w = m["mix_ai"].median(), m["mix_wfh"].median()
    m["hi_ai"] = (m["mix_ai"] > med_a).astype(int)
    m["hi_wfh"] = (m["mix_wfh"] > med_w).astype(int)
    rows = [{"item": "n_employers_both_scores", "value": float(len(m))},
            {"item": "spearman", "value": float(rho)},
            {"item": "pearson", "value": float(pear)},
            {"item": "median_ai", "value": float(med_a)},
            {"item": "median_wfh", "value": float(med_w)}]
    for ha in (0, 1):
        for hw in (0, 1):
            n = int(((m["hi_ai"] == ha) & (m["hi_wfh"] == hw)).sum())
            rows.append({"item": f"cell_ai{ha}_wfh{hw}", "value": float(n)})
    same_q = float((m["fq_ai"] == m["fq_wfh"]).mean())
    rows.append({"item": "share_same_quartile", "value": same_q})
    print(f"  the two scores: Spearman {rho:+.3f}, Pearson {pear:+.3f}, "
          f"{same_q:.1%} of employers in the same quartile on both")
    NOTES.append(f"the two firm scores correlate at Spearman {rho:+.3f} "
                 f"(Pearson {pear:+.3f}); {same_q:.1%} share a quartile")
    save(rows, "wfh_scores_overlap.csv", count_col="")
    return m[["employer_id", "hi_ai", "hi_wfh"]]


def fit_in_half(frame, expo, half_ids, s67, s76, s78, j47, tag, suffix=""):
    """
    Equation (2) on the young band, restricted to one half of employers.

    `expo` carries the exposure whose step is wanted; `half_ids` is the
    employer set defined by the OTHER score. The term set is 78's own,
    so the coefficient is the one Table 1 prints and is comparable with
    it in construction if not in sample.
    """
    skel = s67.build_skeleton_sex(s76.collapse(frame), s76.YOUNG, j47, "n_emp")
    if skel.empty:
        print(f"  {tag}: empty skeleton")
        return None
    b = s78.with_exposure(skel, expo)
    del skel
    gc.collect()
    if b.empty:
        print(f"  {tag}: no firm on this panel carries the quartile")
        return None
    b = b[b["employer_id"].isin(half_ids)]
    if b.empty:
        print(f"  {tag}: the half is empty on this panel")
        return None
    b, terms = s78.eq2_terms(b, "high", suffix)
    n_firms = int(b["employer_id"].nunique())
    print(f"  {tag}: {len(b):,} rows, {n_firms:,} firms{mc.mem_line(' | ')}")
    try:
        r = mc.run_fepois_multi(b, OUT, tag=f"s89_{tag}", terms=terms,
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
    want = f"post_x_high{suffix}_x_young"
    if want not in g.index:
        return None
    return {"coef": float(g.loc[want, "coef"]),
            "se": float(g.loc[want, "se"]), "n_firms": n_firms}


def part_a(allf, ai, wfh, halves, s67, s76, s78, j47) -> list:
    """
    The four fits. 1 and 3 discriminate; 2 and 4 are reported without
    comment, because read rule 1 promised every cell whichever way it
    falls.
    """
    lo_wfh = set(halves.loc[halves["hi_wfh"] == 0, "employer_id"])
    hi_wfh = set(halves.loc[halves["hi_wfh"] == 1, "employer_id"])
    lo_ai = set(halves.loc[halves["hi_ai"] == 0, "employer_id"])
    hi_ai = set(halves.loc[halves["hi_ai"] == 1, "employer_id"])
    plan = [("ai_in_low_wfh", ai["exposure"], lo_wfh, "", True),
            ("ai_in_high_wfh", ai["exposure"], hi_wfh, "", False),
            ("wfh_in_low_ai", wfh["exposure"], lo_ai, "wfh", True),
            ("wfh_in_high_ai", wfh["exposure"], hi_ai, "wfh", False)]
    rows = []
    for tag, expo, ids, suffix, discriminating in plan:
        res = fit_in_half(allf, expo, ids, s67, s76, s78, j47, tag, suffix)
        if res is None:
            FAILURES.append(f"A/{tag}")
            continue
        t = res["coef"] / res["se"] if res["se"] else np.nan
        rows.append({"fit": tag, "score": "ai" if not suffix else "wfh",
                     "held": "low_wfh" if tag.endswith("low_wfh") else
                             ("high_wfh" if tag.endswith("high_wfh") else
                              ("low_ai" if tag.endswith("low_ai") else "high_ai")),
                     "discriminating": int(discriminating),
                     "coef": res["coef"], "se": res["se"], "t": t,
                     "n_firms": res["n_firms"], "status": "ok"})
        print(f"    {tag:<16} {res['coef']:+.4f} ({res['se']:.4f}) "
              f"t {t:+.2f}   firms {res['n_firms']:,}")
        save(rows, "wfh_offdiagonal.csv")
    return rows


def part_b(allf, ai, wfh, s67, s76, s78, j47) -> list:
    """
    Both gradients on one quarterly path, 2019Q1 on, 2022Q1 omitted, no
    calendar terms, at 22-25. Descriptive: read rule 3 pre-commits
    nothing, so the export carries every quarter for both scores and the
    reading is done on the plot.
    """
    skel = s67.build_skeleton_sex(s76.collapse(allf), s76.YOUNG, j47, "n_emp")
    if skel.empty:
        FAILURES.append("B/empty skeleton")
        return []
    b = s78.with_exposure(skel, ai["exposure"])
    del skel
    gc.collect()
    # with_exposure merges the QUARTILE alone, so neither continuous
    # mean is on the panel yet. Both are merged on here by name.
    a_mix = ai["exposure"][["employer_id", "mix"]].rename(
        columns={"mix": "mix_ai"})
    w_mix = wfh["exposure"][["employer_id", "mix"]].rename(
        columns={"mix": "mix_wfh"})
    b = b.merge(a_mix, on="employer_id", how="inner")
    b = b.merge(w_mix, on="employer_id", how="inner")
    if b.empty:
        FAILURES.append("B/no overlap")
        return []
    # standardised, so the two gradients are on one axis and comparable
    for col, src in (("z_ai", "mix_ai"), ("z_wfh", "mix_wfh")):
        if src not in b.columns:
            FAILURES.append(f"B/missing {src}")
            return []
        s = b[src].astype(float)
        b[col] = (s - s.mean()) / (s.std(ddof=0) or 1.0)
    ym = b["year_month"].astype(str)
    b["q"] = ym.str[:4] + "Q" + (((ym.str[5:7].astype(int) - 1) // 3) + 1).astype(str)
    terms = []
    for q in sorted(b["q"].unique()):
        if q < "2019Q1" or q == "2022Q1":
            continue
        col = f"t_{q}_ai"
        b[col] = (b["q"] == q).astype(int) * b["z_ai"] * b["young"]
        terms.append(col)
        col = f"t_{q}_wfh"
        b[col] = (b["q"] == q).astype(int) * b["z_wfh"] * b["young"]
        terms.append(col)
    n_firms = int(b["employer_id"].nunique())
    print(f"  part B: {len(b):,} rows, {n_firms:,} firms, "
          f"{len(terms)} terms{mc.mem_line(' | ')}")
    try:
        r = mc.run_fepois_multi(b, OUT, tag="s89_timing", terms=terms,
                                fes=j47.FES, cluster="employer_id")
    except BaseException as ex:
        print(f"  part B FAILED: {type(ex).__name__}: {ex}")
        traceback.print_exc()
        r = pd.DataFrame()
    del b
    gc.collect()
    if r.empty:
        FAILURES.append("B/fit")
        return []
    rows = []
    for _, x in r.iterrows():
        nm = str(x["term"])
        if not nm.startswith("t_"):
            continue
        q, which = nm[2:-3], nm.rsplit("_", 1)[-1]
        rows.append({"quarter": q, "score": which, "coef": float(x["coef"]),
                     "se": float(x["se"]), "n_firms": n_firms, "status": "ok"})
    save(rows, "wfh_timing_path.csv")
    return rows


def main():
    mc.Tee(OUT / "89_log.txt")
    t0 = time.time()
    print("=" * 70)
    print("89: AI EXPOSURE AGAINST WORKING FROM HOME, OFF THE DIAGONAL")
    print("=" * 70)
    print("\n".join(READ_RULES))
    print(mc.mem_line("  "))

    s82, s61, s67, s76, s78, l47, l70, j47 = load_modules()
    h47 = j47._h47()

    missing = [y for y in s76.YEARS
               if not (CACHE / f"L_counts_sex_edu_{y}.parquet").exists()]
    if missing:
        raise RuntimeError(
            f"L_counts_sex_edu_* not cached for {missing}. This lane does "
            f"no SQL; run 76 in a lane that may pull, or restore the cache.")

    ai, wfh = build_both(s82, l47, l70, j47)
    halves = overlap(ai["exposure"], wfh["exposure"])

    frames = []
    for y in s76.YEARS:
        c = mc.read_cache(CACHE / f"L_counts_sex_edu_{y}.parquet",
                          require=s76.EDU_COLS + ["n_emp"])
        if c is None:
            raise RuntimeError(f"L_counts_sex_edu_{y}.parquet unreadable.")
        frames.append(s76.tag_frame(c, h47))
        del c
        gc.collect()
    allf = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    print("\n  PART A, the off-diagonal:")
    a_rows = part_a(allf, ai, wfh, halves, s67, s76, s78, j47)
    print("\n  PART B, the two gradients on one path:")
    b_rows = part_b(allf, ai, wfh, s67, s76, s78, j47)
    del allf
    gc.collect()
    drain(s78, "78")

    L = ["AI EXPOSURE AGAINST WORKING FROM HOME, ON THE OCCUPATION ROUTE",
         "=" * 62, "",
         "Both firm scores come from 82's build_exposure() on the same",
         "employers, the same freeze year, the same incumbents aged 31 to",
         "69 and the same floor. Only the occupation score book differs:",
         "the DAIOE generative-AI percentile for one, the Dingel and",
         "Neiman teleworkable share for the other.", ""]
    for n in NOTES:
        if n.startswith("the two firm scores"):
            L += [f"HOW CLOSE THE SCORES ARE: {n}", ""]
    if a_rows:
        L += ["PART A, THE OFF-DIAGONAL. The two discriminating cells are",
              "marked; the diagonal cells are reported without comment."]
        for r in a_rows:
            mark = "  <- discriminating" if r["discriminating"] else ""
            L.append(f"  {r['fit']:<16} {r['coef']:+.4f} ({r['se']:.4f}) "
                     f"t {r['t']:+.2f}   firms {r['n_firms']:,}{mark}")
        L += ["  A cell that is imprecise is imprecise, never a zero;",
              "  every standard error is printed above for that reason."]
        L += [f"  For reference only, and not a gate: the headline at "
              f"{YOUNG_BAND} on the whole panel is {HEADLINE_22_25[0]:+.4f} "
              f"({HEADLINE_22_25[1]:.4f}). Each fit above is a half of that",
              "  panel and cannot reproduce it.", ""]
    if b_rows:
        L += [f"PART B, TIMING: {len(b_rows)} quarter-by-score coefficients in",
              "wfh_timing_path.csv, both scores standardised, 2022Q1 omitted,",
              "no calendar terms. Nothing here is pre-committed: the AI and",
              "teleworkability gradients are plotted together and read off the",
              "figure. A WFH gradient that opens LATER than the AI one is the",
              "return-to-office backlash, which is a finding and not a defect.",
              ""]
    L += ["WHAT THIS LANE DOES NOT CLAIM:",
          "  The 50-and-over gain. Online Appendix III.2 already reports that",
          "  on the continuous route the oldest band's gain loads as strongly",
          "  on teleworkability and is not specific to AI exposure. The young-",
          "  worker margin is kept and the oldest band is conceded. Fixed",
          "  before the run.", ""]
    if NOTES:
        L += ["NOTES:"] + [f"  {n}" for n in NOTES] + [""]
    if FAILURES:
        L += [f"WHAT FAILED: {', '.join(FAILURES)}",
              "A missing row is a missing fit, never a zero.", ""]
    L += READ_RULES + ["",
                       f"Runtime {(time.time() - t0) / 60:.1f} min. "
                       + mc.mem_line("")]
    (OUT / "89_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
