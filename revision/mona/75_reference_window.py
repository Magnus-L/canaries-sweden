#!/usr/bin/env python3
"""
75_reference_window.py -- the headline read against the months BEFORE the
                          rate hike, so the level after adoption has a
                          standard error of its own.

======================================================================
  RUNS IN MONA. No SQL: reads the caches 47h, 47L and 54 already wrote.
  Writes output_75/. Four fits, two to three hours.
======================================================================

WHAT 68 MEASURES, STATED EXACTLY, AND WHY THIS SCRIPT EXISTS.

Script 68 carries `rb x high x young`, equal to one in EVERY month from
April 2022 onward, beside the interim, post, year and quarter terms. The
Riksbank term therefore stays switched on through the whole post period,
and every later coefficient is a step measured from the level of April to
November 2022, the seven tightening months before ChatGPT. That is the
right estimand for the paper's question (does anything ADDITIONAL happen
when generative AI arrives?) and it is exactly how beta_2 on the posting
margin is read. But it is not the number a reader hears when the paper
says "employment falls 4.0 per cent": that reader hears a change from
before the hike.

At 22-25 the sequence in 68 is +0.0214 (0.0078) from April 2022, then
-0.0143 (0.0095) for the thirteen months after the launch, then -0.0408
(0.0150) from January 2024. The level after adoption relative to the
months before the hike is the sum of the first and the last, about
-0.019, and 68 exported no covariance, so that sum has no standard error.
This script gives it one, the cheap way: the Riksbank term becomes a
WINDOW, one in April to November 2022 and zero after, so the interim and
post terms read directly against January 2021 to March 2022.

Nothing else changes. Same skeleton (61), same exposure (47h's OL_daioe,
frozen 2019, incumbents 31+), same three fixed effects (47j), same three
quarter-of-year interactions with the fourth quarter omitted (68), same
Poisson wrapper. The four fits are the ones the paper quotes: the stock
at 22-25 and 26-30, and hires and separations at 22-25.

THE READ RULE, FIXED BEFORE THE RUN.

  1. RECONCILED if, for each fit, the window-specification post
     coefficient lies within one of its own standard errors of the sum
     of 68's cumulative rb and post coefficients. The two
     specifications describe the same fitted means, so this is
     arithmetic, and a failure means the panels differ and nothing here
     may be quoted.
  2. The paper's headline stays the ADOPTION STEP from 68 (-0.0408),
     stated as the step from the tightening level, with gamma_1 beside
     it. The number here is the LEVEL after adoption relative to the
     pre-hike months and goes in Table 1 and the response letter as
     such. Both are reported; neither replaces the other.
  3. For hires, the window arm says whether hiring of the young in
     exposed firms after adoption is above, at or below its pre-hike
     level. 68's -0.0032 says only that it did not fall from its 2022
     level. Write whichever the interval supports, with the bound.

The clustered covariance is now exported by the R wrapper for every fit
(vcov_<tag>.csv), so any further combination of terms can be given a
standard error without another trip.

Output (output_75/):
  reference_window.csv   every term, every fit, both arms of the
                         reconciliation
  vcov_s75_*.csv         the clustered covariance per fit
  75_summary.txt
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
OUT = HERE / "output_75"
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR

POST_FROM = "2024-01"
RB_FROM, RB_TO = mc.RIKSBANK_YM, mc.CHATGPT_YM     # the window: 2022-04 .. 2022-11
JOBS = [("22-25", "stock"), ("26-30", "stock"),
        ("22-25", "hires"), ("22-25", "seps")]

# 68's cumulative coefficients (rb, interim, post), from the export of
# 21 September 21:52, round3_20260921-2152-lane14-seasonal-complete.
# Used ONLY for the reconciliation check; never written into a result.
S68 = {
    ("22-25", "stock"): (0.0214, -0.0143, -0.0408),
    ("26-30", "stock"): (0.0222, -0.0035, -0.0394),
    ("22-25", "hires"): (0.0523, -0.0270, -0.0032),
    ("22-25", "seps"):  (0.0081,  0.0332,  0.0787),
}
FAILURES = []


def _mod(fname, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def quarter_of_year(ym: pd.Series) -> pd.Series:
    return ((ym.str.slice(5, 7).astype(int) - 1) // 3) + 1


def add_window_terms(bal: pd.DataFrame) -> tuple:
    """
    68's pooled term set with ONE change: the Riksbank term is a window.

      rbw      April to November 2022 only. The seven tightening months
               before ChatGPT, against January 2021 to March 2022.
      q1..q3   the calendar cycle, fourth quarter omitted, as in 68.
      interim  December 2022 to December 2023, against the pre-hike
               months directly.
      post     January 2024 onward, against the pre-hike months directly.
               This is the level after adoption.
    """
    ym = bal["year_month"].astype(str)
    hy = bal["high"] * bal["young"]
    q = quarter_of_year(ym)
    post_any = ym >= mc.CHATGPT_YM
    bal["rbw_x_high_x_young"] = ((ym >= RB_FROM) & (ym < RB_TO)).astype(int) * hy
    terms = ["rbw_x_high_x_young"]
    for qq in (1, 2, 3):
        col = f"q{qq}_x_high_x_young"
        bal[col] = (q == qq).astype(int) * hy
        terms.append(col)
    bal["interim_x_high_x_young"] = (post_any & (ym < POST_FROM)).astype(int) * hy
    bal["post_x_high_x_young"] = (ym >= POST_FROM).astype(int) * hy
    terms += ["interim_x_high_x_young", "post_x_high_x_young"]
    return bal, terms


def reconcile(rows: list) -> list:
    """Rule 1: window post against 68's rb + post, within one SE."""
    out = []
    df = pd.DataFrame(rows)
    if df.empty:
        return ["NO VERDICT: no fit came back."]
    for band, label in JOBS:
        sub = df[(df.young_band == band) & (df.outcome == label)]
        if sub.empty:
            out.append(f"  {band} {label}: not fitted")
            continue
        g = sub.set_index("term")
        if "post_x_high_x_young" not in g.index:
            out.append(f"  {band} {label}: post term missing")
            continue
        pw, se = float(g.loc["post_x_high_x_young", "coef"]), \
            float(g.loc["post_x_high_x_young", "se"])
        rb68, _, post68 = S68[(band, label)]
        expect = rb68 + post68
        ok = abs(pw - expect) <= se
        out.append(f"  {band} {label}: level after adoption vs pre-hike months "
                   f"{pw:+.4f} ({se:.4f}) t {pw/max(se,1e-12):+.2f}; 68's rb + post "
                   f"= {expect:+.4f}; {'RECONCILED' if ok else 'NOT RECONCILED'}")
        if "rbw_x_high_x_young" in g.index:
            rw, rse = float(g.loc["rbw_x_high_x_young", "coef"]), \
                float(g.loc["rbw_x_high_x_young", "se"])
            out.append(f"      tightening window Apr-Nov 2022 {rw:+.4f} ({rse:.4f}); "
                       f"68's cumulative rb {rb68:+.4f}")
        if "interim_x_high_x_young" in g.index:
            iw, ise = float(g.loc["interim_x_high_x_young", "coef"]), \
                float(g.loc["interim_x_high_x_young", "se"])
            out.append(f"      Dec 2022 to Dec 2023 vs pre-hike {iw:+.4f} ({ise:.4f})")
    return out


def main():
    mc.Tee(OUT / "75_log.txt")
    t0 = time.time()
    print("=" * 70)
    print("75: THE HEADLINE AGAINST THE PRE-HIKE MONTHS")
    print("=" * 70)
    print("  68's Riksbank term is cumulative, so its post coefficients read")
    print("  from the April-November 2022 level. Here the term is a window,")
    print("  so interim and post read from January 2021 to March 2022.")
    print(mc.mem_line("  "))

    s61 = _mod("61_redated_triple.py", "s61")
    j47 = s61._j47()
    h47 = j47._h47()

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
    frame19 = mc.read_cache(CACHE / "edu_hr_2019.parquet",
                            require=h47.YEAR_COLS + ["n_emp"])
    if frame19 is None:
        raise RuntimeError("edu_hr_2019.parquet missing: run 47h first.")
    expo, _ = j47.incumbent_exposure(frame19, book, "OL_daioe", spec, "true",
                                     s61.TRUNC)
    print(f"  exposure: {len(expo):,} firms")
    del frame19
    gc.collect()

    def load(prefix, require=None):
        out = []
        for y in s61.PANEL_YEARS:
            c = mc.read_cache(CACHE / f"{prefix}_{y}.parquet", require=require)
            if c is None:
                return None
            out.append(c)
        return pd.concat(out, ignore_index=True) if out else None

    counts = load("L_counts")
    if counts is None:
        raise RuntimeError("L_counts_* missing: run 47L first.")
    flows = load("flows")
    rows = []

    for band, label in JOBS:
        src, value = {"stock": (counts, "n_emp"), "hires": (flows, "n_hire"),
                      "seps": (flows, "n_sep")}[label]
        if src is None:
            print(f"  {band} {label}: no cache, skipped")
            FAILURES.append(f"{band}/{label}/no cache")
            continue
        t1 = time.time()
        s = src.rename(columns={value: "n_emp"}) if value != "n_emp" else src
        skel = s61.build_skeleton(s, band, j47)
        if value != "n_emp":
            del s
        if skel.empty:
            FAILURES.append(f"{band}/{label}/empty")
            continue
        b = skel.merge(expo[["employer_id", "fq"]], on="employer_id", how="inner")
        del skel
        gc.collect()
        b["high"] = (b["fq"] == 4).astype(int)
        b, terms = add_window_terms(b)
        n_firms = int(b["employer_id"].nunique())
        print(f"\n  {band} {label}: {len(b):,} rows, {n_firms:,} firms "
              f"({(time.time()-t1)/60:.1f} min to build){mc.mem_line(' | ')}")
        tag = f"s75_{band.replace('-', '_')}_{label}"
        try:
            r = mc.run_fepois_multi(b, OUT, tag=tag, terms=terms, fes=j47.FES)
        except BaseException as ex:
            print(f"  {tag} FAILED: {type(ex).__name__}: {ex}")
            traceback.print_exc()
            r = pd.DataFrame()
        del b
        gc.collect()
        if r.empty:
            FAILURES.append(f"{band}/{label}")
            print("    FAILED, recorded and skipped")
            continue
        g = r.set_index("term")
        for t_ in terms:
            if t_ not in g.index:
                continue
            rows.append({"young_band": band, "outcome": label, "term": t_,
                         "coef": float(g.loc[t_, "coef"]),
                         "se": float(g.loc[t_, "se"]),
                         "n_obs": int(g.loc[t_, "n_obs"]), "n_firms": n_firms,
                         "status": str(g.loc[t_].get("status", "ok"))})
        pd.DataFrame(rows).to_csv(OUT / "reference_window.csv", index=False)
        x = g.loc["post_x_high_x_young"]
        print(f"    post (level after adoption vs pre-hike) {float(x['coef']):+.4f} "
              f"({float(x['se']):.4f})   vcov: {r.attrs.get('vcov', 'not written')}")

    lines = ["THE HEADLINE AGAINST THE PRE-HIKE MONTHS", "=" * 52, "",
             "68's specification with the Riksbank term as a WINDOW (April to",
             "November 2022) instead of a cumulative switch. Interim and post",
             "then read directly against January 2021 to March 2022, and the",
             "post coefficient is the LEVEL after adoption, with its own",
             "standard error. The calendar cycle is removed exactly as in 68.", "",
             "RECONCILIATION (rule 1: window post within one SE of 68's rb + post):"]
    lines += reconcile(rows)
    lines += ["",
              "READ THIS BEFORE QUOTING ANY OF IT:",
              "  1. The paper's headline stays 68's ADOPTION STEP, -0.0408 at",
              "     22-25, and must be written as the additional change from the",
              "     tightening level, with 68's +0.0214 rb term beside it.",
              "  2. The post coefficient here is the level after adoption",
              "     relative to the pre-hike months. It goes in Table 1 and the",
              "     response letter as that, and nowhere as 'the effect'.",
              "  3. Hires: 68 says hiring did not fall from its 2022 level. The",
              "     window arm says where it stands against the pre-hike level.",
              "     Quote the interval, not the point.",
              "  4. The clustered covariance of every fit is in vcov_s75_*.csv;",
              "     any further linear combination gets its SE from there.",
              ""]
    if FAILURES:
        lines += ["FITS THAT FAILED: " + "; ".join(FAILURES),
                  "A missing row is a missing fit, never a zero.", ""]
    lines += [f"Runtime {(time.time()-t0)/60:.1f} min. " + mc.mem_line()]
    (OUT / "75_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    mc.runlog("75_reference_window", 0, (time.time() - t0) / 60)
    print("\n75 done.")


if __name__ == "__main__":
    main()
