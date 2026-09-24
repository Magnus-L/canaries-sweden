#!/usr/bin/env python3
"""
75_reference_window.py: the headline read against the months before the
rate rise, so that the level after adoption has a standard error.

QUESTION
Script 68 keeps the Riksbank interaction switched on through the post
period, so its adoption coefficient is the step from the level of April to
November 2022. A reader who hears "employment falls 4 per cent" hears a
change from before the rate rise, which is the sum of the tightening step
and the adoption step and has no standard error in script 68's export.
This script re-estimates the same specification with the Riksbank term as
a window, so that the interim and post terms read directly against
January 2021 to March 2022, and exports the clustered covariance of every
fit.

DESIGN
Panel, exposure and fixed effects are script 68's: script 61's skeleton
from January 2021 to June 2025, the headline classification, employer-by-
month, employer-by-age and month-by-age effects, Poisson pseudo-maximum
likelihood, standard errors clustered by employer. Terms
(add_window_terms): RBW x High x Young equal to one in April to November
2022 only; Q1, Q2 and Q3 x High x Young with the fourth quarter omitted;
Interim x High x Young for December 2022 to December 2023; Post x High x
Young from January 2024. The post coefficient is the level after adoption
relative to the pre-hike months, and the difference between the post and
interim coefficients equals gamma_2 minus gamma_0 of Equation (2), with
its standard error from the exported covariance. Four fits: the stock at
22-25 and 26-30, hires and separations at 22-25.

Reconciliation rule fixed before the run: the window post coefficient
must lie within one of its own standard errors of the sum of script 68's
cumulative Riksbank and post coefficients, since the two specifications
describe the same fitted means; a failure means the panels differ and
nothing is quoted. The headline stays script 68's adoption step; the
number here is the level, reported beside it.

INPUTS AND OUTPUTS
Reads the caches edu_hr_weights_2019 to 2021 and edu_hr_2019 (script 47h),
L_counts_2021 to 2025 (script 47L) and flows_2021 to 2025 (script 54);
performs no SQL. Writes to output_75/: reference_window.csv (every term
of every fit), vcov_s75_<band>_<outcome>.csv and 75_summary.txt.

IN THE PAPER
The coefficients written here are on the education route and are read by
the exhibit builders only for comparison. add_window_terms is imported by
script 83 (Part B of that script), which gives the occupation-route level
after adoption against the pre-hike months that Table 1 and Online
Appendix III.2 (Table A11) report.
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

# Script 68's cumulative coefficients (rb, interim, post), from its export.
# Used only for the reconciliation check; never written into a result.
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
