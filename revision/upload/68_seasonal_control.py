#!/usr/bin/env python3
"""
68_seasonal_control.py -- the headline, with the seasonal taken out.

======================================================================
  RUNS IN MONA. No SQL: reads the caches 47h, 47L, 54 and 67 wrote.
  Writes output_68/. Budget two to three hours.
======================================================================

WHY THIS IS NOT OPTIONAL.

Script 64 ran the quarterly path and the pre-period failed its
pre-committed test in all three specifications. The cause is visible in
the coefficients and it is not a trend: the exposure-differential
young-to-older ratio has a strong CALENDAR cycle. The fourth quarter is
positive in every year, +0.076, +0.092, +0.062 and +0.025 from 2021 to
2024, and the first quarter is around -0.065 to -0.077 in every year
including the two before any treatment.

Month-by-age fixed effects do not remove this. They absorb the seasonal
pattern COMMON to all firms; what is left is the part that differs
between exposed and unexposed firms, and that part is what the triple
interaction picks up. Pool a post window whose quarters are mixed
differently from the pre window and the seasonal walks straight into the
estimate. Our pre-period holds one Q4 against two of every other
quarter, so the pooled estimate is inflated.

Comparing like quarters across years gives about -0.03 at 22-25 rather
than the -0.051 the pooled specification reported, with nothing in Q1
and -0.03 to -0.06 in the other three. So the effect survives a crude
seasonal correction and is smaller. This script does it properly.

WHAT IT CHANGES, AND IT IS ONE LINE OF ALGEBRA.

Three extra terms: quarter-of-year interacted with high and young, with
the fourth quarter omitted. The treatment term is then identified from
variation WITHIN calendar quarter across years, which is the comparison
the crude correction above makes by hand. Everything else, the panel,
the exposure, the fixed effects and the dating, is unchanged from 61.

This is also the employment analogue of what Referee 1 asked for on the
posting side, where the answer was industry-by-month fixed effects.

WHAT ELSE IT SETTLES, so that no further run is needed.

  THE MECHANISM. Script 67 found hiring at 22-25 indistinguishable from
  zero on the headline classification while separations rose, which
  contradicts the paper's abstract and reverses what script 63 found on
  the occupational measure. All three margins are re-estimated here with
  the seasonal out, at both young bands, so the mechanism sentence rests
  on one treatment variable and one specification.

  TWO OF THE THREE FITS THAT CRASHED on 21 September, namely separations
  at 22-25 and hires at 26-30. The third, 65's step function at 26-30,
  is not re-run: 65 is now a robustness check on the register rather
  than a headline, its pooled coefficients at both bands already exist,
  and the step decomposition adds nothing there.

  THE GENDER CLAIM, on the stock at 22-25, so the one result that came
  through 67 cleanly is not left resting on a specification whose
  pre-period failed.

  THE ANNUAL PATH, net of the seasonal, as the exhibit the paper needs
  in place of a quarterly figure that a reader cannot interpret without
  this correction.

Output (output_68/):
  seasonal_pooled.csv   the headline by band, outcome and arm
  seasonal_path.csv     year coefficients net of the quarterly cycle
  seasonal_gender.csv   the female differential with the seasonal out
  68_summary.txt
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
OUT = HERE / "output_68"
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR
POST_FROM = "2024-01"
REF_QUARTER = 4              # omitted, so the others read against Q4
# The baseline for every specification: everything before the launch.
# Nothing in this script assumes when the effect began.
BASELINE = f"2021-01 to the month before {mc.CHATGPT_YM}"
FAILURES = []


def opt(label, fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except BaseException as ex:
        print(f"  [optional] {label} FAILED ({type(ex).__name__}: {ex})")
        traceback.print_exc()
        return None


def _mod(name, alias):
    import importlib.util
    spec = importlib.util.spec_from_file_location(alias, HERE / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def quarter_of_year(ym: pd.Series) -> pd.Series:
    return ((ym.str.slice(5, 7).astype(int) - 1) // 3) + 1


def add_seasonal_terms(bal: pd.DataFrame, extra: str = "post") -> tuple:
    """
    The treatment, the Riksbank control, and the calendar cycle.

    THE BASELINE IS THE PRE-CHATGPT WINDOW, and that is the point of this
    version. An earlier draft compared the adoption window against
    everything before January 2024, which assumes the onset is January
    2024. If the effect in fact began in 2023 then part of the baseline is
    treated and every estimate is attenuated. We do not know the onset,
    only that it is later than November 2022 and earlier than 2025, so
    nothing here may assume it.

    So the reference is always 2021-01 to 2022-11, and 2022-12 onward is
    described rather than assumed:

      post     two steps, an interim from the launch and the adoption
               window from POST_FROM, both against the pre-ChatGPT
               baseline. The interim is estimated, not set to zero.
      year     2023, 2024 and 2025 against the same baseline.
      quarter  every calendar quarter from 2022Q4 onward.
      month    the same, month by month.

    Any pooled estimate for a candidate date the reader prefers is a
    weighted average of the quarter or month coefficients, so one path fit
    answers every dating rather than one.

    Why the paths are built this way rather than by adding a control to an
    ordinary event study: a full set of event-time dummies spanning the
    WHOLE panel already spans the calendar cycle, so a seasonal control
    beside them is collinear. Here the dummies start at the launch, the
    cycle is identified off the twenty-three pre-launch months, which see
    each season twice, and each later period reads against the seasonally
    adjusted baseline.

    The fourth quarter, and December, are the omitted seasons. That is a
    normalisation and not a claim about the world.
    """
    ym = bal["year_month"].astype(str)
    hy = bal["high"] * bal["young"]
    q = quarter_of_year(ym)
    post_any = ym >= mc.CHATGPT_YM              # the baseline is everything before
    terms = ["rb_x_high_x_young"]
    bal["rb_x_high_x_young"] = (ym >= mc.RIKSBANK_YM).astype(int) * hy
    if extra == "month":
        mo = ym.str.slice(5, 7).astype(int)
        for mm in range(1, 12):                 # December omitted
            col = f"m{mm:02d}_x_high_x_young"
            bal[col] = (mo == mm).astype(int) * hy
            terms.append(col)
    else:
        for qq in (1, 2, 3):                    # Q4 omitted
            col = f"q{qq}_x_high_x_young"
            bal[col] = (q == qq).astype(int) * hy
            terms.append(col)
    if extra == "post":
        bal["interim_x_high_x_young"] = (post_any & (ym < POST_FROM)).astype(int) * hy
        bal["post_x_high_x_young"] = (ym >= POST_FROM).astype(int) * hy
        terms += ["interim_x_high_x_young", "post_x_high_x_young"]
    elif extra == "year":
        yr = ym.str.slice(0, 4).astype(int)
        for y in (2023, 2024, 2025):
            col = f"y{y}_x_high_x_young"
            bal[col] = ((yr == y) & post_any).astype(int) * hy
            terms.append(col)
        # 2022-12 is neither baseline nor a full year; give it its own term
        bal["dec22_x_high_x_young"] = (ym == mc.CHATGPT_YM).astype(int) * hy
        terms.append("dec22_x_high_x_young")
    elif extra == "quarter":
        lab = ym.str.slice(0, 4) + "Q" + q.astype(str)
        for qq in sorted(lab[post_any].unique()):
            col = f"pq_{qq}_x_high_x_young"
            bal[col] = ((lab == qq) & post_any).astype(int) * hy
            terms.append(col)
    elif extra == "month":
        for mm in sorted(ym[post_any].unique()):
            col = f"pm_{mm.replace('-', '_')}_x_high_x_young"
            bal[col] = ((ym == mm) & post_any).astype(int) * hy
            terms.append(col)
    return bal, terms


def main():
    mc.Tee(OUT / "68_log.txt")
    t0 = time.time()
    print("=" * 70)
    print("68: THE HEADLINE, WITH THE SEASONAL TAKEN OUT")
    print("=" * 70)
    print("  64's pre-period failed its pre-committed test. The cause is a")
    print("  calendar cycle in the exposure-differential ratio: Q4 positive")
    print("  in every year, Q1 negative in every year, before any treatment.")
    print("  Three quarter interactions remove it. Nothing else changes.")
    print(mc.mem_line("  "))

    s61 = _mod("61_redated_triple.py", "s61")
    s67 = _mod("67_gender_on_the_new_design.py", "s67")
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
    expos = {}
    for arm in ("true", "asof"):
        e, _ = j47.incumbent_exposure(frame19, book, "OL_daioe", spec, arm,
                                      s61.TRUNC)
        expos[arm] = e
        print(f"  exposure {arm}: {len(e):,} firms")
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
    pooled_rows, path_rows, gender_rows = [], [], []

    # ---- the headline and the mechanism, all three margins -----------
    for band in j47.YOUNG_BANDS:
        for label, src, value in (("stock", counts, "n_emp"),
                                  ("hires", flows, "n_hire"),
                                  ("seps", flows, "n_sep")):
            if src is None:
                print(f"  {band} {label}: no cache, skipped")
                continue
            t1 = time.time()
            s = src.rename(columns={value: "n_emp"}) if value != "n_emp" else src
            skel = s61.build_skeleton(s, band, j47)
            if value != "n_emp":
                del s
            if skel.empty:
                continue
            print(f"\n  {band} {label}: skeleton {len(skel):,} rows "
                  f"({(time.time()-t1)/60:.1f} min)")
            arms = ("true", "asof") if (band == "22-25"
                                        and label == "stock") else ("true",)
            for arm in arms:
                b = skel.merge(expos[arm][["employer_id", "fq"]],
                               on="employer_id", how="inner")
                if b.empty:
                    continue
                b["high"] = (b["fq"] == 4).astype(int)
                b, terms = add_seasonal_terms(b, "post")
                t2 = time.time()
                r = mc.run_fepois_multi(
                    b, OUT, tag=f"s68_{band.replace('-','_')}_{label}_{arm}",
                    terms=terms, fes=j47.FES)
                if r.empty:
                    FAILURES.append(f"{band}/{label}/{arm}")
                    print(f"    {arm}: FAILED, recorded and skipped")
                else:
                    g = r.set_index("term")
                    for t_ in terms:
                        if t_ not in g.index:
                            continue
                        pooled_rows.append(
                            {"young_band": band, "outcome": label, "arm": arm,
                             "term": t_, "coef": float(g.loc[t_, "coef"]),
                             "se": float(g.loc[t_, "se"]),
                             "n_obs": int(g.loc[t_, "n_obs"]),
                             "status": str(g.loc[t_].get("status", "ok"))})
                    pd.DataFrame(pooled_rows).to_csv(
                        OUT / "seasonal_pooled.csv", index=False)
                    x = g.loc["post_x_high_x_young"]
                    print(f"    {arm:<5} post {float(x['coef']):+.4f} "
                          f"({float(x['se']):.4f}) t "
                          f"{float(x['coef'])/max(float(x['se']),1e-12):+.2f}"
                          f"   [{(time.time()-t2)/60:.1f} min]")
                    for qq in (1, 2, 3):
                        c = g.loc[f"q{qq}_x_high_x_young"]
                        print(f"          Q{qq} against Q4 "
                              f"{float(c['coef']):+.4f} ({float(c['se']):.4f})")
                # The paths, all cleaned of the cycle, on the stock only.
                #
                # MONTHLY NOW RUNS AT 26-30 TOO. It was 22-25 alone,
                # because monthly carries thirty terms on a thirty-plus
                # million row panel and 22-25 was "the cell the claim is
                # about". The claim has since become the SPREADING
                # pattern, so 26-30 is half of it, and the lead figure
                # has to mix a monthly series with a quarterly one to
                # show two bands at all.
                #
                # The memory reasoning has also moved on. What kills a
                # fit is the number of FIXED EFFECTS, not the number of
                # terms: 30.5M rows with three effects fitted on
                # 21 September while 28.5M with four died. Terms are
                # columns in X, not demeaned dimensions. And every fit
                # now falls 8 -> 2 -> 1 threads before giving up, so a
                # squeeze costs wall-clock rather than the coefficient.
                # If it still dies it is recorded and the rest of the
                # lane is unaffected.
                if label == "stock" and arm == "true":
                    shapes = ["year", "quarter", "month"]
                    for shape in shapes:
                        b, pterms = add_seasonal_terms(b, shape)
                        t3 = time.time()
                        rp = mc.run_fepois_multi(
                            b, OUT,
                            tag=f"p68_{shape}_{band.replace('-','_')}",
                            terms=pterms, fes=j47.FES)
                        if rp.empty:
                            FAILURES.append(f"path/{shape}/{band}")
                            print(f"    {shape} path: FAILED, recorded")
                            continue
                        gp = rp.set_index("term")
                        pref = {"year": "y2", "quarter": "pq_",
                                "month": "pm_"}[shape]
                        got = [t_ for t_ in pterms
                               if t_.startswith(pref) and t_ in gp.index]
                        for t_ in got:
                            lab = (t_.split("_x_high")[0]
                                   .removeprefix("pq_").removeprefix("pm_")
                                   .removeprefix("y"))
                            path_rows.append(
                                {"young_band": band, "shape": shape,
                                 "period": lab.replace("_", "-"),
                                 "coef": float(gp.loc[t_, "coef"]),
                                 "se": float(gp.loc[t_, "se"]),
                                 "status": str(gp.loc[t_].get("status", "ok"))})
                        pd.DataFrame(path_rows).to_csv(
                            OUT / "seasonal_path.csv", index=False)
                        print(f"    {shape} path, cycle removed "
                              f"[{(time.time()-t3)/60:.1f} min]:")
                        for d in path_rows:
                            if d["young_band"] == band and d["shape"] == shape:
                                print(f"          {d['period']:<8} "
                                      f"{d['coef']:+.4f} ({d['se']:.4f})")
                del b
                gc.collect()
            del skel
            gc.collect()

    # ---- the gender differential, seasonal out -----------------------
    if counts is not None:
        stock_sex = load("L_counts_sex",
                         require=["employer_id", "year_month", "age_group",
                                  "gender", "n_emp"])
        if stock_sex is None:
            print("\n  gender: L_counts_sex_* missing, skipped")
            FAILURES.append("gender/no cache")
        else:
            t1 = time.time()
            skel = s67.build_skeleton_sex(stock_sex, "22-25", j47, "n_emp")
            del stock_sex
            gc.collect()
            if not skel.empty:
                b = skel.merge(expos["true"][["employer_id", "fq"]],
                               on="employer_id", how="inner")
                b["high"] = (b["fq"] == 4).astype(int)
                ym = b["year_month"].astype(str)
                q = quarter_of_year(ym)
                hy, hyf = b["high"] * b["young"], b["high"] * b["young"] * b["female"]
                b["rb_x_high_x_young"] = (ym >= mc.RIKSBANK_YM).astype(int) * hy
                b["post_x_high_x_young"] = (ym >= POST_FROM).astype(int) * hy
                b["post_x_high_x_female"] = ((ym >= POST_FROM).astype(int)
                                             * b["high"] * b["female"])
                b["post_x_high_x_young_x_female"] = (ym >= POST_FROM).astype(int) * hyf
                terms = ["rb_x_high_x_young", "post_x_high_x_young",
                         "post_x_high_x_female",
                         "post_x_high_x_young_x_female"]
                for qq in (1, 2, 3):
                    b[f"q{qq}_x_high_x_young"] = (q == qq).astype(int) * hy
                    b[f"q{qq}_x_high_x_young_x_female"] = (q == qq).astype(int) * hyf
                    terms += [f"q{qq}_x_high_x_young",
                              f"q{qq}_x_high_x_young_x_female"]
                print(f"\n  gender, 22-25 stock: {len(b):,} rows "
                      f"({(time.time()-t1)/60:.1f} min to build)")
                r = mc.run_fepois_multi(b, OUT, tag="g68_22_25_stock",
                                        terms=terms, fes=j47.FES)
                if r.empty:
                    FAILURES.append("gender/22-25/stock")
                else:
                    g = r.set_index("term")
                    for t_ in ("post_x_high_x_young",
                               "post_x_high_x_young_x_female"):
                        if t_ in g.index:
                            gender_rows.append(
                                {"term": t_, "coef": float(g.loc[t_, "coef"]),
                                 "se": float(g.loc[t_, "se"]),
                                 "status": str(g.loc[t_].get("status", "ok"))})
                    pd.DataFrame(gender_rows).to_csv(
                        OUT / "seasonal_gender.csv", index=False)
                    for d in gender_rows:
                        print(f"    {d['term']:<32} {d['coef']:+.4f} "
                              f"({d['se']:.4f}) t "
                              f"{d['coef']/max(d['se'],1e-12):+.2f}")
                del b, skel
                gc.collect()

    # ---- summary ------------------------------------------------------
    P = pd.DataFrame(pooled_rows)
    lines = ["THE HEADLINE, WITH THE SEASONAL TAKEN OUT", "=" * 52, "",
             "61's design and dating, plus three quarter-of-year",
             "interactions with the fourth quarter omitted. The treatment is",
             "then identified within calendar quarter across years.", "",
             "Compare every number here with 61 and 67. Where they differ,",
             "the difference is the seasonal, and this one is the estimate to",
             "quote.", ""]
    if not P.empty:
        pp = P[P["term"] == "post_x_high_x_young"]
        lines.append("POOLED from 2024-01, seasonal removed:")
        for _, r in pp.iterrows():
            lines.append(f"  {r['young_band']:<6} {r['outcome']:<6} "
                         f"{r['arm']:<5} {r['coef']:+.4f} ({r['se']:.4f}) t "
                         f"{r['coef']/max(r['se'],1e-12):+.2f}")
        lines.append("")
        qs = P[P["term"].str.startswith("q")]
        if not qs.empty:
            lines += ["THE SEASONAL ITSELF, each quarter against Q4. This is",
                      "the thing 64's pre-period test caught, now measured:"]
            for (band, oc, arm), grp in qs.groupby(["young_band", "outcome",
                                                    "arm"]):
                bits = "  ".join(f"{r['term'][:2]} {r['coef']:+.4f}"
                                 for _, r in grp.iterrows())
                lines.append(f"  {band:<6} {oc:<6} {arm:<5} {bits}")
            lines.append("")
        t = pp[(pp.arm == "true") & (pp.young_band == "22-25")
               & (pp.outcome == "stock")]
        a = pp[(pp.arm == "asof") & (pp.young_band == "22-25")
               & (pp.outcome == "stock")]
        if len(t) and len(a):
            lines += [f"ARTEFACT at 22-25 on the stock: "
                      f"{float(a['coef'].iloc[0]) - float(t['coef'].iloc[0]):+.4f}",
                      ""]
    if path_rows:
        lines += ["PATHS NET OF THE SEASONAL. Every coefficient reads against",
                  "the seasonally adjusted PRE-CHATGPT baseline, 2021-01 to",
                  "2022-11. Nothing here assumes when the effect began; the",
                  "path is what tells you. Note that 2022Q4 is part launch",
                  "month and part baseline, so read it as neither.", ""]
        for shape in ("year", "quarter", "month"):
            for band in sorted({d["young_band"] for d in path_rows}):
                rows = [d for d in path_rows
                        if d["young_band"] == band and d["shape"] == shape]
                if not rows:
                    continue
                lines.append(f"  {shape}, {band}:")
                for d in rows:
                    star = "" if abs(d["coef"]) < 2 * d["se"] else "  *"
                    lines.append(f"    {d['period']:<8} {d['coef']:+.4f} "
                                 f"({d['se']:.4f}){star}")
                lines.append("")
    if gender_rows:
        lines += ["GENDER at 22-25 on the stock, seasonal removed:"]
        for d in gender_rows:
            nm = ("young, pooled over sex" if d["term"].endswith("young")
                  else "female differential")
            lines.append(f"  {nm:<26} {d['coef']:+.4f} ({d['se']:.4f}) t "
                         f"{d['coef']/max(d['se'],1e-12):+.2f}")
        lines.append("")
    if FAILURES:
        lines += ["FITS THAT FAILED: " + "; ".join(FAILURES),
                  "A missing row is a missing fit, never a zero.", ""]
    lines += [
        "READ THIS BEFORE QUOTING ANY OF IT:",
        "  1. These supersede 61 and 67 for anything quoted in the paper.",
        "     61's pooled estimates are not wrong arithmetic; they are a",
        "     specification whose pre-period failed a test we had written",
        "     down beforehand, and the failure was seasonal.",
        "  2. Expect the estimates to SHRINK. A hand comparison of like",
        "     quarters across years put 22-25 near -0.03 against 61's",
        "     -0.051. If the seasonal-controlled figure is far from -0.03",
        "     something other than the seasonal is going on and it should",
        "     be chased before publication.",
        "  3. The Q1-to-Q3 coefficients are worth reporting in the paper.",
        "     An exposure-differential seasonal is itself a fact about how",
        "     exposed firms hire and shed young workers over the year.",
        "  4. The mechanism sentence depends on the hires and seps rows",
        "     here and on nothing else. 63's flow results use a different",
        "     exposure measure and cannot settle it.",
        "  5. 2025 is a half year, and 2022Q4 straddles the launch, so",
        "     neither should be read as a clean period.",
        "  6. NOTHING HERE ASSUMES THE ONSET. The baseline is the",
        "     pre-ChatGPT window and every later period is estimated, so a",
        "     reader who thinks the effect began in mid-2023 can read that",
        "     off the path rather than argue with our dating. The pooled",
        "     estimate for any candidate date is a weighted average of the",
        "     quarter or month coefficients.",
        "  7. The paths read against the baseline as a whole, not against",
        "     an adjacent period, so they answer when the level shifted and",
        "     not how fast it moved. A full event study cannot be cleaned",
        "     this way: event-time dummies spanning the whole panel already",
        "     span the calendar cycle. What identifies the cycle separately",
        "     is that the twenty-three pre-launch months see each season",
        "     twice.",
        "", f"Runtime {(time.time()-t0)/60:.1f} min. " + mc.mem_line()]
    (OUT / "68_summary.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print("\n68 done.")


if __name__ == "__main__":
    main()
