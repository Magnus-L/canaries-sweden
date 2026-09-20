#!/usr/bin/env python3
"""
60_treatment_date.py -- when did it start, and does the answer depend on
                        assuming it?

======================================================================
  RUNS IN MONA. No SQL. Reads the caches 47L and 54 already wrote.
  Writes output_60/.
======================================================================

THE PROBLEM WITH EVERY ESTIMATE WE HAVE SO FAR.

All of them define the post period as December 2022, the ChatGPT launch.
That pools thirteen months of 2023 with eighteen months of 2024 and 2025.
If the labour-market response began later than the launch, which the
qualitative evidence from firms, unions and student associations suggests
and which is what diffusion usually looks like, then every pooled
coefficient averages an untreated period with a treated one. That
attenuates towards zero mechanically, and the longer the untreated period
the worse it gets.

So the tight bound we have reported, roughly -0.01 to +0.02 on the stock,
is a bound on the AVERAGE over the whole window. It is not evidence
against a late effect, and reading it as though it were would be a
mistake.

WHAT THIS SCRIPT DOES.

PART 1, PRE-SPECIFIED. Four candidate dates, each justified by something
outside our data and outside our outcome, all four reported whatever they
show. Choosing among them after seeing the results would destroy the
inference, so the choice is made here, in the source, before the run.

    2022-12  ChatGPT public launch, 30 November 2022. A capability event.
    2023-04  the quarter after GPT-4, 14 March 2023. The first release
             widely described as useful for professional work.
    2023-10  the midpoint of the interval over which SWEDISH FIRM ADOPTION
             more than doubled. See below.
    2024-01  the far end of that interval, and the date the qualitative
             accounts point to.

Employment responds to firms USING the technology, not to a model being
released, so the diffusion anchor is the one to weight, and for Sweden it
is measured rather than inferred. SCB's IT-anvandning i foretag puts the
share of enterprises with ten or more employees using AI technology at

    10.4 per cent in 2023
    25.2 per cent in 2024
    35.0 per cent in 2025

so adoption roughly two and a half times over between the 2023 and 2024
observations, the largest proportional jump in the series, and then grew
more slowly. Whatever the survey's reference convention, and SCB's public
pages do not state whether the figure describes the survey year or the
year before it, the transition is centred on the second half of 2023 and
the first half of 2024. That is the interval the grid below is built to
cover, and 2023-10 and 2024-01 bracket it.

Verified against SCB's own statistical news releases on 20 Sep 2026. The
Bick, Blandin and Deming series measures individual use in the United
States and is NOT used here: it dates a different population.

PART 2, EXPLORATORY. The same coefficient over a grid of candidate dates,
reported as a profile.

This is not only a robustness check. The profile has a known shape. If
the true break is at date D, assuming an earlier date puts untreated
months in the post window, and assuming a later one puts treated months
in the pre window; both attenuate towards zero. So the profile should
have an INTERIOR MINIMUM at the truth, and the data can date the effect
rather than us asserting it. A profile that simply declines towards the
end of the window is itself evidence for a late onset even where no
single coefficient is significant.

TWO CAUTIONS, BUILT IN.

Searching over dates invalidates naive inference. The four pre-specified
dates carry the inference; the grid is exploratory and is labelled so in
every output. A proper sup-Wald test with Andrews critical values would
fix this and is not attempted here.

The post window shrinks as the date moves later, so standard errors grow
along the profile for arithmetic reasons that have nothing to do with the
effect. Every profile row carries its standard error and its number of
post months, and a reader who ignores them will misread the picture.

Output (output_60/):
  prespecified.csv   four dates x outcome x age band, with SEs
  profile.csv        the grid, with SEs and post-window length
  60_summary.txt
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
OUT = HERE / "output_60"
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR

YOUNG = "22-25"
FES = ("fe_emp_t", "fe_emp_age", "fe_t_age")

# Pre-specified, with the event each one marks. Fixed before the run.
PRESPEC = {
    "2022-12": "ChatGPT public launch (30 Nov 2022)",
    "2023-04": "quarter after GPT-4 (14 Mar 2023)",
    "2023-10": "midpoint of SCB's 10.4 to 25.2 per cent adoption jump",
    "2024-01": "far end of that jump; the qualitative accounts' onset",
}
# Exploratory grid: every quarter from the launch to a year before the end,
# so the shortest post window is still four quarters.
GRID = [f"{y}-{m:02d}" for y in range(2022, 2025) for m in (1, 4, 7, 10)]
GRID = [d for d in GRID if "2022-12" <= d <= "2024-07"]
MIN_POST_MONTHS = 11
FAILURES = []


def opt(label, fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except BaseException as ex:
        print(f"  [optional] {label} FAILED ({type(ex).__name__}: {ex})")
        traceback.print_exc()
        return None


def _mod(name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name[:4], HERE / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fit_at(bal: pd.DataFrame, outcome: str, date: str, tag: str) -> dict:
    """
    One coefficient on post(date) x exposure x young, with the Riksbank
    control kept at its own date so the two shocks are not conflated.

    The age interaction is what the paper is about, so the term estimated
    is the YOUNG DIFFERENTIAL, identified inside the employer against its
    own other age bands, exactly as everywhere else in this round.
    """
    b = bal.copy()
    ym = b["year_month"].astype(str)
    b["is_young"] = (b["age_group"] == YOUNG).astype(int)
    b["post_rb_x_expo"] = (ym >= mc.RIKSBANK_YM).astype(int) * b["expo_z"]
    post = (ym >= date).astype(int)
    b["post_x_expo"] = post * b["expo_z"]
    b["post_x_expo_y"] = post * b["expo_z"] * b["is_young"]
    b["n_emp"] = b[outcome]
    n_post = ym[post == 1].nunique()
    terms = ["post_rb_x_expo", "post_x_expo", "post_x_expo_y"]
    r = mc.run_fepois_multi(b, OUT, tag=tag, terms=terms, fes=FES)
    del b
    gc.collect()
    row = {"date": date, "outcome": outcome, "n_post_months": n_post,
           "coef": np.nan, "se": np.nan, "status": "no_output"}
    if not r.empty and (r["term"] == "post_x_expo_y").any():
        g = r[r["term"] == "post_x_expo_y"].iloc[0]
        row.update(coef=float(g["coef"]), se=float(g["se"]),
                   n_obs=int(g["n_obs"]), status=str(g.get("status", "ok")))
    else:
        FAILURES.append(tag)
    return row


def main():
    mc.Tee(OUT / "60_log.txt")
    t0 = time.time()
    print("=" * 70)
    print("60: WHEN DID IT START?")
    print("=" * 70)
    print("  Every earlier estimate pools 2023, when the qualitative")
    print("  accounts say nothing was happening, with 2024 and 2025, when")
    print("  they say it was. That attenuates towards zero for arithmetic")
    print("  reasons. This re-dates the treatment instead of assuming it.")
    print("  PRE-SPECIFIED DATES, fixed before the run:")
    for d, why in PRESPEC.items():
        print(f"    {d}  {why}")
    print("  The grid below is EXPLORATORY: searching over dates invalidates")
    print("  naive inference, so the four above carry it.")
    print(mc.mem_line("  "))

    l47, s54 = _mod("47L_age_baseline_exposure.py"), _mod("54_hiring_flows.py")
    base = mc.read_cache(CACHE / "L_baseline_2019.parquet")
    if base is None:
        raise RuntimeError("L_baseline_2019.parquet missing: run 47L first.")
    daioe = pd.read_stata(str(Path(mc.SHARE) / "daioe_quartiles.dta"))
    daioe["ssyk4"] = daioe["ssyk4"].astype(str).str.zfill(4)
    daioe = daioe.rename(columns={"pctl_rank_genai": "score"})[["ssyk4",
                                                               "score"]]
    expo = l47.build_exposure(base, daioe)

    panels = {}
    fl = [f for f in (mc.read_cache(CACHE / f"flows_{y}.parquet",
                                    require=s54.FLOW_COLS)
                      for y in s54.YEARS) if f is not None]
    if fl:
        bf = s54.build_panel(pd.concat(fl, ignore_index=True), expo)
        panels["hires"] = (bf, "n_hire")
        panels["seps"] = (bf, "n_sep")
    del fl
    cnt = [c for c in (mc.read_cache(CACHE / f"L_counts_{y}.parquet")
                       for y in l47.YEARS) if c is not None]
    if cnt:
        panels["stock"] = (l47.build_panel(pd.concat(cnt, ignore_index=True),
                                           expo), "n_emp")
    del cnt
    gc.collect()
    if not panels:
        raise RuntimeError("no cached panel: run 54 and 47L first.")

    # ---- Part 1: the four pre-specified dates ----
    print("\nPRE-SPECIFIED DATES")
    rows = []
    for date in PRESPEC:
        for label, (bal, outcome) in panels.items():
            t1 = time.time()
            r = fit_at(bal, outcome, date, f"d60_pre_{date.replace('-','_')}_{label}")
            r["spec"], r["why"] = "prespecified", PRESPEC[date]
            r["outcome"] = label
            rows.append(r)
            pd.DataFrame(rows).to_csv(OUT / "prespecified.csv", index=False)
            t = r["coef"] / max(r["se"], 1e-12) if np.isfinite(r["coef"]) else np.nan
            print(f"  {date}  {label:<5} {r['coef']:+.4f} (SE {r['se']:.4f}) "
                  f"t {t:+.1f}  post months {r['n_post_months']}  "
                  f"{(time.time()-t1)/60:.1f} min")

    # ---- Part 2: the exploratory profile, hires only ----
    print("\nEXPLORATORY PROFILE (hires only; searching over dates, so this "
          "carries no inference)")
    prof = []
    if "hires" in panels:
        bal, outcome = panels["hires"]
        for date in GRID:
            r = fit_at(bal, outcome, date,
                       f"d60_grid_{date.replace('-','_')}")
            if r["n_post_months"] < MIN_POST_MONTHS:
                print(f"  {date}  skipped: only {r['n_post_months']} post "
                      f"months, below the {MIN_POST_MONTHS} floor")
                continue
            r["spec"], r["outcome"] = "grid", "hires"
            prof.append(r)
            pd.DataFrame(prof).to_csv(OUT / "profile.csv", index=False)
            t = r["coef"] / max(r["se"], 1e-12) if np.isfinite(r["coef"]) else np.nan
            print(f"  {date}  {r['coef']:+.4f} (SE {r['se']:.4f}) t {t:+.1f}"
                  f"  post months {r['n_post_months']}")

    # ---- summary ----
    pre = pd.DataFrame(rows)
    lines = ["WHEN DID IT START?", "=" * 52, "",
             "22-25 differential, by assumed treatment date.",
             "Identified inside the employer against its own other ages.", "",
             "PRE-SPECIFIED (these carry the inference):"]
    for d, why in PRESPEC.items():
        lines.append(f"  {d}  {why}")
    lines.append("")
    if not pre.empty:
        piv = pre.pivot_table(index="date", columns="outcome", values="coef")
        se = pre.pivot_table(index="date", columns="outcome", values="se")
        lines += [piv.round(4).to_string(), "", "standard errors:",
                  se.round(4).to_string(), ""]
    if prof:
        P = pd.DataFrame(prof)
        lines += ["EXPLORATORY PROFILE, hires. Searching over dates, so no",
                  "p-value here means anything. Read the SHAPE.", ""]
        for _, r in P.iterrows():
            lines.append(f"  {r['date']}  {r['coef']:+.4f} (SE {r['se']:.4f})"
                         f"  post months {int(r['n_post_months'])}")
        good = P[np.isfinite(P["coef"])]
        if len(good) > 2:
            lo = good.loc[good["coef"].idxmin()]
            interior = lo["date"] not in (good["date"].iloc[0],
                                          good["date"].iloc[-1])
            lines += ["",
                      f"  most negative at {lo['date']}: {lo['coef']:+.4f} "
                      f"(SE {lo['se']:.4f})",
                      "  " + ("INTERIOR minimum, which is what a real break "
                              "at that date would produce"
                              if interior else
                              "at an ENDPOINT, which is what a trend rather "
                              "than a break looks like, and is also what "
                              "shrinking precision alone can produce")]
    if FAILURES:
        lines += ["", "FITS THAT FAILED AND ARE ABSENT ABOVE: "
                  + "; ".join(FAILURES),
                  "A missing row is a missing fit, never a zero."]
    lines += ["",
              "READ THIS BEFORE QUOTING ANY OF IT:",
              "  1. The four pre-specified dates were fixed in the source",
              "     before the run and all four are reported. The grid was",
              "     not, and nothing in it carries a p-value.",
              "  2. The post window shrinks as the date moves later, so",
              "     standard errors grow along the profile for reasons that",
              "     have nothing to do with the effect. Every row carries",
              "     its post-month count; use it.",
              "  3. A late date does not rescue a null. It tests a different",
              "     and more specific hypothesis, and it can fail.",
              "  4. Exposure is still frozen in 2019 and 2025 is still the",
              "     preliminary file. Re-dating the treatment changes",
              "     neither.",
              "", f"Runtime {(time.time()-t0)/60:.1f} min. " + mc.mem_line()]
    (OUT / "60_summary.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print("\n60 done.")


if __name__ == "__main__":
    main()
