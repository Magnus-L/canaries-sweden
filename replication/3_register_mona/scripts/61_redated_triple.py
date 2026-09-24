#!/usr/bin/env python3
"""
61_redated_triple.py: the within-employer age design with the treatment
dated at adoption, and the panel construction every later script reuses.

QUESTION
Statistics Sweden's survey of enterprises records firm AI use at 10 per
cent in 2023, 25 in 2024 and 35 in 2025, so most of the diffusion came
after the ChatGPT launch of November 2022. A design dated at the launch
pools thirteen months in which few firms had adopted with the months in
which they did. This script estimates script 47j's design with the
adoption window opening in January 2024, on an outcome panel that runs to
June 2025.

DESIGN
Panel (build_skeleton): employer by age band by month counts from script
47L's caches, January 2021 to June 2025, for one young band (22-25 or
26-30) beside the four incumbent bands 31-34, 35-40, 41-49 and 50-69. The
panel starts in 2021 so that 2019, the scoring year, stays out of the
outcome window and the pandemic year does not enter the reference. An
employer enters if it holds the young band and at least one incumbent
band; cells are zero-filled over the window; employer-band cells that are
zero in every month are dropped, and so is an employer left with one
band. The three fixed-effect keys (employer by month, employer by age,
month by age) are integer codes. The skeleton does not depend on the
exposure, so it is built once per young band and the quartile is merged
in afterwards (attach_exposure).

Exposure: script 47j's incumbent_exposure on script 47h's 2019 frame, for
the OL_daioe score book (the paper's) and the entrant score book, with
the 2019 incumbents scored from the education register as it stood in
2019 (true arm) and, for OL_daioe, as it stood in 2021 (as-of arm).

Terms (add_terms): PostRB x High x Young from April 2022, kept on through
the window, and either three disjoint windows (launch, December 2022 to
October 2023; Copilot, November and December 2023; adoption, from January
2024) or one step from January 2024. The window boundaries were fixed
before any estimate was seen. Poisson pseudo-maximum likelihood, standard
errors clustered by employer. The script refuses to run if the counts end
before the adoption window opens.

INPUTS AND OUTPUTS
Reads the caches edu_hr_weights_2019 to 2021, edu_hr_2019 (script 47h) and
L_counts_2021 to 2025 (script 47L); performs no SQL. Writes to output_61/:
redated_step.csv (the three windows), redated_pooled.csv (the single step)
and 61_summary.txt.

IN THE PAPER
build_skeleton, attach_exposure and add_terms are the panel of Equation
(2). build_skeleton and PANEL_YEARS are used by every later register
script, including the occupation-route scripts 82 to 92 that produce the
paper's estimates. The coefficients written here are on the education
route and are not quoted in the paper.
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
OUT = HERE / "output_61"
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR

# The three disjoint windows. Boundaries are the pre-specified dates of
# script 60, which were fixed before any of this was estimated.
STEPS = [("launch", "2022-12", "2023-11"),    # ChatGPT to Copilot GA
         ("copilot", "2023-11", "2024-01"),   # Copilot GA to the adoption jump
         ("adoption", "2024-01", "9999-99")]  # SCB's jump onward
POOLED_FROM = "2024-01"
DESIGNS = ("OL_daioe", "entrant")
ARMS = ("true", "asof")
# Which (design, arm) pairs are actually estimated. The as-of arm exists to
# measure the register artefact at the new dating, and one design is enough
# for that: running it on the second as well would cost two more fits of a
# forty-million-row panel and answer the same question twice.
JOBS = (("OL_daioe", "true"), ("OL_daioe", "asof"), ("entrant", "true"))
TRUNC = 2021

# The panel starts here rather than in 2019: 2019 is the scoring year and
# 2020 the pandemic year, twenty-three months of pre-launch data identify
# the employer x age and month x age effects, and the 2019 to 2025 panel
# would run to about fifty-five million rows. A choice fixed before the
# estimate was seen.
PANEL_FROM = "2021-01"
PANEL_YEARS = list(range(2021, 2026))
FAILURES = []


def opt(label, fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except BaseException as ex:
        print(f"  [optional] {label} FAILED ({type(ex).__name__}: {ex})")
        traceback.print_exc()
        return None


def _j47():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "j47", HERE / "47j_within_employer_triple.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build_skeleton(counts: pd.DataFrame, young: str, j47) -> pd.DataFrame:
    """
    The balanced employer x band x month panel, WITHOUT exposure.

    47j rebuilds this for every design and arm, which is wasted work: the
    counts, the band structure, the zero-filling and the three fixed-effect
    strings do not depend on which education measure classified the firm.
    Only the quartile does. Building the skeleton once per young band and
    merging the quartile in afterwards turns eight panel builds into two.
    """
    bands = [young] + j47.INCUMBENT_BANDS
    p = counts[counts["age_group"].astype(str).isin(bands)]
    p = p[p["year_month"].astype(str) >= PANEL_FROM]
    p = (p.groupby(["employer_id", "age_group", "year_month"], observed=True)
         ["n_emp"].sum().reset_index())
    p["age_group"] = p["age_group"].astype(str)
    p["year_month"] = p["year_month"].astype(str)
    # an employer must hold the young band AND at least one older band, or
    # it contributes nothing to a within-employer age comparison
    have = p.groupby("employer_id")["age_group"].agg(set)
    keep = have[have.apply(lambda v: young in v
                           and bool(v & set(j47.INCUMBENT_BANDS)))].index
    p = p[p["employer_id"].isin(keep)]
    if p.empty:
        return p
    months = sorted(p["year_month"].unique())
    emp = p["employer_id"].drop_duplicates().to_numpy()
    full = pd.MultiIndex.from_product([emp, bands, months],
                                      names=["employer_id", "age_group",
                                             "year_month"])
    bal = (p.groupby(["employer_id", "age_group", "year_month"], observed=True)
           ["n_emp"].sum().reindex(full, fill_value=0).reset_index())
    bal["n_emp"] = bal["n_emp"].astype(int)
    bal = j47._drop_dead_cells(bal)
    if bal.empty:
        return bal
    bal["young"] = (bal["age_group"] == young).astype(int)
    # The fixed-effect keys are INTEGERS here, not the usual pasted
    # strings. A fixed effect is a grouping label and any bijection of the
    # pairs does the same job, and mona_common factorises these columns to
    # int32 before they reach R in any case. On a forty-million-row panel
    # three columns of Python strings cost several gigabytes and are
    # copied again by every merge; three int64 columns cost 900 MB. That
    # is the difference between fitting under the memory cap and not.
    ec = pd.factorize(bal["employer_id"], sort=False)[0].astype("int64")
    tc = pd.factorize(bal["year_month"], sort=False)[0].astype("int64")
    ac = pd.factorize(bal["age_group"], sort=False)[0].astype("int64")
    n_t, n_a = int(tc.max()) + 1, int(ac.max()) + 1
    bal["fe_emp_t"] = ec * n_t + tc
    bal["fe_emp_age"] = ec * n_a + ac
    bal["fe_t_age"] = tc * n_a + ac
    return bal


def attach_exposure(skel: pd.DataFrame, expo: pd.DataFrame) -> tuple:
    """Merge the quartile onto the skeleton and build the interactions."""
    b = skel.merge(expo[["employer_id", "fq"]], on="employer_id", how="inner")
    if b.empty:
        return b, [], []
    b["high"] = (b["fq"] == 4).astype(int)
    return add_terms(b, copy=False)


def add_terms(bal: pd.DataFrame, copy: bool = True) -> tuple:
    """
    Step and pooled interactions on a panel that already carries `high`,
    `young` and the fixed effects. The Riksbank control keeps its own date
    so the two shocks are never conflated.
    """
    b = bal.copy() if copy else bal
    ym = b["year_month"].astype(str)
    hy = b["high"] * b["young"]
    b["post_rb_x_high_x_young"] = (ym >= mc.RIKSBANK_YM).astype(int) * hy
    step = []
    for name, lo, hi in STEPS:
        b[f"s_{name}"] = ((ym >= lo) & (ym < hi)).astype(int) * hy
        step.append(f"s_{name}")
    b["post2024_x_high_x_young"] = (ym >= POOLED_FROM).astype(int) * hy
    return b, ["post_rb_x_high_x_young"] + step, \
        ["post_rb_x_high_x_young", "post2024_x_high_x_young"]


def main():
    mc.Tee(OUT / "61_log.txt")
    t0 = time.time()
    print("=" * 70)
    print("61: THE WITHIN-EMPLOYER DESIGN, DATED WHERE ADOPTION HAPPENED")
    print("=" * 70)
    print("  windows: " + " | ".join(f"{n} {lo}..{hi}" for n, lo, hi in STEPS))
    print(f"  panel from {PANEL_FROM} to the end of the AGI file")
    print("  the skeleton is built once per band; the dates enter as one")
    print("  disjoint step function rather than as four regressions.")
    print(mc.mem_line("  "))

    j47 = _j47()
    h47 = j47._h47()

    # Exactly 47j's own setup, so the two cannot drift apart. No SQL: every
    # cache must already exist, and a missing one is an error rather than a
    # silent pull, because this script is meant to be cheap.
    counts = {}
    for y in (2019, 2020, 2021):
        w = mc.read_cache(CACHE / f"edu_hr_weights_{y}.parquet",
                          require=h47.WEIGHT_COLS)
        if w is None:
            raise RuntimeError(f"edu_hr_weights_{y}.parquet missing: run 47h "
                               f"first. This script performs no SQL.")
        counts[y] = w
    book = h47.ScoreBook(counts, h47.load_key(), h47.load_scores())
    designs = {k: dict(h47.DESIGNS[k]) for k in DESIGNS}
    for nm, sp in designs.items():
        book.build(nm, sp)
    print(f"  scorebook built for {', '.join(designs)}")

    frame19 = mc.read_cache(CACHE / "edu_hr_2019.parquet",
                            require=h47.YEAR_COLS + ["n_emp"])
    if frame19 is None:
        raise RuntimeError("edu_hr_2019.parquet missing: run 47h first.")
    print(f"  exposure frame: {len(frame19):,} cells (2019 only)")

    # The outcome comes from 47L's monthly counts, which reach June 2025.
    # 47h's frames stop in 2023 and would leave the adoption window empty.
    cnt = []
    for y in PANEL_YEARS:
        c = mc.read_cache(CACHE / f"L_counts_{y}.parquet")
        if c is None:
            raise RuntimeError(f"L_counts_{y}.parquet missing: run 47L first. "
                               f"This script performs no SQL.")
        cnt.append(c)
    counts = pd.concat(cnt, ignore_index=True)
    del cnt
    gc.collect()
    last = str(counts["year_month"].max())
    print(f"  counts: {len(counts):,} employer-age-months, ending {last}")
    if last < POOLED_FROM:
        raise RuntimeError(
            f"the counts end at {last} and the adoption window opens at "
            f"{POOLED_FROM}; this script would estimate its headline "
            f"coefficient on no data. Refusing to run.")

    # exposure first: it depends on the design and arm but not on the band
    expos = {}
    for nm, arm in JOBS:
        e, _ = j47.incumbent_exposure(frame19, book, nm, designs[nm],
                                      arm, TRUNC)
        expos[(nm, arm)] = e
        print(f"  exposure {nm:<10} {arm:<5} {len(e):,} firms")
    del frame19
    gc.collect()

    step_rows, pooled_rows = [], []
    for band in j47.YOUNG_BANDS:
        t0b = time.time()
        skel = build_skeleton(counts, band, j47)
        if skel.empty:
            print(f"  {band}: empty panel, skipped")
            continue
        print(f"  skeleton {band}: {len(skel):,} rows "
              f"({time.time()-t0b:.0f}s, built once for every arm)")
        for nm, arm in JOBS:
            t1 = time.time()
            b, step_terms, pool_terms = attach_exposure(skel,
                                                        expos[(nm, arm)])
            if b.empty:
                print(f"  {nm} {arm} {band}: no firms matched, skipped")
                continue
            print(f"  {nm} {arm} {band}: panel {len(b):,} rows "
                  f"({time.time()-t1:.0f}s to attach)")

            for label, terms, sink in (("step", step_terms, step_rows),
                                       ("pooled", pool_terms, pooled_rows)):
                t2 = time.time()
                r = mc.run_fepois_multi(
                    b, OUT, tag=f"r61_{label}_{nm}_{arm}_{band.replace('-','_')}",
                    terms=terms, fes=j47.FES)
                if r.empty:
                    FAILURES.append(f"{label}/{nm}/{arm}/{band}")
                    print(f"    {label}: FAILED, recorded and skipped")
                    continue
                for _, x in r.iterrows():
                    sink.append({"design": nm, "arm": arm,
                                 "young_band": band, "term": x["term"],
                                 "coef": float(x["coef"]),
                                 "se": float(x["se"]),
                                 "n_obs": int(x["n_obs"]),
                                 "status": str(x.get("status", "ok"))})
                pd.DataFrame(step_rows).to_csv(OUT / "redated_step.csv",
                                               index=False)
                pd.DataFrame(pooled_rows).to_csv(OUT / "redated_pooled.csv",
                                                 index=False)
                show = r[r["term"].str.startswith(("s_", "post2024"))]
                for _, x in show.iterrows():
                    t = x["coef"] / max(x["se"], 1e-12)
                    print(f"    {label:<6} {x['term']:<26} "
                          f"{x['coef']:+.4f} (SE {x['se']:.4f}) t {t:+.2f}"
                          f"  [{(time.time()-t2)/60:.1f} min]")
            del b
            gc.collect()
        del skel
        gc.collect()

    # ---- summary ----
    S = pd.DataFrame(step_rows)
    P = pd.DataFrame(pooled_rows)
    lines = ["THE WITHIN-EMPLOYER DESIGN, RE-DATED", "=" * 52, "",
             "Young against older workers inside one employer in one month.",
             "Exposure from the education mix of incumbents aged 31+ in 2019.",
             "No young worker is ever classified.", "",
             "Windows: " + "; ".join(f"{n} {lo} to {hi}"
                                     for n, lo, hi in STEPS), ""]
    if not S.empty:
        st = S[S["term"].str.startswith("s_")]
        for band in sorted(st["young_band"].unique()):
            lines.append(f"STEP FUNCTION, young = {band}:")
            for arm in sorted(st["arm"].unique()):
                for nm in sorted(st["design"].unique()):
                    d = st[(st.young_band == band) & (st.arm == arm)
                           & (st.design == nm)]
                    if d.empty:
                        continue
                    bits = "  ".join(
                        f"{r['term'][2:]} {r['coef']:+.4f}"
                        f"({r['coef']/max(r['se'],1e-12):+.1f})"
                        for _, r in d.iterrows())
                    lines.append(f"  {nm:<10} {arm:<5} {bits}")
            lines.append("")
    if not P.empty:
        pp = P[P["term"] == "post2024_x_high_x_young"]
        lines += ["POOLED from 2024-01, coefficient (SE) t:"]
        for _, r in pp.iterrows():
            lines.append(f"  {r['design']:<10} {r['arm']:<5} "
                         f"{r['young_band']:<6} {r['coef']:+.4f} "
                         f"({r['se']:.4f}) t "
                         f"{r['coef']/max(r['se'],1e-12):+.2f}")
        lines.append("")
        tr = pp[pp.arm == "true"].set_index(["design", "young_band"])["coef"]
        af = pp[pp.arm == "asof"].set_index(["design", "young_band"])["coef"]
        common = tr.index.intersection(af.index)
        if len(common):
            lines += ["ARTEFACT AT THE NEW DATING (as-of minus true). A design",
                      "clean at one date is not automatically clean at another:"]
            for k in common:
                lines.append(f"  {k[0]:<10} {k[1]:<6} {af[k]-tr[k]:+.4f}")
            lines.append("")
    if FAILURES:
        lines += ["FITS THAT FAILED AND ARE ABSENT ABOVE: " + "; ".join(FAILURES),
                  "A missing row is a missing fit, never a zero.", ""]
    lines += [
        "READ THIS BEFORE QUOTING ANY OF IT:",
        "  1. The window boundaries were fixed by script 60 before any of",
        "     this was estimated. They were not chosen to suit the answer.",
        "  2. The adoption window is the shortest, so its coefficient is the",
        "     least precise. Compare the windows on their t, not their size.",
        "  3. A later dating does not rescue a null. It tests a sharper",
        "     hypothesis and it can fail.",
        "  4. 2025 is the preliminary AGI file and stops in June.",
        f"  5. The panel starts at {PANEL_FROM}, not 2019. The earlier years",
        "     identify nothing here and cost enough rows to crash R.",
        "", f"Runtime {(time.time()-t0)/60:.1f} min. " + mc.mem_line()]
    (OUT / "61_summary.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print("\n61 done.")


if __name__ == "__main__":
    main()
