#!/usr/bin/env python3
"""
53_freshcode_panel.py -- the headline design on codes that are ACTUALLY
                         contemporaneous, 2019-2023.

======================================================================
  RUNS IN SCB's MONA SECURE ENVIRONMENT ONLY. Standalone: submit this
  file itself. Writes output_53/. Caches under cache/.
======================================================================

WHY THIS EXISTS.

The as-of backtest (45) showed the occupation artefact is -0.307 at the
2021 truncation and -0.163 at 2022, against a published headline of -0.174
(43, Poisson, 22-25). The artefact is the size of the finding. Every route
tried since has either inherited the problem (47b/47h, education) or
answered a different question (47L, across-firm).

One question has never been asked, and it is the obvious one. The backtest
manufactures staleness by truncating the register. But the register is
ALREADY partly stale in every year: script 50's M3 moment measured that only
63.7 per cent of 22-25 codes in 2019 were assigned in 2019, 58.2 in 2020,
61.8 in 2021, 66.0 in 2022 and 68.0 in 2023. `Ssyk4_2012_J16` being present
has never meant the code describes this year's job.

So: run the paper's own design on the worker-months whose code was assigned
in the observation year, and stop the panel at 2023, the last year with its
own Individ table. That is the estimand the paper claims -- young workers
more exposed than other young workers, inside the same employer, monthly --
measured with codes that are genuinely contemporaneous. The window loses
2024 and 2025 but keeps thirteen months after ChatGPT.

NOTE the vintage column in `panel_vintage.parquet` does NOT answer this.
It records which Individ TABLE supplied the code (own / 2023 / 2022 / 2021).
A code from the year's own table can still have been assigned four years
earlier. `SsykAr_J16` is the assignment year and is what this script pulls.

WHAT IT REPORTS. Three arms on one window, so the contrast is internal and
not across specifications:

    all     every coded worker-month        (= script 43's design, 2019-2023)
    fresh   SsykAr_J16 == observation year
    stale   coded, but carried forward

`fresh` is the estimate we want. `stale` is the direct, real-data reading of
what carry-forward does to this coefficient, which until now we only had
from a simulated truncation.

THE HONEST PROBLEM, STATED UP FRONT. Freshness is not random. A code is
assigned when the register observes someone in an occupation, so freshly
coded workers over-represent movers and entrants, which is exactly the
margin AI is supposed to act on. Restricting to fresh codes trades a
measurement bias for a selection bias, and this script does not pretend
otherwise.

What decides whether the trade is acceptable is NOT whether freshness is
selected, but whether it is DIFFERENTIALLY selected across the exposure
dimension over time: gamma2 is an interaction, so a level difference in
freshness between Q4 and the rest cannot move it, and a common change over
time cannot either. Only a change in the GAP can. The script measures that
directly and pre-commits to a read rule (SELECTION_DID_LIMIT below).

Output (output_53/):
  fresh_pooled.csv        arm x age_group x term, Poisson pooled DiD
  fresh_es.csv            arm x age_group x halfyear, 22-25 and 50+
  freshness_shares.csv    year_month x age_group x quartile, share fresh
  selection_did.csv       the Q4-vs-rest freshness gap, post minus pre
  panel_sizes.csv         cells, employers, zero share, by arm and age
  53_summary.txt
"""

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mona_common as mc

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_53"
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR
CACHE.mkdir(exist_ok=True)

YEARS = (2019, 2020, 2021, 2022, 2023)   # 2023 is the last own-Individ year
ARMS = ("all", "fresh", "stale")
ES_AGES = ("22-25", "50+")               # the estimand and its placebo
STEP1_MIN_CUMULATIVE = 5                 # script 43's Step-1 threshold

# Pre-committed read rule: the fresh arm is clean if its 50+ coefficient is
# within this of zero. Older workers' codes are selected for freshness the
# same way younger workers' are, but they do not churn (M2: 8.8 per cent
# change code a year at 50+, against 24.1 at 22-25), so the placebo isolates
# the restriction's own effect from the churn it is correcting for.
PLACEBO_LIMIT = 0.03

# Reported, NOT gated. The Q4-versus-rest freshness gap is the obvious
# selection diagnostic and it does not work, which the synthetic test
# established before this ever ran on real data. Misplacement moves the gap
# MECHANICALLY: if carry-forward files high-exposure workers under a
# low-exposure code, the surviving Q4 cell is left fresher and Q1-3 staler,
# so the gap moves whenever the artefact is present. It therefore cannot
# separate the threat (differential selection) from the thing we are
# correcting for (churn plus lag). Read it as a description of the sample.
SELECTION_DID_LIMIT = 0.02

# Schema the year cache must carry, checked on read (mona_common.read_cache).
YEAR_COLS = ["employer_id", "year_month", "ssyk4", "freshness",
             "age_group", "n_emp"]


def opt(label: str, fn, *a, **kw):
    """
    Stata's `capture noisily`. For work that is INESSENTIAL: a diagnostic,
    a side table, a print. A failure is reported loudly and the run goes on.
    Never wrap an estimate or a primary export in this.
    """
    try:
        return fn(*a, **kw)
    except BaseException as ex:
        print(f"  [optional] {label} FAILED ({type(ex).__name__}: {ex})")
        traceback.print_exc()
        return None


def pull_year_fresh(year: int, conn) -> pd.DataFrame:
    """
    One year of AGI joined to that year's OWN Individ table, aggregated to
    employer x year_month x ssyk4 x freshness x age_group.

    freshness:
        'fresh'  TRY_CAST(SsykAr_J16 AS INT) == year
        'stale'  coded, but SsykAr_J16 is an earlier year (or unreadable)
        'none'   no code at all; carries ssyk4 '____' and is kept only so
                 the freshness denominators are right

    TRY_CAST because SsykAr_J16 is a char column that carries blanks and
    '****' like every other char column in this register; a plain CAST
    raises on the first bad row and loses the year.
    """
    monthly = []
    for month in range(1, 13):
        ym = f"{year}{month:02d}"
        monthly.append(f"""
            SELECT
                agi.P1207_LOPNR_PEORGNR AS employer_id,
                agi.PERIOD AS period,
                ind.Ssyk4_2012_J16 AS ssyk4,
                CASE WHEN ind.Ssyk4_2012_J16 IS NULL
                       OR LTRIM(RTRIM(ind.Ssyk4_2012_J16)) = '' THEN 'none'
                     WHEN TRY_CAST(ind.SsykAr_J16 AS INT) = {year}
                       THEN 'fresh'
                     ELSE 'stale' END AS freshness,
                ind.FodelseAr AS birth_year,
                agi.P1207_LOPNR_PERSONNR AS person_id
            FROM dbo.Arb_AGIIndivid{ym}_def agi
            LEFT JOIN dbo.Individ_{year} ind
                ON agi.P1207_LOPNR_PERSONNR = ind.P1207_LopNr_PersonNr
        """)
    union = "\nUNION ALL\n".join(monthly)
    age_case = """CASE
            WHEN age BETWEEN 22 AND 25 THEN '22-25'
            WHEN age BETWEEN 26 AND 30 THEN '26-30'
            WHEN age BETWEEN 31 AND 34 THEN '31-34'
            WHEN age BETWEEN 35 AND 40 THEN '35-40'
            WHEN age BETWEEN 41 AND 49 THEN '41-49'
            WHEN age BETWEEN 50 AND 69 THEN '50+'
            ELSE NULL END"""
    query = f"""
    WITH base AS ({union}),
    age_calc AS (
        SELECT employer_id, period,
               COALESCE(RIGHT('0000'+CAST(ssyk4 AS VARCHAR(4)),4), '____')
                   AS ssyk4,
               freshness, person_id,
               CAST(LEFT(period,4) AS INT) - birth_year AS age
        FROM base
        WHERE birth_year IS NOT NULL
    )
    SELECT employer_id,
           LEFT(period,4) + '-' + SUBSTRING(period,5,2) AS year_month,
           ssyk4, freshness,
           {age_case} AS age_group,
           COUNT(DISTINCT person_id) AS n_emp
    FROM age_calc
    WHERE age BETWEEN 22 AND 69
    GROUP BY employer_id, period, ssyk4, freshness, {age_case}
    """
    df = pd.read_sql(query, conn)
    for c in ("ssyk4", "freshness", "age_group", "year_month"):
        df[c] = df[c].astype("string").astype("category")
    df["n_emp"] = df["n_emp"].astype("int32")
    return df


def arm_rows(panel: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Coded worker-months belonging to one arm, vintage tag summed out."""
    if arm == "all":
        keep = panel["freshness"].isin(["fresh", "stale"])
    else:
        keep = panel["freshness"] == arm
    sub = panel[keep & (panel["ssyk4"] != "____")]
    out = (sub.groupby(["employer_id", "year_month", "ssyk4", "age_group"],
                       observed=True)["n_emp"].sum().reset_index())
    out["year_month"] = out["year_month"].astype(str)
    out["ssyk4"] = out["ssyk4"].astype(str)
    out["age_group"] = out["age_group"].astype(str)
    return out


def build_age_panel(agg, all_months, age_label):
    """Script 43's build, unchanged: Step-1 threshold, balance, treatment."""
    sub = agg[agg["age_group"] == age_label]
    cum = sub.groupby("employer_id")["n_emp"].sum()
    sub = sub[sub["employer_id"].isin(cum[cum >= STEP1_MIN_CUMULATIVE].index)]
    bal = mc.add_treatment(mc.balance_panel(sub, all_months))
    bal["halfyear"] = mc.assign_halfyear(bal["year_month"])
    return bal


def freshness_diagnostics(panel: pd.DataFrame, daioe: pd.DataFrame):
    """
    Share of coded worker-months that are fresh, by month x age x quartile,
    and the one number that decides whether the fresh restriction can itself
    manufacture gamma2: the Q4-versus-rest gap, post minus pre.
    """
    coded = panel[panel["ssyk4"] != "____"].copy()
    coded["ssyk4"] = coded["ssyk4"].astype(str).str.zfill(4)
    coded["year_month"] = coded["year_month"].astype(str)
    coded = coded.merge(daioe, on="ssyk4", how="inner")
    g = (coded.groupby(["year_month", "age_group", "exposure_quartile",
                        "freshness"], observed=True)["n_emp"].sum()
         .unstack("freshness").fillna(0.0))
    for c in ("fresh", "stale"):
        if c not in g.columns:
            g[c] = 0.0
    g["n_coded"] = g["fresh"] + g["stale"]
    g["share_fresh"] = np.where(g["n_coded"] > 0, g["fresh"] / g["n_coded"],
                                np.nan)
    shares = g.reset_index()[["year_month", "age_group", "exposure_quartile",
                              "n_coded", "share_fresh"]]

    rows = []
    for age, a in shares.groupby("age_group", observed=True):
        a = a.dropna(subset=["share_fresh"])
        post = a["year_month"] >= mc.CHATGPT_YM
        hi = a["exposure_quartile"] == 4

        def wm(mask):
            s = a[mask]
            return (np.average(s["share_fresh"], weights=s["n_coded"])
                    if s["n_coded"].sum() > 0 else np.nan)

        q4_pre, q4_post = wm(hi & ~post), wm(hi & post)
        lo_pre, lo_post = wm(~hi & ~post), wm(~hi & post)
        rows.append({"age_group": age,
                     "q4_pre": q4_pre, "q4_post": q4_post,
                     "rest_pre": lo_pre, "rest_post": lo_post,
                     "gap_pre": q4_pre - lo_pre,
                     "gap_post": q4_post - lo_post,
                     "selection_did": (q4_post - lo_post) - (q4_pre - lo_pre)})
    return shares, pd.DataFrame(rows)


def main():
    mc.Tee(OUT / "53_log.txt")
    t_start = time.time()
    print("=" * 70)
    print("53: THE HEADLINE DESIGN ON CONTEMPORANEOUS CODES, 2019-2023")
    print("=" * 70)
    print("  plan: 5 year pulls x ~3 min | 18 pooled fits | 6 event studies")
    print(mc.mem_line("  "))

    conn = None
    frames = []
    for y in YEARS:
        cf = CACHE / f"freshcode_{y}.parquet"
        f = mc.read_cache(cf, require=YEAR_COLS)
        if f is None:
            conn = conn or mc.connect()
            t0 = time.time()
            f = pull_year_fresh(y, conn)
            mc.write_cache(f, cf)
            print(f"  {y}: {len(f):,} cells pulled ({time.time()-t0:.0f}s)")
        else:
            print(f"  {y}: cached ({len(f):,} cells)")
        frames.append(f)
    panel = pd.concat(frames, ignore_index=True)
    del frames
    # year_month is stored categorical to keep the parquet small, but it is
    # COMPARED with >= against CHATGPT_YM downstream and an unordered
    # categorical raises on that. Convert once, here, explicitly.
    panel["year_month"] = panel["year_month"].astype(str)

    # --- how much of the register is actually contemporaneous ---
    tot = (panel[panel["ssyk4"] != "____"]
           .groupby(["year_month", "age_group", "freshness"],
                    observed=True)["n_emp"].sum().unstack("freshness")
           .fillna(0.0))
    tot["share_fresh"] = tot["fresh"] / (tot["fresh"] + tot["stale"])
    yr = tot.reset_index()
    yr["year"] = yr["year_month"].astype(str).str[:4]
    print("\nSHARE OF CODED WORKER-MONTHS THAT ARE CONTEMPORANEOUS")
    piv = (yr.groupby(["year", "age_group"], observed=True)
           .apply(lambda d: np.average(d["share_fresh"],
                                       weights=d["fresh"] + d["stale"]),
                  include_groups=False)
           .unstack("age_group"))
    print(piv.round(3).to_string())

    daioe = mc.load_daioe()
    shares, seldid = freshness_diagnostics(panel, daioe)
    opt("freshness_shares.csv",
        lambda: mc.enforce_min_cell(shares, count_col="n_coded").to_csv(
            OUT / "freshness_shares.csv", index=False))
    seldid.to_csv(OUT / "selection_did.csv", index=False)
    print("\nSELECTION CHECK: does the Q4-vs-rest freshness gap MOVE?")
    print(seldid.round(4).to_string(index=False))
    worst = seldid.loc[seldid["selection_did"].abs().idxmax()]
    print(f"  largest |gap movement| {abs(worst['selection_did']):.4f} "
          f"at {worst['age_group']}  -- DESCRIPTIVE, not a gate: this gap "
          f"moves mechanically whenever misplacement is present")

    # --- estimation ---
    pooled_rows, es_frames, size_rows = [], [], []
    for arm in ARMS:
        print(f"\n{'=' * 70}\nARM: {arm}\n{'=' * 70}")
        agg = arm_rows(panel, arm)
        agg = mc.merge_daioe_and_filter(agg, daioe)
        agg = mc.aggregate_to_quartile(agg)
        all_months = sorted(agg["year_month"].unique())
        for age in mc.AGE_GROUPS:
            t0 = time.time()
            bal = build_age_panel(agg, all_months, age)
            if bal.empty:
                print(f"  {age}: EMPTY after the Step-1 threshold, skipped")
                continue
            size_rows.append({"arm": arm, "age_group": age,
                              "cells": len(bal),
                              "employers": bal["employer_id"].nunique(),
                              "zero_share": float((bal["n_emp"] == 0).mean()),
                              "n_emp_total": int(bal["n_emp"].sum())})
            res = mc.run_fepois(bal, OUT, tag=f"f53_{arm}_{age}")
            if res.empty:
                raise RuntimeError(
                    f"the pooled Poisson returned nothing for {arm}/{age}; "
                    f"this is a primary estimate, not a diagnostic")
            for _, r in res.iterrows():
                pooled_rows.append({"arm": arm, "age_group": age,
                                    **r.to_dict()})
            g2 = res.loc[res["term"] == "post_gpt_x_high"]
            if not g2.empty:
                print(f"  [{arm:<5}] {age:<5} gamma2 "
                      f"{g2['coef'].iloc[0]:+.4f} "
                      f"(SE {g2['se'].iloc[0]:.4f}) "
                      f"n {int(g2['n_obs'].iloc[0]):,} "
                      f"{(time.time()-t0)/60:.1f} min")
            if age in ES_AGES:
                e = opt(f"event study {arm}/{age}",
                        mc.run_fepois_es, bal, OUT, tag=f"e53_{arm}_{age}")
                if e is not None and not e.empty:
                    e = e.copy()
                    e["arm"], e["age_group"] = arm, age
                    es_frames.append(e)
            del bal
        del agg

    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(OUT / "fresh_pooled.csv", index=False)
    pd.DataFrame(size_rows).to_csv(OUT / "panel_sizes.csv", index=False)
    if es_frames:
        pd.concat(es_frames, ignore_index=True).to_csv(
            OUT / "fresh_es.csv", index=False)

    # --- summary ---
    g2 = (pooled[pooled["term"] == "post_gpt_x_high"]
          .pivot_table(index="age_group", columns="arm", values="coef"))
    se = (pooled[pooled["term"] == "post_gpt_x_high"]
          .pivot_table(index="age_group", columns="arm", values="se"))
    order = [a for a in mc.AGE_GROUPS if a in g2.index]
    g2, se = g2.reindex(order), se.reindex(order)

    lines = ["FRESH-CODE PANEL 2019-2023 -- gamma2 (post-GPT x Q4)",
             "=" * 62,
             "Codes assigned in the observation year, against the same",
             "design on all codes and on carried-forward codes only.",
             "", g2.round(4).to_string(), "",
             "standard errors:", se.round(4).to_string(), ""]
    if {"all", "fresh"} <= set(g2.columns) and "22-25" in g2.index:
        d = g2.loc["22-25", "fresh"] - g2.loc["22-25", "all"]
        lines += [f"22-25: fresh {g2.loc['22-25', 'fresh']:+.4f} vs all "
                  f"{g2.loc['22-25', 'all']:+.4f}  (difference {d:+.4f})"]
    if "stale" in g2.columns and "22-25" in g2.index:
        lines += [f"carry-forward arm at 22-25: {g2.loc['22-25','stale']:+.4f}"
                  "  -- the staleness channel measured on real data, not "
                  "simulated by truncation"]
    lines += ["", f"PRE-COMMITTED READ RULE (placebo, limit {PLACEBO_LIMIT}):"]
    if "fresh" in g2.columns and "50+" in g2.index:
        pl = g2.loc["50+", "fresh"]
        ok = abs(pl) < PLACEBO_LIMIT
        lines.append(f"  fresh arm at 50+: {pl:+.4f}   "
                     + ("within limit, the fresh arm is clean"
                        if ok else
                        "EXCEEDS THE LIMIT -- the restriction is doing "
                        "something of its own and the 22-25 number must "
                        "not be quoted alone"))
    else:
        lines.append("  fresh arm at 50+ NOT ESTIMATED -- the read rule "
                     "cannot be applied and nothing here is quotable")
    lines += ["", "SELECTION CHECK (descriptive, NOT a gate -- see the note "
              "in the source):"]
    for _, r in seldid.iterrows():
        lines.append(f"  {r['age_group']:<5} Q4-vs-rest freshness gap moves "
                     f"{r['selection_did']:+.4f}")
    lines += ["",
              "READ THIS BEFORE QUOTING ANY OF IT:",
              "  1. The window ends in 2023, so this cannot speak to 2024 or",
              "     2025. It answers whether the finding was ever there, not",
              "     whether it is there now.",
              "  2. Freshness is not random: a code is assigned when the",
              "     register observes someone in an occupation, so the fresh",
              "     arm over-represents movers and entrants. That cannot move",
              "     an INTERACTION unless it moves differentially across the",
              "     exposure dimension over time. The freshness-gap table",
              "     below does NOT test that, because the gap moves whenever",
              "     misplacement is present; the 50+ placebo does.",
              "  3. The 50+ row is the placebo and is not a robustness check:",
              "     if it moves in the fresh arm, the arm is not clean.",
              "  4. The 'all' arm here is script 43's design on a shorter",
              "     window, so it will NOT equal the published -0.174, which",
              "     runs to 2025. Compare arms within this table, never one",
              "     of these numbers against the published one.",
              f"", f"Total runtime {(time.time()-t_start)/60:.1f} min. "
              + mc.mem_line()]
    (OUT / "53_summary.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print("\n53 done.")


if __name__ == "__main__":
    main()
