#!/usr/bin/env python3
"""
98_backtest_common.py -- the three-arm as-of backtest: completed-vintage
                         codes on the full sample (A), completed-vintage
                         codes on the workers the as-of construction keeps
                         (B), as-of codes on that same retained sample (C).

======================================================================
  RUNS IN MONA (lane 37b, FIRST stage; measurement content of the
  review's Lane 1, placed here to balance runtime). Output folder
  CANARIES_98_OUT (default output_98). No SQL if script 45's caches
  cache/panel_dual_T2021.parquet and panel_dual_T2022.parquet are on the
  share; otherwise 45's own pull rebuilds them (~17 min a truncation).
======================================================================

QUESTION
Online Appendix IV.3: the submitted design returns +0.0193 (0.0129) with
completed-vintage codes and -0.2875 (0.0171) with the codes a register
truncated at 2021 would give (2022 cutoff: +0.0176 (0.0111) and -0.1452
(0.0122)). The as-of arm differs in two ways at once: it codes workers
with stale codes, and it keeps only the workers a truncated register can
code. The review (25 Sep 2026) asks for the arms that separate them.

THE ARMS (for each cutoff T in 2021, 2022)
  A  completed-vintage (own-year) codes, every worker who has one: 45's
     true arm.
  B  completed-vintage codes, restricted to the workers the as-of
     construction retains (a DAIOE-scorable as-of code), and who have a
     scorable own-year code (without one they cannot enter B at all).
  C  as-of codes on exactly B's workers.
  45's as-of arm (as-of codes, every worker with one) is refitted as
  well, because it is half of the gate.
  B - A is sample inclusion, C - B is coding on fixed workers, C - A is
  the two together.

THE ESTIMATOR, unchanged from 45: outcome employment at 22-25 in
employer x exposure-quartile x month cells, 2019-01 to 2023-12; employers
with a cumulative count of at least five, balanced and zero-filled,
restricted to employers holding a top-quartile cell and a lower one;
PostRB x High and PostGPT x High with the pseudo-dates of 45 (hike April
of T-1, launch December of T-1, so the post window crosses into T+1 and
T+2 as the production window crosses into 2024); Poisson with
employer-by-quartile and employer-by-month effects, clustered by
employer; the reference period is everything before the pseudo-hike.
Coding cascade of the as-of arm: own-year code for years up to T, then
Individ T, T-1, T-2. The DAIOE quartile file is the paper's.

HARMONISED CELLS AND THE DIFFERENCES
Each arm is fitted on its own support and then on the HARMONISED support:
the employers present in all three arms' estimation panels. On the
harmonised support the three arms are stacked in one Poisson fit with
arm-specific fixed effects (employer-by-quartile x arm, employer-by-month
x arm) and arm-specific treatment terms, clustered by employer, so the
point estimates are the separate fits' and the covariance ACROSS arms is
estimated: B - A, C - B and C - A get standard errors. Support is
documented per arm: employers, cells, cells PPML used (the rest are
separated or singleton cells fixest drops), and person-months at 22-25.

THE GATES (hard stop)
1. The headline: the paper's panel with script 82's score reproduces
   Table 1 at 22-25 within 0.0005: tau -0.0399 (0.0102). This lane's
   first act, before anything is varied.
2. A and 45's as-of arm reproduce 45's asof_estimates.csv within 0.0005
   at both cutoffs.

READ RULE: no verdict. The pieces and their SEs are reported; if C - B is
under half of C - A at the 2021 cutoff, the paper attributes the artefact
to sample inclusion as much as to stale codes.

EXPORT (output_98/)
  backtest_common.csv  per cutoff, arm and support: gamma2, se, cells,
                       cells used, employers, person-months; then the
                       three differences with SEs
  98_summary.txt, 98_log.txt; vcov_s98_*.csv (the stacked covariances)

IN THE PAPER
Online Appendix IV.3 and tableIV3_backtest.tex (A, B, C and the
differences); the response letter's backtest paragraph.

    python 98_backtest_common.py
"""

import gc
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mona_common as mc  # noqa: E402

OUT = HERE / os.environ.get("CANARIES_98_OUT", "output_98")
OUT.mkdir(exist_ok=True)
os.environ.setdefault("CANARIES_82_OUT", str(OUT))
SKIP_HEADLINE_GATE = os.environ.get("CANARIES_98_NO_HEADLINE_GATE") == "1"

TRUNCATIONS = (2021, 2022)
AGE = "22-25"
STEP1_MIN_CUMULATIVE = 5          # 45's constant
DUAL_COLS = ["employer_id", "year_month", "ssyk_true", "ssyk_asof",
             "age_group", "n_emp"]
GATE = {(2021, "A"): (0.0193, 0.0129), (2021, "asof_all"): (-0.2875, 0.0171),
        (2022, "A"): (0.0176, 0.0111), (2022, "asof_all"): (-0.1452, 0.0122)}
HEADLINE = {"post": (-0.0578, 0.0155), "tau": (-0.0399, 0.0102)}
GATE_TOL = 0.0005
ARMS = ("A", "asof_all", "B", "C")
STACK = ("A", "B", "C")
DIFFS = (("B_minus_A", "B", "A"), ("C_minus_B", "C", "B"),
         ("C_minus_A", "C", "A"))

NOTES: list = []
FAILURES: list = []
ROWS: list = []
T0 = time.time()


def dual_panel(trunc: int) -> pd.DataFrame:
    """45's cached dual panel, schema-checked and read with categorical
    text (about 120 million rows); rebuilt by 45's own pull only if
    missing, so a run on the caches touches nothing of 45's."""
    cf = mc.CACHE_DIR / f"panel_dual_T{trunc}.parquet"
    p = None
    if cf.exists():
        try:
            import pyarrow.parquet as pq
            have = pq.read_schema(cf).names
            missing = [c for c in DUAL_COLS if c not in have]
            if missing:
                print(f"  panel_dual_T{trunc} lacks {missing}; rebuilt")
            else:
                p = pq.read_table(cf, columns=DUAL_COLS).to_pandas(
                    strings_to_categorical=True)
        except Exception as ex:
            print(f"  panel_dual_T{trunc} unreadable ({type(ex).__name__}: "
                  f"{ex}); rebuilt")
            p = None
    if p is None:
        print(f"  panel_dual_T{trunc} is not on the share: 45's pull rebuilds "
              f"it (SQL, about 17 minutes)")
        NOTES.append(f"panel_dual_T{trunc} was rebuilt by 45's pull")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "s45", HERE / "45_asof_backtest.py")
        s45 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(s45)
        conn = mc.connect()
        try:
            p = s45.get_dual_panel(conn, trunc)
        finally:
            conn.close()
    else:
        print(f"  panel_dual_T{trunc}: cached, {len(p):,} cells")
    for c in ("ssyk_true", "ssyk_asof", "age_group", "year_month"):
        if not isinstance(p[c].dtype, pd.CategoricalDtype):
            p[c] = p[c].astype(str).astype("category")
    return p[DUAL_COLS]


def arm_rows(panel: pd.DataFrame, arm: str, scorable: set) -> tuple:
    """(rows the arm keeps, the code column it scores)."""
    if arm == "A":
        return panel["ssyk_true"] != "____", "ssyk_true"
    if arm == "asof_all":
        return panel["ssyk_asof"] != "____", "ssyk_asof"
    both = (panel["ssyk_true"].astype(str).isin(scorable)
            & panel["ssyk_asof"].astype(str).isin(scorable))
    return both, ("ssyk_true" if arm == "B" else "ssyk_asof")


def build(panel, keep, col, daioe, trunc) -> pd.DataFrame:
    """45's estimation panel for one assignment, line for line."""
    pseudo_gpt, pseudo_rb = f"{trunc - 1}-12", f"{trunc - 1}-04"
    agg = (panel[keep]
           .groupby(["employer_id", "year_month", col, "age_group"],
                    observed=True)["n_emp"].sum().reset_index()
           .rename(columns={col: "ssyk4"}))
    for c in ("year_month", "age_group", "ssyk4"):
        agg[c] = agg[c].astype(str)
    agg = mc.merge_daioe_and_filter(agg, daioe)
    agg = mc.aggregate_to_quartile(agg)
    months = sorted(agg["year_month"].unique())
    sub = agg[agg["age_group"] == AGE]
    cum = sub.groupby("employer_id")["n_emp"].sum()
    sub = sub[sub["employer_id"].isin(cum[cum >= STEP1_MIN_CUMULATIVE].index)]
    bal = mc.balance_panel(sub, months)
    bal["post_rb"] = (bal["year_month"] >= pseudo_rb).astype(int)
    bal["post_gpt"] = (bal["year_month"] >= pseudo_gpt).astype(int)
    bal["high"] = (bal["exposure_quartile"] == 4).astype(int)
    bal["post_rb_x_high"] = bal["post_rb"] * bal["high"]
    bal["post_gpt_x_high"] = bal["post_gpt"] * bal["high"]
    bal["fe_emp_bin"] = (bal["employer_id"].astype(str) + "_"
                         + bal["exposure_quartile"].astype(str))
    bal["fe_emp_t"] = (bal["employer_id"].astype(str) + "_"
                       + bal["year_month"])
    return bal


def fit_one(bal, trunc, arm, support) -> None:
    row = {"trunc": trunc, "arm": arm, "support": support, "gamma2": np.nan,
           "se2": np.nan, "cells": len(bal), "cells_used": np.nan,
           "n_firms": int(bal["employer_id"].nunique()),
           "person_months_22_25": float(bal["n_emp"].sum()),
           "status": "empty"}
    if len(bal):
        try:
            pres = mc.run_fepois(bal, OUT, tag=f"s98_{arm}_{support}_T{trunc}")
        except BaseException as ex:
            print(f"  [{arm} {support} T{trunc}] FAILED: {type(ex).__name__}")
            traceback.print_exc()
            pres = pd.DataFrame()
        g2 = pres.loc[pres["term"] == "post_gpt_x_high"] if len(pres) else pres
        if len(g2):
            row.update(gamma2=float(g2["coef"].iloc[0]),
                       se2=float(g2["se"].iloc[0]),
                       cells_used=int(g2["n_obs"].iloc[0]), status="ok")
        else:
            FAILURES.append(f"{arm}_{support}_T{trunc}")
            row["status"] = "failed"
    print(f"  [{arm} {support} T{trunc}] {row['gamma2']:+.4f} "
          f"({row['se2']:.4f}); {row['cells']:,} cells, {row['cells_used']} "
          f"used, {row['n_firms']:,} employers")
    ROWS.append(row)
    save()


def stacked(panels: dict, trunc: int) -> None:
    """A, B and C on the harmonised support in one fit, arm-specific
    effects and terms, clustered by employer: the differences get SEs."""
    parts = []
    for i, arm in enumerate(STACK):
        b = panels[arm][["employer_id", "year_month", "n_emp", "post_rb_x_high",
                         "post_gpt_x_high", "fe_emp_bin", "fe_emp_t"]].copy()
        b["arm_i"] = i
        for a2 in STACK:
            on = int(a2 == arm)
            b[f"rb_{a2}"] = b["post_rb_x_high"] * on
            b[f"gpt_{a2}"] = b["post_gpt_x_high"] * on
        parts.append(b.drop(columns=["post_rb_x_high", "post_gpt_x_high"]))
    s = pd.concat(parts, ignore_index=True)
    del parts
    s["fe_bin_arm"] = pd.factorize(s["fe_emp_bin"])[0].astype("int64") * 3 \
        + s["arm_i"]
    s["fe_t_arm"] = pd.factorize(s["fe_emp_t"])[0].astype("int64") * 3 \
        + s["arm_i"]
    s = s.drop(columns=["fe_emp_bin", "fe_emp_t"])
    terms = [f"{p}_{a}" for a in STACK for p in ("rb", "gpt")]
    print(f"  stacked T{trunc}: {len(s):,} rows")
    try:
        r = mc.run_fepois_multi(s, OUT, tag=f"s98_stacked_T{trunc}",
                                terms=terms, fes=("fe_bin_arm", "fe_t_arm"),
                                cluster="employer_id")
    except BaseException as ex:
        print(f"  stacked T{trunc} FAILED: {type(ex).__name__}: {ex}")
        traceback.print_exc()
        r = pd.DataFrame()
    if r.empty:
        FAILURES.append(f"stacked_T{trunc}")
        return
    g = r.set_index("term")
    v = pd.read_csv(r.attrs["vcov"]).set_index("term") \
        if "vcov" in r.attrs else None
    for arm in STACK:
        t = f"gpt_{arm}"
        sep = next((x["gamma2"] for x in ROWS if x["trunc"] == trunc
                    and x["arm"] == arm and x["support"] == "harmonised"), np.nan)
        if t in g.index and sep == sep and abs(float(g.loc[t, "coef"]) - sep) > 1e-4:
            NOTES.append(f"T{trunc}: stacked {arm} {float(g.loc[t, 'coef']):+.5f}"
                         f" differs from its separate fit {sep:+.5f}")
    for name, a, b in DIFFS:
        ta, tb = f"gpt_{a}", f"gpt_{b}"
        c = float(g.loc[ta, "coef"]) - float(g.loc[tb, "coef"])
        se = np.nan
        if v is not None:
            var = float(v.loc[ta, ta] + v.loc[tb, tb] - 2 * v.loc[ta, tb])
            se = float(np.sqrt(var)) if var > 0 else np.nan
        ROWS.append({"trunc": trunc, "arm": name, "support": "harmonised",
                     "gamma2": c, "se2": se, "cells": len(s),
                     "cells_used": int(g["n_obs"].max()), "n_firms": np.nan,
                     "person_months_22_25": np.nan, "status": "derived"})
        print(f"  T{trunc} {name}: {c:+.4f} ({se:.4f})")
    save()


def save() -> pd.DataFrame:
    df = pd.DataFrame(ROWS)
    if not df.empty:
        had = df["n_firms"].notna()
        df = mc.enforce_min_cell(df, count_col="n_firms", floor=5)
        small = had & df["n_firms"].isna()
        if small.any():
            df.loc[small, ["gamma2", "se2", "person_months_22_25"]] = np.nan
    df.to_csv(OUT / "backtest_common.csv", index=False)
    return df


def val(trunc, arm, support="own"):
    for r in ROWS:
        if r["trunc"] == trunc and r["arm"] == arm and r["support"] == support:
            return r["gamma2"], r["se2"]
    return np.nan, np.nan


def check_gate(trunc) -> None:
    bad = []
    for arm in ("A", "asof_all"):
        c, s = val(trunc, arm)
        wc, ws = GATE[(trunc, arm)]
        if not (abs(c - wc) <= GATE_TOL and abs(s - ws) <= GATE_TOL):
            bad.append(f"T{trunc} {arm}: this run {c:+.4f} ({s:.4f}), 45 "
                       f"{wc:+.4f} ({ws:.4f})")
    if bad:
        msg = "THE BACKTEST GATE FAILED. Nothing is quotable. " + "; ".join(bad)
        print(f"\n  {msg}")
        FAILURES.append(msg)
        write_summary()
        raise SystemExit("98: the gate failed; stopping.")
    print(f"  THE BACKTEST GATE PASSES at T{trunc}")


def headline_gate() -> None:
    """Table 1's tau at 22-25, on the paper's panel, before anything else."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "s82", HERE / "82_occupation_route.py")
    s82 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s82)
    s82.OUT = OUT
    s61, s67, s74, s78, s80, l47, l70, j47 = s82.load_modules()
    expo = s82.build_exposure(l47, l70, j47)["exposure"]
    counts = s82.load_counts("L_counts", s61.PANEL_YEARS,
                             require=["employer_id", "year_month",
                                      "age_group", "n_emp"])
    if counts is None:
        raise RuntimeError("L_counts_2021-2025 missing")
    b = s78.with_exposure(s61.build_skeleton(counts, AGE, j47), expo)
    del counts
    b, terms = s78.eq2_terms(b)
    r = mc.run_fepois_multi(b, OUT, tag="s98_headline_gate", terms=terms,
                            fes=j47.FES, cluster="employer_id")
    g = r.set_index("term")
    v = pd.read_csv(r.attrs["vcov"]).set_index("term")
    p_, i_ = "post_x_high_x_young", "interim_x_high_x_young"
    c = float(g.loc[p_, "coef"] - g.loc[i_, "coef"])
    s = float(np.sqrt(v.loc[p_, p_] + v.loc[i_, i_] - 2 * v.loc[p_, i_]))
    pc, ps = float(g.loc[p_, "coef"]), float(g.loc[p_, "se"])
    ROWS.append({"trunc": 0, "arm": "headline_tau_22_25", "support": "paper",
                 "gamma2": c, "se2": s, "cells": len(b),
                 "cells_used": int(g["n_obs"].max()),
                 "n_firms": int(b["employer_id"].nunique()),
                 "person_months_22_25": np.nan, "status": "gate"})
    save()
    bad = [f"{k}: this run {x:+.4f} ({y:.4f}), Table 1 {HEADLINE[k][0]:+.4f} "
           f"({HEADLINE[k][1]:.4f})" for k, (x, y) in
           (("post", (pc, ps)), ("tau", (c, s)))
           if not (abs(x - HEADLINE[k][0]) <= GATE_TOL
                   and abs(y - HEADLINE[k][1]) <= GATE_TOL)]
    if bad:
        FAILURES.append("THE HEADLINE GATE FAILED: " + "; ".join(bad))
        write_summary()
        raise SystemExit("98: the headline gate failed; stopping.")
    print(f"  THE HEADLINE GATE PASSES: tau {c:+.4f} ({s:.4f})")


def write_summary() -> None:
    L = ["THE THREE-ARM AS-OF BACKTEST", "=" * 28, "",
         "Estimator: 45's submitted design (employment at 22-25 in employer x",
         "exposure-quartile x month cells, 2019-2023; PostRB x High and",
         "PostGPT x High; employer-by-quartile and employer-by-month effects;",
         "Poisson; clustered by employer). Pseudo-dates for cutoff T: hike",
         "April T-1, launch December T-1; reference period before the hike.",
         "As-of cascade: own-year code to T, then Individ T, T-1, T-2.",
         "A = completed codes, full sample; B = completed codes, workers the",
         "as-of construction retains; C = as-of codes, same workers as B.", ""]
    h = [r for r in ROWS if r["arm"] == "headline_tau_22_25"]
    if h:
        L.append(f"HEADLINE GATE: tau {h[0]['gamma2']:+.4f} ({h[0]['se2']:.4f})")
        L.append("")
    for trunc in TRUNCATIONS:
        L.append(f"CUTOFF {trunc}:")
        for r in ROWS:
            if r["trunc"] != trunc:
                continue
            used = r["cells_used"]
            drop = (r["cells"] - used) if used == used else np.nan
            L.append(f"  {r['arm']:<10} {r['support']:<10} {r['gamma2']:+.4f} "
                     f"({r['se2']:.4f})  cells {r['cells']:,}, dropped by "
                     f"PPML {drop if drop == drop else 'n/a'}"
                     + (f", employers {int(r['n_firms']):,}"
                        if r["n_firms"] == r["n_firms"] else ""))
        L.append("")
    L += ["READ RULE: no verdict; if C - B is under half of C - A at the",
          "2021 cutoff, the artefact is sample inclusion as much as coding."]
    if NOTES:
        L += ["", "NOTES:"] + [f"  {n}" for n in NOTES]
    if FAILURES:
        L += ["", "FAILED: " + " | ".join(FAILURES),
              "A missing row is a missing fit, never a zero."]
    L += ["", f"Runtime {(time.time() - T0) / 60:.1f} min. " + mc.mem_line("")]
    (OUT / "98_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))


def run_cutoff(trunc: int, daioe, scorable) -> None:
    print(f"\n=== cutoff T = {trunc} ===")
    panel = dual_panel(trunc)
    panels = {}
    for arm in ARMS:
        keep, col = arm_rows(panel, arm, scorable)
        panels[arm] = build(panel, keep, col, daioe, trunc)
        fit_one(panels[arm], trunc, arm, "own")
        if arm == "asof_all":
            check_gate(trunc)
    del panel
    gc.collect()
    common = set.intersection(*(set(panels[a]["employer_id"]) for a in STACK))
    NOTES.append(f"T{trunc}: the harmonised support holds {len(common):,} "
                 f"employers (A {panels['A']['employer_id'].nunique():,}, "
                 f"B {panels['B']['employer_id'].nunique():,}, "
                 f"C {panels['C']['employer_id'].nunique():,})")
    h = {a: panels[a][panels[a]["employer_id"].isin(common)].copy()
         for a in STACK}
    del panels
    gc.collect()
    for arm in STACK:
        fit_one(h[arm], trunc, arm, "harmonised")
    stacked(h, trunc)


def main() -> int:
    global T0
    mc.Tee(OUT / "98_log.txt")
    T0 = time.time()
    print("=" * 70)
    print("98: THE THREE-ARM AS-OF BACKTEST")
    print("=" * 70)
    rc = 0
    try:
        if not SKIP_HEADLINE_GATE:
            headline_gate()
        daioe = mc.load_daioe()
        scorable = set(daioe["ssyk4"].astype(str))
        for trunc in TRUNCATIONS:
            run_cutoff(trunc, daioe, scorable)
    except SystemExit:
        mc.runlog("98_backtest_common", 2, (time.time() - T0) / 60)
        raise
    except BaseException as ex:
        print(f"98 FAILED: {type(ex).__name__}: {ex}")
        traceback.print_exc()
        FAILURES.append(f"main/{type(ex).__name__}: {ex}")
        rc = 1
    save()
    write_summary()
    rc = rc or (1 if FAILURES else 0)
    mc.runlog("98_backtest_common", rc, (time.time() - T0) / 60)
    print("\n98 done.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
