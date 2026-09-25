#!/usr/bin/env python3
"""
98_backtest_common.py -- the as-of backtest re-run on a common sample, so
                         that what the stale register CODES and whom it
                         KEEPS are separated rather than summed.

======================================================================
  RUNS IN MONA (lane 37c, second stage). Output folder CANARIES_98_OUT
  (default output_98). No SQL if script 45's caches
  cache/panel_dual_T2021.parquet and panel_dual_T2022.parquet are still on
  the share (written 18 Sep 2026; no round has retired them). If one is
  missing, 45's own pull rebuilds it (about 17 minutes a truncation).
======================================================================

QUESTION
Online Appendix IV.3 reports the as-of backtest of the submitted design:
on 2019-2023, where every worker's own-year occupation is observed, the
design returns +0.0193 (SE 0.0129) with true codes and -0.2875 (0.0171)
with the codes a register truncated at 2021 would have given, an artefact
of -0.3068 (+0.0176 against -0.1452 at the 2022 truncation). The review
(25 Sep 2026) noted that the two arms differ in two ways at once: the
as-of arm CODES workers with stale codes, and it KEEPS only the workers a
truncated register can code at all (the true arm keeps everyone with an
own-year code). The artefact is therefore coding plus sample inclusion,
and the paper's "completed-vintage codes" wording needs to know which.

DESIGN
45's estimator unchanged (employer by exposure quartile by month cells at
22-25, employers with a cumulative count of at least five, balanced,
restricted to employers in both the top quartile and a lower one;
PostRB x High and PostGPT x High with the pseudo-dates of 45; Poisson
with employer-by-quartile and employer-by-month effects, clustered by
employer), on four arms of the same dual panel:
  true_all     own-year codes, every worker who has one (45's true arm)
  asof_all     as-of codes, every worker who has one (45's as-of arm)
  true_common  own-year codes, restricted to the COMMON sample: workers
               carrying a DAIOE-scorable code in BOTH assignments
  asof_common  as-of codes on the same common sample
so that
  asof_all - true_all = (true_common - true_all)     sample inclusion
                      + (asof_common - true_common)  coding, same workers
                      + (asof_all - asof_common)     workers only the
                                                     as-of arm can score
The three pieces sum to the artefact by construction; each is reported.

THE GATE (hard stop)
true_all and asof_all must reproduce 45's asof_estimates.csv within
0.0005 on coefficient and standard error at both truncations: T2021
+0.0193 (0.0129) and -0.2875 (0.0171); T2022 +0.0176 (0.0111) and
-0.1452 (0.0122). A miss means a moved panel and nothing is quoted.

READ RULE, FIXED BEFORE THE RUN
Reported, no verdict: the three pieces and each one's share of the
artefact at each truncation. If the coding piece is under half the
artefact at T2021, the paper's sentence must attribute the artefact to
sample inclusion as much as to stale codes; if it is over half, "stale
codes" stands with the sample-inclusion caveat.

EXPORT (output_98/)
  backtest_common.csv  per truncation and arm: gamma2, se, n_obs, the
                       employers in the fit and the person-months at
                       22-25 behind it; then the three pieces
  98_summary.txt, 98_log.txt (the vcov files stay on the share)

IN THE PAPER
Online Appendix IV.3 (Table IV.3 gains the two common-sample columns)
and the response letter's backtest paragraph ("completed-vintage codes",
the sample-inclusion caveat).

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

TRUNCATIONS = (2021, 2022)
AGE = "22-25"
STEP1_MIN_CUMULATIVE = 5          # 45's constant
DUAL_COLS = ["employer_id", "year_month", "ssyk_true", "ssyk_asof",
             "age_group", "n_emp"]
GATE = {(2021, "true_all"): (0.0193, 0.0129), (2021, "asof_all"): (-0.2875, 0.0171),
        (2022, "true_all"): (0.0176, 0.0111), (2022, "asof_all"): (-0.1452, 0.0122)}
GATE_TOL = 0.0005
ARMS = ("true_all", "asof_all", "true_common", "asof_common")

NOTES: list = []
FAILURES: list = []
ROWS: list = []
T0 = time.time()


def dual_panel(trunc: int) -> pd.DataFrame:
    """45's cached dual panel, schema-checked; rebuilt by 45's own pull
    only if missing (45 is imported only then, so a run on the caches
    touches nothing of 45's)."""
    cf = mc.CACHE_DIR / f"panel_dual_T{trunc}.parquet"
    p = None
    if cf.exists():
        # About 120 million rows with four text columns: read them as
        # categoricals straight from Arrow, or the frame alone would hold
        # some 25 GB of Python strings. The schema is checked first, as
        # mona_common.read_cache would.
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
    """(rows of the dual panel the arm keeps, the code column it scores).
    The common sample keeps a row only if BOTH codes are DAIOE-scorable,
    so the two common arms hold exactly the same workers."""
    if arm == "true_all":
        return panel["ssyk_true"] != "____", "ssyk_true"
    if arm == "asof_all":
        return panel["ssyk_asof"] != "____", "ssyk_asof"
    both = (panel["ssyk_true"].astype(str).isin(scorable)
            & panel["ssyk_asof"].astype(str).isin(scorable))
    return both, ("ssyk_true" if arm == "true_common" else "ssyk_asof")


def estimate(panel, keep, col, daioe, trunc, arm) -> dict:
    """45's estimate_both for one assignment, line for line, on the rows
    `keep` selects and the code column `col`; the pooled Poisson only."""
    pseudo_gpt = f"{trunc - 1}-12"
    pseudo_rb = f"{trunc - 1}-04"
    agg = (panel[keep]
           .groupby(["employer_id", "year_month", col, "age_group"],
                    observed=True)["n_emp"].sum().reset_index()
           .rename(columns={col: "ssyk4"}))
    agg["year_month"] = agg["year_month"].astype(str)
    agg["age_group"] = agg["age_group"].astype(str)
    agg["ssyk4"] = agg["ssyk4"].astype(str)
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
    print(f"  [{arm} T{trunc}] {len(bal):,} cells, "
          f"{bal['employer_id'].nunique():,} employers")
    row = {"trunc": trunc, "arm": arm, "gamma2": np.nan, "se2": np.nan,
           "n_obs": len(bal), "n_firms": int(bal["employer_id"].nunique()),
           "person_months_22_25": float(bal["n_emp"].sum()), "status": "empty"}
    if bal.empty:
        return row
    try:
        pres = mc.run_fepois(bal, OUT, tag=f"s98_{arm}_T{trunc}")
    except BaseException as ex:
        print(f"  [{arm} T{trunc}] FAILED: {type(ex).__name__}: {ex}")
        traceback.print_exc()
        pres = pd.DataFrame()
    g2 = pres.loc[pres["term"] == "post_gpt_x_high"] if len(pres) else pres
    if len(g2):
        row.update(gamma2=float(g2["coef"].iloc[0]),
                   se2=float(g2["se"].iloc[0]), status="ok")
    else:
        FAILURES.append(f"{arm}_T{trunc}")
        row["status"] = "failed"
    return row


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


def val(trunc, arm):
    for r in ROWS:
        if r["trunc"] == trunc and r["arm"] == arm:
            return r["gamma2"], r["se2"]
    return np.nan, np.nan


def check_gate(trunc) -> None:
    bad = []
    for arm in ("true_all", "asof_all"):
        c, s = val(trunc, arm)
        wc, ws = GATE[(trunc, arm)]
        if not (abs(c - wc) <= GATE_TOL and abs(s - ws) <= GATE_TOL):
            bad.append(f"T{trunc} {arm}: this run {c:+.4f} ({s:.4f}), 45 "
                       f"{wc:+.4f} ({ws:.4f})")
    if bad:
        msg = "THE GATE FAILED. Nothing from this run is quotable. " + \
              "; ".join(bad)
        print(f"\n  {msg}")
        FAILURES.append(msg)
        write_summary()
        raise SystemExit("98: the gate failed; stopping.")
    print(f"  THE GATE PASSES at T{trunc}")


def pieces(trunc) -> dict:
    ta, _ = val(trunc, "true_all")
    aa, _ = val(trunc, "asof_all")
    tc, _ = val(trunc, "true_common")
    ac, _ = val(trunc, "asof_common")
    return {"artefact": aa - ta, "sample_inclusion": tc - ta,
            "coding_same_workers": ac - tc,
            "asof_only_workers": aa - ac}


def write_summary() -> None:
    L = ["THE AS-OF BACKTEST ON A COMMON SAMPLE", "=" * 38, "",
         "45's estimator on four arms of one dual panel (22-25, submitted",
         "design). The common sample keeps a worker-month only if both the",
         "own-year and the as-of code are DAIOE-scorable.", ""]
    for trunc in TRUNCATIONS:
        L.append(f"T{trunc}:")
        for arm in ARMS:
            c, s = val(trunc, arm)
            r = next((x for x in ROWS if x["trunc"] == trunc
                      and x["arm"] == arm), None)
            if r is not None:
                L.append(f"  {arm:<12} {c:+.4f} ({s:.4f})  {r['n_firms']:,} "
                         f"employers, {r['person_months_22_25']:,.0f} "
                         f"person-months")
        pc = pieces(trunc)
        if all(v == v for v in pc.values()):
            art = pc["artefact"]
            for k in ("sample_inclusion", "coding_same_workers",
                      "asof_only_workers"):
                sh = pc[k] / art if art else np.nan
                L.append(f"  {k:<22} {pc[k]:+.4f}  ({sh:.0%} of the artefact "
                         f"{art:+.4f})")
        L.append("")
    L += ["READ RULE: no verdict; if the coding piece is under half the",
          "artefact at T2021 the paper attributes the artefact to sample",
          "inclusion as much as to stale codes."]
    if NOTES:
        L += ["", "NOTES:"] + [f"  {n}" for n in NOTES]
    if FAILURES:
        L += ["", "FAILED: " + " | ".join(FAILURES),
              "A missing row is a missing fit, never a zero."]
    L += ["", f"Runtime {(time.time() - T0) / 60:.1f} min. " + mc.mem_line("")]
    (OUT / "98_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))


def main() -> int:
    global T0
    mc.Tee(OUT / "98_log.txt")
    T0 = time.time()
    print("=" * 70)
    print("98: THE AS-OF BACKTEST ON A COMMON SAMPLE")
    print("=" * 70)
    rc = 0
    try:
        daioe = mc.load_daioe()
        scorable = set(daioe["ssyk4"].astype(str))
        for trunc in TRUNCATIONS:
            print(f"\n=== truncation T = {trunc} ===")
            panel = dual_panel(trunc)
            for arm in ARMS:
                keep, col = arm_rows(panel, arm, scorable)
                ROWS.append(estimate(panel, keep, col, daioe, trunc, arm))
                save()
                if arm == "asof_all":
                    check_gate(trunc)
            del panel
            gc.collect()
            pc = pieces(trunc)
            for k, v in pc.items():
                ROWS.append({"trunc": trunc, "arm": f"piece_{k}",
                             "gamma2": v, "se2": np.nan, "n_obs": np.nan,
                             "n_firms": np.nan,
                             "person_months_22_25": np.nan,
                             "status": "derived"})
            save()
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
