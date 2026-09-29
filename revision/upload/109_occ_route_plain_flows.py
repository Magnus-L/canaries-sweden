#!/usr/bin/env python3
"""
109_occ_route_plain_flows.py -- raw hires and separations by exposure
                                quartile and age band, on the paper's
                                occupation-route score.

======================================================================
  RUNS IN MONA (lane 40). Output folder CANARIES_109_OUT (default
  output_109). No SQL and no R: it reads the caches lane 28a and script
  54 left (L_baseline_2019*, L_counts_2021..2025, flows_2021..2025).
  Minutes, not hours.
======================================================================

QUESTION (ML, 29 Sep 2026)
Online Appendix Table A12 shows that young HEADCOUNT fell at exposed and
less exposed employers alike. Did young HIRING also fall alike, and how
much do the young depend on hiring compared with older workers? The only
raw flows we hold (script 66) sit on the education-route score, which the
paper no longer reports. This puts them on the reported score.

WHAT IT BUILDS
The score is 82.build_exposure(), called through script 85's own
load_modules(), so the quartile is the paper's and is not rebuilt. The
descriptive function is script 66's describe(), called on three
outcomes: headcount (n_emp, L_counts), hires and separations (n_hire,
n_sep, flows). A hire is an employer-person pair present in a month and
absent from that employer the month before; a separation the reverse
(script 54). Windows as in 66 and 85: pre 2022-01..2023-12 (24 months),
post 2024-01..2025-06 (18 months). Every change is computed per month.

READ RULES, fixed before the run:
  1. THE GATE. The headcount totals must reproduce script 85's exported
     occ_route_descriptive_full.csv (Table A12) EXACTLY in every cell.
     If they do not, the score or the panel has moved, and no flow
     number from this run is quoted.
  2. NO VERDICT. Raw totals with composition, firm size and the cycle
     inside them; seasonality is not balanced (the post window holds one
     extra first half-year). Read beside the estimates, never instead.
  3. The hire rate (monthly hires over monthly headcount, pre window) is
     reported by band and quartile, because a proportional fall in
     hiring shrinks a band's stock in proportion to that rate.
  Employer counts below 5 are suppressed before anything leaves MONA.

OUTPUTS (output_109/)
  occ_route_plain_flows.csv   fq, age_group, period, mean_value, total,
                              n_firms, n_cells, outcome (n_emp, n_hire,
                              n_sep)
  109_summary.txt             the gate, per-month changes, hire rates
"""

import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mona_common as mc

HERE = Path(__file__).resolve().parent
OUT = HERE / os.environ.get("CANARIES_109_OUT", "output_109")
OUT.mkdir(exist_ok=True)
os.environ.setdefault("CANARIES_85_OUT", str(OUT))
os.environ.setdefault("CANARIES_82_OUT", str(OUT))
CACHE = mc.CACHE_DIR

FLOOR = 5
MONTHS = {"pre": 24, "post": 18}
# Where script 85's Table A12 export may be found, for the gate.
PRIOR = ("output_85", "output_85d", ".")
PRIOR_FILE = "occ_route_descriptive_full.csv"
FAILURES, NOTES = [], []


def _mod(fname: str, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def per_month_change(g: pd.DataFrame) -> pd.DataFrame:
    """Per-month change from pre to post, by quartile and band."""
    piv = g.pivot_table(index=["fq", "age_group"], columns="period",
                        values="total")
    if not {"pre", "post"} <= set(piv.columns):
        return pd.DataFrame()
    piv["change"] = (piv["post"] / MONTHS["post"]) / \
                    (piv["pre"] / MONTHS["pre"]) - 1.0
    return piv.reset_index()


def gate(stock: pd.DataFrame) -> str:
    """Rule 1: headcount must equal Table A12's export cell for cell."""
    for d in PRIOR:
        p = HERE / d / PRIOR_FILE
        if p.exists():
            prior = pd.read_csv(p)
            break
    else:
        FAILURES.append("no Table A12 export on the share: gate not run")
        return "NO GATE: occ_route_descriptive_full.csv not found"
    key = ["fq", "age_group", "period"]
    # Normalise the key on both sides: age_group arrives categorical from
    # the cache and as text from the CSV (failure class 3).
    a, b = stock.copy(), prior.copy()
    for d in (a, b):
        d["fq"] = d["fq"].astype(int)
        d["age_group"] = d["age_group"].astype(str)
        d["period"] = d["period"].astype(str)
    m = a.merge(b, on=key, how="outer", suffixes=("", "_a12"),
                    indicator=True)
    unmatched = int((m["_merge"] != "both").sum())
    moved = m[(m["_merge"] == "both") &
              (m["total"].round(0) != m["total_a12"].round(0))]
    print(f"  gate: {len(m)} cells, {unmatched} unmatched, "
          f"{len(moved)} with a different total")
    if unmatched or len(moved):
        FAILURES.append("gate")
        for _, r in moved.head(10).iterrows():
            NOTES.append(f"moved: Q{r['fq']} {r['age_group']} {r['period']} "
                         f"{r['total']:.0f} against {r['total_a12']:.0f}")
        return ("THE PANEL HAS MOVED: headcount does not reproduce Table "
                "A12, NO FLOW NUMBER IS QUOTED")
    return "PASSED: headcount reproduces Table A12 in every cell"


def main():
    mc.Tee(OUT / "109_log.txt")
    t0 = time.time()
    print("=" * 70)
    print("109: RAW HIRES AND SEPARATIONS ON THE OCCUPATION-ROUTE SCORE")
    print("=" * 70)
    print(mc.mem_line("  "))

    s85 = _mod("85_occupation_route_plain_profile.py", "s85")
    s85.OUT = OUT
    s82, s61, s66, s74, s78, l47, l70, j47 = s85.load_modules()
    s66.OUT = OUT
    if tuple(s66.PRE) != ("2022-01", "2023-12") or s66.POST_FROM != "2024-01":
        raise RuntimeError("66's windows have moved; refusing to run.")
    built = s82.build_exposure(l47, l70, j47)
    occ = built["exposure"]
    print(f"  the score: {len(occ):,} employers on the {built['arm']} arm")

    bands_y, bands_o = j47.YOUNG_BANDS, j47.INCUMBENT_BANDS
    counts = s85.load_counts("L_counts", s61.PANEL_YEARS)
    if counts is None:
        raise RuntimeError("L_counts_* missing: run 47L first.")
    stock = s66.describe(counts, occ, "n_emp", bands_y, bands_o)
    del counts
    gc.collect()
    g_verdict = gate(stock)
    print(f"  {g_verdict}")

    flows = s85.load_counts("flows", s61.PANEL_YEARS)
    if flows is None:
        raise RuntimeError("flows_* missing: run 54 first.")
    parts = [stock.assign(outcome="n_emp")]
    for col in ("n_hire", "n_sep"):
        if col not in flows.columns:
            FAILURES.append(f"{col} not in flows cache")
            continue
        parts.append(s66.describe(flows, occ, col, bands_y, bands_o)
                     .assign(outcome=col))
    del flows
    gc.collect()
    allg = pd.concat(parts, ignore_index=True)
    allg = mc.enforce_min_cell(allg, count_col="n_firms", floor=FLOOR)
    allg.to_csv(OUT / "occ_route_plain_flows.csv", index=False)

    L = ["RAW HIRES AND SEPARATIONS ON THE OCCUPATION-ROUTE SCORE",
         "=" * 56, "", f"THE GATE: {g_verdict}", ""]
    for col, name in (("n_emp", "HEADCOUNT"), ("n_hire", "HIRES"),
                      ("n_sep", "SEPARATIONS")):
        ch = per_month_change(allg[allg["outcome"] == col])
        if ch.empty:
            continue
        L.append(f"{name}, per month, change pre to post (Q1 .. Q4):")
        for band in list(bands_y) + list(bands_o):
            r = ch[ch["age_group"].astype(str) == band].sort_values("fq")
            L.append(f"  {band:6s} " + "  ".join(
                f"Q{int(q)} {c:+6.1%}" for q, c in zip(r["fq"], r["change"])))
        L.append("")
    s = allg[(allg.outcome == "n_emp") & (allg.period == "pre")] \
        .set_index(["fq", "age_group"])["total"]
    h = allg[(allg.outcome == "n_hire") & (allg.period == "pre")] \
        .set_index(["fq", "age_group"])["total"]
    rate = (h / s).dropna()
    if len(rate):
        L.append("MONTHLY HIRE RATE, pre window (hires / headcount):")
        for band in list(bands_y) + list(bands_o):
            r = rate[rate.index.get_level_values(1).astype(str) == band]
            L.append(f"  {band:6s} " + "  ".join(
                f"Q{int(i[0])} {v:6.1%}" for i, v in r.items()))
        L.append("")
    if FAILURES:
        L += [f"WHAT FAILED: {', '.join(FAILURES)}", ""]
    if NOTES:
        L += ["NOTES:"] + [f"  {n}" for n in NOTES] + [""]
    L += ["Raw totals: composition, firm size and the cycle are inside them.",
          "The post window holds one extra first half-year, so seasonality",
          "is not balanced. Cells on fewer than 5 employers are dropped.",
          "", f"Runtime {(time.time()-t0)/60:.1f} min. {mc.mem_line('')}"]
    (OUT / "109_summary.txt").write_text("\n".join(L) + "\n",
                                         encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
