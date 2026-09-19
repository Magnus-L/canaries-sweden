#!/usr/bin/env python3
"""
56_dynamics_by_age.py -- the effect of AI exposure BY AGE and OVER TIME, on
                         measurement the register lag cannot reach.

======================================================================
  RUNS IN MONA. Reads the caches 47L and 54 already wrote, so it runs
  no SQL if either has run. Writes output_56/.
======================================================================

WHY THIS EXISTS.

The question is what generative AI has done to workers of different ages
since late 2022. Answering it needs three things at once: an exposure
measure the register lag cannot corrupt, a breakdown by age, and a path
through time. We have the first two. 47L and 54 each report one number per
age band, pooled over the whole post period, and a pooled number cannot
distinguish a shock that arrived with ChatGPT from a trend that was already
running in 2019. That distinction is the difference between an effect and a
correlation, so this script supplies it.

WHAT IT ESTIMATES. For each outcome, two Poisson event studies on the same
panel and the same fixed effects as the design that produced it:

  (a) POOLED       E(f,a) x half-year, one coefficient per half-year,
                   referenced to 2022H1. The average dynamic response.
  (b) BY AGE       the same, plus E(f,a) x half-year x 1[age = 22-25].
                   The second set is the YOUNG DIFFERENTIAL over time, and
                   it is the object the paper is about: whether the young
                   moved differently from everyone else in the same firm,
                   and when.

Three outcomes, each from a design that fails differently:

  stock    employment headcount        (47L's panel)
  hires    starts, the fast margin     (54's panel)
  seps     separations                 (54's panel)

WHY THE AGE BANDS ARE INTERACTED RATHER THAN SPLIT. The design absorbs
employer x month. Estimated on one age band at a time, that fixed effect
has one observation per employer-month and absorbs the outcome entirely,
so nothing is identified. The age contrast has to live inside a single fit
on the multi-age panel. This is not a convenience; splitting the sample
here would silently produce zeros.

HOW TO READ IT, decided before the numbers exist:

  1. The pre-period coefficients are the test. If the young differential is
     already trending before 2022H2, the design is picking up something
     that predates the shock and the post-period coefficients do not carry
     a causal reading. Say so plainly rather than quoting the post period.
  2. A shock that arrives with ChatGPT should appear in 2022H2 or 2023H1
     and persist. One isolated half-year is noise.
  3. The stock and the flow should disagree in a specific way if firms
     adjusted through hiring: the hire path turns down first and the stock
     path follows slowly, or does not follow at all inside this window.
     If the stock moves first, the story is not entry-level.
  4. Exposure is frozen in 2019 and is a stale proxy for who is exposed in
     2025. That attenuates every coefficient here toward zero and the
     attenuation grows with distance from 2019, so a FLAT path late in the
     window is weaker evidence of no effect than a flat path early in it.

Output (output_56/):
  dynamics_pooled.csv      outcome x halfyear
  dynamics_young.csv       outcome x halfyear, the 22-25 differential
  56_summary.txt
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
OUT = HERE / "output_56"
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR

YOUNG = "22-25"
REF = mc.REF_HALFYEAR          # 2022H1, the house reference
FES = ("fe_emp_t", "fe_emp_age", "fe_t_age")


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


def add_event_terms(bal: pd.DataFrame) -> tuple:
    """
    E(f,a) x half-year, and the same interacted with the young indicator.
    The reference half-year is omitted from both sets, so every coefficient
    reads against 2022H1, the last half-year wholly before ChatGPT.
    """
    b = bal.copy()
    b["halfyear"] = mc.assign_halfyear(pd.Series(b["year_month"].astype(str)))
    b["is_young"] = (b["age_group"] == YOUNG).astype(int)
    hys = sorted(h for h in b["halfyear"].unique() if h != REF)
    pooled, young = [], []
    for h in hys:
        d = (b["halfyear"] == h).astype(int)
        pn = f"expo_{h}"
        yn = f"expo_{h}_young"
        b[pn] = b["expo_z"] * d
        b[yn] = b["expo_z"] * d * b["is_young"]
        pooled.append(pn)
        young.append(yn)
    return b, pooled, young


def run_es(bal, outcome: str, terms, tag: str) -> pd.DataFrame:
    """Poisson with the event-study term list. Returns the coefficients."""
    b = bal.copy()
    b["n_emp"] = b[outcome]
    r = mc.run_fepois_multi(b, OUT, tag=tag, terms=terms, fes=FES)
    if r.empty:
        return r
    r = r[r["term"].isin(terms)].copy()
    r["is_young_term"] = r["term"].str.endswith("_young")
    r["halfyear"] = (r["term"].str.replace("_young", "", regex=False)
                     .str.replace("expo_", "", regex=False))
    return r[["term", "is_young_term", "halfyear", "coef", "se", "pvalue",
              "n_obs", "status"]]


def reference_row(df: pd.DataFrame) -> pd.DataFrame:
    """Put the omitted half-year back in, at zero, so a plot is complete."""
    if df.empty:
        return df
    row = {"term": f"expo_{REF}", "is_young_term": False,
           "halfyear": REF, "coef": 0.0, "se": 0.0, "pvalue": np.nan,
           "n_obs": df["n_obs"].iloc[0], "status": "reference"}
    return (pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            .sort_values("halfyear").reset_index(drop=True))


def main():
    mc.Tee(OUT / "56_log.txt")
    t0 = time.time()
    print("=" * 70)
    print("56: AI EXPOSURE BY AGE AND OVER TIME")
    print("=" * 70)
    print(f"  reference half-year {REF}; young band {YOUNG}")
    print(mc.mem_line("  "))

    l47 = _mod("47L_age_baseline_exposure.py")
    s54 = _mod("54_hiring_flows.py")

    base = mc.read_cache(CACHE / "L_baseline_2019.parquet")
    if base is None:
        raise RuntimeError(
            "L_baseline_2019.parquet is missing: run 47L (lane 4) first. "
            "This script deliberately performs no SQL of its own.")
    daioe = pd.read_stata(str(Path(mc.SHARE) / "daioe_quartiles.dta"))
    daioe["ssyk4"] = daioe["ssyk4"].astype(str).str.zfill(4)
    daioe = daioe.rename(columns={"pctl_rank_genai": "score"})[["ssyk4",
                                                               "score"]]
    expo = l47.build_exposure(base, daioe)
    print(f"  exposure: {len(expo):,} firm-age cells, "
          f"{expo['employer_id'].nunique():,} firms")

    panels = {}

    # --- the stock, from 47L's counts ---------------------------------
    cnt = []
    for y in l47.YEARS:
        c = mc.read_cache(CACHE / f"L_counts_{y}.parquet")
        if c is not None:
            cnt.append(c)
    if cnt:
        bal = l47.build_panel(pd.concat(cnt, ignore_index=True), expo)
        panels["stock"] = (bal, "n_emp")
        print(f"  stock panel: {len(bal):,} cells")
    else:
        print("  stock panel: L_counts_* missing, skipped (run 47L first)")
    del cnt
    gc.collect()

    # --- the flows, from 54 -------------------------------------------
    fl = []
    for y in s54.YEARS:
        f = mc.read_cache(CACHE / f"flows_{y}.parquet", require=s54.FLOW_COLS)
        if f is not None:
            fl.append(f)
    if fl:
        balf = s54.build_panel(pd.concat(fl, ignore_index=True), expo)
        panels["hires"] = (balf, "n_hire")
        panels["seps"] = (balf, "n_sep")
        print(f"  flow panel: {len(balf):,} cells")
    else:
        print("  flow panels: flows_* missing, skipped (run 54 first)")
    del fl
    gc.collect()

    if not panels:
        raise RuntimeError("no panel available: run 47L or 54 before this.")

    pooled_out, young_out = [], []
    for label, (bal, outcome) in panels.items():
        print(f"\n--- {label} ---")
        b, p_terms, y_terms = add_event_terms(bal)
        t1 = time.time()
        rp = run_es(b, outcome, p_terms, f"es56_{label}_pooled")
        if rp.empty:
            raise RuntimeError(f"the pooled event study for {label} returned "
                               f"nothing; this is a primary estimate")
        rp = reference_row(rp)
        rp["outcome"] = label
        pooled_out.append(rp)
        pd.concat(pooled_out, ignore_index=True).to_csv(
            OUT / "dynamics_pooled.csv", index=False)
        print(f"  pooled ({time.time()-t1:.0f}s):")
        for _, r in rp.iterrows():
            print(f"    {r['halfyear']}  {r['coef']:+.4f} "
                  f"(SE {r['se']:.4f})  {r['status']}")

        t1 = time.time()
        ry = opt(f"young differential ({label})", run_es, b, outcome,
                 p_terms + y_terms, f"es56_{label}_young")
        if ry is not None and not ry.empty:
            # only the young-interacted terms are the differential. Filter
            # by NAME: fixest does not promise to return coefficients in
            # the order they were passed, so taking the last k rows would
            # silently mix the two sets if that order ever changed.
            ry = ry[ry["is_young_term"]].copy()
            if ry.empty:
                raise RuntimeError(
                    "the young-interacted terms are all absent from the "
                    "fit: they were collinear or dropped, and the age "
                    "contrast is not identified on this panel")
            ry = reference_row(ry)
            ry["outcome"] = label
            young_out.append(ry)
            pd.concat(young_out, ignore_index=True).to_csv(
                OUT / "dynamics_young.csv", index=False)
            print(f"  22-25 differential ({time.time()-t1:.0f}s):")
            for _, r in ry.iterrows():
                print(f"    {r['halfyear']}  {r['coef']:+.4f} "
                      f"(SE {r['se']:.4f})  {r['status']}")
        del b
        gc.collect()

    # --- summary -------------------------------------------------------
    lines = ["AI EXPOSURE BY AGE AND OVER TIME", "=" * 52,
             f"Poisson event studies, reference {REF}, exposure frozen 2019.",
             "Absorbed: employer x month, employer x age, month x age.", ""]
    if pooled_out:
        P = pd.concat(pooled_out, ignore_index=True)
        lines += ["POOLED over ages, coefficient on E(f,a) x half-year:",
                  P.pivot_table(index="halfyear", columns="outcome",
                                values="coef").round(4).to_string(), ""]
    if young_out:
        Y = pd.concat(young_out, ignore_index=True)
        piv = Y.pivot_table(index="halfyear", columns="outcome",
                            values="coef").round(4)
        lines += ["22-25 DIFFERENTIAL, coefficient on "
                  "E(f,a) x half-year x young:", piv.to_string(), ""]
        pre = [h for h in piv.index if h < "2022H2" and h != REF]
        if pre:
            worst = piv.loc[pre].abs().max()
            lines += ["PRE-PERIOD, which is the test:",
                      "  largest |coefficient| before 2022H2, by outcome:",
                      "  " + "  ".join(f"{c} {worst[c]:.4f}"
                                       for c in piv.columns),
                      "  A young differential already moving here means the",
                      "  post-period coefficients are not a shock response.",
                      ""]
    lines += [
        "READ THIS BEFORE QUOTING ANY OF IT:",
        "  1. The pre-period is the test, not the decoration. Quote it.",
        "  2. A shock arriving with ChatGPT shows in 2022H2 or 2023H1 and",
        "     persists. One isolated half-year is noise.",
        "  3. Stock and flow should disagree if firms adjusted through",
        "     hiring: hires turn first, the stock follows slowly or not at",
        "     all within this window. If the stock moves first, the story",
        "     is not entry-level.",
        "  4. Exposure is frozen in 2019, so attenuation toward zero GROWS",
        "     with distance from the baseline. A flat path late in the",
        "     window is weaker evidence of no effect than a flat path",
        "     early in it.",
        "  5. The age contrast is estimated inside one fit on the",
        "     multi-age panel. It cannot be estimated by splitting the",
        "     sample: employer x month is a singleton within one age band",
        "     and would absorb the outcome.",
        "", f"Runtime {(time.time()-t0)/60:.1f} min. " + mc.mem_line()]
    (OUT / "56_summary.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print("\n56 done.")


if __name__ == "__main__":
    main()
