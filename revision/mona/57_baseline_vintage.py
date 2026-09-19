#!/usr/bin/env python3
"""
57_baseline_vintage.py -- the effect is expected LATE, and a 2019 baseline
                          is weakest late. Measure that, and fix what can
                          be fixed.

======================================================================
  RUNS IN MONA. One SQL pull per baseline year; reuses 47L's counts
  and 54's flow caches for everything else. Writes output_57/.
======================================================================

THE PROBLEM THIS ADDRESSES.

47L and 54 freeze exposure at what a firm's age-a workers did in 2019.
That is what makes them immune to the register lag, and it is also their
weakness: by 2025 the 2019 assignment is a six-year-old proxy for who is
exposed now. Classical measurement error in a continuous regressor
attenuates the coefficient toward zero, and here the error GROWS with
distance from the baseline. So the design is least able to see an effect
in exactly the years where generative AI is most likely to have produced
one.

That is not a caveat to be written into a footnote. It is a quantity, and
both parts of it are measurable in these data.

PART 1 -- HOW FAST DOES THE 2019 ASSIGNMENT DECAY?

For every year with its own occupation register, rebuild E(f,a)
contemporaneously and regress it on the 2019 version across firm-age
cells. The slope is the reliability ratio lambda(y): the share of the
2019 signal still present in year y. Attenuation is multiplicative, so a
coefficient estimated with the 2019 baseline in year y is approximately
lambda(y) times the coefficient a contemporaneous baseline would give,
and dividing by lambda(y) undoes it. lambda is measurable for 2020
through 2023 and must be extrapolated beyond, which the output states
rather than hides.

PART 2 -- A BASELINE CLOSER TO WHERE THE EFFECT IS

Re-run the design with exposure frozen in 2022 instead of 2019. 2022 is
still PRE-TREATMENT (ChatGPT is November 2022 and the register's
reference month is November, so the 2022 occupation mix is not a response
to it), and it is three years closer to the outcome years that matter.
If the effect is late, this is the baseline that can see it.

The two baselines answer different questions and both are reported:
  2019  cleanest pre-period, longest pre-trend, most attenuated late
  2022  weakest pre-period (one half-year), least attenuated late

WHAT WOULD MAKE PART 2 WRONG. A 2022 baseline sits after the Riksbank
tightening of April 2022, so a firm whose occupation mix had already
responded to that shock carries it into the exposure measure. The
occupation mix inside a firm-age cell moves slowly, so this is a second-
order worry, but it is a real one and it is why the 2019 baseline stays
the headline and 2022 is the check on attenuation, not the replacement.

Output (output_57/):
  reliability.csv        lambda(y) by year and age band
  vintage_estimates.csv  pooled gamma per baseline year x outcome
  vintage_gradient.csv   age gradient per baseline year x outcome
  57_summary.txt
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
OUT = HERE / "output_57"
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR

BASE_YEARS = (2019, 2021, 2022, 2023)   # every year with its own register
HEADLINE_BASE = 2019
LATE_BASE = 2022
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


def q_baseline_year(year: int, conn) -> pd.DataFrame:
    """
    47L's baseline pull, parameterised by year. November of the base year,
    matching the occupation register's reference month.
    """
    age_case = "\n".join(
        f"             WHEN {year} - TRY_CAST(i.FodelseAr AS INT) "
        f"BETWEEN {lo} AND {hi} THEN '{lab}'"
        for lab, (lo, hi) in mc.AGE_GROUPS.items())
    age_case = f"CASE\n{age_case}\n             ELSE NULL END"
    code = ("""CASE WHEN i.Ssyk4_2012_J16 IS NULL
                      OR LTRIM(i.Ssyk4_2012_J16) = ''
                      OR LEFT(LTRIM(i.Ssyk4_2012_J16), 1) = '*'
                 THEN '____'
                 ELSE RIGHT('0000' + CAST(i.Ssyk4_2012_J16 AS VARCHAR(4)), 4)
                 END""")
    q = f"""
    SELECT agi.P1207_LOPNR_PEORGNR AS employer_id,
           {age_case} AS age_group,
           {code} AS ssyk4,
           LTRIM(RTRIM(i.SsykStatus_J16)) AS ssyk_status,
           COUNT(DISTINCT agi.P1207_LOPNR_PERSONNR) AS n
    FROM dbo.Arb_AGIIndivid{year}11_def agi
    LEFT JOIN dbo.Individ_{year} i
      ON agi.P1207_LOPNR_PERSONNR = i.P1207_LopNr_PersonNr
    WHERE {year} - TRY_CAST(i.FodelseAr AS INT) BETWEEN 22 AND 69
    GROUP BY agi.P1207_LOPNR_PEORGNR, {age_case}, {code},
             LTRIM(RTRIM(i.SsykStatus_J16))
    """
    return pd.read_sql(q, conn)


def reliability(expos: dict) -> pd.DataFrame:
    """
    lambda(y): the slope of E_y on E_2019 across firm-age cells, overall
    and by age band, with the correlation and the cell count beside it.

    The slope, not the correlation, is the attenuation factor: for a
    regressor measured with error, the probability limit of the estimated
    coefficient is the true one times Cov(E_2019, E_y)/Var(E_2019), and
    that ratio IS the slope of this regression.
    """
    ref = expos[HEADLINE_BASE][["employer_id", "age_group", "expo"]].rename(
        columns={"expo": "e0"})
    rows = []
    for y, e in sorted(expos.items()):
        if y == HEADLINE_BASE:
            rows.append({"year": y, "age_group": "ALL", "lam": 1.0,
                         "corr": 1.0, "n_cells": len(ref)})
            for a in mc.AGE_GROUPS:
                rows.append({"year": y, "age_group": a, "lam": 1.0,
                             "corr": 1.0,
                             "n_cells": int((ref["age_group"] == a).sum())})
            continue
        m = ref.merge(e[["employer_id", "age_group", "expo"]],
                      on=["employer_id", "age_group"], how="inner")
        for a in ["ALL"] + list(mc.AGE_GROUPS):
            d = m if a == "ALL" else m[m["age_group"] == a]
            if len(d) < 50 or d["e0"].var() == 0:
                rows.append({"year": y, "age_group": a, "lam": np.nan,
                             "corr": np.nan, "n_cells": len(d)})
                continue
            lam = np.cov(d["e0"], d["expo"])[0, 1] / d["e0"].var()
            rows.append({"year": y, "age_group": a, "lam": float(lam),
                         "corr": float(np.corrcoef(d["e0"], d["expo"])[0, 1]),
                         "n_cells": len(d)})
    return pd.DataFrame(rows)


def main():
    mc.Tee(OUT / "57_log.txt")
    t0 = time.time()
    print("=" * 70)
    print("57: BASELINE VINTAGE AND ATTENUATION")
    print("=" * 70)
    print("  the effect is expected late; a 2019 baseline is weakest late.")
    print(mc.mem_line("  "))

    l47 = _mod("47L_age_baseline_exposure.py")
    s54 = _mod("54_hiring_flows.py")
    conn = None

    daioe = pd.read_stata(str(Path(mc.SHARE) / "daioe_quartiles.dta"))
    daioe["ssyk4"] = daioe["ssyk4"].astype(str).str.zfill(4)
    daioe = daioe.rename(columns={"pctl_rank_genai": "score"})[["ssyk4",
                                                               "score"]]

    # --- baselines -----------------------------------------------------
    expos = {}
    for y in BASE_YEARS:
        cf = (CACHE / "L_baseline_2019.parquet" if y == 2019
              else CACHE / f"L_baseline_{y}.parquet")
        b = mc.read_cache(cf)
        if b is None:
            conn = conn or mc.connect()
            t1 = time.time()
            b = q_baseline_year(y, conn)
            mc.write_cache(b, cf)
            print(f"  baseline {y}: {len(b):,} rows ({time.time()-t1:.0f}s)")
        else:
            print(f"  baseline {y}: cached ({len(b):,} rows)")
        expos[y] = l47.build_exposure(b, daioe)
        print(f"    -> {len(expos[y]):,} firm-age cells, "
              f"{expos[y]['employer_id'].nunique():,} firms")

    # --- PART 1: the decay curve ---------------------------------------
    rel = reliability(expos)
    rel.to_csv(OUT / "reliability.csv", index=False)
    print("\nRELIABILITY OF THE 2019 ASSIGNMENT, lambda(y)")
    piv = rel.pivot_table(index="year", columns="age_group", values="lam")
    cols = ["ALL"] + [a for a in mc.AGE_GROUPS if a in piv.columns]
    print(piv[cols].round(3).to_string())
    lam_all = rel[(rel["age_group"] == "ALL")].set_index("year")["lam"]
    if lam_all.notna().sum() >= 3:
        yrs = lam_all.dropna().index.values.astype(float)
        vals = lam_all.dropna().values
        slope = np.polyfit(yrs - HEADLINE_BASE, vals, 1)[0]
        proj = {y: max(1.0 + slope * (y - HEADLINE_BASE), 0.0)
                for y in (2024, 2025)}
        print(f"  linear decay {slope:+.4f} a year; EXTRAPOLATED "
              f"lambda(2024) {proj[2024]:.3f}, lambda(2025) {proj[2025]:.3f}")
        print("  (extrapolated, not measured: there is no 2024 or 2025 "
              "occupation register to check it against)")

    # --- PART 2: the same design on two baselines ----------------------
    cnt = [c for c in (mc.read_cache(CACHE / f"L_counts_{y}.parquet")
                       for y in l47.YEARS) if c is not None]
    fl = [f for f in (mc.read_cache(CACHE / f"flows_{y}.parquet",
                                    require=s54.FLOW_COLS)
                      for y in s54.YEARS) if f is not None]
    counts = pd.concat(cnt, ignore_index=True) if cnt else None
    flows = pd.concat(fl, ignore_index=True) if fl else None
    del cnt, fl
    gc.collect()

    rows, grads = [], []
    for base in (HEADLINE_BASE, LATE_BASE):
        expo = expos[base]
        jobs = []
        if counts is not None:
            jobs.append(("stock", l47.build_panel(counts, expo), "n_emp"))
        if flows is not None:
            bf = s54.build_panel(flows, expo)
            jobs += [("hires", bf, "n_hire"), ("seps", bf, "n_sep")]
        for label, bal, outcome in jobs:
            if bal.empty:
                continue
            b = bal.copy()
            b["n_emp"] = b[outcome]
            r = mc.run_fepois_multi(b, OUT, tag=f"v57_{base}_{label}",
                                    terms=s54.TERMS, fes=FES)
            if r.empty or not (r["term"] == "post_gpt_x_expo").any():
                raise RuntimeError(f"no estimate for {base}/{label}")
            g = r[r["term"] == "post_gpt_x_expo"].iloc[0]
            rows.append({"baseline": base, "outcome": label,
                         "gamma": float(g["coef"]), "se": float(g["se"]),
                         "n_obs": int(g["n_obs"])})
            pd.DataFrame(rows).to_csv(OUT / "vintage_estimates.csv",
                                      index=False)
            print(f"  [base {base} {label:<5}] PostGPT x exposure "
                  f"{g['coef']:+.4f} (SE {g['se']:.4f})")
            # KEYWORDS, not positions: run_fepois_multi's signature is
            # (panel, workdir, tag, terms, cluster=..., fes=...), so a
            # positional FES lands in `cluster` and the fit dies on a
            # KeyError naming the DEFAULT fixed effects, which is a very
            # confusing way to find out.
            gr = opt(f"gradient {base}/{label}", mc.run_fepois_multi, b,
                     OUT, tag=f"vg57_{base}_{label}",
                     terms=s54.TERMS_GRAD, fes=FES)
            if gr is not None and not gr.empty:
                want = {s54.age_term(a): a for a in s54.AGES}
                gr = gr[gr["term"].isin(want)].copy()
                gr["age_group"] = gr["term"].map(want)
                gr["baseline"], gr["outcome"] = base, label
                grads.append(gr)
                pd.concat(grads, ignore_index=True).to_csv(
                    OUT / "vintage_gradient.csv", index=False)
            del b
            gc.collect()
        del jobs
        gc.collect()

    # --- summary --------------------------------------------------------
    est = pd.DataFrame(rows)
    lines = ["BASELINE VINTAGE AND ATTENUATION", "=" * 52, "",
             "lambda(y): share of the 2019 exposure signal still present",
             piv[cols].round(3).to_string(), "",
             "Same design, two pre-treatment baselines:",
             (est.pivot_table(index="outcome", columns="baseline",
                              values="gamma").round(4).to_string()
              if not est.empty else "  none estimated"), ""]
    if not est.empty and {HEADLINE_BASE, LATE_BASE} <= set(est["baseline"]):
        w = est.pivot_table(index="outcome", columns="baseline",
                            values="gamma")
        for o in w.index:
            d = w.loc[o, LATE_BASE] - w.loc[o, HEADLINE_BASE]
            lines.append(f"  {o:<6} 2022 baseline moves the estimate "
                         f"{d:+.4f} against the 2019 one")
    lines += ["",
              "READ THIS BEFORE QUOTING ANY OF IT:",
              "  1. lambda is measured for years with their own occupation",
              "     register. For 2024 and 2025 it is EXTRAPOLATED, because",
              "     there is no register to measure it against. An",
              "     attenuation correction for those years rests on that",
              "     extrapolation and must say so.",
              "  2. The 2019 baseline stays the headline: it has the",
              "     longest clean pre-period. The 2022 baseline is the",
              "     check on attenuation, not a replacement, because it",
              "     sits after the April 2022 tightening and leaves only",
              "     one pre-treatment half-year.",
              "  3. If the two baselines agree, attenuation is not what is",
              "     driving the result. If the 2022 baseline finds more,",
              "     the 2019 estimate is an attenuated version of a real",
              "     effect and should be reported as a LOWER BOUND on its",
              "     magnitude, not as a null.",
              "  4. Dividing an estimate by lambda corrects the point",
              "     estimate and inflates its standard error by the same",
              "     factor. Do both or neither.",
              "", f"Runtime {(time.time()-t0)/60:.1f} min. " + mc.mem_line()]
    (OUT / "57_summary.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print("\n57 done.")


if __name__ == "__main__":
    main()
