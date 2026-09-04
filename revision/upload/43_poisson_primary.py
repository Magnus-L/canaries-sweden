#!/usr/bin/env python3
"""
43_poisson_primary.py -- T4/E6 (MONA run M4): the new primary battery.

======================================================================
  RUNS IN SCB's MONA SECURE ENVIRONMENT ONLY.
  Requires cache/panel_vintage.parquet (script 39 writes it) (script 39).
======================================================================

Poisson PML becomes the paper's main estimator (decided with Koch,
28 Aug). This script produces everything the switch needs:

  1. Poisson pooled DiD, all six age groups        (headline gamma2)
  2. Poisson half-year event study, all six groups (headline endpoint;
     pre-committed: the 2025H1 coefficient is the endpoint quoted)
  3. OLS+1 pooled DiD, all six groups              (bridge-table column;
     coefficients only, never converted to per cent)
  4. Extensive margin: LPM on P(n_emp = 0), same FE (demeaned OLS)
  5. Poisson pre-period joint test inputs: the ES CSV carries every
     pre-2022H1 coefficient and SE; the chi2 is computed locally

Sample: the Step-1 threshold (employers with >= 5 cumulative person-
months in the age group), balanced zero-filled panel, Q4-and-below
identification restriction -- the submitted paper's sample, unchanged.

Output (output_43/):
  poisson_pooled.csv        age_group x term
  poisson_es.csv            age_group x halfyear
  olsplus1_pooled.csv       age_group x term
  extensive_lpm.csv         age_group x term
  bridge_inputs.csv         side-by-side gamma2: OLS+1 vs Poisson
  43_summary.txt
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mona_common as mc

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_43"
OUT.mkdir(exist_ok=True)
CACHE = mc.PANEL_CACHE

STEP1_MIN_CUMULATIVE = 5


def build_age_panel(agg, all_months, age_label):
    sub = agg[agg["age_group"] == age_label]
    cum = sub.groupby("employer_id")["n_emp"].sum()
    sub = sub[sub["employer_id"].isin(cum[cum >= STEP1_MIN_CUMULATIVE].index)]
    bal = mc.add_treatment(mc.balance_panel(sub, all_months))
    bal["halfyear"] = mc.assign_halfyear(bal["year_month"])
    return bal


def olsplus1(bal):
    """Within-transformation OLS on ln(n+1); coefficients only."""
    import statsmodels.api as sm
    b = bal.copy()
    b["ln_emp"] = np.log(b["n_emp"] + 1)
    for col in ("ln_emp", "post_rb_x_high", "post_gpt_x_high"):
        b[f"{col}_d1"] = b.groupby("fe_emp_bin")[col].transform(
            lambda x: x - x.mean())
    for col in ("ln_emp", "post_rb_x_high", "post_gpt_x_high"):
        b[f"{col}_dm"] = b.groupby("fe_emp_t")[f"{col}_d1"].transform(
            lambda x: x - x.mean())
    res = sm.OLS(b["ln_emp_dm"].values,
                 b[["post_rb_x_high_dm", "post_gpt_x_high_dm"]].values
                 ).fit(cov_type="cluster",
                       cov_kwds={"groups": b["fe_emp_bin"].values})
    return {"gamma1": res.params[0], "se1": res.bse[0], "p1": res.pvalues[0],
            "gamma2": res.params[1], "se2": res.bse[1], "p2": res.pvalues[1],
            "n_obs": len(b)}


def extensive_lpm(bal):
    """LPM on the zero-employment indicator, same FE, same demeaning."""
    import statsmodels.api as sm
    b = bal.copy()
    b["zero"] = (b["n_emp"] == 0).astype(float)
    for col in ("zero", "post_rb_x_high", "post_gpt_x_high"):
        b[f"{col}_d1"] = b.groupby("fe_emp_bin")[col].transform(
            lambda x: x - x.mean())
    for col in ("zero", "post_rb_x_high", "post_gpt_x_high"):
        b[f"{col}_dm"] = b.groupby("fe_emp_t")[f"{col}_d1"].transform(
            lambda x: x - x.mean())
    res = sm.OLS(b["zero_dm"].values,
                 b[["post_rb_x_high_dm", "post_gpt_x_high_dm"]].values
                 ).fit(cov_type="cluster",
                       cov_kwds={"groups": b["fe_emp_bin"].values})
    return {"gamma1": res.params[0], "se1": res.bse[0], "p1": res.pvalues[0],
            "gamma2": res.params[1], "se2": res.bse[1], "p2": res.pvalues[1],
            "mean_zero": b["zero"].mean(), "n_obs": len(b)}


def main():
    mc.Tee(OUT / "43_log.txt")
    print("=" * 70)
    print("43: POISSON PRIMARY BATTERY (E6)")
    print("=" * 70)
    if not CACHE.exists():
        print("FATAL: run 39 first")
        sys.exit(1)

    panel = pd.read_parquet(CACHE)
    agg = mc.collapse_vintage(panel)
    agg = mc.merge_daioe_and_filter(agg, mc.load_daioe())
    agg = mc.aggregate_to_quartile(agg)
    all_months = sorted(agg["year_month"].unique())

    pooled_rows, es_frames, ols_rows, lpm_rows = [], [], [], []
    for age in mc.AGE_GROUPS:
        print(f"\n--- {age} ---")
        t0 = time.time()
        bal = build_age_panel(agg, all_months, age)
        print(f"  {len(bal):,} cells, {bal['employer_id'].nunique():,} "
              f"employers, {(bal['n_emp'] == 0).mean():.1%} zeros")

        # 1. Poisson pooled
        pres = mc.run_fepois(bal, OUT, tag=f"pool_{age}")
        for _, r in pres.iterrows():
            pooled_rows.append({"age_group": age, **r.to_dict()})

        # 2. Poisson ES
        eres = mc.run_fepois_es(bal, OUT, tag=f"es_{age}")
        if not eres.empty:
            eres["age_group"] = age
            es_frames.append(eres)

        # 3. OLS+1 bridge column
        ols_rows.append({"age_group": age, **olsplus1(bal)})

        # 4. extensive margin
        lpm_rows.append({"age_group": age, **extensive_lpm(bal)})
        print(f"  done in {(time.time()-t0)/60:.1f} min")

    pd.DataFrame(pooled_rows).to_csv(OUT / "poisson_pooled.csv", index=False)
    if es_frames:
        pd.concat(es_frames).to_csv(OUT / "poisson_es.csv", index=False)
    ols = pd.DataFrame(ols_rows)
    ols.to_csv(OUT / "olsplus1_pooled.csv", index=False)
    pd.DataFrame(lpm_rows).to_csv(OUT / "extensive_lpm.csv", index=False)

    # 5. bridge inputs
    poi = pd.DataFrame(pooled_rows)
    poi2 = (poi[poi["term"] == "post_gpt_x_high"]
            [["age_group", "coef", "se", "pvalue"]]
            .rename(columns={"coef": "poisson_g2", "se": "poisson_se",
                             "pvalue": "poisson_p"}))
    bridge = ols[["age_group", "gamma2", "se2", "p2"]].rename(
        columns={"gamma2": "olsplus1_g2", "se2": "olsplus1_se",
                 "p2": "olsplus1_p"}).merge(poi2, on="age_group")
    bridge["poisson_mult_pct"] = 100 * (np.exp(bridge["poisson_g2"]) - 1)
    bridge.to_csv(OUT / "bridge_inputs.csv", index=False)

    lines = ["POISSON PRIMARY -- SUMMARY", "=" * 40,
             bridge.to_string(index=False)]
    (OUT / "43_summary.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
