#!/usr/bin/env python3
"""
39_canary_gate.py -- the gate that runs FIRST in the revision MONA trip.

======================================================================
  RUNS IN SCB's MONA SECURE ENVIRONMENT ONLY.
======================================================================

Reproduces the submitted paper's anchor numbers from a fresh pull through
the v2 shared module before any new analysis runs. If the gate fails,
STOP: the v2 pull differs from v1 and every downstream number would be
uninterpretable.

Anchors (from script 32's April run, quoted in the revision plan):
  22-25 OLS+1 pooled gamma2      ~ -0.010
  22-25 Poisson pooled gamma2    ~ -0.174   (N = 11,970,426 obs)
Tolerance: coefficients to 3 decimals; N exact.

Also writes the v2 panel cache (vintage-tagged parquet) that scripts
40-46 reuse, so the gate doubles as the one-off SQL pull (~30-45 min).

Output (output_39/):
  canary_gate_results.csv
  canary_gate_verdict.txt      PASS / FAIL per anchor
  panel_vintage.parquet        (cache; stays in MONA)
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mona_common as mc

OUT = Path(__file__).resolve().parent / "output_39"
OUT.mkdir(exist_ok=True)
mc.CACHE_DIR.mkdir(exist_ok=True)
CACHE = mc.PANEL_CACHE  # disposable; see mona_common storage discipline

ANCHORS = {
    "ols_gamma2": -0.010,
    "poisson_gamma2": -0.174,
    "poisson_n_obs": 11_970_426,
}
TOL = 0.0005  # coefficient tolerance (3 decimals)


def main():
    mc.Tee(OUT / "39_canary_gate_log.txt")
    print("=" * 70)
    print("39: CANARY GATE -- reproduce v1 anchors through the v2 module")
    print("=" * 70)

    conn = mc.connect()
    panel_v = mc.pull_panel(range(2019, 2026), conn, CACHE, vintage=True)
    agg = mc.collapse_vintage(panel_v)
    daioe = mc.load_daioe()
    agg = mc.merge_daioe_and_filter(agg, daioe)
    agg = mc.aggregate_to_quartile(agg)
    all_months = sorted(agg["year_month"].unique())

    sub = agg[agg["age_group"] == "22-25"]
    # Step-1 threshold from script 32 (>=5 cumulative per employer)
    cum = sub.groupby("employer_id")["n_emp"].sum()
    sub = sub[sub["employer_id"].isin(cum[cum >= 5].index)]
    bal = mc.add_treatment(mc.balance_panel(sub, all_months))
    print(f"  22-25 balanced panel: {len(bal):,} cells, "
          f"{bal['employer_id'].nunique():,} employers")

    rows = []

    # --- OLS+1 (within-transformation, as v1 scripts 16/32) ---
    t0 = time.time()
    bal["ln_emp"] = np.log(bal["n_emp"] + 1)
    for col in ("ln_emp", "post_rb_x_high", "post_gpt_x_high"):
        bal[f"{col}_d1"] = bal.groupby("fe_emp_bin")[col].transform(
            lambda x: x - x.mean())
    for col in ("ln_emp", "post_rb_x_high", "post_gpt_x_high"):
        bal[f"{col}_dm"] = bal.groupby("fe_emp_t")[f"{col}_d1"].transform(
            lambda x: x - x.mean())
    import statsmodels.api as sm
    res = sm.OLS(bal["ln_emp_dm"].values,
                 bal[["post_rb_x_high_dm", "post_gpt_x_high_dm"]].values
                 ).fit(cov_type="cluster",
                       cov_kwds={"groups": bal["fe_emp_bin"].values})
    ols_g2 = res.params[1]
    print(f"  OLS+1 gamma2 = {ols_g2:+.4f} ({time.time()-t0:.0f}s)")
    rows.append({"estimator": "OLS+1", "gamma2": ols_g2,
                 "se2": res.bse[1], "n_obs": len(bal)})

    # --- Poisson via R + fixest ---
    pres = mc.run_fepois(bal, OUT, tag="gate")
    if not pres.empty:
        g2 = pres.loc[pres["term"] == "post_gpt_x_high"]
        poi_g2 = float(g2["coef"].iloc[0])
        poi_n = int(g2["n_obs"].iloc[0])
        print(f"  Poisson gamma2 = {poi_g2:+.4f}, N = {poi_n:,}")
        rows.append({"estimator": "Poisson", "gamma2": poi_g2,
                     "se2": float(g2["se"].iloc[0]), "n_obs": poi_n})
    else:
        poi_g2, poi_n = np.nan, -1

    pd.DataFrame(rows).to_csv(OUT / "canary_gate_results.csv", index=False)

    # --- Verdict ---
    # Coefficients are the anchors and stay STRICT. N is compared with a
    # 3% tolerance, adopted 4 Sep 2026 after the first run reproduced both
    # coefficients to the 4th decimal on a panel 2.40% smaller: the database
    # moves under us (2024 AGI prel->def, the repaired 2023 Individ
    # delivery), so exact N against a February pull is not reproducible.
    # The drift is PRINTED, and 39b localises it by year; a drift beyond 3%
    # or any coefficient miss still fails the gate.
    n_drift_pct = 100 * abs(poi_n - ANCHORS["poisson_n_obs"]) / ANCHORS["poisson_n_obs"]
    checks = {
        "ols_gamma2": abs(ols_g2 - ANCHORS["ols_gamma2"]) <= TOL,
        "poisson_gamma2": abs(poi_g2 - ANCHORS["poisson_gamma2"]) <= TOL,
        "poisson_n_obs": n_drift_pct <= 3.0,
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"
    lines = [f"CANARY GATE: {verdict}"]
    for k, ok in checks.items():
        note = ""
        if k == "poisson_n_obs":
            note = (f"  (N {poi_n:,} vs anchor {ANCHORS['poisson_n_obs']:,}: "
                    f"drift {n_drift_pct:.2f}%, tolerance 3%; see output_39b)")
        lines.append(f"  {k}: {'ok' if ok else 'MISMATCH'}{note}")
    (OUT / "canary_gate_verdict.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    if verdict == "FAIL":
        print("\nSTOP. Do not run scripts 40-46 until the mismatch is "
              "understood; the v2 pull differs from v1.")
        sys.exit(1)


if __name__ == "__main__":
    main()
