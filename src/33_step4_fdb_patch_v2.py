#!/usr/bin/env python3
"""
33_step4_fdb_patch_v2.py -- standalone Step 4 with FDB fix AND vectorized aggregation.

WHY V2 EXISTS
=============
v1 (33_step4_fdb_patch.py) inherited script 32's `step4_poisson_reweighted`
function. That function contains a `groupby(...).apply(lambda g: ...)`
on the cell-level panel which becomes catastrophically slow at full
panel size (millions of groups, Python-level callback per group). That
code path was never exercised in the previous run (Step 4 aborted at
the FDB SQL error), so the bottleneck was discovered only when v1
finally got past the SQL.

v2 monkey-patches BOTH the FDB pull AND `step4_poisson_reweighted`
with a fully-vectorized version that produces identical numerical
output but runs in seconds, not hours.

Vectorization trick (the only change to the math): the original code
computes a per-cell weighted mean of `w` weighted by `n_emp` via
groupby-apply. Equivalent vectorized pattern:
  grain[w_x_n] = grain[w] * grain[n_emp]
  cell = grain.groupby(...).agg(n_emp=sum, w_x_n=sum)
  cell[w] = cell[w_x_n] / cell[n_emp]

This collapses two summations and a division -- pure numpy, no Python
callback per group.

USAGE IN MONA
=============
1. Upload as 33_step4_fdb_patch_v2.txt; rename to .py.
2. Required co-located: 32_mona_kauhanen_robustness.py
3. Required cache: output_32/step0_panel_cache.csv (already present
   from yesterday's run).
4. Required share file: finland_marginals_2022.txt at FINLAND_MARGINALS_PATH.
5. Run:
       python 33_step4_fdb_patch_v2.py

Expected runtime: 5-10 min total. The FDB pull is ~1 min; the
vectorized aggregation is seconds; six Poisson regressions via R
subprocess take 30-60s each.

If you have a v1 run still pending: kill the Batch job before starting
v2 (otherwise the two will write to the same output_32/ files).

EXPORT-SAFETY
=============
Identical to v1. All outputs are aggregated counts and regression
coefficients; cell-count safeguards already inside script 32.
"""

import importlib.util
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pyodbc

SCRIPT_DIR = Path(__file__).resolve().parent
KAUHANEN_PY = SCRIPT_DIR / "32_mona_kauhanen_robustness.py"

if not KAUHANEN_PY.exists():
    print(f"FATAL: 32_mona_kauhanen_robustness.py not found at {KAUHANEN_PY}")
    sys.exit(1)

_spec = importlib.util.spec_from_file_location("kauhanen", str(KAUHANEN_PY))
kauhanen = importlib.util.module_from_spec(_spec)
sys.modules["kauhanen"] = kauhanen
_spec.loader.exec_module(kauhanen)


# ----------------------------------------------------------------------
# Patch 1: FDB pull (same as v1)
# ----------------------------------------------------------------------

def _pull_employer_nace_fdb_je(conn):
    """
    Pull (employer_id, NACE-2) from FDB_JE_2014_2021 + FDB_JE_2022_2024.
    Most-recent year per employer; robust to dot/space/padding in ng1.
    """
    print("  [v2] Pulling employer -> NACE-2 from FDB_JE...")
    t0 = time.time()
    query = """
        SELECT
            P1207_Lopnr_peorgnr AS employer_id,
            ng1 AS ng1_raw,
            CAST(ar AS INT) AS year
        FROM dbo.FDB_JE_2022_2024
        WHERE ng1 IS NOT NULL

        UNION ALL

        SELECT
            P1207_Lopnr_peorgnr AS employer_id,
            ng1 AS ng1_raw,
            CAST(ar AS INT) AS year
        FROM dbo.FDB_JE_2014_2021
        WHERE ng1 IS NOT NULL AND ar >= '2018'
    """
    df = pd.read_sql(query, conn)
    print(f"  [v2] Pulled {len(df):,} firm-year rows in {time.time()-t0:.0f}s")

    df = df.sort_values(["employer_id", "year"], ascending=[True, False])
    df = df.drop_duplicates(subset=["employer_id"], keep="first")

    df["nace2"] = (
        df["ng1_raw"].astype(str)
        .str.replace(r"[^0-9]", "", regex=True)
        .str[:2]
        .str.zfill(2)
    )
    bad = ~df["nace2"].str.match(r"^[0-9]{2}$")
    if int(bad.sum()):
        df = df[~bad]

    # Drop nace2 == '00' (NACE has no division 00). These cells originate
    # from non-numeric or empty ng1 raw values and have no Swedish/Finnish
    # marginal, so they contribute zero-weight noise downstream.
    n_zero = int((df["nace2"] == "00").sum())
    if n_zero:
        print(f"  [v2] Dropping {n_zero:,} employers with nace2 == '00' "
              f"(empty/non-numeric ng1)")
        df = df[df["nace2"] != "00"]

    print(f"  [v2] Retrieved NACE-2 for {len(df):,} employers")
    return df[["employer_id", "nace2"]]


# ----------------------------------------------------------------------
# Patch 2: vectorized step4_poisson_reweighted
# ----------------------------------------------------------------------

def step4_poisson_reweighted_v2(panel_ssyk_age, daioe):
    """
    Drop-in replacement for kauhanen.step4_poisson_reweighted.

    Same math as the original:
      - Per-worker weight w = (p_fin / p_swe)_isco1 * (p_fin / p_swe)_nace2
      - Cell-level weight = mean of worker-level weights, weighted by n_emp
      - Poisson PML on (employer x quartile x age x ym) with cell weight

    Vectorization difference: the cell-level weighted mean is computed via
    two summations and a division (numpy-vectorized) rather than a Python
    callback per group. Output is numerically identical.
    """
    print("\n" + "=" * 70)
    print("STEP 4 v2: Poisson PML, reweighted to Finnish composition (vectorized)")
    print("=" * 70)
    t_start = time.time()

    finland = kauhanen._load_finland_marginals()
    if finland is None:
        return pd.DataFrame()

    # Pull NACE-2 (uses the v2 monkey-patched function)
    try:
        conn = pyodbc.connect(kauhanen.SQL_CONN_STRING)
        nace = kauhanen._pull_employer_nace(conn)
        conn.close()
    except BaseException as e:
        print(f"  [v2] NACE pull failed: {e}")
        return pd.DataFrame()

    # SSYK-4 -> ISCO-1
    print(f"  [v2] Building ISCO-1 from SSYK-4...")
    t0 = time.time()
    panel = panel_ssyk_age  # no .copy() -- we mutate in place via assignment
    panel = panel.assign(isco1=panel["ssyk4"].astype(str).str[0])
    print(f"    done in {time.time()-t0:.1f}s; rows={len(panel):,}")

    # Merge NACE-2
    print(f"  [v2] Merging NACE-2 to panel...")
    t0 = time.time()
    panel = panel.merge(nace, on="employer_id", how="left")
    n_pre = len(panel)
    panel = panel.dropna(subset=["nace2"])
    print(f"    done in {time.time()-t0:.1f}s; "
          f"{n_pre - len(panel):,} rows dropped for missing NACE-2; "
          f"rows={len(panel):,}")

    # Swedish marginals from the panel (population-weighted by n_emp)
    print(f"  [v2] Computing Swedish ISCO-1 + NACE-2 marginals...")
    t0 = time.time()
    swe_isco = (
        panel.groupby("isco1")["n_emp"].sum()
        .pipe(lambda s: s / s.sum())
        .to_dict()
    )
    swe_nace = (
        panel.groupby("nace2")["n_emp"].sum()
        .pipe(lambda s: s / s.sum())
        .to_dict()
    )
    print(f"    done in {time.time()-t0:.1f}s; "
          f"|ISCO1|={len(swe_isco)}, |NACE2|={len(swe_nace)}")

    # Compute per-cell weight w = (p_fin / p_swe)_occ * (p_fin / p_swe)_ind
    print(f"  [v2] Computing per-cell weights...")
    t0 = time.time()

    def _w_occ(c):
        pf = finland["ISCO1"].get(c, 0.0)
        ps = swe_isco.get(c, 0.0)
        return pf / ps if ps > 0 else 0.0

    def _w_nace(c):
        pf = finland["NACE2"].get(c, 0.0)
        ps = swe_nace.get(c, 0.0)
        return pf / ps if ps > 0 else 0.0

    panel["w_occ"] = panel["isco1"].map(_w_occ).astype(float)
    panel["w_nace"] = panel["nace2"].map(_w_nace).astype(float)
    panel["w"] = panel["w_occ"] * panel["w_nace"]
    raw_w = panel["w"]
    print(f"    done in {time.time()-t0:.1f}s; "
          f"raw w: min={raw_w.min():.3f}, "
          f"p99={raw_w.quantile(0.99):.3f}, "
          f"max={raw_w.max():.3f}, "
          f"mean={raw_w.mean():.3f}, "
          f"median={raw_w.median():.3f}")

    # Winsorize the weights to prevent single-cell tail-domination of the
    # weighted Poisson. Cap at min(99th percentile, 10*median). The
    # rationale is standard for inverse-probability weighting under
    # finite-sample marginals: extreme weights inflate variance and can
    # pull point estimates toward a handful of cells. The cap preserves
    # the reweighting direction while stabilising estimation.
    cap_p99 = float(raw_w.quantile(0.99))
    cap_10x_median = 10.0 * float(raw_w.median())
    weight_cap = min(cap_p99, cap_10x_median)
    n_capped = int((panel["w"] > weight_cap).sum())
    panel["w"] = panel["w"].clip(upper=weight_cap)
    print(f"    Winsorized at {weight_cap:.3f} "
          f"(min(p99={cap_p99:.3f}, 10*median={cap_10x_median:.3f})); "
          f"{n_capped:,} cells capped")
    print(f"    Total reweighted person-months (capped): "
          f"{(panel['n_emp']*panel['w']).sum():,.0f}")
    print(f"    Total raw person-months:                 {panel['n_emp'].sum():,.0f}")

    # Merge DAIOE quartiles
    print(f"  [v2] Merging DAIOE quartiles...")
    t0 = time.time()
    panel = panel.merge(daioe, on="ssyk4", how="inner")
    print(f"    done in {time.time()-t0:.1f}s; rows={len(panel):,}")

    # First aggregation: collapse over SSYK4 within
    # (employer, quartile, age, ym, isco1, nace2). 'w' is constant within
    # this finer group because isco1 and nace2 are functions of ssyk4 and
    # employer respectively, so 'first' is correct here (matches original).
    print(f"  [v2] First aggregation: (employer, q, age, ym, isco1, nace2)...")
    t0 = time.time()
    grain = (
        panel.groupby(
            ["employer_id", "daioe_quartile", "age_group",
             "year_month", "isco1", "nace2"],
            sort=False, observed=True,
        )
        .agg(n_emp=("n_emp", "sum"), w=("w", "first"))
        .reset_index()
    )
    print(f"    done in {time.time()-t0:.1f}s; grain rows={len(grain):,}")

    # Second aggregation: collapse to (employer, quartile, age, ym).
    # VECTORIZED REPLACEMENT for the slow .apply() in the original:
    #   original: .apply(lambda g: dict(n_emp=sum, w=weighted_mean(w, n_emp)))
    #   v2:       precompute w*n_emp; sum both; divide.
    print(f"  [v2] Second aggregation (vectorized): (employer, q, age, ym)...")
    t0 = time.time()
    grain["w_x_n"] = grain["w"] * grain["n_emp"]
    cell = (
        grain.groupby(
            ["employer_id", "daioe_quartile", "age_group", "year_month"],
            sort=False, observed=True, as_index=False,
        )
        .agg(n_emp=("n_emp", "sum"), w_x_n=("w_x_n", "sum"))
    )
    # Weighted mean; original returned 1.0 when n_emp == 0
    cell["w"] = np.where(cell["n_emp"] > 0,
                         cell["w_x_n"] / cell["n_emp"],
                         1.0)
    cell = cell.drop(columns=["w_x_n"])
    print(f"    done in {time.time()-t0:.1f}s; cell rows={len(cell):,}")
    print(f"    cell w summary: min={cell['w'].min():.3f}, "
          f"max={cell['w'].max():.3f}, mean={cell['w'].mean():.3f}")

    # Free grain memory before per-age-group estimation
    del grain, panel
    import gc
    gc.collect()

    # Per-age-group Poisson estimation (uses script 32's helpers; fast)
    results = []
    for age_label in kauhanen.AGE_GROUPS:
        print(f"\n  Age group: {age_label}")
        sub = kauhanen.filter_step1(cell, age_label)
        if len(sub) == 0:
            print(f"    No employers; skip")
            continue
        balanced = kauhanen.build_balanced_panel(sub, "daioe_quartile", n_bins=4)
        if len(balanced) == 0:
            print(f"    No employers span Q4 and Q1-Q3; skip")
            continue

        weights_lookup = sub[
            ["employer_id", "daioe_quartile", "year_month", "w"]
        ].drop_duplicates(
            subset=["employer_id", "daioe_quartile", "year_month"]
        )
        balanced = balanced.merge(
            weights_lookup,
            on=["employer_id", "daioe_quartile", "year_month"],
            how="left",
        )
        balanced["w"] = balanced["w"].fillna(1.0)

        result = kauhanen._estimate_poisson_weighted(
            balanced, "daioe_quartile", n_bins=4,
            age_label=age_label, spec_label="STEP4_reweighted",
        )
        if result is not None:
            results.append(result)
            pd.DataFrame(results).to_csv(
                kauhanen.OUTPUT_DIR / "step4_poisson_reweighted.csv", index=False
            )

    print(f"\n  [v2] Step 4 total elapsed: {time.time()-t_start:.0f}s")
    return pd.DataFrame(results)


# ----------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------

def _maybe_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main():
    print("=" * 70)
    print("Step 4 patch runner v2 -- FDB fix + vectorized step4")
    print("=" * 70)
    print(f"Script 32 module: {KAUHANEN_PY}")
    print(f"Output dir:       {kauhanen.OUTPUT_DIR}")

    # Apply both monkey-patches
    kauhanen._pull_employer_nace = _pull_employer_nace_fdb_je
    kauhanen.step4_poisson_reweighted = step4_poisson_reweighted_v2
    print("  [v2] Patched _pull_employer_nace -> _pull_employer_nace_fdb_je")
    print("  [v2] Patched step4_poisson_reweighted -> step4_poisson_reweighted_v2")

    # Load cache + DAIOE
    panel_ssyk_age = kauhanen.step0_pull_or_load_panel()
    daioe = kauhanen.load_daioe_quartiles()

    # Run Step 4 (now vectorized)
    step4 = kauhanen.step4_poisson_reweighted(panel_ssyk_age, daioe)

    if step4 is None or len(step4) == 0:
        print("\nStep 4 returned no results. Most likely causes:")
        print("  (a) finland_marginals_2022.txt not found on the share")
        print("  (b) FDB pull empty after dedup")
        print("  (c) Reweighted panel had insufficient employers per age group")
        print("Check the log lines above to identify which.")
        return

    # Re-load Steps 1, 2, 3, 5 to rebuild comparison and prose summary
    out_dir = kauhanen.OUTPUT_DIR
    step1 = _maybe_csv(out_dir / "step1_poisson_current.csv")
    step2 = _maybe_csv(out_dir / "step2_poisson_threshold.csv")
    step3 = _maybe_csv(out_dir / "step3_poisson_kauhanen.csv")
    step5 = _maybe_csv(out_dir / "step5_poisson_no_ict.csv")

    kauhanen.write_comparison_table(step1, step2, step3, step4, step5)
    kauhanen.write_prose_summary(step1, step2, step3, step4, step5)

    print("\nDone. Updated outputs in output_32/:")
    print("  step4_poisson_reweighted.csv  (new)")
    print("  kauhanen_comparison.csv       (rewritten, now 5 steps)")
    print("  kauhanen_summary.txt          (rewritten, now 5 steps)")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        import traceback
        print("\nFATAL: unhandled exception")
        traceback.print_exc()
        sys.exit(2)
