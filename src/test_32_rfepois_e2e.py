#!/usr/bin/env python3
"""
test_32_rfepois_e2e.py -- End-to-end test for the R + fixest::fepois
subprocess wrapper introduced after pyfixest could not be installed in
MONA (SCB support email 2026-04-28).

What it tests:
  - src/r_fepois.R is callable via Rscript
  - estimate_poisson() correctly serialises a cell-level panel,
    invokes R, and parses results back
  - The recovered Poisson coefficients are within tolerance of the
    known DGP coefficients (recovery test, not a unit-equality test)

Skipped if Rscript or fixest are not available on the local machine.
This test exists so we have local confidence the pipeline works before
shipping it to MONA.

Run:
    python3 src/test_32_rfepois_e2e.py
"""

import sys
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Skip if R or fixest unavailable
# ----------------------------------------------------------------------

if shutil.which("Rscript") is None:
    print("SKIP: Rscript not on PATH. Install R + fixest to run this test.")
    sys.exit(0)

proc = subprocess.run(
    ["Rscript", "-e",
     'suppressMessages(library(fixest)); '
     'cat(as.character(packageVersion("fixest")))'],
    capture_output=True, text=True, timeout=60,
)
if proc.returncode != 0:
    print("SKIP: fixest not installed in R. "
          "Install with: Rscript -e 'install.packages(\"fixest\")'")
    sys.exit(0)

print(f"R + fixest version: {proc.stdout.strip()}")
print()

# ----------------------------------------------------------------------
# Import the module (will run pre-flight check; safe since R is present)
# ----------------------------------------------------------------------

# Stub pyodbc so the SQL connection setup doesn't blow up
import types
pyodbc_stub = types.ModuleType("pyodbc")
pyodbc_stub.connect = lambda *a, **kw: None
sys.modules["pyodbc"] = pyodbc_stub

import importlib.util
spec = importlib.util.spec_from_file_location(
    "kauhanen", Path(__file__).parent / "32_mona_kauhanen_robustness.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
estimate_poisson = mod.estimate_poisson
_run_rfepois = mod._run_rfepois

# ----------------------------------------------------------------------
# Synthetic panel with known DGP
# ----------------------------------------------------------------------

def make_balanced_panel(seed=42):
    """
    Generate a balanced (employer x quartile x month) panel with a known
    Poisson DGP:

        log(E[n_emp]) = alpha_emp_bin + alpha_emp_t
                       + 0.05 * post_rb_x_high
                       + 0.03 * post_gpt_x_high

    Each employer spans Q1 and Q4 (satisfies the identification
    restriction). Cell counts are ~Poisson(mu) with mu around 10-50,
    large enough that fepois converges easily.

    Treatment dates match the script: RIKSBANK_YM=2022-04, CHATGPT_YM=2022-12.
    """
    rng = np.random.default_rng(seed)

    n_employers = 80
    months = pd.period_range("2021-01", "2023-12", freq="M").strftime("%Y-%m").tolist()
    quartiles = [1, 2, 3, 4]

    rows = []
    for emp_idx in range(n_employers):
        emp_id = f"emp{emp_idx:03d}"
        # Each employer-bin pair has a baseline log-mean; vary across employers
        alpha_emp = rng.normal(loc=2.5, scale=0.3)  # ~exp(2.5)=12 baseline
        for q in quartiles:
            # Bin shifts intensity slightly (Q4 employers a bit larger here)
            alpha_bin = 0.1 * (q - 1)
            for m in months:
                # Month FE: smooth seasonality
                month_idx = months.index(m)
                alpha_t = 0.02 * (month_idx % 12 - 6)
                # Treatment effects (true coefficients)
                post_rb = int(m >= "2022-04")
                post_gpt = int(m >= "2022-12")
                high = int(q == 4)
                gamma1_true = 0.05
                gamma2_true = 0.03
                log_mu = (alpha_emp + alpha_bin + alpha_t
                          + gamma1_true * post_rb * high
                          + gamma2_true * post_gpt * high)
                mu = np.exp(log_mu)
                n_emp = rng.poisson(mu)
                rows.append((emp_id, q, m, n_emp))

    df = pd.DataFrame(
        rows, columns=["employer_id", "daioe_quartile", "year_month", "n_emp"]
    )
    return df


# ----------------------------------------------------------------------
# Run the test
# ----------------------------------------------------------------------

print("=" * 70)
print("E2E TEST: estimate_poisson() via R + fixest::fepois subprocess")
print("=" * 70)

panel = make_balanced_panel(seed=42)
print(f"\nGenerated panel: {len(panel):,} rows, "
      f"{panel['employer_id'].nunique()} employers, "
      f"{panel['year_month'].nunique()} months, "
      f"{panel['n_emp'].sum():,} total person-months")
print(f"True coefficients: gamma1 (PostRB x High) = +0.0500, "
      f"gamma2 (PostGPT x High) = +0.0300")

print("\nCalling estimate_poisson...")
result = estimate_poisson(
    panel, bin_col="daioe_quartile", n_bins=4,
    age_label="22-25", spec_label="E2E_TEST",
)

print()
if result is None:
    print("FAIL: estimate_poisson returned None")
    sys.exit(1)

# ----------------------------------------------------------------------
# Assertions
# ----------------------------------------------------------------------

n_fail = 0

def check(label, cond, detail=""):
    global n_fail
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {label}{(' -- ' + detail) if detail else ''}")
    if not cond:
        n_fail += 1

# 1. Result has the expected keys
required_keys = {
    "spec", "age_group", "gamma1", "se1", "p1",
    "gamma2", "se2", "p2", "n_obs", "n_emp_total",
    "n_employers", "estimator", "fe", "vcov", "elapsed_s",
}
check(
    "result dict contains all required keys",
    required_keys.issubset(result.keys()),
    detail=f"missing: {required_keys - set(result.keys())}",
)

# 2. Estimator string mentions R + fixest
check(
    "estimator string mentions R + fixest",
    "fixest" in result["estimator"] and "R " in result["estimator"],
    detail=f"got: {result['estimator']!r}",
)

# 3. Coefficient recovery (true values: 0.05, 0.03)
g1 = result["gamma1"]
g2 = result["gamma2"]
# Allow generous tolerance: large panel, but Poisson noise makes
# point estimates wander. 4 SE band is generous.
g1_in_band = abs(g1 - 0.05) < 4 * result["se1"]
g2_in_band = abs(g2 - 0.03) < 4 * result["se2"]
check(
    f"gamma1={g1:+.4f} (SE={result['se1']:.4f}) within 4 SE of true 0.0500",
    g1_in_band,
)
check(
    f"gamma2={g2:+.4f} (SE={result['se2']:.4f}) within 4 SE of true 0.0300",
    g2_in_band,
)

# 4. Standard errors are positive and finite
check(
    "se1 finite and positive",
    np.isfinite(result["se1"]) and result["se1"] > 0,
    detail=f"se1={result['se1']}",
)
check(
    "se2 finite and positive",
    np.isfinite(result["se2"]) and result["se2"] > 0,
    detail=f"se2={result['se2']}",
)

# 5. p-values in [0, 1]
check(
    "p1 in [0, 1]",
    0.0 <= result["p1"] <= 1.0,
    detail=f"p1={result['p1']}",
)
check(
    "p2 in [0, 1]",
    0.0 <= result["p2"] <= 1.0,
    detail=f"p2={result['p2']}",
)

# 6. n_obs matches input (after high/post construction the panel keeps all rows)
check(
    "n_obs matches input panel size",
    result["n_obs"] == len(panel),
    detail=f"got {result['n_obs']}, expected {len(panel)}",
)

# 7. n_emp_total matches sum of n_emp
check(
    "n_emp_total matches input sum",
    result["n_emp_total"] == int(panel["n_emp"].sum()),
    detail=f"got {result['n_emp_total']}, expected {int(panel['n_emp'].sum())}",
)

# ----------------------------------------------------------------------
# Weighted variant
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print("E2E TEST: _estimate_poisson_weighted (cell-level w=1 -> equivalent)")
print("=" * 70)

panel_w = panel.copy()
panel_w["w"] = 1.0  # All-ones weights -> result should equal unweighted

# Build the full balanced panel with treatment vars (mirroring the script)
# and pass it to the helper directly to test the weighted path.
panel_w_full = panel_w.copy()
panel_w_full["high"] = (panel_w_full["daioe_quartile"] == 4).astype(int)
panel_w_full["post_rb"] = (panel_w_full["year_month"] >= mod.RIKSBANK_YM).astype(int)
panel_w_full["post_gpt"] = (panel_w_full["year_month"] >= mod.CHATGPT_YM).astype(int)
panel_w_full["post_rb_x_high"] = panel_w_full["post_rb"] * panel_w_full["high"]
panel_w_full["post_gpt_x_high"] = panel_w_full["post_gpt"] * panel_w_full["high"]
panel_w_full["fe_emp_bin"] = (
    panel_w_full["employer_id"].astype(str) + "_"
    + panel_w_full["daioe_quartile"].astype(str)
)
panel_w_full["fe_emp_t"] = (
    panel_w_full["employer_id"].astype(str) + "_" + panel_w_full["year_month"]
)

res_w = _run_rfepois(panel_w_full, weighted=True,
                     age_label="22-25", spec_label="E2E_W_TEST")

if res_w is None:
    check("weighted call returned a dict", False)
else:
    # With w=1 everywhere, coefficients should match unweighted within
    # numerical tolerance (fepois may not produce literally identical
    # values across weighted/unweighted internal paths).
    check(
        f"weighted gamma1={res_w['gamma1']:+.4f} close to "
        f"unweighted {g1:+.4f}",
        abs(res_w["gamma1"] - g1) < 0.01,
    )
    check(
        f"weighted gamma2={res_w['gamma2']:+.4f} close to "
        f"unweighted {g2:+.4f}",
        abs(res_w["gamma2"] - g2) < 0.01,
    )

# ----------------------------------------------------------------------
# Failure-mode test: deliberately bad input
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print("FAILURE-MODE TEST: empty panel -> graceful None return")
print("=" * 70)

empty_panel = pd.DataFrame(columns=[
    "n_emp", "post_rb_x_high", "post_gpt_x_high",
    "fe_emp_bin", "fe_emp_t", "employer_id"
])
res_empty = _run_rfepois(empty_panel, weighted=False,
                         age_label="empty", spec_label="EMPTY_TEST")
check(
    "empty panel returns None (does not crash)",
    res_empty is None,
)

# ----------------------------------------------------------------------
# Final report
# ----------------------------------------------------------------------

print()
print("=" * 70)
if n_fail == 0:
    print(f"ALL E2E TESTS PASS -- R + fixest subprocess pipeline is sound.")
    sys.exit(0)
else:
    print(f"FAILED: {n_fail} test(s) -- see above.")
    sys.exit(1)
