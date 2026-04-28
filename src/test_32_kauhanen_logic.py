#!/usr/bin/env python3
"""
test_32_kauhanen_logic.py -- Local unit tests for script 32's panel logic.

Tests the data-manipulation pieces of 32_mona_kauhanen_robustness.py that
DON'T need MONA SQL or pyfixest -- threshold filters, balanced panel
construction, treatment variable wiring. Run this BEFORE uploading
to MONA. If any test fails, fix it locally first; round-trips to MONA
cost ~15 min each.

Run:
    python3 src/test_32_kauhanen_logic.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make pyfixest optional for the import
sys.modules.setdefault("pyfixest", type(sys)("pyfixest"))
sys.modules.setdefault("pyodbc", type(sys)("pyodbc"))

# Skip the _Tee logger setup (it tries to write to a path that may not be
# writable in the test environment); patch with a no-op instead
_orig_argv = sys.argv

# Import the functions we want to test
sys.path.insert(0, str(Path(__file__).parent))

# To avoid running the script's top-level _Tee/pyodbc/pyfixest code, we
# load the module by file path with side effects suppressed by replacing
# the relevant imports with stubs in sys.modules. The functions are pure.

import importlib.util
import builtins

# Pre-stub pyodbc.connect to avoid import-time call attempts
import types
pyodbc_stub = types.ModuleType("pyodbc")
pyodbc_stub.connect = lambda *a, **kw: None
sys.modules["pyodbc"] = pyodbc_stub

# pyfixest stub kept for backward-compat (no longer imported by script 32,
# which now calls R + fixest::fepois via subprocess). Kept defensively in
# case future edits reintroduce pyfixest imports.
pyfixest_stub = types.ModuleType("pyfixest")
pyfixest_stub.fepois = lambda *a, **kw: None
sys.modules["pyfixest"] = pyfixest_stub

# Mock subprocess.run so the script 32 R+fixest pre-flight passes even on
# machines without Rscript installed. We only need the data-prep logic
# tested; the R subprocess is exercised separately via end-to-end tests
# that require local R + fixest.
import subprocess as _subprocess
_orig_run = _subprocess.run
class _FakeProc:
    def __init__(self):
        self.returncode = 0
        self.stdout = "0.13.2\n"
        self.stderr = ""
def _fake_run(cmd, *a, **kw):
    # Only intercept the pre-flight Rscript call. For any other use, fall
    # back to the real subprocess.run so future tests can call it freely.
    # Match by basename so that absolute paths returned by Rscript
    # auto-discovery (e.g. /opt/homebrew/bin/Rscript or
    # E:\Programs\R-4.5.3\bin\x64\Rscript.exe) are also intercepted.
    if isinstance(cmd, list) and len(cmd) > 0:
        first = str(cmd[0])
        base = os.path.basename(first).lower()
        if base.startswith("rscript"):
            return _FakeProc()
    return _orig_run(cmd, *a, **kw)
import os  # for basename
_subprocess.run = _fake_run

spec = importlib.util.spec_from_file_location(
    "kauhanen", Path(__file__).parent / "32_mona_kauhanen_robustness.py"
)
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except SystemExit:
    print("(SystemExit caught at module load -- expected if pyfixest stub "
          "triggered the pre-flight check; testing functions individually.)")

# Pull functions
filter_step1 = mod.filter_step1
filter_kauhanen = mod.filter_kauhanen
build_balanced_panel = mod.build_balanced_panel
RIKSBANK_YM = mod.RIKSBANK_YM
CHATGPT_YM = mod.CHATGPT_YM
KAUHANEN_MIN_MEAN_PER_MONTH = mod.KAUHANEN_MIN_MEAN_PER_MONTH
KAUHANEN_MIN_CUMULATIVE = mod.KAUHANEN_MIN_CUMULATIVE
STEP1_MIN_CUMULATIVE = mod.STEP1_MIN_CUMULATIVE


# ======================================================================
#   SYNTHETIC PANEL FIXTURE
# ======================================================================

def make_synthetic_panel():
    """
    Build a small synthetic (employer, quartile, age, month) panel with
    deliberate edge cases:

      Employer 'big_q4_only'  -- 50 workers/month, only in Q4 -> dropped by
                                  identification restriction (no Q1-Q3).
      Employer 'small'        -- 1 worker total -> dropped by all thresholds.
      Employer 'mid'          -- 12 workers/month, mostly Q4 plus some Q2.
                                  Cumulative ~150. Passes all thresholds.
      Employer 'kauhanen_fail'-- 8 workers/month (mean), spans Q4 and Q1.
                                  Cumulative ~80. Passes Step 1 (>=5) but
                                  FAILS Kauhanen (mean<10 AND cum<100).
      Employer 'edge'         -- 10 workers/month, spans Q4 and Q3.
                                  Cumulative exactly 100 -> right at boundary.
    """
    months = ["2021-06", "2021-07", "2021-08", "2021-09", "2021-10",
              "2022-01", "2022-04", "2022-07", "2022-12", "2023-06"]
    rows = []

    # big_q4_only: only Q4 -> drop by identification restriction
    for m in months:
        rows.append(("big_q4_only", 4, "22-25", m, 50))

    # small: 1 worker total -> drop by all thresholds
    rows.append(("small", 2, "22-25", "2021-06", 1))

    # mid: 12 workers/month, Q4 + Q2 -> passes Kauhanen
    for m in months:
        rows.append(("mid", 4, "22-25", m, 10))
        rows.append(("mid", 2, "22-25", m, 5))

    # kauhanen_fail: 8/month, Q4 + Q1, cum 80 -> Step 1 yes, Kauhanen no
    for m in months:
        rows.append(("kauhanen_fail", 4, "22-25", m, 5))
        rows.append(("kauhanen_fail", 1, "22-25", m, 3))

    # edge: 10/month, Q4 + Q3, cum 100 -> on Kauhanen boundary
    for m in months:
        rows.append(("edge", 4, "22-25", m, 7))
        rows.append(("edge", 3, "22-25", m, 3))

    # sparse: Q4 only in first half of period, Q1 only in second half.
    # Tests that the cross-join + left-join correctly zero-fills the
    # missing (employer, bin, month) cells. 10 expected zero cells.
    for m in months[:5]:
        rows.append(("sparse", 4, "22-25", m, 6))
    for m in months[5:]:
        rows.append(("sparse", 1, "22-25", m, 6))

    df = pd.DataFrame(
        rows,
        columns=["employer_id", "daioe_quartile", "age_group", "year_month", "n_emp"]
    )
    return df


# ======================================================================
#   TESTS
# ======================================================================

def _ok(label, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {label}{(' -- ' + detail) if detail else ''}")
    if not cond:
        global _N_FAIL
        _N_FAIL += 1


_N_FAIL = 0


def test_filter_step1():
    print("\n--- test_filter_step1 ---")
    df = make_synthetic_panel()
    out = filter_step1(df, "22-25")
    employers = set(out["employer_id"].unique())
    _ok("'small' (cum=1) dropped",
        "small" not in employers,
        f"got {employers}")
    _ok("'mid' (cum=150) kept",
        "mid" in employers)
    _ok("'kauhanen_fail' (cum=80) kept by Step 1",
        "kauhanen_fail" in employers)
    _ok("'edge' (cum=100) kept",
        "edge" in employers)
    _ok("'big_q4_only' (cum=500) kept (no identification restriction yet)",
        "big_q4_only" in employers)


def test_filter_kauhanen():
    print("\n--- test_filter_kauhanen ---")
    df = make_synthetic_panel()
    out = filter_kauhanen(df, "22-25")
    employers = set(out["employer_id"].unique())

    _ok("'small' dropped",          "small" not in employers)
    _ok("'mid' kept",               "mid" in employers,
        f"mean monthly = 15, cum = 150 (passes >=10 and >=100)")
    _ok("'kauhanen_fail' dropped",  "kauhanen_fail" not in employers,
        f"mean monthly = 8 < 10, cum = 80 < 100 -- both fail")
    _ok("'edge' kept",              "edge" in employers,
        f"mean monthly = 10 (boundary), cum = 100 (boundary)")
    _ok("'big_q4_only' kept",       "big_q4_only" in employers,
        f"mean monthly = 50, cum = 500")


def test_build_balanced_panel():
    print("\n--- test_build_balanced_panel ---")
    df = make_synthetic_panel()
    sub = filter_step1(df, "22-25")
    balanced = build_balanced_panel(sub, "daioe_quartile", n_bins=4)
    employers = set(balanced["employer_id"].unique())

    _ok("'big_q4_only' dropped (only Q4)",
        "big_q4_only" not in employers,
        "identification restriction: must span Q4 and Q1-Q3")
    _ok("'mid' kept (Q4 + Q2)",
        "mid" in employers)
    _ok("'kauhanen_fail' kept (Q4 + Q1)",
        "kauhanen_fail" in employers)
    _ok("'edge' kept (Q4 + Q3)",
        "edge" in employers)

    # Zero-fill: each employer-bin pair x all months
    months_in_data = sorted(df["year_month"].unique())
    sample = balanced[balanced["employer_id"] == "mid"]
    expected_rows = 2 * len(months_in_data)  # 2 quartiles x 10 months
    _ok("balanced cells for 'mid' = 2 quartiles x 10 months",
        len(sample) == expected_rows,
        f"got {len(sample)}, expected {expected_rows}")
    sparse_rows = balanced[balanced["employer_id"] == "sparse"]
    sparse_zeros = (sparse_rows["n_emp"] == 0).sum()
    _ok("zero-fill present for 'sparse' (10 expected)",
        sparse_zeros == 10,
        f"got {sparse_zeros} zero cells in sparse")
    _ok("non-negative counts",
        (balanced["n_emp"] >= 0).all())


def test_treatment_dates():
    print("\n--- test_treatment_dates ---")
    _ok("RIKSBANK_YM is 2022-04",
        RIKSBANK_YM == "2022-04",
        f"got {RIKSBANK_YM}")
    _ok("CHATGPT_YM is 2022-12",
        CHATGPT_YM == "2022-12",
        f"got {CHATGPT_YM}")

    # Sanity: month-string comparison works as expected
    _ok("'2022-03' < RIKSBANK_YM",  "2022-03" < RIKSBANK_YM)
    _ok("'2022-04' >= RIKSBANK_YM", "2022-04" >= RIKSBANK_YM)
    _ok("'2022-11' < CHATGPT_YM",   "2022-11" < CHATGPT_YM)
    _ok("'2022-12' >= CHATGPT_YM",  "2022-12" >= CHATGPT_YM)


def test_quintile_logic():
    """Verify pd.qcut with 5 quantiles produces 5 bins, label 5 = highest."""
    print("\n--- test_quintile_logic ---")
    np.random.seed(42)
    scores = pd.Series(np.random.uniform(0, 1, 100))
    quintiles = pd.qcut(scores, 5, labels=[1, 2, 3, 4, 5]).astype(int)
    _ok("quintile labels span 1..5",
        set(quintiles.unique()) == {1, 2, 3, 4, 5})
    # Highest scores should map to quintile 5
    top_score = scores[quintiles == 5].min()
    bot_score = scores[quintiles == 1].max()
    _ok("quintile 5 contains higher scores than quintile 1",
        top_score > bot_score,
        f"min(Q5) = {top_score:.3f}, max(Q1) = {bot_score:.3f}")


def test_kauhanen_threshold_constants():
    print("\n--- test_kauhanen_threshold_constants ---")
    _ok("KAUHANEN_MIN_MEAN_PER_MONTH == 10",
        KAUHANEN_MIN_MEAN_PER_MONTH == 10)
    _ok("KAUHANEN_MIN_CUMULATIVE == 100",
        KAUHANEN_MIN_CUMULATIVE == 100)
    _ok("STEP1_MIN_CUMULATIVE == 5",
        STEP1_MIN_CUMULATIVE == 5)


def main():
    print("=" * 60)
    print("Local tests for script 32 (Kauhanen staged robustness)")
    print("=" * 60)
    test_filter_step1()
    test_filter_kauhanen()
    test_build_balanced_panel()
    test_treatment_dates()
    test_quintile_logic()
    test_kauhanen_threshold_constants()

    print("\n" + "=" * 60)
    if _N_FAIL == 0:
        print("ALL TESTS PASS -- script 32 logic is sound.")
        print("=" * 60)
        return 0
    else:
        print(f"{_N_FAIL} TEST(S) FAILED -- fix before uploading to MONA.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
