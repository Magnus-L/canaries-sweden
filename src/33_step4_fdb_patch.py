#!/usr/bin/env python3
"""
33_step4_fdb_patch.py -- standalone Step 4 runner with FDB table-name fix.

WHY THIS SCRIPT EXISTS
======================
Script 32 (Kauhanen staged robustness) ran successfully on 2026-04-28
for Steps 1, 2, 3, 5 + attrition diagnostic. Step 4 (Finnish-composition
reweighting) failed because its NACE pull used `dbo.Foretag_FDB`, which
does not exist in P1207. The Foretagsdatabasen tables in P1207 are
year-partitioned and split by unit level:

    dbo.FDB_AE_*  -- workplace level (Arbetsstaelle)
    dbo.FDB_JE_*  -- legal-entity level (Juridisk Enhet)

For employer-level analysis (matching AGI's P1207_LOPNR_PEORGNR), we
want JE. Two partitions cover our 2019-2025 panel:

    dbo.FDB_JE_2014_2021
    dbo.FDB_JE_2022_2024

Column names verified 2026-04-29 via Object Explorer:
    P1207_Lopnr_peorgnr (int)   -- legal-entity employer ID
    ar (varchar(4))             -- reference year
    ng1 (char(5))               -- primary SNI 2007 5-digit code

NACE-2 = first two digits of ng1, after stripping any non-digit
characters (defensive against dots, padding, or formatting artefacts).

WHAT THIS SCRIPT DOES
=====================
1. Imports script 32 as a module (via importlib, since '32_*' is not
   a valid Python identifier).
2. Monkey-patches its broken `_pull_employer_nace` with a working one.
3. Loads the cached AGI panel and DAIOE quartiles (no SQL pull).
4. Runs Step 4 (`step4_poisson_reweighted`) end to end.
5. Re-reads Steps 1, 2, 3, 5 from disk and rewrites the comparison
   table and prose summary so they include all five steps.

USAGE IN MONA
=============
1. Upload as 33_step4_fdb_patch.txt; rename to .py.
2. Confirm the existing files are in place:
       32_mona_kauhanen_robustness.py           (must be alongside)
       output_32/step0_panel_cache.csv          (from previous run)
       finland_marginals_2022.txt on the share
3. Run:
       python 33_step4_fdb_patch.py

Expected runtime: 5-15 minutes (no SQL pull from cache; one short FDB
query; six Poisson regressions).

EXPORT-SAFE
===========
All output is aggregated counts and regression coefficients. NACE-2
extraction includes a sanity check that drops malformed rows.
"""

import importlib.util
import re
import sys
import pandas as pd
import pyodbc
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
KAUHANEN_PY = SCRIPT_DIR / "32_mona_kauhanen_robustness.py"

if not KAUHANEN_PY.exists():
    print(f"FATAL: 32_mona_kauhanen_robustness.py not found at {KAUHANEN_PY}")
    print("       This script must live in the same folder as script 32.")
    sys.exit(1)

# Load script 32 as the module 'kauhanen' -- importlib lets us bypass
# the digit-prefix restriction on Python module names.
_spec = importlib.util.spec_from_file_location("kauhanen", str(KAUHANEN_PY))
kauhanen = importlib.util.module_from_spec(_spec)
sys.modules["kauhanen"] = kauhanen
_spec.loader.exec_module(kauhanen)


def _pull_employer_nace_fdb_je(conn):
    """
    Replacement for kauhanen._pull_employer_nace.

    Pulls (employer_id, NACE-2) from FDB_JE_2014_2021 + FDB_JE_2022_2024.
    Most-recent year per employer is kept. NACE-2 is extracted by
    stripping non-digit characters from ng1 and taking the first 2 digits.
    """
    print("  Pulling employer -> NACE-2 from FDB_JE (2014-2021 + 2022-2024)...")
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
    print(f"    Pulled {len(df):,} firm-year rows total")

    # Most-recent year per employer
    df = df.sort_values(["employer_id", "year"], ascending=[True, False])
    df = df.drop_duplicates(subset=["employer_id"], keep="first")

    # Robust NACE-2: strip non-digits, take first 2, zero-pad to width 2
    df["nace2"] = (
        df["ng1_raw"].astype(str)
        .str.replace(r"[^0-9]", "", regex=True)
        .str[:2]
        .str.zfill(2)
    )

    bad = ~df["nace2"].str.match(r"^[0-9]{2}$")
    n_bad = int(bad.sum())
    if n_bad:
        print(f"    WARNING: {n_bad:,} rows had malformed ng1; dropping")
        df = df[~bad]

    nace_codes = sorted(df["nace2"].unique())
    if nace_codes:
        print(f"    NACE-2 codes seen: {len(nace_codes)} distinct, "
              f"range {nace_codes[0]}-{nace_codes[-1]}")
    print(f"    Retrieved NACE-2 for {len(df):,} employers")
    return df[["employer_id", "nace2"]]


def _maybe_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main():
    print("=" * 70)
    print("Step 4 patch runner -- FDB_JE table-name fix")
    print("=" * 70)
    print(f"Script 32 module: {KAUHANEN_PY}")
    print(f"Output dir:       {kauhanen.OUTPUT_DIR}")

    # 1. Monkey-patch the broken pull on the imported module.
    kauhanen._pull_employer_nace = _pull_employer_nace_fdb_je
    print("  Monkey-patched kauhanen._pull_employer_nace -> _pull_employer_nace_fdb_je")

    # 2. Reuse the cached panel and DAIOE quartiles (no SQL re-pull).
    panel_ssyk_age = kauhanen.step0_pull_or_load_panel()
    daioe = kauhanen.load_daioe_quartiles()

    # 3. Run Step 4 only.
    step4 = kauhanen.step4_poisson_reweighted(panel_ssyk_age, daioe)

    if step4 is None or len(step4) == 0:
        print("\nStep 4 returned no results. Most likely causes:")
        print("  (a) finland_marginals_2022.txt not found on the share")
        print("  (b) FDB pull returned zero rows after dedup")
        print("  (c) Reweighted panel had insufficient employers per age group")
        print("Check the log lines above for which.")
        return

    # 4. Re-load Steps 1, 2, 3, 5 from disk so the comparison and prose
    #    summary include the full picture.
    out_dir = kauhanen.OUTPUT_DIR
    step1 = _maybe_csv(out_dir / "step1_poisson_current.csv")
    step2 = _maybe_csv(out_dir / "step2_poisson_threshold.csv")
    step3 = _maybe_csv(out_dir / "step3_poisson_kauhanen.csv")
    step5 = _maybe_csv(out_dir / "step5_poisson_no_ict.csv")

    # 5. Rewrite comparison table + prose summary, now including Step 4.
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
