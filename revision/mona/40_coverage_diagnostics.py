#!/usr/bin/env python3
"""
40_coverage_diagnostics.py -- T1/E3 (MONA run M1)

======================================================================
  RUNS IN SCB's MONA SECURE ENVIRONMENT ONLY.  ** SPEC STUB **
  Status 2026-08-28: specified, NOT yet implemented. Implementation is
  scheduled before the MONA trip (week of 7 Sept); every stub imports
  mona_common and consumes output_39/panel_vintage.parquet.
======================================================================

WHAT THIS SCRIPT WILL DO
========================
Coverage diagnostics from the vintage-tagged panel (output_39/panel_vintage.parquet): match rates by month x age group x DAIOE quartile of last-known code; the incumbent / recent-hire / new-entrant split (first AGI appearance logic from script 30); the share of workers classified with 2023 / 2022 / 2021 codes per month and group; counts and shares excluded because no vintage carries a code (vintage == none). Exports aggregate tables only, min cell 5.

SHARED INFRASTRUCTURE
=====================
- mona_common.pull_panel / collapse_vintage (cache from script 39)
- mona_common.balance_panel / add_treatment (bit-identical to v1)
- Poisson via mona_common.run_fepois / run_fepois_es (R + fixest)
- Export floor: mona_common.enforce_min_cell (counts >= 5)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mona_common as mc  # noqa: F401

OUT = Path(__file__).resolve().parent / "output_40"


def main():
    raise NotImplementedError(
        "40_coverage_diagnostics: spec staged 2026-08-28; implement before the MONA trip. "
        "See the docstring and the revision plan (T-rows).")


if __name__ == "__main__":
    main()
