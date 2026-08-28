#!/usr/bin/env python3
"""
42_frozen_cohort.py -- T3/E5 (MONA run M3)

======================================================================
  RUNS IN SCB's MONA SECURE ENVIRONMENT ONLY.  ** SPEC STUB **
  Status 2026-08-28: specified, NOT yet implemented. Implementation is
  scheduled before the MONA trip (week of 7 Sept); every stub imports
  mona_common and consumes output_39/panel_vintage.parquet.
======================================================================

WHAT THIS SCRIPT WILL DO
========================
Coverage-immune cohort design: freeze the population holding any cascade code by Dec 2023; follow that fixed person set forward 2024-25; employment ES on the frozen cohort with the standard FE. No coverage deterioration can enter, because membership and exposure are fixed before the coverage break. Report cohort size and attrition from death/emigration separately.

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

OUT = Path(__file__).resolve().parent / "output_42"


def main():
    raise NotImplementedError(
        "42_frozen_cohort: spec staged 2026-08-28; implement before the MONA trip. "
        "See the docstring and the revision plan (T-rows).")


if __name__ == "__main__":
    main()
