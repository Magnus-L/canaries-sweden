#!/usr/bin/env python3
"""
41_vintage_event_studies.py -- T2/E4 (MONA run M2)

======================================================================
  RUNS IN SCB's MONA SECURE ENVIRONMENT ONLY.  ** SPEC STUB **
  Status 2026-08-28: specified, NOT yet implemented. Implementation is
  scheduled before the MONA trip (week of 7 Sept); every stub imports
  mona_common and consumes output_39/panel_vintage.parquet.
======================================================================

WHAT THIS SCRIPT WILL DO
========================
Separate half-year event studies for workers classified using 2023, 2022 and 2021 codes (vintage panel split before aggregation), plus incumbent vs new person-employer match ES, plus same-employer vs job-changer carry-forward split. Poisson via mona_common.run_fepois_es; OLS+1 companion for comparability. The editor reads these as: is the 2024-25 decline concentrated where assignments are older?

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

OUT = Path(__file__).resolve().parent / "output_41"


def main():
    raise NotImplementedError(
        "41_vintage_event_studies: spec staged 2026-08-28; implement before the MONA trip. "
        "See the docstring and the revision plan (T-rows).")


if __name__ == "__main__":
    main()
