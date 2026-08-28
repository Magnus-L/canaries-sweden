#!/usr/bin/env python3
"""
45_asof_backtest.py -- T3/E5 centrepiece (MONA run M6)

======================================================================
  RUNS IN SCB's MONA SECURE ENVIRONMENT ONLY.  ** SPEC STUB **
  Status 2026-08-28: specified, NOT yet implemented. Implementation is
  scheduled before the MONA trip (week of 7 Sept); every stub imports
  mona_common and consumes output_39/panel_vintage.parquet.
======================================================================

WHAT THIS SCRIPT WILL DO
========================
The as-of backtest: on the fully-covered 2019-2023 window, pretend the register ends at 2021 (and 2022), rebuild the exact cascade under that truncation, re-estimate the 22-25 ES, and report the ARTIFICIAL coefficient latency alone generates vs the true-code estimate. Then the two-dimensional frontier: grid over the unknown Q4 share of unmatched workers x misclassification, marking which combinations would erase the observed effect and where the backtest calibrates us. Plus the backdated-treatment-date variant (R1.14, Facius-Iacono style placebo). This is the response letter's central exhibit.

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

OUT = Path(__file__).resolve().parent / "output_45"


def main():
    raise NotImplementedError(
        "45_asof_backtest: spec staged 2026-08-28; implement before the MONA trip. "
        "See the docstring and the revision plan (T-rows).")


if __name__ == "__main__":
    main()
