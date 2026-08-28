#!/usr/bin/env python3
"""
43_poisson_primary.py -- T4/E6 (MONA run M4)

======================================================================
  RUNS IN SCB's MONA SECURE ENVIRONMENT ONLY.  ** SPEC STUB **
  Status 2026-08-28: specified, NOT yet implemented. Implementation is
  scheduled before the MONA trip (week of 7 Sept); every stub imports
  mona_common and consumes output_39/panel_vintage.parquet.
======================================================================

WHAT THIS SCRIPT WILL DO
========================
The new primary battery: Poisson pooled DiD + half-year ES for all six age groups (run_fepois / run_fepois_es), producing the headline multiplicative effect and its endpoint; the bridge-table inputs (OLS+1 vs Poisson, both windows); the extensive-margin LPM (P(n_emp=0)); occupation- and employer-level clustering variants. Every ln(n+1) percentage in the paper is replaced from this output.

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

OUT = Path(__file__).resolve().parent / "output_43"


def main():
    raise NotImplementedError(
        "43_poisson_primary: spec staged 2026-08-28; implement before the MONA trip. "
        "See the docstring and the revision plan (T-rows).")


if __name__ == "__main__":
    main()
