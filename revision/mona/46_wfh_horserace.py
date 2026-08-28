#!/usr/bin/env python3
"""
46_wfh_horserace.py -- T10/R1.7 (MONA run M8)

======================================================================
  RUNS IN SCB's MONA SECURE ENVIRONMENT ONLY.  ** SPEC STUB **
  Status 2026-08-28: specified, NOT yet implemented. Implementation is
  scheduled before the MONA trip (week of 7 Sept); every stub imports
  mona_common and consumes output_39/panel_vintage.parquet.
======================================================================

WHAT THIS SCRIPT WILL DO
========================
WFH horse race: telework exposure (Dingel-Neiman SSYK from data/raw, uploaded to the share) interacted with year effects ALONGSIDE the DAIOE treatment, both margins; plus the split-sample variant from v1 script 26 rerun under Poisson. Azar, Gine and Sanz-Espin (2026) find dropping the WFH control makes their estimate MORE negative; report our analogue.

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

OUT = Path(__file__).resolve().parent / "output_46"


def main():
    raise NotImplementedError(
        "46_wfh_horserace: spec staged 2026-08-28; implement before the MONA trip. "
        "See the docstring and the revision plan (T-rows).")


if __name__ == "__main__":
    main()
