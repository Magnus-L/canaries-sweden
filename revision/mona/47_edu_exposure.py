#!/usr/bin/env python3
"""
47_edu_exposure.py -- T12/R2.3: education-based exposure (Erik's construction).

======================================================================
  RUNS IN SCB's MONA SECURE ENVIRONMENT ONLY.  ** AWAITING INPUT **
  Status 2026-08-28: waits for Erik Engberg's do-files (education-level
  AI exposure built from DAIOE + the occupational composition of workers
  by educational degree). Port them to Python here, mirroring float32
  storage per gen/egen/replace (the daioe-pipeline lesson: Stata stores
  float32; a float64 port drifts ~1e-7/step and compounds), and VALIDATE
  against his Stata output before estimating anything.
======================================================================

DESIGN CONSTRAINTS (from the April failure of script 34 -- the results
note "EduQuartile is broken", notes/eduquartile-results-2026-04-29.md):
  1. Use the MOST RECENT education record, not a 3-year lag; report the
     retention rate for 22-25 explicitly (34 silently kept ~47%).
  2. Fix the education-to-occupation mapping on PRE-SHOCK flows
     (answers Koch's drift caveat).
  3. Headline the EVENT-STUDY SHAPE and the total post effect; do NOT
     headline the Riksbank/ChatGPT split -- education cells mix
     destination occupations with different responses to the two shocks,
     which is what destroyed 34's timing decomposition.
  4. Compare Erik's construction against 34's diagnosis before running.

The estimation stage then mirrors 43: balanced employer x edu-bin panel,
Poisson pooled + ES, same FE and clustering.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mona_common as mc  # noqa: F401


def main():
    raise NotImplementedError("47: waiting for Erik's do-files (R2.4).")


if __name__ == "__main__":
    main()
