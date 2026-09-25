#!/usr/bin/env python3
"""
run_lane37c.py -- LANE 37c, AGE AND POLICY DIAGNOSTICS (lower priority:
they qualify interpretation, not causal separation). Submit this file to
BatchClient.

  95  pension ages: the profile with 50-59, 60-64 and 65-69 on one
      employer sample (tau and gamma_2), and tau with the reference cut
      to 31-59 and 31-49 beside the full-sample estimate          ~2 h
  96  the payroll reduction on FIXED, DISJOINT birth cohorts (reference
      born 1956-1990, never covered 1994-1997, ever covered 1998-2003,
      no age filter): cohort-specific exposure gradients; the dose last
      and optional                                                ~1 h

Independent stages; each reproduces Table 1's tau before varying
anything. SQL: 95 one eight-band counts pull (~6 min); 96 one
fixed-cohort pull (~15 min).

Tested locally in revision/local/test_95_pension_reference.py and
test_96_payroll_cohorts.py.
EXPORT: output_95/95_summary.txt, pension_reference.csv, vcov_s95_*;
output_96/96_summary.txt, payroll_cohorts.csv, vcov_s96_*.
"""
import os
import sys
from pathlib import Path

os.environ["CANARIES_95_OUT"] = "output_95"
os.environ["CANARIES_96_OUT"] = "output_96"
os.environ["CANARIES_82_OUT"] = "output_95"
os.environ["CANARIES_RWORK_TAG"] = "_37c"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("95_pension_reference.py", "output_95/95_summary.txt", 120),
    ("96_payroll_cohorts.py", "output_96/96_summary.txt", 60),
]

if __name__ == "__main__":
    _lane.run("37c", STAGES)
