#!/usr/bin/env python3
"""
run_lane37b.py -- LANE 37b. Submit this file to BatchClient.

  96  the youth payroll reduction by BIRTH COHORT: tau for the never-
      covered cohorts (born 1994-1997), the ever-covered (1998-2003), the
      months of eligibility as a dose, and the female differential among
      the never covered                                          ~1.75 h

SQL. One pull, 47L's counts query with 67's sex column and a cell of its
own for every birth year from 1994, 2021 to 2025, cached as
cache/L_counts_cohortsex_YYYY (about three minutes a year). A rerun reads
the cache and issues no SQL.

THE GATE. The pull re-banded by age must reproduce Table 1 at 22-25
within 0.0005 (tau -0.0399 (0.0102)); a miss stops the script. The sex
part also needs the female differential, tau -0.0714 (0.0109).

INDEPENDENT OF LANES 37a AND 37c. Tested locally in
revision/local/test_96_payroll_cohorts.py.

EXPORT: output_96/96_summary.txt and output_96/payroll_cohorts.csv.
"""
import os
import sys
from pathlib import Path

os.environ["CANARIES_96_OUT"] = "output_96"
os.environ["CANARIES_82_OUT"] = "output_96"
os.environ["CANARIES_RWORK_TAG"] = "_37b"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("96_payroll_cohorts.py", "output_96/96_summary.txt", 105),
]

if __name__ == "__main__":
    _lane.run("37b", STAGES)
