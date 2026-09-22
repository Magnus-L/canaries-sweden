#!/usr/bin/env python3
"""
run_lane26c.py -- LANE 26c. Submit this file to BatchClient.

  79  part C: the uncounted payslips, a count and no fit

WHY THREE RUNNERS. Script 79 answers three questions script 78 opened, and
they share nothing but the caches they read. MONA allows three batch jobs
at once, so each part has its own runner, its own output folder and its own
R exchange directory, and the three run together rather than in sequence:
about three hours of wall clock for all three instead of five. Submit 26a,
26b and 26c together. run_lane26.py runs all three in one job and is kept
for the record; do not submit it as well.

WHAT RUNS HERE.
  C  no fit at all: one pull per year from 2019 to 2025 over the employer
     declarations joined to Individ_2023, 2021 and 2019, aggregated in the
     server to employer by status by band and cached per year. 47L's
     comparable pull ran at about seventy seconds a year, so budget well
     under an hour for the seven, plus the aggregation. Each year is
     cached as it lands, so a kill costs only the year that was running.

SQL. One INFORMATION_SCHEMA probe of the January declaration tables and seven year pulls over the declarations. Safe beside a job that is not itself pulling declarations.

RUNS BESIDE THE OTHER TWO AND BESIDE LANE 25a. CANARIES_RWORK_TAG gives
this job its own R exchange directory on the batch node, so no two jobs can
share an input file or sweep each other's away.

Export: output_79c/79_summary.txt and uncounted_share.csv, every count floored
at five before it is written.
"""

import os
import sys
from pathlib import Path

PARTS = "C"
os.environ["CANARIES_79_PARTS"] = PARTS
os.environ["CANARIES_79_OUT"] = "output_79c"
# One exchange directory per job, so lane 26 can run beside lane 25a.
os.environ["CANARIES_RWORK_TAG"] = "_26c"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("79_last_gaps.py", "output_79c/79_summary.txt", 90),
]

if __name__ == "__main__":
    _lane.run("26c", STAGES)
