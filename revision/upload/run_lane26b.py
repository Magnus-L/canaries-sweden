#!/usr/bin/env python3
"""
run_lane26b.py -- LANE 26b. Submit this file to BatchClient.

  79  part B: the industry test at 26 to 30, with the calendar terms

WHY THREE RUNNERS. Script 79 answers three questions script 78 opened, and
they share nothing but the caches they read. MONA allows three batch jobs
at once, so each part has its own runner, its own output folder and its own
R exchange directory, and the three run together rather than in sequence:
about three hours of wall clock for all three instead of five. Submit 26a,
26b and 26c together. run_lane26.py runs all three in one job and is kept
for the record; do not submit it as well.

WHAT RUNS HERE.
  B  the 26-30 stock with the three calendar terms and industry by age by
     month effects, beside its own baseline on the same industry-linked
     sample. Two fits, about two hours, the second the heavier. Lane 25
     ran this at 22-25 only, and 26-30 is the band whose step survives
     industry clustering, so the paper needs the pair.

SQL. One read of LISA's Ftg_2019 for the industry code, the read script 73 makes.

RUNS BESIDE THE OTHER TWO AND BESIDE LANE 25a. CANARIES_RWORK_TAG gives
this job its own R exchange directory on the batch node, so no two jobs can
share an input file or sweep each other's away.

Export: output_79b/79_summary.txt and industry_seasonal_2630.csv.
"""

import os
import sys
from pathlib import Path

PARTS = "B"
os.environ["CANARIES_79_PARTS"] = PARTS
os.environ["CANARIES_79_OUT"] = "output_79b"
# One exchange directory per job, so lane 26 can run beside lane 25a.
os.environ["CANARIES_RWORK_TAG"] = "_26b"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("79_last_gaps.py", "output_79b/79_summary.txt", 150),
]

if __name__ == "__main__":
    _lane.run("26b", STAGES)
