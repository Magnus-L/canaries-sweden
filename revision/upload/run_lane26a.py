#!/usr/bin/env python3
"""
run_lane26a.py -- LANE 26a. Submit this file to BatchClient.

  79  part A: the sexes clustered by three-digit industry

WHY THREE RUNNERS. Script 79 answers three questions script 78 opened, and
they share nothing but the caches they read. MONA allows three batch jobs
at once, so each part has its own runner, its own output folder and its own
R exchange directory, and the three run together rather than in sequence:
about three hours of wall clock for all three instead of five. Submit 26a,
26b and 26c together. run_lane26.py runs all three in one job and is kept
for the record; do not submit it as well.

WHAT RUNS HERE.
  A  the sex panel of 67, every term of Equation (2) interacted with
     female, clustered by the employer's 2019 three-digit industry. One
     Poisson fit on the 40,520,358-row sex panel, two to three hours. If
     lane 25's output_78b/gender_eq2.csv and its covariance are on the
     share they supply the employer-clustered numbers beside it; if they
     are not, that fit is repeated here and the part takes four to five
     hours. The summary says which happened.
     This is the longest of the three, so submit it first.

SQL. One read of LISA's Ftg_2019 for the industry code, the read script 73 makes.

RUNS BESIDE THE OTHER TWO AND BESIDE LANE 25a. CANARIES_RWORK_TAG gives
this job its own R exchange directory on the batch node, so no two jobs can
share an input file or sweep each other's away.

Export: output_79a/79_summary.txt, gender_cluster_industry.csv and the
vcov_s79_*.csv files for the sex terms.
"""

import os
import sys
from pathlib import Path

PARTS = "A"
os.environ["CANARIES_79_PARTS"] = PARTS
os.environ["CANARIES_79_OUT"] = "output_79a"
# One exchange directory per job, so lane 26 can run beside lane 25a.
os.environ["CANARIES_RWORK_TAG"] = "_26a"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("79_last_gaps.py", "output_79a/79_summary.txt", 240),
]

if __name__ == "__main__":
    _lane.run("26a", STAGES)
