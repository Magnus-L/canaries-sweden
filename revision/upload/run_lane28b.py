#!/usr/bin/env python3
"""
run_lane28b.py -- LANE 28b. Submit this file to BatchClient.

  82  part B: the headline and the age profile on the occupation route

WHY THREE RUNNERS. Script 82 re-estimates the paper's findings on a firm
score that uses no education record at all (see run_lane28.py for the
question). The three parts share nothing but the caches they read and the
score itself, which each of them builds from the same cached baseline.
MONA allows three batch jobs at once, so each part has its own runner, its
own output folder and its own R exchange directory, and the three run
together rather than in sequence. Submit 28a, 28b and 28c together.
run_lane28.py runs all three in one job and is kept for the record; do not
submit it as well.

WHAT RUNS HERE.
  B  Equation (2) on the employment stock at 22-25 and at 26-30, with the
     tightening switch, the interim window, the adoption step and the
     three calendar-quarter terms, exactly as script 68 estimates it and
     with script 78's term builder, on the occupation-route quartile.
     Then the six-band profile against 41-49, as script 74 builds it on
     script 70's six-band skeleton.
     Three fits. On the education route the two stock fits ran in fifteen
     to thirty minutes each and the six-band profile in about twenty, and
     every panel here is a SUBSET of those: the occupation route scores
     about 65,000 employers against the education route's 311,000, so no
     fit in this lane is larger than one that has already fitted. Budget
     an hour, ninety minutes at the outside.

READ RULES, fixed before the run. There is no coefficient gate: this is a
different measure and the estimates will differ. Part B answers two of the
three questions, and reports the numbers whichever way they fall.
  1. REPRODUCES if the adoption step at 22-25 is negative and
     distinguishable from zero at the five per cent level on employer
     clustering.
  2. THE PROFILE REPRODUCES if the 50-and-over band gains against 41-49
     and the young band is the lowest or second lowest of the six.
The education-route numbers are printed beside every estimate: -0.0408
(0.0150) at 22-25, -0.0394 (0.0102) at 26-30, -0.0099 (0.0121) for 22-25
against 41-49 and +0.0589 (0.0062) for 50 and over.

SQL. None, if 47L's caches are on the share; one fourteen-second read of
the November 2019 declarations otherwise. Safe beside any job.

Export: output_82b/82_summary.txt, occ_route_headline.csv,
occ_route_profile.csv and the three vcov_s82_*.csv files.
"""

import os
import sys
from pathlib import Path

os.environ["CANARIES_82_PARTS"] = "B"
os.environ["CANARIES_82_OUT"] = "output_82b"
os.environ["CANARIES_RWORK_TAG"] = "_28b"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("82_occupation_route.py", "output_82b/82_summary.txt", 90),
]

if __name__ == "__main__":
    _lane.run("28b", STAGES)
