#!/usr/bin/env python3
"""
run_lane28a.py -- LANE 28a. Submit this file to BatchClient.

  82  part A: the occupation-route score, and what it covers

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
  A  the score and no fit at all. The coverage of the 2019 occupation code
     among incumbent head count, by age band, both as a coded share and as
     the share whose code also carries a DAIOE percentile; the employers
     each route can score, alone and on the two young panels; the
     occupation quartile against the education quartile among the
     employers both routes score, with the share on the diagonal and the
     Spearman rank correlation of the underlying scores; and the size and
     longevity of the employers one route scores and the other does not,
     read off the panels the fits themselves build.
     Runtime: fifteen to twenty minutes, nearly all of it the two panel
     rebuilds. This is the shortest of the three and the only one that
     also builds the education score, so submit it first if the three must
     be staggered.

SQL. None, if 47L's L_baseline_2019 cache and 47h's education caches are
on the share; one fourteen-second read of the November 2019 declarations
otherwise. No declarations beyond that, so this is safe beside any job.

Export: output_82a/82_summary.txt and occ_route_coverage.csv. Employer and
person counts below five are suppressed before anything leaves MONA, and a
share is suppressed with its own numerator.
"""

import os
import sys
from pathlib import Path

os.environ["CANARIES_82_PARTS"] = "A"
os.environ["CANARIES_82_OUT"] = "output_82a"
os.environ["CANARIES_RWORK_TAG"] = "_28a"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("82_occupation_route.py", "output_82a/82_summary.txt", 20),
]

if __name__ == "__main__":
    _lane.run("28a", STAGES)
