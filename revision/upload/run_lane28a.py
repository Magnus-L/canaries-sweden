#!/usr/bin/env python3
"""
run_lane28a.py -- LANE 28a. Submit this file to BatchClient.

  82  part A: the occupation-route score, and what it covers

WHY THREE RUNNERS. Script 82 re-estimates the paper's findings on a firm
score that uses no education record at all (see run_lane28.py for the
question and for what was wrong with the first version of the score). The
three parts share nothing but the caches they read and the occupation
cascade, which each of them builds from the same pull and which is cached
the moment it is built. MONA allows three batch jobs at once, so each part
has its own runner, its own output folder and its own R exchange
directory. Submit 28a first and give it a few minutes to get past the
cascade pull, then submit 28b and 28c: three jobs pulling at once write the
same frame and the write is atomic, so a race is harmless, but it costs the
pull three times. run_lane28.py runs all three in one job and is kept for
the record; do not submit it as well.

WHAT RUNS HERE.
  A  the score and no fit at all. Where the cascade resolves each
     incumbent, step by step; the coverage of the code among incumbents by
     age band, BEFORE and AFTER the cascade, as a coded share and as the
     share whose code also carries a DAIOE percentile; the decomposition
     of the employers script 65's rule lost into those the floor lost,
     those missing codes lost after the full backward cascade, and those
     both lost; the floor sensitivity at 1, 3 and 5; the employers each
     route can score, alone and on the two young panels; the occupation
     quartile against the education quartile among the employers both
     routes score, with the share on the diagonal and the Spearman rank
     correlation; and the size and longevity of the employers one route
     scores and the other does not, read off the panels the fits
     themselves build.
     Runtime: thirty to forty minutes. The cascade pull is a few minutes
     and the 2019 monthly counts about seventy seconds when they are not
     already on the share; the rest is the two panel rebuilds. This is the
     shortest of the three, it is the only one that also builds the
     education score, and it is the one that caches the cascade, so submit
     it first.

SQL. One read of Arb_AGIIndivid201911_def joined to the Individ tables of
2015 to 2021 (one INFORMATION_SCHEMA probe first, so a vintage the
delivery does not hold shortens the cascade rather than failing the pull),
and one read of the 2019 monthly declarations through 47L's own query if
L_counts_2019 is not on the share. Both are cached. No other declarations
are touched, so this is safe beside any job.

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
    ("82_occupation_route.py", "output_82a/82_summary.txt", 40),
]

if __name__ == "__main__":
    _lane.run("28a", STAGES)
