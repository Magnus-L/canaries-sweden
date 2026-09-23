#!/usr/bin/env python3
"""
run_lane32.py -- LANE 32. Submit this file to BatchClient.

  86  the plain quarterly path from 2019 and the pre-launch drift test,
      on lane 28's occupation-route score

WHY LANE 32 EXISTS. Online Appendix III.2 draws the quarterly path of
the design on the plain specification, 2019Q1 to 2025Q2 with 2022Q1
omitted and no calendar terms, and reports the drift test beneath it.
Script 78's part A produces both, and it has only ever run on the
education-mix score. So the figure and its table classified employers on
a route the paper no longer reports, while the drift numbers printed
beside them belong to the occupation route, through Table 1. The page
carried both: 0.00060 and 0.00187 in the appendix against 0.0004 and
0.0016 in the table. This lane puts the figure, its table and the test
on one measure.

THE DRIFT REFIT IS THE GATE. The drift already exists on this route, in
lane 29b's occ_rest_drift.csv. Part A fits the path and the drift
together on one frame, and this lane does not edit 78, so the drift is
produced again. That is the gate: the refit is checked against lane 29b
to four decimals on every term of both bands, and a match is the only
evidence that the panel behind the new path is the panel Table 1 sits
on. IF THE CHECK FAILS THE PATH IS NOT DRAWN and the summary says so at
the top.

THE SCORE IS LANE 28'S. The quartile comes from
82_occupation_route.build_exposure(), the primary arm: uniform3, the
backward cascade, a floor of five incumbent person-months, as in lanes
29, 30 and 31.

WHAT RUNS HERE. Four Poisson fits, two per young band: the quarterly
path on the two-band panel (the young band and the four incumbent
bands), 25 quarter dummies with 2022Q1 omitted and no calendar terms,
about 48 million cells at 22-25 and 55 million at 26-30; then the drift
fit on the pre-launch months alone, January 2021 to November 2022, with
the cycle, the tightening window and a linear monthly trend. Employer-by-
month, employer-by-age and month-by-age effects throughout, clustered by
employer.

  Runtime: lane 25a ran parts A, D and G in 155 minutes and part A was
  most of it. Budget ninety minutes and do not be surprised by two
  hours. The path fits are the large ones.

THE WINDOW. The path runs from 2019Q1 only if L_counts_2019 and
L_counts_2020 are on the share. Lane 25a's summary records that they
were, so they should be. If they are not, the script runs the path from
2021-01 and says so in those words; IT DOES NOT PULL TO EXTEND IT. A
full read of the monthly declarations is not something to begin by
accident, and a 2021-start path is still usable, it is simply not the
window the appendix figure draws.

SQL. None, provided lane 28a's cascade cache and 47L's counts are on the
share, which they are. Safe beside any job.

Export: output_86/86_summary.txt, occ_route_prepath.csv,
occ_route_predrift.csv and the vcov_s78_prepath_*/vcov_s78_predrift_*
files. Part A is 78's code, so its fits keep 78's tags; the two files it
writes under its own export names are renamed to the occ_route_* names
before anything leaves, because two exposure routes must never share an
export name.
"""

import os
import sys
from pathlib import Path

os.environ["CANARIES_86_OUT"] = "output_86"
# 82 is imported for its score builder; its own OUT is pointed here so a
# run of this lane leaves no stray output_82 folder behind.
os.environ["CANARIES_82_OUT"] = "output_86"
os.environ["CANARIES_RWORK_TAG"] = "_32"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("86_occupation_route_prepath.py", "output_86/86_summary.txt", 150),
]

if __name__ == "__main__":
    _lane.run("32", STAGES)
