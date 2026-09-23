#!/usr/bin/env python3
"""
run_lane31.py -- LANE 31. Submit this file to BatchClient.

  85  three things on lane 28's occupation-route score: the age profile
      without the calendar terms beside a refit of the arm with them, the
      descriptive counterpart with its totals, and the oldest band split
      at 65

WHY LANE 31 EXISTS. Figure 2 of the paper draws each age band against
41-49 twice: on the paper's specification, with three quarter-of-year
interactions per band, and without them. The pair is the exhibit's point.
It shows that the calendar terms, and not the data, are what remove the
youngest band's significance: on the education route the plain arm gives
-0.0288 (SE 0.0130) at 22-25 and the arm with the cycle removed -0.0099
(0.0121), so a reader sees what the control costs and can decide whether
to believe it.

Lane 28 fitted the profile with the calendar terms only, so the
occupation route has one arm and the figure cannot be drawn as it was.
This lane fits the missing half.

BOTH ARMS RUN HERE, AND THAT IS THE POINT OF THE GATE. The seasonal arm
already exists in lane 28b's export. It is refitted anyway, because two
series of one figure must sit on one panel and the only way to know that
is to fit them in one job on one frame. The refit is then checked against
lane 28b's coefficients to four decimals. IF THE CHECK FAILS THE PANEL
HAS MOVED AND NEITHER ARM IS QUOTED, and the summary says so at the top.
The check costs one fit and buys the only thing that makes the pair
comparable.

THE SCORE IS LANE 28'S AND THE TERMS ARE 74'S. The quartile comes from
82_occupation_route.build_exposure(), the primary arm, as in lanes 29 and
30. The terms come from 74's own build_terms called with seasonal False
and True, so the plain arm is the paper's specification minus the
calendar terms and nothing else.

WHAT RUNS HERE, three parts, chosen with CANARIES_85_PARTS (default PDS).
  P  Two Poisson fits of the six-band panel (22-25, 26-30, 31-34, 35-40,
     41-49 and 50 and over, the reference 41-49), with employer-by-month,
     employer-by-age and month-by-age effects, clustered by employer, on
     the 153,845 employers lane 28b fitted. This is Figure 2.
  D  The descriptive counterpart, script 66's own describe(), which
     returns the TOTAL and the per-cell mean by quartile, age band and
     window. No fit, seconds. Lane 29a called the same function and
     exported only the means, which is why the appendix table's Panel A
     could not be rebuilt and that table is still the education route's.
  S  The oldest band split at 65, script 78's own part_e on the seven-band
     panel: one fit. It settles whether the gain of the oldest band is a
     retirement-age effect. Its counts are 78's own pull and are cached
     from the earlier lane; if they are NOT on the share this part skips
     itself rather than starting a pull, because this lane promises no
     SQL and a full read of the monthly declarations is not something to
     begin by accident.
  Runtime: thirty to forty minutes for all three. The score is cached by
  lane 28a and the counts by 47L, so nothing is pulled.

READ RULES, fixed before the run and printed by the script at the start
and in the summary.

  1. THE GATE. The arm with the calendar terms must reproduce lane 28b's
     profile to four decimals at every band. A moved coefficient means a
     moved panel, and then neither arm is quoted and the figure is not
     drawn.
  2. THE PLAIN ARM CARRIES NO VERDICT. It is not a rival estimate of the
     profile and the paper does not read it as one. The calendar terms
     are in the reported specification because the exposure-differential
     ratio has a seasonal cycle present before any treatment, and an arm
     without them inherits it. The arm exists to show the reader how much
     of the profile the control removes.
  3. The difference between the arms is reported at every band whichever
     way it falls, and any band where the two disagree in SIGN is named.

SQL. None, provided lane 28a's cascade cache and 47L's counts are on the
share, which they are. Safe beside any job.

Export: output_85/85_summary.txt, occ_route_profile_arms.csv with the two
vcov_s85_profile_*.csv files, occ_route_descriptive_full.csv, and
occ_route_split65.csv with its vcov.
"""

import os
import sys
from pathlib import Path

os.environ["CANARIES_85_PARTS"] = "PDS"
os.environ["CANARIES_85_OUT"] = "output_85"
# 82 is imported for its score builder; its own OUT is pointed here so a
# run of this lane leaves no stray output_82 folder behind.
os.environ["CANARIES_82_OUT"] = "output_85"
os.environ["CANARIES_RWORK_TAG"] = "_31"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("85_occupation_route_plain_profile.py", "output_85/85_summary.txt", 40),
]

if __name__ == "__main__":
    _lane.run("31", STAGES)
