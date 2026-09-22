#!/usr/bin/env python3
"""
run_lane29b.py -- LANE 29b. Submit this file to BatchClient.

  83  part B: the remaining rows of Table 1 on the occupation route

WHY LANE 29 EXISTS, AND WHY THE SCORE IS NOT REBUILT: see run_lane29a.py.
In one line: lane 28 moved the headline, the profile, the margins and the
sex split onto a firm score with no education record in it, and this lane
moves everything else that uses the exposure quartile, so that no
published number mixes two definitions of the same treatment. Every fit
takes its quartile from 82_occupation_route.build_exposure(), the primary
arm, and nothing here rebuilds it.

WHAT RUNS HERE.
  B  the three remaining rows of Table 1, in order of cost.
     (i) THE REFERENCE WINDOW, script 75's specification at both young
         bands: Equation (2) with the tightening term entered as a WINDOW
         covering April to November 2022 rather than as a cumulative
         switch, so the interim and adoption terms read directly against
         January 2021 to March 2022 and the adoption term is the LEVEL
         after adoption with a standard error of its own. Two fits.
     (ii) THE PRE-LAUNCH DRIFT, script 78's Part A(ii) at both young
         bands: a linear monthly trend interacted with High x Young,
         estimated on January 2021 to November 2022 with the calendar
         cycle and the tightening window in. Two fits, and small ones:
         the panel is twenty-three months rather than fifty-four.
     (iii) THE INDUSTRY CLUSTERING, script 80's Part B applied to this
         quartile: the pooled stock fits at both young bands and the sex
         fit at 22-25, clustered on the COMPLETED three-digit industry
         key that script 80 builds and caches, each beside its own
         employer-clustered run. Six fits. 80's key builder and 80's
         Part B are called, not copied.
     Ten fits. On the education route the two stock fits ran in fifteen
     to thirty minutes each and the sex panel in about fifteen, and every
     panel here is a SUBSET of those, so no fit in this lane is larger
     than one that has already fitted. Budget three to three and a half
     hours.

ONE THING IS SWITCHED OFF IN SCRIPT 80, AND IT MATTERS. 80's Part B
reads lanes 25 and 26's exports to get the employer-clustered run it
checks itself against. Those exports are the EDUCATION route's, so read
on this quartile they would compare our coefficients with someone else's
and the four-decimal gate would fail for a reason that is not a defect.
The script empties 80's prior-export search path before calling it, which
sends 80 down its own documented fallback and makes it refit the
employer-clustered run on THIS panel. That costs three of the six fits
and buys a gate that means what it says.

READ RULES, fixed before the run and printed by the script at the start
and in the summary, with the education-route figure beside every
estimate.

  THE ONLY COEFFICIENT GATE IN LANE 29 IS HERE. The clustered
  coefficients must reproduce their own employer-clustered run to FOUR
  DECIMALS, exactly as script 80 gates: clustering changes the covariance
  and nothing else, so a coefficient that moves means the panel moved,
  and nothing from the clustering arm may then be quoted.

  Nothing else in this part has a gate. This is a different measure of
  the same object, the estimates will differ from the education route,
  and that is expected.

  4. THE REFERENCE WINDOW AGREES IN DIRECTION if the level after
     adoption is negative at both bands. No significance rule is set,
     because neither education-route figure is distinguishable from zero
     (-0.0194 (0.0184) at 22-25 and -0.0172 (0.0121) at 26-30), so a rule
     on significance would be one the education route itself fails.
  5. THE PRE-LAUNCH DRIFT is read as script 78 reads it: FLAT if the
     monthly trend is within two of its own standard errors of zero. The
     education route is FLAT at 22-25 (+0.0006 (0.0008)) and NOT FLAT at
     26-30 (+0.0019 (0.0004), four standard errors from zero), so a drift
     at 26-30 here is agreement with the education route and not a defect
     of this route. Both are reported whichever way they fall.
  6. THE INDUSTRY CLUSTERING. The gate above, and then: THE INFERENCE
     SURVIVES INDUSTRY CLUSTERING if the adoption step at 22-25 and the
     female differential keep their sign and stay distinguishable from
     zero at the five per cent level with the industry standard error.
     The education route's industry standard errors are 0.0302 at 22-25,
     0.0200 at 26-30, 0.0353 for young men, 0.0179 for the female
     differential and 0.0285 for young women.

  THE FIRST STAGE IS IN PART A AND IT GOVERNS THIS PART. If lane 29a's
  summary does not say THE FIRST STAGE REPRODUCES, nothing here is
  quoted, however it falls. Run 29a whether or not you run this one.

SQL. None, once the caches are on the share, EXCEPT the completed
industry key: if I_industry_key.parquet is not there (lane 27 builds and
caches it), script 80's builder runs and reads the firm registers. The
occupation cascade and the 2019 monthly counts are pulled only if lane
28a has not already cached them, which it has. Safe beside any job.

Export: output_83b/83_summary.txt, occ_rest_window.csv,
occ_rest_drift.csv, occ_rest_cluster.csv (every term with the employer
and the industry standard error, the four-decimal reproduction check, the
cluster counts and the education route's industry standard error on the
same row) and the ten vcov_*.csv files.
"""

import os
import sys
from pathlib import Path

os.environ["CANARIES_83_PARTS"] = "B"
os.environ["CANARIES_83_OUT"] = "output_83b"
os.environ["CANARIES_82_OUT"] = "output_83b"
os.environ["CANARIES_RWORK_TAG"] = "_29b"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("83_occupation_route_rest.py", "output_83b/83_summary.txt", 210),
]

if __name__ == "__main__":
    _lane.run("29b", STAGES)
