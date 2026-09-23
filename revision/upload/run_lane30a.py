#!/usr/bin/env python3
"""
run_lane30a.py -- LANE 30a. Submit this file to BatchClient.

  84  the QUARTERLY path of Figure 3, at 22-25 and at 26-30, on lane
      28's occupation-route score

WHY LANE 30 EXISTS. Lanes 28 and 29 moved every register estimate in the
paper onto a firm score built from the employer's own 2019 occupation
mix, with no education record in it. ONE EXHIBIT DID NOT MOVE. Figure 3,
the quarterly path that dates the step, is still drawn from script 68's
education-route fit, so the v3 draft asks a reader to take Table 1 on one
definition of the treatment and the timing evidence on another. The
caption says so, which keeps the draft honest, and this lane is what lets
the caption be deleted.

THE SCORE IS LANE 28'S AND IS NOT REBUILT. The quartile comes from
82_occupation_route.build_exposure(), the primary arm: uniform3, the
backward cascade, a floor of five incumbent person-months. One definition
of the treatment variable, in one place, as in lane 29.

THE TERMS ARE SCRIPT 68'S AND ARE NOT REBUILT EITHER. The path terms come
from 68's own add_seasonal_terms(b, "quarter"), called on 68's module.
Copying the term list into a new script would let the path drift from the
specification the pooled estimates come from, and the point of the figure
is that it decomposes the same fit. The Riksbank interaction stays on in
every path fit, so each coefficient is a step from the level of the
tightening months, which is the reading the caption already states.

WHAT RUNS HERE.
  Q  the quarterly path at both young bands: every calendar quarter from
     2022Q4 onward, the three quarter-of-year terms removing the cycle
     with the fourth quarter omitted, employer-by-month, employer-by-age
     and month-by-age effects, Poisson, clustered by employer. Two fits,
     one per band, on the panels lane 28b already fitted (104,217
     employers at 22-25 and 117,090 at 26-30).
     Runtime: thirty-five to fifty minutes. The score is cached by lane
     28a and the counts by 47L, so nothing is pulled.

READ RULES, fixed before the run and printed by the script at the start
and in the summary. There is NO coefficient gate: this is a
decomposition of a fit already made and reported, not a new estimate of
it, and lane 28's pooled estimates stand whatever the path does. What the
rules decide is which sentences about TIMING survive.

  1. THE DATING REPRODUCES at 22-25 if no quarter before 2024 carries a
     negative coefficient distinguishable from zero at five per cent,
     AND at least one quarter from 2024 onward does. The rule is written
     on the year and not on the quarter, because a rule naming 2024Q3
     would be read off the figure it is meant to judge. The education
     route is flat through 2023Q3, dips in 2023Q4 and opens its shortfall
     in the second half of 2024.
  2. THE LAG REPRODUCES if the first quarter meeting that description at
     26-30 falls LATER than the first at 22-25. The paper's sentence is
     that the shortfall reaches the older of the two young bands a year
     later; if they open together, or the older one first, that sentence
     goes.
  A consistency line is printed and NOT gated: the unweighted mean of the
  quarters from 2024 at 22-25, beside lane 28b's pooled adoption step of
  -0.0578. The two are different statistics, the pooled term weighting
  employer-months and the mean not, so a difference of a few thousandths
  is expected. A difference of the order of the estimate itself would
  mean the path and the pooled fit are not on one panel, and the summary
  says so in those words if it appears.

SQL. None, provided lane 28a's cascade cache and 47L's counts are on the
share, which they are. If the cascade were missing it would be pulled
through 82's own query; nothing else here touches the database, so this
is safe beside any job.

Export: output_84a/84_summary.txt and occ_route_path.csv, with the two
vcov_s84_path_quarter_*.csv files.
"""

import os
import sys
from pathlib import Path

os.environ["CANARIES_84_SHAPES"] = "Q"
os.environ["CANARIES_84_OUT"] = "output_84a"
# 82 is imported for its score builder; its own OUT is pointed here so a
# run of this lane leaves no stray output_82 folder behind.
os.environ["CANARIES_82_OUT"] = "output_84a"
os.environ["CANARIES_RWORK_TAG"] = "_30a"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("84_occupation_route_path.py", "output_84a/84_summary.txt", 50),
]

if __name__ == "__main__":
    _lane.run("30a", STAGES)
