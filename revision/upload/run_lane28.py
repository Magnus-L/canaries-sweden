#!/usr/bin/env python3
"""
run_lane28.py -- LANE 28. Submit this file to BatchClient.

  82  the paper's findings on a route with no education in it
                                                  about four hours for ABC

WHY. The paper's firm exposure is routed through education. An education
group is given the employment-weighted mean DAIOE generative-AI percentile
of the occupations its holders worked in during 2019, and an employer is
ranked by the mean of that score over its incumbents aged 31 to 69.
Education therefore carries the exposure from the occupation to the firm,
and a referee may reasonably ask how much of the result is the education
register rather than the work. Script 82 deletes the intermediate step: an
employer is ranked directly by the employment-weighted mean DAIOE
percentile of the 2019 four-digit occupations of its OWN incumbents aged
31 to 69. Same freeze year, same incumbent restriction, same person floor,
same employment-weighted quartile cuts, no education record anywhere. The
score is script 65's occupation_exposure, which is script 47j's
incumbent_exposure with the register changed and nothing else.

WHAT RUNS, AND FOR HOW LONG. Parts are chosen with CANARIES_82_PARTS
(default and the setting below: ABC) and the folder with CANARIES_82_OUT.

  A  the score and what it covers, no fit. Coverage of the 2019 code by
     age band, coded and scored; the employers each route can place, alone
     and on the two young panels; the occupation quartile against the
     education quartile with the diagonal share and the rank correlation;
     and the size and longevity of the employers one route places and the
     other does not. Fifteen to twenty minutes, nearly all of it the two
     panel rebuilds.
  B  the headline. Equation (2) on the stock at both young bands with the
     calendar terms, exactly as script 68 estimates it, then the six-band
     profile against 41-49 as script 74 builds it. Three fits, about an
     hour.
  C  the sex split, the margins and the vintage check. The sex
     specification of Equation (2) at 22-25, hires and separations at
     22-25, and the same 2019 incumbents re-scored from the 2021 register
     with both arms fitted on one panel. Five fits and one small pull,
     ninety minutes to two hours.

About four hours for ABC. A stage that dies costs its own part. Every
panel here is a SUBSET of the education route's: that route scores about
311,000 employers and this one about 65,000, so no fit in this lane is
larger than one that has already fitted, and these budgets are the
conservative end.

SQL. Part C pulls the November 2019 declarations once, joined to
Individ_2019 for the birth year and Individ_2021 for the occupation code,
and caches it as L_baseline_2019_asof2021.parquet; 47L's comparable pull
took fourteen seconds. Parts A and B pull nothing if 47L's and 47h's
caches are on the share. No monthly declarations are touched, so this is
safe beside any job.

THREE SLOTS ARE BETTER. The three parts share nothing but the caches, so
run_lane28a, 28b and 28c run them in three jobs at once: about two hours
of wall clock instead of four. Submit those three rather than this file;
this one is kept for the record and for a single-slot night.

READ RULES, fixed before the runs and printed by the script at the start
and in the summary. There is NO coefficient gate: this is a different
measure of the same object, so the estimates will differ from the
education route and that is expected. What is judged is whether the
paper's findings survive the change of route, on three questions settled
in advance.
  1. REPRODUCES if the adoption step at 22-25 is negative and
     distinguishable from zero at the five per cent level on employer
     clustering.
  2. THE PROFILE REPRODUCES if the 50-and-over band gains against 41-49
     and the young band is the lowest or second lowest of the six.
  3. THE SEX RESULT REPRODUCES if the female differential is negative and
     distinguishable from zero at the one per cent level.
Each verdict is reported explicitly and whatever the numbers are, and
every point estimate is reported beside the education-route one: -0.0408
(0.0150) at 22-25 and -0.0394 (0.0102) at 26-30; -0.0099 (0.0121) for
22-25 against 41-49 and +0.0589 (0.0062) for 50 and over; -0.0746 (0.0142)
for the female differential; +0.0787 (0.0203) for separations and -0.0032
(0.0363) for hires.

Output goes to output_82/. Export: output_82/82_summary.txt,
occ_route_coverage.csv, occ_route_headline.csv, occ_route_profile.csv,
occ_route_gender.csv, occ_route_flows.csv, occ_route_vintage.csv and the
vcov_s82_*.csv files (eight, small). Employer and person counts below five
are suppressed before anything leaves MONA.
"""

import os
import sys
from pathlib import Path

PARTS = "ABC"
os.environ.setdefault("CANARIES_82_PARTS", PARTS)
os.environ.setdefault("CANARIES_82_OUT", "output_82")
# One exchange directory per job, so lane 28 can run beside anything else.
os.environ["CANARIES_RWORK_TAG"] = "_28"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("82_occupation_route.py", "output_82/82_summary.txt", 240),
]

if __name__ == "__main__":
    _lane.run("28", STAGES)
