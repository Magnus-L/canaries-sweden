#!/usr/bin/env python3
"""
run_lane28c.py -- LANE 28c. Submit this file to BatchClient.

  82  part C: the sex split, the margins and the vintage check

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
  C  the sex specification of Equation (2) at 22-25 on script 67's panel,
     every term entered as High x Young, High x Female and
     High x Young x Female with sex-specific employer-by-age and
     month-by-age effects; hires and separations at 22-25 on script 54's
     flows with the same terms as the stock; and the vintage check, in
     which the same November 2019 incumbents are re-scored from the
     occupation code the 2021 Individ register holds for them, both arms
     fitted on one panel so the difference is the re-scoring and not the
     sample.
     Five fits and one pull. The sex panel is the heavy one but it is a
     subset of the 40.5-million-row education-route panel, which fit in
     about fifteen minutes; the flows and the two vintage arms are
     smaller again. Budget ninety minutes to two hours. THE LONGEST OF
     THE THREE, so submit it first if the three must be staggered and 28a
     has already run.

READ RULE, fixed before the run. No coefficient gate. Part C answers the
third question:
  3. THE SEX RESULT REPRODUCES if the female differential is negative and
     distinguishable from zero at the one per cent level.
The education route prints -0.0746 (0.0142) for the differential, +0.0787
(0.0203) for separations and -0.0032 (0.0363) for hires, and each is on
the record beside its counterpart here. The vintage movement is reported
whatever it is, beside the education route's own re-scoring, which moved
the adoption step +0.0113.

SQL. One read of Arb_AGIIndivid201911_def joined to Individ_2019 for the
birth year and Individ_2021 for the code, unless the
L_baseline_2019_asof2021 cache is already on the share. 47L's comparable
pull took fourteen seconds. No monthly declarations are touched, so this
is safe beside any job.

Export: output_82c/82_summary.txt, occ_route_gender.csv,
occ_route_flows.csv, occ_route_vintage.csv and the five vcov_s82_*.csv
files. Employer counts below five are suppressed before anything leaves
MONA.
"""

import os
import sys
from pathlib import Path

os.environ["CANARIES_82_PARTS"] = "C"
os.environ["CANARIES_82_OUT"] = "output_82c"
os.environ["CANARIES_RWORK_TAG"] = "_28c"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("82_occupation_route.py", "output_82c/82_summary.txt", 120),
]

if __name__ == "__main__":
    _lane.run("28c", STAGES)
