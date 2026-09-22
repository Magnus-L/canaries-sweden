#!/usr/bin/env python3
"""
run_lane28c.py -- LANE 28c. Submit this file to BatchClient.

  82  part C: the sex split, the margins and the vintage check

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


THE SCORING ARM. Every incumbent is scored at THREE digits, from a book
built once as the 2019 national employment-weighted mean of the
four-digit DAIOE scores within each three-digit group, read from the
register's own three-digit column and never by truncating the four-digit
one. The workers lacking a four-digit code are not a random subset, so a
mixed four-then-three rule would give a sharp score to firms whose coding
is complete and a smoothed one to firms whose coding is not, and quartile
assignment would then depend partly on coding completeness: a bias
channel into the treatment variable, not merely noise. Under the uniform
rule the smoothing is common to every firm and the ranking survives it,
and from 2019 every coded occupation carries at least three digits, so
the uniform level is the near-complete one. mixed43 and four_only are
fitted beside it as robustness and settle nothing; the read rules are
read on uniform3 and on nothing else.

WHAT RUNS HERE.
  C  the sex specification of Equation (2) at 22-25 on script 67's panel,
     every term entered as High x Young, High x Female and
     High x Young x Female with sex-specific employer-by-age and
     month-by-age effects; hires and separations at 22-25 on script 54's
     flows with the same terms as the stock; and the vintage check in
     THREE arms, the reported backward cascade, the 2019 code alone and
     the same November 2019 incumbents re-scored from the 2021 register.
     Three and not two, because the as-of arm has no cascade: against the
     reported score it would sum the re-coding with the loss of the
     cascade's coverage and call the total an artefact. All three fit on
     one panel and on the employers all three can score, so the
     difference between them is the score and not the sample.
     Every fit is on the reported score, the uniform three-digit arm.
     Six fits and one small pull. The sex panel is the heavy one but it is
     a subset of the 40.5-million-row education-route panel, which fit in
     about fifteen minutes; the flows and the three vintage arms are
     smaller again. Budget two to two and a half hours. THE LONGEST OF THE
     THREE.

READ RULE, fixed before the run. No coefficient gate. Part C answers the
third question:
  3. THE SEX RESULT REPRODUCES if the female differential is negative and
     distinguishable from zero at the one per cent level.
The education route prints -0.0746 (0.0142) for the differential, +0.0787
(0.0203) for separations and -0.0032 (0.0363) for hires, and each is on
the record beside its counterpart here. The re-coding artefact and the
cascade's own contribution are reported separately, each against the 2019
code alone and whatever they are, beside the education route's own
re-scoring, which moved the adoption step +0.0113.

SQL. One read of Arb_AGIIndivid201911_def joined to Individ_2019 for the
birth year and Individ_2021 for the code, unless the
L_baseline_2019_asof2021 cache is already on the share; 47L's comparable
pull took fourteen seconds. Plus the cascade pull and the 2019 monthly
counts if lane 28a has not already cached them. No monthly declarations
beyond that, so this is safe beside any job.

Export: output_82c/82_summary.txt, occ_route_gender.csv,
occ_route_flows.csv, occ_route_vintage.csv and the six vcov_s82_*.csv
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
    ("82_occupation_route.py", "output_82c/82_summary.txt", 150),
]

if __name__ == "__main__":
    _lane.run("28c", STAGES)
