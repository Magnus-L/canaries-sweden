#!/usr/bin/env python3
"""
run_lane29a.py -- LANE 29a. Submit this file to BatchClient.

  83  part A: does the occupation-route score predict reported AI use,
      and what did an exposed employer actually experience

WHY LANE 29 EXISTS. Lane 28 showed that the headline, the age profile,
the margins and the sex split survive a firm score built from the
employer's own 2019 occupation mix, with no education record anywhere.
EVERYTHING ELSE in the paper that uses the exposure quartile is still on
the education definition: the first stage against reported AI use, the
descriptive counterpart of the headline, the reference-window
specification, the pre-launch drift, the industry standard errors, the
industry-by-age-by-month test and the credit test. Until those move too,
a table in the paper can mix two definitions of the same treatment.
Script 83 closes that gap, so that no published number mixes two
exposure definitions.

THE SCORE IS LANE 28'S AND IS NOT REBUILT. Every fit in lane 29 takes
its quartile from 82_occupation_route.build_exposure(), the primary arm:
uniform3, the backward cascade, a floor of five incumbent person-months.
One definition of the treatment variable, in one place. A second
construction would drift from the first the moment either was corrected,
and a published table would then carry two scores under one name. Lane
28 was refactored to expose that builder; nothing about the score
changed.

WHY THREE RUNNERS. The three parts share nothing but the caches they
read. MONA allows three batch jobs at once, so each part has its own
runner, its own output folder and its own R exchange directory. Submit
29a first: it is much the shortest, it is the only one that touches the
survey tables, and it is the one that decides whether anything else may
be quoted. run_lane29.py runs all three in one job and is kept for the
record; do not submit it as well.

WHAT RUNS HERE.
  A  two arms, and NOT ONE PANEL FIT, which is why this part is first:
     it is the cheapest of the three and it is the one that settles
     whether the rest is worth reading.
     (i) THE FIRST STAGE, script 71's two arms run on the occupation
         quartile: the firm-level association between the quartile and
         reported AI use in the 2023 ITFtg wave, the individual-level
         association in the 2024 BITA wave, and the same firm-level
         association in every earlier wave the delivery holds, which is
         how the pre-ChatGPT gap of 2019, 2021 and 2023 is read. BOTH
         ROUTES ARE RUN ON THE SAME TABLES, so the comparison is a
         contrast within one regression sample rather than between two
         runs, and the education route's recorded figures are printed
         beside its re-estimate here as a check that the two agree.
     (ii) THE DESCRIPTIVE COUNTERPART of script 66: the mean headcount
         per employer-band-month cell by exposure quartile and age band,
         in the pre window (January 2022 to December 2023) and in the
         adoption window (January 2024 onward), and the change between
         them. Raw means; nothing is controlled for.
     Runtime: forty-five to sixty minutes. Most of it is the one read of
     the November 2024 declarations that joins the BITA respondents to
     their employers, and the two passes over the counts frame that the
     descriptives make. If lane 28a has already cached the occupation
     cascade and the 2019 monthly counts, which it has, neither is
     pulled again.

READ RULES, fixed before the run and printed by the script at the start
and in the summary, with the education-route figure beside every
estimate. There is no coefficient gate in this part.

  1. THE FIRST STAGE, AND IT DECIDES WHETHER ANYTHING IN LANE 29 OR LANE
     28 MAY BE QUOTED AT ALL. THE FIRST STAGE REPRODUCES if the
     occupation-route quartile predicts reported AI use with the same
     sign as the education route and an association at least HALF its
     size, on the 2023 firm-level gap (education route +21.5 points) and
     on the 2024 individual gap (+23.5 points). Both must hold; if one
     does it is PARTLY and the summary names which. IF IT DOES NOT
     REPRODUCE, NOTHING IN EITHER LANE IS QUOTED, and the summary says
     so at the top in those words: a score that does not predict who
     uses AI is not an AI exposure measure, whatever its coefficients
     do.
     Script 71's own sample gates travel with 71's code -- 800 firms
     with an outcome and 100 of them exposed, 600 respondents and 80
     exposed. This route scores fewer employers than the education
     route, so BELOW THRESHOLD is a live outcome and is accepted as one:
     it is reported as NO VERDICT, never as a failure and never as a
     pass. Lane 28a scored 262,089 employers, so the firm arm should
     clear its gate comfortably; the individual arm depends on how many
     BITA respondents work for a scored employer.
  2. The pre-ChatGPT any-AI gap in 2019, 2021 and 2023 is a DIAGNOSTIC
     and settles nothing. The education route reports it flat at 20.3,
     19.8 and 21.5 points; ours is printed beside it whatever it is. No
     threshold is set on it, because none can be read off three
     coefficients with no interaction test behind them.
  3. The descriptives carry NO verdict. They are raw means, with
     composition, firm size and the business cycle inside them. The
     education route's shape is stated (exposed employers grew in every
     band and grew the oldest band fastest) and ours is described beside
     it in the same words.
  Employer, firm and person counts below five are suppressed before
  anything leaves MONA, and a share is suppressed with its own
  numerator.

SQL. One INFORMATION_SCHEMA probe of the survey catalogue; one read of
each AI survey table it finds; and one read of Arb_AGIIndivid202411_def
restricted to the person and employer columns, to join the BITA
respondents to their employers. Plus the occupation cascade and the 2019
monthly counts if lane 28a has not already cached them. No other monthly
declarations are touched, so this is safe beside any job.

Export: output_83a/83_summary.txt, occ_rest_firststage.csv,
occ_rest_firststage_overlap.csv and occ_rest_descriptive.csv.
"""

import os
import sys
from pathlib import Path

os.environ["CANARIES_83_PARTS"] = "A"
os.environ["CANARIES_83_OUT"] = "output_83a"
# 82 is imported for its score builder; its own OUT is pointed here so a
# run of this lane leaves no stray output_82 folder behind.
os.environ["CANARIES_82_OUT"] = "output_83a"
os.environ["CANARIES_RWORK_TAG"] = "_29a"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("83_occupation_route_rest.py", "output_83a/83_summary.txt", 60),
]

if __name__ == "__main__":
    _lane.run("29a", STAGES)
