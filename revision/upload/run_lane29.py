#!/usr/bin/env python3
"""
run_lane29.py -- LANE 29. Submit this file to BatchClient.

  83  the rest of the paper on lane 28's occupation-route score
                                        about nine hours for ABCD

WHY. Lane 28 re-estimated the headline, the six-band age profile, the
margins and the sex split on a firm score built from the employer's own
2019 occupation mix, with no education record anywhere, and showed that
they survive the change of route. EVERYTHING ELSE in the paper that uses
the exposure quartile is still on the education definition: the first
stage against reported AI use, the descriptive counterpart of the
headline, the reference-window specification, the pre-launch drift, the
industry standard errors, the industry-by-age-by-month test and the
credit test. Until those move too, a table in the paper can mix two
definitions of the same treatment. This lane closes that gap, so that no
published number mixes two exposure definitions.

THE SCORE IS LANE 28'S AND IS NOT REBUILT. Every fit takes its quartile
from 82_occupation_route.build_exposure(), the primary arm: uniform3,
the backward cascade, a floor of five incumbent person-months. One
definition of the treatment variable, in one place. A second
construction would drift from the first the moment either was corrected,
and a published table would then carry two scores under one name. Lane
28 was refactored to expose that builder as a function; nothing about
the score itself changed, and lane 28's own dry run still passes
unaltered.

NOTHING ELSE IS COPIED EITHER. Script 71's two first-stage arms, script
66's descriptive cells, script 75's window term set, script 78's drift
term set, script 80's completed industry key, its clustered fits and its
industry-absorbed fits, and script 73's leverage builder, coverage gate
and credit arm are all IMPORTED AND CALLED. Lane 29 is those scripts with
one column changed, which is what a comparison requires; a copied term
list could drift away from the estimates the paper quotes.

WHAT RUNS, AND FOR HOW LONG. Parts are chosen with CANARIES_83_PARTS
(default and the setting below: ABCD) and the folder with
CANARIES_83_OUT.

  A  the first stage and the descriptive counterpart. NO PANEL FIT AT
     ALL, which is why it is first: it is much the cheapest of the four
     and it is the one that decides whether anything in this lane or in
     lane 28 may be quoted. Forty-five to sixty minutes.
  B  the remaining rows of Table 1: the reference window at both young
     bands, the pre-launch drift at both, and the industry clustering of
     the pooled fits and the sex fit on the completed key, each beside
     its own employer-clustered run. Ten fits, three to three and a half
     hours.
  C  the two robustness tests Results names: industry by age band by
     month at both bands beside a same-sample baseline, and the credit
     test on the employers with a 2019 balance sheet. Ten fits, two and a
     half to three hours.
  D  the firm-size robustness, asked for after lane 28a landed. The
     reliability of the firm score as a function of the number of
     incumbents behind it (no fit); the adoption step at a floor of sixty
     incumbent person-months at both bands, beside the reported floor of
     five; and the adoption step by firm-size tercile at both bands. The
     quartile is the national one in every arm and is never recut. Ten
     fits, two and a half to three hours.

About nine hours for ABCD. A stage that dies costs its own part.

SEPARATE SLOTS ARE BETTER. The four parts share nothing but the caches
they read, so run_lane29a, 29b, 29c and 29d run them in separate jobs:
about three and a half hours of wall clock instead of nine. MONA allows
three jobs at once, so submit 29a first -- it is much the shortest and it
is the one that settles the first stage -- then 29b and 29c, and 29d when
29a comes back. Submit those four rather than this file; this one is kept
for the record and for a single-slot night.

READ RULES, fixed before the runs and printed by the script at the start
and in the summary, with the education-route figure beside every
estimate. THERE IS NO COEFFICIENT GATE ANYWHERE EXCEPT the clustering
fits of Part B, where the coefficients must reproduce their own
employer-clustered run to four decimals, exactly as script 80 gates.

The one rule with a coefficient in it is the clustering gate above; read
rule 9, the firm-size robustness of Part D, PASSES if the adoption step at
22-25 at a floor of sixty incumbent person-months keeps its sign and sits
within one standard error of the reported one, and if no size tercile
carries the whole step.

The rule that governs the lane: THE FIRST STAGE REPRODUCES if the
occupation-route quartile predicts reported AI use with the same sign as
the education route and an association at least HALF its size, on the
2023 firm-level gap (education route +21.5 points) and on the 2024
individual gap (+23.5 points). IF IT DOES NOT REPRODUCE, NOTHING IN LANE
29 OR LANE 28 IS QUOTED, and the summary says so at the top in those
words. The other eight rules are in the four part runners and in the
script's own READ_RULES, and every one of them is reported explicitly
whichever way it falls.

SQL. In Part A, one INFORMATION_SCHEMA probe of the survey catalogue,
one read of each AI survey table it finds, and one read of
Arb_AGIIndivid202411_def to join the BITA respondents to their
employers. In Part C, one probe of the firm registers and one read of
the 2019 balance sheets. In Parts B and C, the completed industry key if
I_industry_key.parquet is not on the share. The occupation cascade and
the 2019 monthly counts are pulled only if lane 28a has not cached them,
which it has. No other monthly declarations are touched, so this is safe
beside any job.

Output goes to output_83/. Export: 83_summary.txt,
occ_rest_firststage.csv, occ_rest_firststage_overlap.csv,
occ_rest_descriptive.csv, occ_rest_window.csv, occ_rest_drift.csv,
occ_rest_cluster.csv, occ_rest_industry.csv, occ_rest_credit.csv,
occ_rest_size.csv, occ_rest_reliability.csv and the thirty vcov_*.csv
files (small). Employer, firm and person counts below
five are suppressed before anything leaves MONA.
"""

import os
import sys
from pathlib import Path

PARTS = "ABCD"
os.environ.setdefault("CANARIES_83_PARTS", PARTS)
os.environ.setdefault("CANARIES_83_OUT", "output_83")
os.environ.setdefault("CANARIES_82_OUT", "output_83")
# One exchange directory per job, so lane 29 can run beside anything else.
os.environ["CANARIES_RWORK_TAG"] = "_29"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("83_occupation_route_rest.py", "output_83/83_summary.txt", 560),
]

if __name__ == "__main__":
    _lane.run("29", STAGES)
