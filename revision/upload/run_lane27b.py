#!/usr/bin/env python3
"""
run_lane27b.py -- LANE 27b. Submit this file to BatchClient.

  80  part B: the industry clustering redone on the complete key

WHY THREE RUNNERS. Script 80 repairs one defect and then re-runs the two
exercises that rested on it (see run_lane27.py for the defect and the
cascade). The three parts share nothing but the caches they read and the
key itself, which the first of them to need it caches. MONA allows three
batch jobs at once, so each part has its own runner, its own output folder
and its own R exchange directory, and the three run together rather than in
sequence: about four hours of wall clock instead of six. Submit 27a, 27b
and 27c together. run_lane27.py runs all three in one job and is kept for
the record; do not submit it as well.

WHAT RUNS HERE.
  B  the industry-clustered standard errors of script 78's Part C (the
     pooled stock fit at 22-25 and at 26-30) and of script 79's Part A
     (the full sex specification at 22-25, every term of Equation (2)
     interacted with female), on the identical panels with the identical
     terms, so that the standard errors belong to the numbers already in
     Table 1. An employer the cascade cannot resolve joins ONE residual
     group rather than becoming a cluster of one, which is the hybrid this
     script exists to remove; the size of that group is reported.
     Three fits. On lanes 25 and 26 the two stock fits ran in fifteen to
     thirty minutes each and the 40,520,358-row sex panel in two to three
     hours, so budget three to four hours. THE LONGEST OF THE THREE, so
     submit it first if the three must be staggered and 27a has already
     written the key. If lane 25's output_78b/cluster_industry.csv and
     gender_eq2.csv are not on the share, the employer-clustered runs are
     repeated here and the part takes five to six hours instead; the
     summary says which happened.

READ RULE, fixed before the run: the coefficients must reproduce the
employer-clustered run to four decimals or nothing from this part is
quoted. Clustering changes the covariance and nothing else, so a moved
coefficient means a moved panel. The new standard errors are reported
beside the employer-clustered ones and beside the earlier hybrid ones
whatever they show.

SQL. None, if lane 27a has already cached the key; the eleven pulls of the
cascade otherwise. No declarations either way.

Export: output_80b/80_summary.txt, cluster_industry_v2.csv and the three
vcov_s80_*.csv files, which is what gives the women's step, the male step
plus the differential, a standard error of its own.
"""

import os
import sys
from pathlib import Path

os.environ["CANARIES_80_PARTS"] = "B"
os.environ["CANARIES_80_OUT"] = "output_80b"
os.environ["CANARIES_RWORK_TAG"] = "_27b"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("80_industry_key.py", "output_80b/80_summary.txt", 240),
]

if __name__ == "__main__":
    _lane.run("27b", STAGES)
