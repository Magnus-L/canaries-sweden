#!/usr/bin/env python3
"""
run_lane33.py -- LANE 33. Submit this file to BatchClient.

  87  the female differential split into composition and within, on lane
      28's occupation-route score

WHY LANE 33 EXISTS. Table 1 carries one row the rest of the table does
not share: the part of the female differential that survives within broad
education tracks, -0.0508. It is script 76's, and 76 scores an employer
by its 2019 EDUCATION mix, so that row is the last estimate anywhere in
the paper still on the route we left. Printed under the occupation
route's -0.0858 it invites a division that means nothing: its own pooled
counterpart, on 76's panel and specification, is -0.0659, which is where
the paper's "three quarters" comes from. This lane puts the row and the
row above it on one measure.

IT DOES NOT USE A YOUNG WORKER'S OCCUPATION CODE, AND COULD NOT. The cut
stays education, and has to: the reported design classifies no young
worker by occupation after 2019, which is the point of freezing exposure
on the employer's incumbents aged 31 to 69, and the as-of backtest of
Part IV is what closed the designs that did classify them. Education is
the only register that can cut the young here. What changes is the
FIRM's score, from the 2019 education mix to the 2019 occupation mix of
its own incumbents aged 31 to 69.

TWO THINGS THAT COME WITH IT.
  1. The specification is Equation (2) itself, through 78's
     gender_eq2_terms, so the pooled differential this produces is the
     one Table 1 prints. 76 used 68's shorter term set, which is why its
     pooled was -0.0659.
  2. The weights move. The split weights young women's track shares in
     EXPOSED employers, and which employers are exposed is exactly what
     this change redefines, so the descriptive composition is rebuilt
     here rather than carried over. That also puts Online Appendix
     Tables A13 and A14 on this route, which they are not today.

THE GATE. The all-track differential is fitted on the education frame
collapsed over education and must reproduce Table 1's -0.0858 to four
decimals. A match is the evidence that this frame is the one Table 1
sits on; a miss means nothing here is quoted and the summary says so at
the top.

WHAT RUNS HERE. Six Poisson fits of the sex panel at 22-25: one on all
workers, which is the gate, and one inside each of the five broad tracks,
plus the descriptive composition, which is not a fit. 76 took 76 minutes
for the same six fits on the same machine, so budget ninety minutes.

SQL. NONE, and the script refuses to open a connection. The counts by
sex and education are 76's own pull and are cached from lane 21-22; if
they are not on the share the script stops rather than starting a full
read of the monthly declarations. Safe beside any job.

Export: output_87/87_summary.txt, occ_route_gender_split.csv,
occ_route_gender_by_track.csv, occ_route_education_mix_by_sex.csv and
the vcov_s87_* files. 76's export names are not reused, because two
exposure routes must never share an export name.
"""

import os
import sys
from pathlib import Path

os.environ["CANARIES_87_OUT"] = "output_87"
# 82 is imported for its score builder; its own OUT is pointed here so a
# run of this lane leaves no stray output_82 folder behind.
os.environ["CANARIES_82_OUT"] = "output_87"
os.environ["CANARIES_RWORK_TAG"] = "_33"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("87_occupation_route_gender_split.py", "output_87/87_summary.txt", 150),
]

if __name__ == "__main__":
    _lane.run("33", STAGES)
