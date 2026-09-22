#!/usr/bin/env python3
"""
run_lane27c.py -- LANE 27c. Submit this file to BatchClient.

  80  part C: the industry-by-age-by-month test redone on the complete key

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
  C  script 78's Part F at 22-25 and script 79's Part B at 26-30, both on
     the complete key: the stock fit with the three calendar terms and
     three-digit industry by age band by month effects, the month-by-age
     effect dropped because it is nested inside them, beside a baseline on
     the same industry-linked sample. Employers the cascade cannot resolve
     leave the sample, as they did in 78 and 79, so both fits run on
     exactly the same firms and the ratio between them is a retained share
     rather than a sample difference.
     Four fits, two per band, the industry one the heavier of each pair.
     Lane 25's pair at 22-25 took about forty-five minutes and lane 26's at
     26-30 about twenty, and both samples are now larger, so budget one and
     a half to two hours.

READ RULE: no gate. The retained share of the adoption step against the
same-sample baseline is reported whatever it is, and so is the share of
firms in each fit whose code came from a source other than Ftg_2019. That
second number is part of the reading, not decoration: a carried-forward
code is noisier than a contemporaneous one, noise in the absorbing
dimension makes it absorb less, and a retained share that rose for that
reason would flatter us.

SQL. None, if lane 27a has already cached the key; the eleven pulls of the
cascade otherwise. No declarations either way.

Export: output_80c/80_summary.txt, industry_seasonal_v2.csv and the four
vcov_s80_*.csv files.
"""

import os
import sys
from pathlib import Path

os.environ["CANARIES_80_PARTS"] = "C"
os.environ["CANARIES_80_OUT"] = "output_80c"
os.environ["CANARIES_RWORK_TAG"] = "_27c"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("80_industry_key.py", "output_80c/80_summary.txt", 120),
]

if __name__ == "__main__":
    _lane.run("27c", STAGES)
