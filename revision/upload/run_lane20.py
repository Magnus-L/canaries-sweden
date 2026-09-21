#!/usr/bin/env python3
"""
run_lane20.py -- LANE 20. Submit this file to BatchClient.

  74  the age profile, with and without the cycle        ~150 min

WHAT CHANGED SINCE THIS LANE LAST RAN, 21 September 14:01.

It ran then on THREE bands, 22-25, 26-30 and 41-49, 120,359 firms, and
its read rule fired: removing the calendar cycle took the 22-25 against
41-49 contrast from -0.0357 to -0.0153, below half its size AND out of
significance. The paper accepted that. It no longer claims the young are
distinctively hit relative to the prime-aged, and the title no longer
says "age gradient".

THIS RUN ASKS A DIFFERENT QUESTION, and it is exploratory.

Script 70 was restored to six bands at 20:38 and ran clean at 172,396
firms. So 31-34, 35-40 and 50+ now have a plain-arm contrast against
41-49, and on that arm the picture is 22-25 -0.0288, 26-30 -0.0146,
31-34 +0.0162, 35-40 +0.0087, 50+ +0.0616. None of those three middle
and older bands has EVER been estimated with the calendar cycle out.

This lane estimates all five with the cycle out, on one skeleton, so the
whole profile comes from one specification for the first time.

HOW TO READ WHAT COMES BACK.

The paper's base does not depend on this. It currently claims a
shortfall of the under-31s against the older workforce pooled, and no
gradient. So:

  * If the profile is flat or all nulls once the cycle is out, nothing
    changes and nothing is added to the paper. That is a perfectly good
    outcome and the likeliest one, since the cycle removal cut the one
    contrast we have tested by 57 per cent.
  * If an ordered profile survives, it can go in, and the framing and
    possibly the title can claim it.
  * Watch 50+ specifically. Its plain coefficient is +0.0616 with an SE
    of 0.0065, so it is the most likely to survive. A profile whose only
    robust feature is 50+ gaining is NOT a gradient in the young, and it
    would put the oldest band at the centre of a paper about the young.
    Do not reach for it.

The plain arm must still reproduce lane 16's six-band numbers within a
standard error. If it does not, the two runs are on different samples
and the summary says VOID.

Two fits on ONE skeleton, so the arms differ in the term list and in
nothing else. BANDS is set at the top of 74 and checked against 70's
CONTRAST_BANDS; the run aborts if they disagree.

No SQL. Reads 47h's education caches and 47L's counts. Safe beside
anything.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("74_contrast_seasonal.py", "output_74/74_summary.txt", 150),
]

if __name__ == "__main__":
    _lane.run("20", STAGES)
