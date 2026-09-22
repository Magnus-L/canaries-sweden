#!/usr/bin/env python3
"""
run_lane27a.py -- LANE 27a. Submit this file to BatchClient.

  80  part A: the industry key, and what it covers

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
  A  the key, and no fit at all. Ftg_2019 first, then Ftg_2018, Ftg_2020,
     2017, 2021, 2016, 2022, 2015, 2023, then Serrano's own firm table at
     the 2019 accounting year, then the business register at reference
     year 2019. The first source that answers keeps the firm, and the
     source that answered travels with it, so Part B and Part C can say
     how many of their firms carry a code from somewhere other than 2019.
     Then the two stock panels and the sex panel are rebuilt exactly as
     the fits build them, and the export says how many of those employers
     each step places, and how large and how long lived the ones Ftg_2019
     missed are. If the reference-date story is right they are small and
     short lived, and the summary says so or says the opposite.
     Runtime: forty-five to sixty minutes, nearly all of it the three
     panel rebuilds; the eleven pulls are a few minutes together.
     This is the shortest of the three but it is also the one that writes
     the cached key, so submit it first if the three must be staggered.

SQL. Up to nine reads of Ftg_2015 to Ftg_2023 (three columns each), one of
Serrano_Serrano_20230614 at ser_year 2019, one of FDB_JE_2014_2021 at
ar 2019, and one INFORMATION_SCHEMA probe. No declarations, so this is safe
beside any job.

Export: output_80a/80_summary.txt and industry_key_coverage.csv. Employer
counts below five are suppressed before anything leaves MONA, and a share
is suppressed with its own numerator.
"""

import os
import sys
from pathlib import Path

os.environ["CANARIES_80_PARTS"] = "A"
os.environ["CANARIES_80_OUT"] = "output_80a"
os.environ["CANARIES_RWORK_TAG"] = "_27a"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("80_industry_key.py", "output_80a/80_summary.txt", 60),
]

if __name__ == "__main__":
    _lane.run("27a", STAGES)
