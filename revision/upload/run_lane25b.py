#!/usr/bin/env python3
"""
run_lane25b.py -- LANE 25b. Submit this file to BatchClient.

  78  parts BC of the last checks                 B and C: the sex split on Equation (2) and the industry-clustered standard errors (two to four hours)

WHY THREE RUNNERS. Script 78 has seven switchable parts (see run_lane25.py
for what each one fits and why the vetting round asked for it). MONA
allows three batch jobs at once, so the parts are split across three
runners by runtime, each writing to its own folder so the summaries do not
overwrite each other: 25a runs BC, 25b runs BC, 25c runs EF. The parts
share nothing but the caches they read; only Part E (25c) pulls counts.
Submit all three together; export each folder's 78_summary.txt, its CSVs
and its vcov_s78_*.csv files.

Output: output_78b/. Reads 47h's, 47L's and 67's caches; SQL only as stated in
run_lane25.py for the parts here.
"""

import os
import sys
from pathlib import Path

os.environ["CANARIES_78_PARTS"] = "BC"
os.environ["CANARIES_78_OUT"] = "output_78b"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("78_final_checks.py", "output_78b/78_summary.txt", 300),
]

if __name__ == "__main__":
    _lane.run("25b", STAGES)
