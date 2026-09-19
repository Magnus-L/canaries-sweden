#!/usr/bin/env python3
"""
run_lane3.py -- LANE 3 of 3. Submit this file to BatchClient.

THE DIAGNOSIS LANE, and the long one. 47h establishes WHICH features of an
education-based classifier survive register lag; 47i and 47j then read its
caches, which is why all three sit in one lane and run in order.

  47h eight education designs, each through the backtest  ~330 min
  47i firm-mix exposure (supporting evidence)              ~12 min
  47j within-employer triple difference (supporting)       ~12 min

About six hours. If it is killed, resubmit: 47h resumes from its caches and
the finished stages are skipped.

Note on what 47i and 47j are FOR, after the cross-vendor review: their
backtest artefact is near zero almost by construction, because their
exposure is fixed in 2019 and their outcome uses only payroll and birth
year. That is an invariance check, not a validation. They are supporting
evidence and the summaries say so.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("47h_edu_horserace.py",          "output_47h/47h_summary.txt",  330),
    ("47i_firmmix.py",                "output_47i/47i_summary.txt",   12),
    ("47j_within_employer_triple.py", "output_47j/47j_summary.txt",   12),
]

if __name__ == "__main__":
    _lane.run("3", STAGES)
