#!/usr/bin/env python3
"""
run_lane30.py -- LANE 30, both shapes in one job. KEPT FOR THE RECORD.

Submit run_lane30a.py and run_lane30b.py instead. This file runs the
quarterly path and the monthly path in one slot, about two hours, and
exists for a night when only one slot is free and both are wanted. If
only one slot is free and a choice must be made, submit 30a: the
quarterly path is Figure 3, and the monthly path is an appendix
diagnostic that carries no verdict.

Do not submit this file beside 30a or 30b. The three write the same
export name into different folders, and two jobs building the same score
at once would pull the cascade twice for nothing.

See run_lane30a.py for why lane 30 exists, what the score is, where the
terms come from and the two dating rules, all of which apply here
unchanged.
"""

import os
import sys
from pathlib import Path

os.environ["CANARIES_84_SHAPES"] = "QM"
os.environ["CANARIES_84_OUT"] = "output_84"
os.environ["CANARIES_82_OUT"] = "output_84"
os.environ["CANARIES_RWORK_TAG"] = "_30"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("84_occupation_route_path.py", "output_84/84_summary.txt", 120),
]

if __name__ == "__main__":
    _lane.run("30", STAGES)
