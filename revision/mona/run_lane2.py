#!/usr/bin/env python3
"""
run_lane2.py -- LANE 2 of 3. Submit this file to BatchClient.

THE ESTIMAND LANE. 47k keeps the paper's question exactly: more versus less
exposed YOUNG workers, inside the same employer, month by month. It gets
there by restricting to young workers whose education record is already
correct rather than stale.

  47k settled-education subsample, three sampling rules,  ~135 min
      both truncations, with the backtest on each

It pulls its own copy of the year frames rather than reading 47h's, because
47h is running in lane 3 at the same time and reading a parquet that another
console is still writing would corrupt both. If 47h has FINISHED by the time
this starts, 47k detects that (its summary file) and reuses the caches
instead, which saves about 75 minutes.

Independent of lanes 1 and 3.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("47k_settled_sample.py", "output_47k/47k_summary.txt", 135),
]

if __name__ == "__main__":
    _lane.run("2", STAGES)
