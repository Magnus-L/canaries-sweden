#!/usr/bin/env python3
"""
run_lane7.py -- LANE 7. Submit this file to BatchClient.

  58  the three things the 2025H1 result rests on                 ~35 min

      1. Is the H1-versus-H1 reading an artefact of choosing the
         slice after seeing the data? Two independent answers: the
         path rebased on its own season, and an H1-only estimate
         with no seasonal model at all.
      2. What are the standard errors? 56 printed none.
      3. Is the 2025 preliminary AGI still the only 2025 data? One
         metadata query lists every monthly AGI table with its
         vintage and row count.

No SQL beyond that one metadata query; everything else reads the
caches 47L and 54 already wrote. Safe to run beside anything.

READ THE PRE-COMMITTED RULE AT THE TOP OF THE LOG BEFORE THE NUMBERS.
It was written before the purged numbers existed and it can refuse.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("58_seasonal_and_vintage.py", "output_58/58_summary.txt", 35),
    ("59_monthly_path.py",         "output_59/59_summary.txt", 50),
    ("60_treatment_date.py",       "output_60/60_summary.txt", 60),
]

if __name__ == "__main__":
    _lane.run("7", STAGES)
