#!/usr/bin/env python3
"""
run_lane1.py -- LANE 1 of 3. Submit this file to BatchClient.

THE ANSWER LANE. Everything here has its own SQL and shares no cache with
the other lanes, so all three can run at once.

  50  calibration and the predictive-validation table   ~20 min
  47L age-specific baseline exposure: the design the     ~75 min
      cross-vendor review identified, and the one that
      keeps the paper's question without classifying
      any current worker
  51  occupation-code vintage, with the clerical cut     ~6 min
      (this one is for AI UNBOXED, not for canaries;
      delete it from STAGES below if you would rather
      not have it in this round)
  48  the gender split, with the char/int fix            ~40 min

About two and a half hours. Independent of lanes 2 and 3.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("50_sim_moments.py",            "output_50/50_summary.txt",   20),
    ("47L_age_baseline_exposure.py", "output_47L/47L_summary.txt", 75),
    ("51_vintage_ai_unboxed.py",     "output_51/51_summary.txt",    6),
    ("48_gender_poisson.py",         "output_48/48_summary.txt",   40),
]

if __name__ == "__main__":
    _lane.run("1", STAGES)
