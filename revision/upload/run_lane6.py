#!/usr/bin/env python3
"""
run_lane6.py -- LANE 6. Submit AFTER lanes 4 and 5 have finished.

THE TIME AND ATTENUATION LANE. Neither script performs a pull that 47L or
54 has not already cached, apart from three baseline years in 57.

  57  baseline vintage and attenuation                          ~35 min
      Measures how fast the 2019 exposure assignment decays as a
      proxy for later years, and re-runs the design on a 2022
      baseline, which is still pre-ChatGPT but three years closer
      to where the effect is expected.

  56  the effect BY AGE and OVER TIME                            ~45 min
      Poisson event studies on the stock, on hires and on
      separations, referenced to 2022H1, with the 22-25
      differential estimated inside the same fit.

WHY THIS LANE EXISTS. A pooled coefficient cannot separate a shock that
arrived with ChatGPT from a trend already running in 2019, and a 2019
baseline is weakest in exactly the years where the effect is most
likely. Both are quantities, not caveats, and both are measured here.

57 RUNS FIRST on purpose: it writes the 2021, 2022 and 2023 baselines
that make its own comparison possible, and its lambda curve is what
tells you how much of 56's late flatness is attenuation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("57_baseline_vintage.py",  "output_57/57_summary.txt", 35),
    ("56_dynamics_by_age.py",   "output_56/56_summary.txt", 45),
]

if __name__ == "__main__":
    _lane.run("6", STAGES)
