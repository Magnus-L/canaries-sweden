#!/usr/bin/env python3
"""
run_lane4.py -- LANE 4. Submit this file to BatchClient.

THE TRIANGULATION LANE. Two designs whose weaknesses do not overlap, which
is the only reason running both is worth more than running either twice.

  53  the headline design on codes that were actually       ~95 min
      assigned in the observation year, 2019-2023.
      Keeps the paper's estimand (young vs young, inside
      the employer, monthly). Fails if freshness is
      differentially selected across the exposure
      dimension; does NOT depend on education at all.

  47L age-specific baseline exposure, RE-RUN for the age    ~25 min
      gradient added tonight. Exposure is frozen in 2019
      and the outcome needs only a birth year and a
      payslip. Fails if firms with more exposed young
      workers in 2019 were on different trends; does NOT
      depend on any post-2019 occupation code.
      Its caches are warm, so this is the fits only.

About two hours. Shares no SQL and no cache with lanes 2 or 3.

47L's summary already exists from the earlier run, so _lane would skip it.
The gradient is new, so this lane forces it: the stage list carries a
sentinel output that only tonight's version writes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("53_freshcode_panel.py",        "output_53/53_summary.txt",     95),
    # the gradient CSV, not the summary: the summary is already on disk from
    # the 19:24 run and would make _lane skip the stage that adds the
    # gradient.
    ("47L_age_baseline_exposure.py", "output_47L/agebase_gradient.csv", 25),
]

if __name__ == "__main__":
    _lane.run("4", STAGES)
