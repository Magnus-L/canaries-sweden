#!/usr/bin/env python3
"""
run_lane13.py -- LANE 13. Submit this file to BatchClient.

  67  the gender split, on the design that survives      ~3-6 h, HAS SQL

"The decline is concentrated among young women" is in the introduction,
the highlights and the cover letter, and it rests on script 48, which
splits the occupation design the as-of backtest closed. It has to be
re-estimated on 61's design or removed from the paper.

This is the only script in the round that pulls SQL, because no cache
carries sex. Two pulls, monthly employment and monthly flows by
employer, age band and sex, both cached as L_counts_sex_YYYY and
flows_sex_YYYY so nothing repeats them. Budget the first run at three to
six hours and any re-run at well under one.

IT RUNS ALONE OR BESIDE READ-ONLY LANES, and it WRITES to the shared
cache, so do not start a second SQL job beside it.

What it produces that script 48 did not: a TEST of the gender
difference. 48 ran men and women separately and the paper then asserted
a concentration, which two separate coefficients cannot establish. The
primary specification here carries post x high x young x female, whose
coefficient is the difference and whose t is the test. Both datings,
all three margins, both young bands, and the register artefact measured
at the cell the claim is about.

If the differential is not significant, the honest sentence is that the
decline is present for both sexes and we cannot distinguish their
magnitudes. That is publishable and it is what 48 should have been
asked.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("67_gender_on_the_new_design.py", "output_67/67_summary.txt", 270),
]

if __name__ == "__main__":
    _lane.run("13", STAGES)
