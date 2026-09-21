#!/usr/bin/env python3
"""
run_lane21.py -- LANE 21. Submit this file to BatchClient.

  75  the headline against the pre-hike months           ~150 min

WHY. Script 68 keeps its Riksbank term switched on through the whole
post period, so every post coefficient it reports is a step from the
level of April to November 2022, not from the months before the hike.
At 22-25 that sequence is +0.0214, -0.0143, -0.0408: the paper's "4.0 per
cent" is the ADDITIONAL step at adoption, and the level after adoption
against the pre-hike months is about -0.019 with no standard error,
because no covariance was exported. 75 redefines the Riksbank term as a
window and reports the level directly, with its SE, for the four fits
the paper quotes (stock at 22-25 and 26-30; hires and separations at
22-25). The R wrapper now also exports the clustered covariance of every
fit, so this class of question never needs a trip again.

READ RULE: 75's post must reconcile with 68's rb + post within one SE,
or nothing is quoted. The paper's headline stays 68's adoption step,
written as the step; 75's number goes in Table 1 as the level.

No SQL. Reads 47h's education caches and 47L's and 54's counts. Safe
beside anything. Export two files plus the covariances:
output_75/75_summary.txt, output_75/reference_window.csv,
output_75/vcov_s75_*.csv.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("75_reference_window.py", "output_75/75_summary.txt", 150),
]

if __name__ == "__main__":
    _lane.run("21", STAGES)
