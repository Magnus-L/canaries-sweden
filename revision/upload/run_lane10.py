#!/usr/bin/env python3
"""
run_lane10.py -- LANE 10. Submit this file to BatchClient.

  63  does the answer depend on the exposure measure?             ~2-3 h

Every design in this round assigns exposure from DAIOE at four-digit
occupation. Four designs that fail differently still fail together if
the measure is wrong, and their agreement would then be worth nothing.
This is the one weakness the design portfolio cannot see, and it is the
largest thing left open.

Three measures, the same panels and the same fixed effects:

  daioe     ours, the one every other design uses
  eloundou  a different team, a different method, the same object
  telework  Dingel and Neiman: NOT an AI measure. If teleworkability
            reproduces the age gradient then we are measuring office
            work, and the interpretation fails however clean the
            identification is.

The three correlate 0.66 to 0.87 across occupations, so the placebo
cannot be expected to return nothing. That is why the script also runs
the horse race, both measures in one regression, pooled and by age band.
The by-age daioe coefficient net of teleworkability is the number worth
quoting, and on the synthetic test it separates the two worlds cleanly
even at a correlation of 0.83.

READS ONLY CACHES: 47L's (L_baseline_2019, L_counts_*) and 54's
(flows_*). Needs eloundou_ssyk4.dta and dingel_neiman_ssyk4.dta in the
project input directory; both are already there.

Safe to run beside lanes 8 and 9.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("63_measure_robustness.py", "output_63/63_summary.txt", 150),
]

# NOTE FOR THE SECOND RUN (20 Sep, evening). The first run finished in 41.7
# minutes and its answer was empty, because it dated the treatment at the
# ChatGPT launch and at that dating the stock gradient at 22-25 is +0.008.
# This version runs every fit at the launch AND at January 2024, and adds
# separations so that a fall in hiring can be told apart from a fall in
# churn. RENAME the existing output_63 folder before resubmitting, or the
# lane sees 63_summary.txt and skips the stage.

if __name__ == "__main__":
    _lane.run("10", STAGES)
