#!/usr/bin/env python3
"""
run_lane30b.py -- LANE 30b. Submit this file to BatchClient.

  84  the MONTHLY path at 22-25 and at 26-30, on lane 28's
      occupation-route score

WHAT IT IS FOR. Online Appendix III.2 carries a monthly version of
Figure 3 as a diagnostic: it is what shows how much the 2025 endpoint
moves from month to month, and it is why the paper's figure is quarterly
and why the paper quotes no monthly coefficient. That diagnostic is
still the education route's, like the figure itself, and this lane moves
it across.

IT CARRIES NO VERDICT, and the script says so in the summary. The
reported specification removes the calendar cycle with quarter-of-year
terms, so a monthly coefficient retains whatever separates the month
from its own quarter's mean; that residual is small at 26-30 and large
at 22-25, which is the point the appendix makes with it. Nothing in the
paper is read off these numbers except that statement.

SUBMIT IT SECOND, or not at all this night. Lane 30a is the one that
matters: it is the figure. If only one slot is free, use it on 30a.

WHAT RUNS HERE.
  M  the monthly path at both young bands: every month from December
     2022 onward, eleven month-of-year terms with December omitted,
     employer-by-month, employer-by-age and month-by-age effects,
     Poisson, clustered by employer. Two fits, one per band, on the
     panels lane 28b already fitted.
     Runtime: forty-five to seventy minutes. The monthly shape carries
     about thirty more terms than the quarterly one, which costs time
     but not memory: what binds a fit is the number of fixed effects,
     not the number of terms.

SQL. None, provided lane 28a's cascade cache and 47L's counts are on the
share, which they are.

Export: output_84b/84_summary.txt and occ_route_path.csv, with the two
vcov_s84_path_month_*.csv files.
"""

import os
import sys
from pathlib import Path

os.environ["CANARIES_84_SHAPES"] = "M"
os.environ["CANARIES_84_OUT"] = "output_84b"
os.environ["CANARIES_82_OUT"] = "output_84b"
os.environ["CANARIES_RWORK_TAG"] = "_30b"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("84_occupation_route_path.py", "output_84b/84_summary.txt", 70),
]

if __name__ == "__main__":
    _lane.run("30b", STAGES)
