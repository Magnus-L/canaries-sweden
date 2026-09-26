#!/usr/bin/env python3
"""
run_lane38b.py -- LANE 38b, POST-FIT SUPPORT OF THE BACKTEST ARMS.
Submit this file to BatchClient.

  98  the three-arm as-of backtest rerun, unchanged in every estimate,
      with the observations each fit RETAINED reported beside the input
      rows. The R wrappers uploaded with lane 38a write n_obs_fit
      (nobs(fit), after fixest drops all-zero fixed-effect groups and
      singletons); 98 now prints "used by the fit / dropped by PPML" from
      it, and checks that the stacked fit and the three separate
      harmonised fits share their support. Until this run the "dropped by
      PPML" count was taken before the fit and read 0 in every row.
                                                            ~1.8 h

SQL: none. Every frame 98 needs is cached from lane 37b
(panel_dual_T2021, panel_dual_T2022, L_counts_*). Output goes to
output_98b so lane 37b's export stays as filed.

THE GATE. Table 1 at 22-25 on the paper's panel within 0.0005 (tau
-0.0399 (0.0102)), then 45's published arms; a miss stops the script.

EXPORT: output_98b/98_summary.txt, backtest_common.csv,
vcov_s98_stacked_T2021.csv, vcov_s98_stacked_T2022.csv.
"""
import os
import sys
from pathlib import Path

os.environ["CANARIES_98_OUT"] = "output_98b"
os.environ["CANARIES_RWORK_TAG"] = "_38b"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("98_backtest_common.py", "output_98b/98_summary.txt", 110),
]

if __name__ == "__main__":
    _lane.run("38b", STAGES)
