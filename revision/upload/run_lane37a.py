#!/usr/bin/env python3
"""
run_lane37a.py -- LANE 37a, MEASUREMENT (the Editor's condition; top
priority). Submit this file to BatchClient.

  99  the headline gate; the headline counts rebuilt from the raw
      declarations and compared cell by cell, tau on the rebuilt counts;
      the denominator reconciled from all declared person-months (by
      month, band, incumbent/new match/entrant, linkage, code source and
      vintage); tau for baseline incumbents and new matches     ~2.5-3.5 h

SQL, all cached so a rerun issues none: the raw rebuild (~3 min a year),
the reconciliation (one query a year joined to the previous year's
declarations, 10-20 min a year, 2020-2025), the baseline-match counts
(~2 min a year). The three-arm backtest of the same review lane runs in
lane 37b to balance the runtime.

THE GATE. Table 1 at 22-25 on the paper's panel within 0.0005 (tau
-0.0399 (0.0102)); a miss stops the script.

Tested locally in revision/local/test_99_measurement.py.
EXPORT: output_99/99_summary.txt, measurement_estimates.csv,
measurement_totals.csv, measurement_reconciliation.csv.
"""
import os
import sys
from pathlib import Path

os.environ["CANARIES_99_OUT"] = "output_99"
os.environ["CANARIES_82_OUT"] = "output_99"
os.environ["CANARIES_RWORK_TAG"] = "_37a"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("99_measurement.py", "output_99/99_summary.txt", 180),
]

if __name__ == "__main__":
    _lane.run("37a", STAGES)
