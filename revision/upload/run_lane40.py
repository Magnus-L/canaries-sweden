#!/usr/bin/env python3
"""
run_lane40.py -- LANE 40, RAW HIRES AND SEPARATIONS ON THE PAPER'S SCORE.
Submit this file to BatchClient.

  109  Raw headcount, hires and separations by exposure quartile and age
       band on the occupation-route score (82.build_exposure via 85),
       2022-23 against Jan 2024 to Jun 2025, per month, plus the monthly
       hire rate by band. Gate: headcount must reproduce 85's Table A12
       export (output_85/occ_route_descriptive_full.csv) exactly.
       No SQL, no R.                                          ~5-15 min

Reads the caches L_baseline_2019*, L_counts_2021..2025, flows_2021..2025.
Output to output_109. Can run beside anything.
Tested locally in revision/local/test_109_occ_route_plain_flows.py (9 checks).
EXPORT: output_109/109_summary.txt, occ_route_plain_flows.csv.
"""
import os
import sys
from pathlib import Path

os.environ["CANARIES_109_OUT"] = "output_109"
os.environ["CANARIES_85_OUT"] = "output_109"
os.environ["CANARIES_82_OUT"] = "output_109"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("109_occ_route_plain_flows.py", "output_109/109_summary.txt", 15),
]

if __name__ == "__main__":
    _lane.run("40", STAGES)
