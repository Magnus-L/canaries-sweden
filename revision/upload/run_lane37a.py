#!/usr/bin/env python3
"""
run_lane37a.py -- LANE 37a. Submit this file to BatchClient.

  95  pension ages: the profile split at 60 and 65, and the headline tau
      against older references that stop at 59 and at 49       ~2 h

SQL. One pull, 47L's counts query with eight age bands, 2021 to 2025,
cached as cache/L_counts_age8_YYYY (about 70 seconds a year). A rerun
reads the cache and issues no SQL.

THE GATE. The pull collapsed to the paper's six bands must reproduce
Table 1 within 0.0005 (22-25 tau -0.0399 (0.0102), 26-30 -0.0403
(0.0067)); a miss stops the script and nothing is quoted. Part P also
needs lane 31's split at 65 reproduced on 104,333 employers.

INDEPENDENT OF LANES 37b AND 37c: its own pull, its own cache names, its
own output folder and R exchange directory. Tested locally in
revision/local/test_95_pension_reference.py.

EXPORT: output_95/95_summary.txt and output_95/pension_reference.csv.
"""
import os
import sys
from pathlib import Path

os.environ["CANARIES_95_OUT"] = "output_95"
os.environ["CANARIES_82_OUT"] = "output_95"
os.environ["CANARIES_RWORK_TAG"] = "_37a"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("95_pension_reference.py", "output_95/95_summary.txt", 120),
]

if __name__ == "__main__":
    _lane.run("37a", STAGES)
