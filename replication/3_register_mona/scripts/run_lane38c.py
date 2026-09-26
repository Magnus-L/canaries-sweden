#!/usr/bin/env python3
"""
run_lane38c.py -- LANE 38c, MONTH-OF-YEAR SEASONALITY (the deferred box
in the response letter, R1.6). Submit this file to BatchClient.

  102  Table 1's tau and the female differential re-estimated with
       eleven month-of-year terms (December omitted) in place of the
       three calendar-quarter terms, on the same panels, clustered by
       employer and by three-digit industry; the gate specification
       under both clusterings beside them. Eight fits.        ~2-2.5 h

SQL: none. Every frame is cached from lane 37b (L_counts_*,
L_counts_sex_*, the score caches, I_industry_key). Output to output_102.

THE GATES. Table 1 at 22-25 within 0.0005 (later -0.0578 (0.0155), tau
-0.0399 (0.0102)) and the female differential (-0.0858 (0.0142), tau
-0.0714 (0.0109)); a miss stops the script.

Tested locally in revision/local/test_102_month_of_year.py.
EXPORT: output_102/102_summary.txt, month_of_year.csv;
vcov_s102_*.csv (tier 2).
"""
import os
import sys
from pathlib import Path

os.environ["CANARIES_102_OUT"] = "output_102"
os.environ["CANARIES_82_OUT"] = "output_102"
os.environ["CANARIES_80_OUT"] = "output_102"
os.environ["CANARIES_73_OUT"] = "output_102"
os.environ["CANARIES_RWORK_TAG"] = "_38c"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("102_month_of_year.py", "output_102/102_summary.txt", 140),
]

if __name__ == "__main__":
    _lane.run("38c", STAGES)
