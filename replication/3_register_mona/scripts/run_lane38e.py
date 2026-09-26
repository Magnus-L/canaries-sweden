#!/usr/bin/env python3
"""
run_lane38e.py -- LANE 38e, THE PAYMENT RULE BEHIND THE PERSON-MONTH
COUNTS. Submit this file to BatchClient.

  104  One SQL pass over the monthly employer declarations, January 2021
       to June 2025 and the twelve months of 2019, collapsed to the
       paper's unit (a distinct person with a record from the employer in
       the month, birth year known, age 22 to 69): the share of counted
       person-months with cash pay subject to employer contributions, by
       age band, exposure group and period; the top-minus-rest difference
       and its change from the interim to the later period; among records
       without cash pay, the share with a pension amount or a taxable
       benefit (columns probed first); the same for the 2019 scoring
       population (person-months aged 31 to 69, by exposure group); and
       a year x band x group reconciliation. Gates: the counts reproduce
       47L's L_counts (2021-2025) and 82's L_counts_2019 to 0.01 per
       cent.                                            ~1-1.5 h, no R

SQL: yes (the declaration tables and Individ_2023/2021/2019). Caches read:
L_counts_2019 and 2021-2025 (gates), 82's score caches (exposure quartile).
Output to output_104. No new cache; nothing is written to the share.

Tested locally in revision/local/test_104_payment_rule.py.
EXPORT: output_104/104_summary.txt, payment_rule.csv, payment_rule_by_year.csv,
payment_rule_2019_incumbents.csv.
"""
import os
import sys
from pathlib import Path

os.environ["CANARIES_104_OUT"] = "output_104"
os.environ["CANARIES_82_OUT"] = "output_104"
os.environ["CANARIES_80_OUT"] = "output_104"
os.environ["CANARIES_73_OUT"] = "output_104"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("104_payment_rule.py", "output_104/104_summary.txt", 80),
]

if __name__ == "__main__":
    _lane.run("38e", STAGES)
