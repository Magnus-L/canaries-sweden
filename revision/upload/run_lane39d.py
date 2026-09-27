#!/usr/bin/env python3
"""
run_lane39d.py -- LANE 39d, WHICH EXPOSURE INDEX PREDICTS REPORTED AI USE.
Submit this file to BatchClient.

  108  DAIOE against the Eloundou rating (GPT-4-rated beta) as predictors
       of reported AI use and of language generation in Statistics
       Sweden's firm surveys (ITFtg, ai_* tables) and the 2024 worker
       survey (BITA), through script 71's own arms: every DAIOE-scored
       employer (the gate, 20.87 points in 2023), both indices on the
       employers both score, and the two disagreement routes (top on one
       index only). No R. SQL on the survey tables.        ~20-40 min

Reads 82's score caches, L_baseline_2019, eloundou_ssyk4.dta at the
project root. Output to output_108. SQL only, so it can run beside an R lane.
Tested locally in revision/local/test_108_adoption_by_index.py (16 checks).
EXPORT: output_108/108_summary.txt, adoption_by_index.csv, adoption_counts.csv.
"""
import os
import sys
from pathlib import Path

os.environ["CANARIES_108_OUT"] = "output_108"
os.environ["CANARIES_103_OUT"] = "output_108"
os.environ["CANARIES_82_OUT"] = "output_108"
os.environ["CANARIES_80_OUT"] = "output_108"
os.environ["CANARIES_73_OUT"] = "output_108"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("108_adoption_by_index.py", "output_108/108_summary.txt", 40),
]

if __name__ == "__main__":
    _lane.run("39d", STAGES)
