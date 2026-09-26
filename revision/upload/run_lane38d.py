#!/usr/bin/env python3
"""
run_lane38d.py -- LANE 38d, THE HEADLINE DESIGN ON THE ELOUNDOU
CLASSIFICATION. Submit this file to BatchClient.

  103  Employers re-classified by the Eloundou et al. (2024) GPT-exposure
       rating through the same chain as the paper's DAIOE score (2019
       cascade, three-digit book, incumbent floor, employment-weighted
       quartile cut); Table 1's tau at 22-25 and 26-30 and the female
       differential re-estimated on the employers both indices score,
       DAIOE and Eloundou side by side, the Eloundou fits clustered by
       employer and by three-digit industry; plus how the two
       classifications agree. Ten fits.                     ~2.5-3 h

SQL: none. Every frame is cached from lane 37b (L_counts_*,
L_counts_sex_*, 82's cascade and baseline caches, I_industry_key); the
input file eloundou_ssyk4.dta is at the project root. Output to
output_103.

THE GATES. Table 1 at 22-25 within 0.0005 (later -0.0578 (0.0155), tau
-0.0399 (0.0102)) and the female differential (-0.0858 (0.0142), tau
-0.0714 (0.0109)); a miss stops the script.

Tested locally in revision/local/test_103_eloundou_classification.py.
EXPORT: output_103/103_summary.txt, eloundou_classification.csv,
classification_agreement.csv; vcov_s103_*.csv (tier 2).
"""
import os
import sys
from pathlib import Path

os.environ["CANARIES_103_OUT"] = "output_103"
os.environ["CANARIES_82_OUT"] = "output_103"
os.environ["CANARIES_80_OUT"] = "output_103"
os.environ["CANARIES_73_OUT"] = "output_103"
os.environ["CANARIES_RWORK_TAG"] = "_38d"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("103_eloundou_classification.py", "output_103/103_summary.txt", 170),
]

if __name__ == "__main__":
    _lane.run("38d", STAGES)
