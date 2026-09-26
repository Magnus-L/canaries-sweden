#!/usr/bin/env python3
"""
run_lane39c.py -- LANE 39c, THE PANDEMIC YEAR. Submit this file to
BatchClient.

  107  The female differential before 2021 on the sex panel from January
       2019: the backdated placebo of the sex specification (36 and 24
       months), the female drift from 2019, and the female quarterly path
       2019Q1 to 2022Q4; the pooled drift from 2019 with a pandemic-year
       term (March 2020 to February 2021); and, from the caches, the
       quarterly head counts of ages 22-25 and 31-69 by exposure group and
       by sex on the headline panel's employers, 2019Q1 to 2025Q2, so that
       the reader sees which group moved in 2020. Gate: the sex
       specification reproduces Table 1's female differential within
       0.0005. Six fits.                                       ~2-3 h

SQL: only if L_counts_sex_2019 / _2020 are not on the share, in which case
67's query pulls them first (about 15 minutes each) and caches them under
those names -- the only thing this lane writes to the share. Everything
else is cached (L_counts_2019-2025, L_counts_sex_2021-2025, 82's caches).
Output to output_107. R lane; do not run beside another R lane.

Tested locally in revision/local/test_107_pandemic_year.py.
EXPORT: output_107/107_summary.txt, pandemic_year.csv, female_prepath.csv,
quarterly_counts.csv; vcov_s107_*.csv (tier 2).
"""
import os
import sys
from pathlib import Path

os.environ["CANARIES_107_OUT"] = "output_107"
os.environ["CANARIES_82_OUT"] = "output_107"
os.environ["CANARIES_80_OUT"] = "output_107"
os.environ["CANARIES_73_OUT"] = "output_107"
os.environ["CANARIES_RWORK_TAG"] = "_39c"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("107_pandemic_year.py", "output_107/107_summary.txt", 170),
]

if __name__ == "__main__":
    _lane.run("39c", STAGES)
