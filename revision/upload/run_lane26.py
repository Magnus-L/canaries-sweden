#!/usr/bin/env python3
"""
run_lane26.py -- LANE 26. Submit this file to BatchClient.

  79  the last three gaps                              about five hours for ABC

WHY. Script 78 answered seven questions and opened three. First, its Part C
showed that the 22-25 adoption step is not distinguishable from zero once
the standard errors allow common three-digit industry disturbances, and the
paper's sex result is estimated on the same employers and has only
employer-clustered standard errors. Second, its Part F put the calendar
terms and industry-by-age-by-month effects in one fit at 22-25 and nowhere
else, and 26-30 is now the band whose step the paper leans on. Third, the
editor's coverage objection has a version we have never quantified: a
worker who reaches no individual register is not counted at all, and if
those workers sit in exposed firms and at young ages, part of our fall is
measurement. Script 79 answers all three.

WHAT RUNS, AND FOR HOW LONG. Parts are chosen with CANARIES_79_PARTS
(default and the setting below: ABC).

  A  the sex panel of 67, every term of Equation (2) interacted with
     female, clustered by the employer's 2019 three-digit industry.
     THE HEAVIEST THING HERE: one Poisson fit on the 40,520,358-row sex
     panel, two to three hours. If lane 25's output_78b/gender_eq2.csv is
     on the share it supplies the employer-clustered numbers beside it; if
     it is not, that fit is repeated here and the part takes four to five
     hours instead. The summary says which happened.
  B  the 26-30 stock with the three calendar terms and industry by age by
     month effects, beside its own baseline on the same industry-linked
     sample. Two fits, about two hours, the second of them the heavier.
  C  no fit at all: one pull per year from 2019 to 2025 over the employer
     declarations joined to Individ_2023, 2021 and 2019, aggregated in the
     server to employer by status by band and cached per year. 47L's
     comparable counts pull ran at about seventy seconds a year, so budget
     well under an hour for the seven, plus the aggregation.

About five hours for ABC, or seven if Part A has to refit. A stage that
dies costs its own part; Part C caches each year as it lands, so a kill
costs the year that was running.

SQL. One read of LISA's Ftg_2019 for the industry code in each of A and B
(the read script 73 makes), one INFORMATION_SCHEMA probe of the January
declaration tables in C, and C's seven year pulls. Nothing else. Safe
beside a job that is not itself pulling declarations.

RUNS BESIDE LANE 25a. CANARIES_RWORK_TAG gives this job its own R exchange
directory on the batch node, so lane 25a's fits and these cannot share an
input file or sweep each other's away.

Output goes to output_79/. Export: output_79/79_summary.txt,
gender_cluster_industry.csv, industry_seasonal_2630.csv,
uncounted_share.csv and the vcov_s79_*.csv files (small). Every count in
uncounted_share.csv is floored at five before it is written.
"""

import os
import sys
from pathlib import Path

PARTS = "ABC"
os.environ.setdefault("CANARIES_79_PARTS", PARTS)
os.environ.setdefault("CANARIES_79_OUT", "output_79")
# One exchange directory per job, so lane 26 can run beside lane 25a.
os.environ["CANARIES_RWORK_TAG"] = "_26"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("79_last_gaps.py", "output_79/79_summary.txt", 300),
]

if __name__ == "__main__":
    _lane.run("26", STAGES)
