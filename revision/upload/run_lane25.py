#!/usr/bin/env python3
"""
run_lane25.py -- LANE 25. Submit this file to BatchClient.

  78  seven checks on the headline design               ~11 hours for ABCDEFG

WHY. The vetting round of 22 September (three referees and a cross-vendor
pass) asked for things the paper's design can answer from the caches
already on the share: a pre-period exhibit for the surviving design (A),
the sex split on Equation (2) exactly so the sexes share Table 1's base
(B), standard errors clustered by three-digit industry beside the employer
ones (C), a skill-intensity placebo that asks whether a 2019 post-secondary
cut reproduces the adoption step (D), the profile with 50 and over split
at the pension age (E), industry x age x month effects beside the calendar
terms (F), and the tightening boundary moved to May 2022, when the
increase took effect (G).

WHICH PARTS RUN. Edit PARTS below before submitting. Rough budgets on the
full panel, one fit at a time: A about 60 minutes (two path fits with up
to twenty-six quarter terms, two small pre-launch fits), B 60 to 120 (one
fit on the 40-million-row sex panel), C 60 to 120 (one fit per band by
industry; the employer-clustered comparison is read from output_68 if it
is on the share, else fitted once more), D about 120 (three fits per
band), E 120 to 180 (one counts pull per panel year, cached as
L_counts_split_YYYY, then one seven-band fit with thirty terms), F about
120 (one baseline and one fit with four effects, the heaviest here), G
under 60 (one fit). Suggested order if time is short: A, B, C, D first
("ABCD", about six hours), E, F and G in a second submission ("EFG").

THE PRE-PERIOD. 47L pulled counts for 2019 to 2025 (its YEARS), so
L_counts_2019 and L_counts_2020 should be on the share. If they are, Part
A(i)'s path runs from 2019Q1 with 2022Q1 as the reference; if they are
not, it runs from 2021Q1, the summary says so, and extending it costs two
more year pulls of 47L's query (the per-year time is in 47L's own log).
The drift test always uses January 2021 to November 2022.

No SQL except one read of LISA's firm table Ftg_2019 for the industry
code (C and F, the read script 73 makes) and, in E, one counts pull per
panel year, cached. Reads 47h's, 47L's and 67's caches. Safe beside
anything. Output goes to output_78/. Export: output_78/78_summary.txt,
prepath_plain.csv, predrift.csv, gender_eq2.csv, cluster_industry.csv,
skill_placebo.csv, prof_split.csv, industry_seasonal.csv,
boundary_may.csv and the vcov_s78_*.csv files (small).
"""

import os
import sys
from pathlib import Path

PARTS = "ABCDEFG"
os.environ.setdefault("CANARIES_78_PARTS", PARTS)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("78_final_checks.py", "output_78/78_summary.txt", 660),
]

if __name__ == "__main__":
    _lane.run("25", STAGES)
