#!/usr/bin/env python3
"""
run_lane27.py -- LANE 27. Submit this file to BatchClient.

  80  the employer industry code completed        about six hours for ABC

WHY. Script 73 reads the employer's three-digit industry from LISA's firm
table for 2019 alone. That table holds 552,099 coded firms, and it is the
right source, but 5,285 of the 111,459 employers on the 22 to 25 panel and
9,106 of the 128,193 on the 26 to 30 panel are not in it. Scripts 73, 78
and 79 then gave each of those firms a cluster of its own, which is why the
industry-clustered runs report 5,545 and 9,368 clusters against 265 real
three-digit groups: nineteen clusters in twenty are one firm, and the
reported inference is a hybrid of industry and employer clustering rather
than the industry clustering the paper claims. The likely mechanism is a
reference-date mismatch: Ftg is LISA's firm table and LISA is built on a
November reference, while our panel comes from the monthly employer
declarations, so a firm that employed somebody earlier in 2019 but nobody
in November is in our panel and absent from Ftg_2019.

WHAT RUNS, AND FOR HOW LONG. Parts are chosen with CANARIES_80_PARTS
(default and the setting below: ABC).

  A  the key, and no fit. The cascade is Ftg_2019, then Ftg_2018, Ftg_2020
     and outward to 2015 and 2023, then Serrano's own firm table at 2019,
     then the business register. Nine small LISA reads, one Serrano read
     and one business-register read, then the two stock panels and the sex
     panel are rebuilt to say how many of THOSE employers each step
     places, and how large and how long lived the ones Ftg_2019 missed
     are. The panel rebuilds dominate: budget forty-five to sixty minutes,
     of which the SQL is a few.
  B  the clustering redone. Three Poisson fits: the 22-25 stock, the 26-30
     stock and the 40,520,358-row sex panel. On lanes 25 and 26 the two
     stock fits ran in fifteen to thirty minutes each and the sex panel in
     two to three hours, so budget three to four hours. If lane 25's
     cluster_industry.csv and gender_eq2.csv are on the share they supply
     the employer-clustered numbers beside the new ones; if they are not,
     those fits are repeated here and the part takes five to six hours.
     The summary says which happened.
  C  the industry effects redone. Four fits, two per band, the industry
     one the heavier of each pair. Lane 25's pair at 22-25 took about
     forty-five minutes and lane 26's at 26-30 about twenty, and both
     samples are now larger. Budget one and a half to two hours.

About six hours for ABC. A stage that dies costs its own part. The key
itself is cached as I_industry_key.parquet as soon as it is built, so a
part that runs after another reads it rather than pulling again.

SQL. Up to nine reads of Ftg_2015 to Ftg_2023 (three columns each), one of
Serrano_Serrano_20230614 at ser_year 2019, one of FDB_JE_2014_2021 at
ar 2019, and one INFORMATION_SCHEMA catalogue probe. No declarations are
touched, so this is safe beside any job.

THREE SLOTS ARE BETTER. The three parts share nothing but the caches and
the key, so run_lane27a, 27b and 27c run them in three jobs at once: about
four hours of wall clock instead of six. Submit those three rather than
this file; this one is kept for the record and for a single-slot night.

Output goes to output_80/. Export: output_80/80_summary.txt,
industry_key_coverage.csv, cluster_industry_v2.csv, industry_seasonal_v2.csv
and the vcov_s80_*.csv files (seven, small). Employer counts below five are
suppressed before anything leaves MONA.
"""

import os
import sys
from pathlib import Path

PARTS = "ABC"
os.environ.setdefault("CANARIES_80_PARTS", PARTS)
os.environ.setdefault("CANARIES_80_OUT", "output_80")
# One exchange directory per job, so lane 27 can run beside anything else.
os.environ["CANARIES_RWORK_TAG"] = "_27"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("80_industry_key.py", "output_80/80_summary.txt", 360),
]

if __name__ == "__main__":
    _lane.run("27", STAGES)
