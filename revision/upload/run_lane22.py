#!/usr/bin/env python3
"""
run_lane22.py -- LANE 22. Submit this file to BatchClient.

  76  the female differential split by education track    ~300 min
  77  the young against 41-49 by education track           ~240 min

WHAT IT ANSWERS (ML, 22 Sep 2026). Two things the paper now says without
the number behind them. First, how much of the -0.0659 female
differential at 22-25 is young women being in different work from young
men inside the same exposed firms, and how much is being hit harder in
the same work: the decomposition the submitted paper carried on
occupation, rebuilt on education because the surviving design classifies
no young worker by occupation after 2019. Second, whether the pooled
null of the young against 41-49 hides a steeper gradient in particular
tracks, ICT above all: the cut is reported in full, as heterogeneity.

IT PULLS SQL ONCE. 76's first stage pulls counts by employer x age x sex
x education x month for 2021-2025 (67's query with the education columns
47h reads) and caches them as L_counts_sex_edu_YYYY; 77 reads the cache.
So 76 runs ALONE or beside read-only lanes, and 77 needs 76's cache
first, which the stage order guarantees. Budget the SQL at about an hour
and the two scripts at eight to nine hours together; resubmitting after
a kill is cheap because _lane skips a finished stage.

READ RULES are fixed in each script's docstring: a reproduction gate on
the all-worker fit (76 against 68's -0.0659; 77 against -0.0153), both
numbers of the split reported whichever way the verdict falls, and no
track promoted to a headline.

Export: output_76/76_summary.txt, education_mix_by_sex.csv,
gender_by_track.csv, gender_split.csv; output_77/77_summary.txt,
contrast_by_track.csv; and the vcov_s76_*.csv / vcov_s77_*.csv files.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("76_gender_decomposition.py", "output_76/76_summary.txt", 300),
    ("77_contrast_by_track.py", "output_77/77_summary.txt", 240),
]

if __name__ == "__main__":
    _lane.run("22", STAGES)
