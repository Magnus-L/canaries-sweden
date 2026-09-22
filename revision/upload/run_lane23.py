#!/usr/bin/env python3
"""
run_lane23.py -- LANE 23. Submit this file to BatchClient.

  77  the young against 41-49 by education track            ~40 min

WHY A LANE OF ITS OWN. Lane 22 finished on 22 September at 07:12 with
one fit lost: the ICT contrast (s77_ict) died with rc=3221225477 at two
threads on a panel of 940,788 rows. Every other fit in 76 and 77 came
back and both reproduction gates passed. The retry ladder in
mona_common._run_r promised an attempt at one thread and made only one
retry; it now walks 8, 2, 1. This lane re-runs 77 alone (about 40 min,
six fits, ICT included) rather than resubmitting lane 22, which would
re-run 76's 76 minutes for nothing.

No SQL: 77 reads the L_counts_sex_edu_YYYY caches 76 wrote. Safe beside
anything. Export: output_77/77_summary.txt, output_77/contrast_by_track.csv,
and output_77/vcov_s77_ict.csv (the other five covariances are already
filed).

If the ICT fit dies again at one thread, the track is reported as not
estimable on this panel, with the row count and the reason, and the paper
says so in the transparency sentence.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("77_contrast_by_track.py", "output_77/77_summary.txt", 40),
]

if __name__ == "__main__":
    _lane.run("23", STAGES)
