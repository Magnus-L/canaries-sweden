#!/usr/bin/env python3
"""
run_lane36.py -- LANE 36. Submit this file to BatchClient.

  93  the uncounted payslips (Online Appendix Table tab:uncounted) re-cut
      on the 2019 occupation-mix quartiles

WHY LANE 36 EXISTS. tab:uncounted was produced by script 79 (lane 26,
part C) before the design moved onto occupations, so its quartiles are
the education-mix ones: the last quantity in the appendix resting on the
education route. 93 reuses 79's cached declaration counts
(U_uncounted_{year}_noage.parquet) and 82's score and changes only the
employer-to-quartile map.

SQL. NONE if 79's caches are still on the share. If a year's cache is
missing, 79's own query pulls it, so budget an SQL slot only then.

THE GATE. The all-employer share does not depend on the quartiles, so it
must reproduce 79's printed figures (0.612, 0.205, 0.187, 0.188, 0.204,
0.195, 0.210 per cent) to three decimals; the quartiles must recompose
to the total exactly. A miss is a different population: quote nothing.

INDEPENDENT OF LANE 35. Run it before, after or alongside.

TIMING. 82's score build plus seven cached reads: 15 to 30 minutes.
Tested locally on synthetic worlds that pass and fail the gate.
"""

import os
import sys
from pathlib import Path

os.environ["CANARIES_93_OUT"] = "output_93"
os.environ["CANARIES_79_OUT"] = "output_93"
os.environ["CANARIES_82_OUT"] = "output_93"
os.environ["CANARIES_RWORK_TAG"] = "_36"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("93_uncounted_occupation.py", "output_93/93_summary.txt", 30),
]

if __name__ == "__main__":
    _lane.run("36", STAGES)
