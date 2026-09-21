#!/usr/bin/env python3
"""
run_lane15.py -- LANE 15. Submit this file to BatchClient.

  69  fixed birth cohorts against moving age bands            ~3-4 h

RUN THIS BEFORE LANE 14. Script 68 removes the Q4/Q1 cycle with
calendar terms, which assumes the cycle is a seasonal. 69 asks whether
it is instead an accounting artefact, and the answer decides whether 68
is the right response or the wrong instrument.

The cause it tests: 47L computes age as calendar year minus birth year,
because FodelseAr is the only age variable in the delivery. Everyone
therefore ages on 1 January, the bands are moving windows, and each
January the 22-25 band loses a whole birth cohort and gains another with
nobody changing job. Cohorts differ in size, so the lump differs by
year, and if exposed firms sit at a different point inside the band the
lump differs by exposure too. Month-by-age effects absorb the common
part and not that.

69 rebuilds the same panel on FIXED BIRTH COHORTS, set once from ages in
2022, so nobody ever crosses a boundary. Both bases are estimated on the
same firms and months, and the read rule is written into the script
before the run: a seasonal that collapses by half or more on cohorts is
MECHANICAL, one that survives within a quarter of its size is REAL, and
anything between is AMBIGUOUS and stays that way.

ONE NEW SQL PULL, cached as L_cohort_YYYY. It reads L_counts_YYYY and
never writes them, so lane 16 can read the same files at the same time.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("69_cohort_basis.py", "output_69/69_summary.txt", 260),
]

if __name__ == "__main__":
    _lane.run("15", STAGES)
