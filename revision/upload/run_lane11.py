#!/usr/bin/env python3
"""
run_lane11.py -- LANE 11. Submit this file to BatchClient.

  64  the within-employer path, one coefficient per quarter      ~45-60 min

The last thing the revision needs. Script 61 gave the headline, -0.0509
at 22-25 and -0.0437 at 26-30 with the treatment dated where SCB's
adoption data put it, and it gave that number as three windows. A step
function cannot show whether the decline arrived as a step at the
boundary we chose or as a path through 2024 and 2025, and it cannot test
the pre-period at all. This replaces the windows with one coefficient per
quarter from 2021Q1 to 2025Q2, with 2022Q3 omitted.

Read the pre-period first. Its two thresholds were fixed before the run:
no single quarter beyond 2.5 of its own standard errors, and no drift
across the six beyond 2 of its. If the pre-period fails, the headline
fails with it and we say so.

READS ONLY CACHES. 47h's (edu_hr_2019, edu_hr_weights_2019 to 2021) and
47L's (L_counts_2021 to 2025). No SQL, no cache writes, so it is safe
beside anything.

Three fits: the true arm at both young bands, and the as-of arm at 22-25
so the register artefact is measured on the path rather than assumed
from the pooled estimate.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("64_within_employer_path.py", "output_64/64_summary.txt", 55),
]

if __name__ == "__main__":
    _lane.run("11", STAGES)
