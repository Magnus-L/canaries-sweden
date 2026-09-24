#!/usr/bin/env python3
"""
run_lane35.py -- LANE 35. Submit this file to BatchClient.

  91  the headline under five adoption boundaries
  92  the reduced youth payroll contribution as a rival

WHY LANE 35 EXISTS. Both stages answer the same class of objection: that
a date or a policy, rather than AI, produces the age gradient. Neither
changes the paper's design; both re-run it with one thing moved.

  91 moves the adoption boundary to 2023-07, 2023-10, 2024-01 (the
     reported one), 2024-04 and 2024-07, and reports the contrast at
     each. The quarterly path shows a GRADUAL decline that is already
     distinguishable from the tightening level in 2023Q4, so the
     boundary does not sit on a break, and a referee will ask what
     moving it does.

  92 splits the 22-25 band at the eligibility line of the reduced youth
     payroll contribution, which ended on 31 March 2023 and reached
     workers who at the start of the year had turned 18 but not 23. The
     22-23 half was partly eligible; 24-25 never was. If the withdrawal
     drives the headline, the decline is in 22-23 alone.

WHAT 91 CLAIMS, AND WHAT IT DOES NOT. Under a smooth path the difference
of the two period means is invariant to the cutoff by construction: the
comparison period and the adoption window slide together. The synthetic
GRADUAL world in test_91 reproduces that, a spread of 0.013 on a step of
-0.156. So a flat sweep shows that THE NUMBER THE PAPER PRINTS does not
hinge on the partition, and shows nothing about when the decline began.
The script reports a RANGE for that reason and runs no equivalence test.
The boundary-free contrasts that do answer the dating question need no
MONA run at all and are already built from script 84's covariance.

SQL. 91 issues NONE; it reuses the L_counts caches. **92 DOES on its
first run**: the panel counts are banded at 22-25 inside the SQL, so the
split needs a new pull of employer x fine age band x month for 2021 to
2025. It opens a connection only if a year is missing, so a resubmission
after a kill reads the caches and needs no SQL slot.

THE GATES. Neither stage may be quoted if its gate fails.
  91 the 2024-01 arm must reproduce -0.0578 (0.0155) and -0.0399 (0.0102)
     at 22-25, and -0.0482 (0.0104) and -0.0403 (0.0067) at 26-30, to
     four decimals. Only the indicators differ across arms, so the
     estimation sample must also be identical across them.
  92 the two sub-bands must sum to the 22-25 counts the paper's own panel
     uses, employer by month, exactly. A split that does not recompose is
     a different population.

READ RULES, FIXED BEFORE THE RUN AND WRITTEN IN EACH SCRIPT.
  91 report the range; claim nothing about timing or cause.
  92 the rival is REJECTED if 24-25 decline distinguishably and land
     within one standard error of 22-23; it SURVIVES if 22-23 decline
     distinguishably and 24-25 do not. An earlier draft keyed on the SIGN
     of 24-25 and the synthetic payroll world caught it: -0.015 (0.015)
     is negative and means nothing.

BOTH WERE TESTED ON SYNTHETIC DATA BEFORE UPLOAD, each on two worlds so
that the verdict can go either way: test_91_dating_sensitivity.py (11
checks) and test_92_youth_payroll.py (9 checks).

TIMING. 91 is ten fits of the headline's size on cached counts; script 84
took 68 minutes for eleven quarterly fits, so budget two to four hours.
92 is one pull of five years plus two fits; the pull dominates and has no
precedent in this project, so budget generously and expect the pull, not
the fits, to be the risk. 92 is second for that reason: if it is killed,
91 has already finished.
"""

import os
import sys
from pathlib import Path

os.environ["CANARIES_91_OUT"] = "output_91"
os.environ["CANARIES_92_OUT"] = "output_92"
# 82 is imported by both for its score builder; its own OUT is pointed at
# 91's folder so the lane leaves no stray output_82 behind.
os.environ["CANARIES_82_OUT"] = "output_91"
os.environ["CANARIES_RWORK_TAG"] = "_35"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("91_dating_sensitivity.py", "output_91/91_summary.txt", 240),
    ("92_youth_payroll_rival.py", "output_92/92_summary.txt", 240),
]

if __name__ == "__main__":
    _lane.run("35", STAGES)
