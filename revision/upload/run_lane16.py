#!/usr/bin/env python3
"""
run_lane16.py -- LANE 16. Submit this file to BatchClient.

  70  three respecifications, no SQL at all                   ~2-3 h

Three things the paper currently asserts without having tested them.

A. The age contrast. 63 reports -0.0318 at 22-25 and -0.0493 at 41-49 on
   the stock, and the paper reads two separately significant coefficients
   as evidence the young are distinctively hit. The point estimates
   actually put 22-25 ABOVE 41-49 by 0.0175. 70 drops the 41-49 band
   deliberately, so every coefficient that comes back is a difference
   from it, correctly signed and with its own standard error. If the
   young-versus-prime-age contrast is not distinguishable from zero the
   framing has to change, and we would rather find that here than in a
   referee report.

B. The youth payroll-tax expiry of 31 March 2023, interacted with
   exposure. 47L already carries the cap, the date and a level control; a
   national policy with a common effect is absorbed by the month-by-age
   terms, so the term that can survive them is the triple, and that is
   the one nobody has run.

C. Why the mechanism reverses between the education and occupation
   routes. Four rungs from education-on-everything to occupation-on-the-
   intersection, so employer coverage is separated from the register
   rather than confounded with it.

READS ONLY CACHES, does no SQL by design, and refuses to run rather than
pulling anything. Safe beside lanes 15 and 17.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("70_respecifications.py", "output_70/70_summary.txt", 200),
]

if __name__ == "__main__":
    _lane.run("16", STAGES)
