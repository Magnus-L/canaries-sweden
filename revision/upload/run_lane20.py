#!/usr/bin/env python3
"""
run_lane20.py -- LANE 20. Submit this file to BatchClient.

  74  the age contrast, with and without the cycle       ~15 min

Lane 16 gave the paper the number its title needs: 22-25 declines 3.6
log points MORE than 41-49, t -2.64, on 120,359 firms. A tested
contrast, not two coefficients read side by side.

But it sits on a specification with no calendar control, while the
headline we quote, -0.0408, has three quarter interactions in it. The
cycle is exposure-differential AND age-specific, so it can move a
contrast between bands and not only a level. Quoting the two beside each
other without checking would be comparing different specifications.

Two fits on ONE skeleton, so the arms differ in the term list and in
nothing else. The plain arm must reproduce lane 16's -0.0357 within a
standard error; if it does not, the summary says VOID and reports no
reading, because the two runs would then be on different samples.

The read rule is fixed in the script. It can lose: if the adjusted
contrast drops below half its size or loses significance, the paper
cannot lead on the young being distinctively hit and the framing moves
to the spreading pattern and the composition of adjustment.

No SQL. Short. Safe beside anything.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("74_contrast_seasonal.py", "output_74/74_summary.txt", 20),
]

if __name__ == "__main__":
    _lane.run("20", STAGES)
