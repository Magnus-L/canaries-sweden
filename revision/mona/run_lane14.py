#!/usr/bin/env python3
"""
run_lane14.py -- LANE 14. Submit this file to BatchClient.

  68  the headline and the paths, seasonal taken out       ~3-4 h

Script 64's pre-period failed the test we had written down beforehand,
in all three specifications. The cause is not a trend: the
exposure-differential young-to-older ratio has a calendar cycle, with
the fourth quarter positive in every year including the two before any
treatment, and the first quarter negative in every year. The post window
is weighted towards the negative quarters and the pre window is not, so
the pooled estimate inherits the difference.

This adds three quarter-of-year interactions with the fourth quarter
omitted. The treatment is then identified within calendar quarter across
years. Nothing else changes: same panel, same exposure, same fixed
effects, same dating.

EXPECT THE ESTIMATE TO SHRINK. Comparing like quarters by hand puts
22-25 near -0.03 against 61's -0.051. If the controlled figure comes
back far from -0.03, something other than the seasonal is at work and it
needs chasing before publication.

AND NOTHING HERE ASSUMES WHEN THE EFFECT BEGAN. The baseline is the
pre-ChatGPT window in every specification, so a reader who thinks the
onset was mid-2023 rather than January 2024 can read that off the path
instead of arguing with our dating. The pooled specification estimates
the interim from the launch to January 2024 rather than assuming it is
zero, and the pooled estimate for any other candidate date is a weighted
average of the quarter or month coefficients. We are not betting the
paper on one month.

It also produces the paths, all cleaned of the cycle: by year, by
quarter and, at 22-25, by month. A full event study cannot be cleaned
this way, because event-time dummies already span the calendar cycle and
a seasonal control beside them is collinear. What makes the cycle
separately identified is that the pre-period runs three years and sees
each season three times, so the seasonal block is estimated there and
each post period reads against the seasonally adjusted pre-period. That
is the object 64 could not produce.

And it settles three other things, so that no further run is needed: the
mechanism on all three margins with one treatment variable rather than
two; the gender differential with the seasonal out; and two of the three
fits that crashed on 21 September. The third, 65's step at 26-30, is not
re-run and is not needed.

READS ONLY CACHES. 47h's, 47L's, 54's and 67's L_counts_sex. No SQL, no
cache writes, safe beside anything.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("68_seasonal_control.py", "output_68/68_summary.txt", 190),
]

if __name__ == "__main__":
    _lane.run("14", STAGES)
