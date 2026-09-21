#!/usr/bin/env python3
"""
run_lane19.py -- LANE 19. Submit this file to BatchClient.

  73  industry x age x month, and the credit channel        ~2-3 h

The two referee points we still answer with argument rather than data.

A. INDUSTRY x AGE x MONTH. Named independently by both external
   reviewers as the most valuable missing specification. Employer x
   month already absorbs industry x month, since industry is fixed
   within an employer; it does NOT absorb an age shock that hits one
   industry harder than another. Industry comes from FDB_JE_2019,
   frozen pre-shock.

B. THE CREDIT CHANNEL. R1.2, R1.3 and R2.1 all raise monetary
   transmission and we reply with a date. If the decline were
   credit-driven it would concentrate in LEVERAGED firms, and that is
   testable. Leverage is 1 - SummaEgetKapital / SummaTillgangar from
   FEK 2019, SCB's own population-level firm accounts, with Serrano's
   statements only as a fallback.

   Two documented traps are handled in the script and reported in its
   summary rather than discovered afterwards: pre-2022 FE tables are
   keyed on the CONSOLIDATED group head unit, not the legal entity the
   AGI panel uses, so a group member carries its group's balance sheet;
   and FEK excludes public, financial and non-profit employers, so the
   credit test runs on private non-financial firms.

C. Serrano's corporate events give bankruptcy and liquidation, so the
   separations margin can be re-estimated with failed firms dropped. A
   worker leaving a firm that went bankrupt is not an AI effect.

The read rule is fixed in the script before the run, and part B can
LOSE: if the credit term takes the effect and the exposure term does
not survive, the summary says MONETARY and we would rather know now.

Every arm counts its matched firms first and refuses below the
threshold. The schema is discovered, not assumed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("73_industry_and_credit.py", "output_73/73_summary.txt", 200),
]

if __name__ == "__main__":
    _lane.run("19", STAGES)
