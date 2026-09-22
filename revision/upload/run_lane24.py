#!/usr/bin/env python3
"""
run_lane24.py -- LANE 24. Submit this file to BatchClient.

  73  the credit channel, Part B only, both bands              ~120 min

WHY. The code review of 22 September (reviewer 2, finding F1) found that
script 73's leverage loader filtered the Serrano balance sheets on a year
column with pd.to_numeric, and that BSLSLUT is a SQL date, so the filter
matched nothing, was skipped without a message, and the credit test ran on
whichever accounting year's balance sheet came first for each firm. The
loader now parses the date, keeps the 2019 close (the latest close inside
2019 if there are several), refuses when 2019 is absent, and writes the
year and the firm count into the summary. Part B also gains a baseline fit
on its own sample, the firms with a balance sheet, so the exposure term is
compared like with like (finding F2).

WHAT RUNS. CANARIES_73_PARTS=B: the baseline fit, the balance-sheet-sample
baseline, and the leverage fit, for 22-25 and 26-30. Industry (Part A) and
the bankruptcy arm (Part C) are not re-run; their exports from lane 19
stand. Output goes to output_73b/ so lane 19's output_73/ is untouched.

No SQL beyond one read of Serrano_bokslut. Reads 47h's education caches
and 47L's counts. Safe beside anything. Export: output_73b/73_summary.txt,
output_73b/credit_test.csv, output_73b/vcov_lev_*.csv and
output_73b/vcov_levbase_*.csv.

The credit sentences and the leverage columns of the online appendix are
out of the paper until this lands; they return quoted against the
same-sample baseline.
"""

import os
import sys
from pathlib import Path

os.environ["CANARIES_73_PARTS"] = "B"
os.environ["CANARIES_73_OUT"] = "output_73b"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("73_industry_and_credit.py", "output_73b/73_summary.txt", 120),
]

if __name__ == "__main__":
    _lane.run("24", STAGES)
