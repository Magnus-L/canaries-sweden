#!/usr/bin/env python3
"""
run_lane37c.py -- LANE 37c. Submit this file to BatchClient.

  97  the credit test on tau (both bands), the female differential's
      pre-launch path and drift, and industry x age x sex x month   ~1.75 h
  98  the as-of backtest on a common sample: coding separated from
      sample inclusion                                             ~40 min

The two stages are independent: a 97 that fails does not stop 98.

SQL. 97 reads Serrano's 2019 balance sheet (script 73's firm_leverage, as
lane 24 did) and, only if cache/I_industry_key is gone, script 80's
industry-key pulls (about an hour). 98 reads script 45's cached dual
panels and pulls them again (about 17 minutes each) only if they are
gone.

THE GATES. 97: Table 1 at both bands before the credit test, and the
female differential (tau -0.0714 (0.0109)) before the sex parts. 98:
45's true and as-of arms (T2021 +0.0193 and -0.2875; T2022 +0.0176 and
-0.1452). Every miss is a hard stop.

INDEPENDENT OF LANES 37a AND 37b. Tested locally in
revision/local/test_97_female_diagnostics.py and test_98_backtest_common.py.

EXPORT: output_97/97_summary.txt, output_97/female_credit_diagnostics.csv,
output_98/98_summary.txt, output_98/backtest_common.csv.
"""
import os
import sys
from pathlib import Path

os.environ["CANARIES_97_OUT"] = "output_97"
os.environ["CANARIES_98_OUT"] = "output_98"
os.environ["CANARIES_82_OUT"] = "output_97"
os.environ["CANARIES_80_OUT"] = "output_97"
os.environ["CANARIES_73_OUT"] = "output_97"
os.environ["CANARIES_RWORK_TAG"] = "_37c"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("97_female_diagnostics.py", "output_97/97_summary.txt", 105),
    ("98_backtest_common.py", "output_98/98_summary.txt", 40),
]

if __name__ == "__main__":
    _lane.run("37c", STAGES)
