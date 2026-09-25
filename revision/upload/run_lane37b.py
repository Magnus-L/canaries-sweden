#!/usr/bin/env python3
"""
run_lane37b.py -- LANE 37b. Submit this file to BatchClient.

  98  the three-arm as-of backtest (A completed codes, full sample; B
      completed codes, as-of retained workers; C as-of codes, same
      workers), own and harmonised support, B-A, C-B, C-A with SEs
      from one stacked fit. Measurement content, run FIRST.   ~1.3 h
  97  headline, sex and exposure checks: continuous exposure and all
      four quartiles; the credit test on tau; the female differential's
      pre-launch path and drift (employer and industry clusters); the
      female differential under industry x age x sex x month    ~2.2 h

The stages are independent: a 98 that fails does not stop 97. Each
reproduces Table 1's tau (-0.0399 (0.0102)) before anything else.

SQL: 97 reads Serrano's 2019 balance sheet, and 80's industry-key pulls
only if cache/I_industry_key is gone (~1 h). 98 pulls 45's dual panels
only if they are gone (~17 min each).

Tested locally in revision/local/test_98_backtest_common.py and
test_97_headline_checks.py.
EXPORT: output_98/98_summary.txt, backtest_common.csv, vcov_s98_stacked_*;
output_97/97_summary.txt, headline_checks.csv, vcov_s97_*.
"""
import os
import sys
from pathlib import Path

os.environ["CANARIES_98_OUT"] = "output_98"
os.environ["CANARIES_97_OUT"] = "output_97"
os.environ["CANARIES_82_OUT"] = "output_97"
os.environ["CANARIES_80_OUT"] = "output_97"
os.environ["CANARIES_73_OUT"] = "output_97"
os.environ["CANARIES_RWORK_TAG"] = "_37b"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("98_backtest_common.py", "output_98/98_summary.txt", 80),
    ("97_headline_checks.py", "output_97/97_summary.txt", 130),
]

if __name__ == "__main__":
    _lane.run("37b", STAGES)
