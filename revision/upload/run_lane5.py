#!/usr/bin/env python3
"""
run_lane5.py -- LANE 5. Submit this file to BatchClient.

THE FAST MARGIN. One script.

  54  hires and separations per employer x age x month, on the exposure  ~90 min
      frozen in 2019. The outcome needs a birth year and a payslip and
      nothing else: no occupation code after 2019, no education register.

WHY IT IS SEPARATE. 47L bounded the effect on the employment STOCK at
about one log point through 2025H1. Headcount is the slowest margin in
the Swedish labour market, and the entry-level claim this paper argues
with is about HIRING. If the effect is anywhere, it is in the inflow.

RUNTIME IS THE LEAST CERTAIN IN THIS ROUND. Nothing of this shape has run
on P1207 before: each month joins two five-million-row AGI tables to each
other rather than a monthly table to a register. The pull is done one
month at a time and cached per year, so a failure names the month and a
restart keeps everything already fetched. 90 minutes is an estimate, not
a measurement. If the first year takes more than 20 minutes on its own,
stop it and tell me: the query needs an index hint, not more patience.

Shares no SQL and no cache with lanes 2, 3 or 4, except that it READS
47L's baseline cache (L_baseline_2019.parquet) if lane 4 has already
written it. Writing is atomic, so reading it while lane 4 runs is safe.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("54_hiring_flows.py", "output_54/54_summary.txt", 90),
]

if __name__ == "__main__":
    _lane.run("5", STAGES)
