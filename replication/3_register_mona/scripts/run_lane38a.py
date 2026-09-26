#!/usr/bin/env python3
"""
run_lane38a.py -- LANE 38a, the Editor's measurement condition, closed.
Submit this file to BatchClient.

  101  provenance of 99's 657 pooled differences: an independent POOLED
       raw rebuild compared with L_counts cell by cell; the persons whose
       register rows disagree (two sex codes, a valid and an invalid one,
       two birth years), cell by cell against every gap; the estimation
       panel's differing cells, explained and retained; pooled tau
       without the two-birth-year persons, the female differential
       without the two-sex persons                          ~1.5-2 h
  100  the tipping point (response A2e): the linkage accounting by
       exposure group and period and by month; the young person-months
       that would have to be missing only at exposed employers in the
       later period to move tau to zero, against the unlinked actually
       there; the extremal allocations; pooled and female differential
                                                             ~2.5-3 h

The stages are independent: a 101 that fails does not stop 100. Each
reproduces Table 1 (pooled -0.0399 (0.0102); female differential -0.0714
(0.0109)) before it varies anything.

SQL, all read only and cached: 101 R_counts_rawpool_2021-2025 (~3 min a
year), P_anom_2021-2025 (~3-5 min a year), an aggregate probe of Individ
2019/2021/2023; 100 T_unlinked_2021-2025 (~3-5 min a year).

Tested locally in revision/local/test_100_tipping_point.py and
test_101_count_provenance.py.
EXPORT: output_101/101_summary.txt, provenance.csv;
output_100/100_summary.txt, tipping_point.csv, unlinked_accounting.csv,
unlinked_by_month.csv, unlinked_accounting_sex.csv,
unlinked_by_month_sex.csv; vcov_s100_*, vcov_s101_* (tier 2).
"""
import os
import sys
from pathlib import Path

os.environ["CANARIES_101_OUT"] = "output_101"
os.environ["CANARIES_100_OUT"] = "output_100"
os.environ["CANARIES_82_OUT"] = "output_100"
os.environ["CANARIES_RWORK_TAG"] = "_38a"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("101_count_provenance.py", "output_101/101_summary.txt", 110),
    ("100_tipping_point.py", "output_100/100_summary.txt", 170),
]

if __name__ == "__main__":
    _lane.run("38a", STAGES)
