#!/usr/bin/env python3
"""
run_lane39.py -- LANE 39, THE EDITOR'S READ: THE PRE-PERIOD, THE PLACEBO,
THE FLOWS BY INDUSTRY, AND AI AGAINST WORKING FROM HOME ON THE HEADLINE.
Submit this file to BatchClient. Two independent stages; a failure in one
does not stop the other.

  105  (a) the drift test on the pre-launch months from January 2019
       (and from 2020) at 22-25 and 26-30; (b) the backdated placebo:
       Equation (2) with every boundary moved back 24 and 36 months and
       the panel ending November 2022, tau_placebo = later minus interim;
       (c) hires and separations at 22-25 refitted with three-digit
       industry as the cluster, the coefficients gated on 97's tau.
       Gate: Table 1 at 22-25 within 0.0005. Nine fits.       ~1.5-2.5 h
  106  the headline design with the employer teleworkability score
       (Dingel and Neiman through 82's chain) beside the DAIOE indicator:
       AI only, joint, teleworkability only on the common sample; the
       four split-sample cells on tau (89's read rule); the female
       differential with the same extension. Gates: Table 1 at 22-25 and
       the female differential within 0.0005. Eleven fits.    ~2.5-3 h

SQL: none in either stage. Every frame is cached: L_counts_2019-2025
(2019 and 2020 were read by lane 25a; 86 drew the 2019Q1 path from them),
flows_2021-2025 (54), L_counts_sex_2021-2025 (67), 82's cascade and
baseline caches, 80's I_industry_key; input file dingel_neiman_ssyk4 at
the project root (the file 89 read). Output to output_105 and output_106.

Tested locally in revision/local/test_105_prepath_placebo.py (30 checks)
and test_106_wfh_headline.py (29 checks).
EXPORT: output_105/105_summary.txt, prepath_placebo.csv, flows_industry.csv;
output_106/106_summary.txt, wfh_headline.csv, wfh_scores_overlap.csv;
vcov_s105_*.csv and vcov_s106_*.csv (tier 2).
"""
import os
import sys
from pathlib import Path

os.environ["CANARIES_105_OUT"] = "output_105"
os.environ["CANARIES_106_OUT"] = "output_106"
os.environ["CANARIES_RWORK_TAG"] = "_39"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("105_prepath_placebo.py", "output_105/105_summary.txt", 130),
    ("106_wfh_headline.py", "output_106/106_summary.txt", 170),
]

if __name__ == "__main__":
    _lane.run("39", STAGES)
