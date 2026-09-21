#!/usr/bin/env python3
"""
run_lane17.py -- LANE 17. Submit this file to BatchClient.

  71  do the firms we call exposed actually adopt AI?          ~30-60 min

The gap both external reviews put first: every estimate classifies a
firm from its 2019 education or occupation mix and then never observes
whether that firm adopted anything.

The delivery contains the missing half and we had not used it.
ITFtg_Stora carries firm AI use with seven technology types including
language generation, 2013 to 2023. ai_itftg_2019 carries the 2019 AI
module. BITA_2024 and 2025 carry INDIVIDUAL generative-AI use, CH1 to
CH3, with survey weights; the delivery reference calls BITA the only
individual-level AI adoption measure we hold.

What it buys is one sentence the paper cannot currently write: firms our
2019 measure places in the top exposure quartile are X points more
likely to report using AI in 2023. It also settles the open
education-versus-occupation decision on an external criterion, by
running the same first stage on both routes, rather than on coverage
counts.

IT COUNTS BEFORE IT ESTIMATES. Both sources are stratified samples and
ITFtg covers only firms with ten or more employees, so the overlap with
our classified firms is an empirical question. The thresholds are fixed
in the script before the run and it refuses rather than lowering them.

It also DISCOVERS the schema before assuming it, because this project
has never read these tables and has already lost time to identifier
spelling. Expect the log to name the keys it found and the match rate on
every join; a zero there means a key mismatch, not a thin sample, and
the script says so.

Short, light, and touches no cache any other lane writes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("71_adoption_validation.py", "output_71/71_summary.txt", 70),
]

if __name__ == "__main__":
    _lane.run("17", STAGES)
