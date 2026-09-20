#!/usr/bin/env python3
"""
run_lane9.py -- LANE 9. Submit this file to BatchClient.

  62  why the two clean designs disagree about age                ~2-3 h

47L says the affected band is the forties: 22-25 comes out at +0.008 and
41-49 at -0.019. 47j says the affected band is 22-25, at -0.013. Both
designs pass their artefact test, so averaging over the disagreement is
not analysis; one of them is answering a different question.

Their exposures differ in three ways at once, which is why the
disagreement cannot be read off directly: the SOURCE (occupation against
education), the UNIT (the age group's own work against the firm's
incumbents) and the FORM (a continuous score against a quartile). This
holds the outcome, the panel and the fixed effects at 47L's and varies
the exposure one ingredient at a time, so the difference is attributed
rather than guessed at.

READS ONLY CACHES: 47L's (L_baseline_2019, L_counts_2019 to 2025) and
47h's (edu_hr_2019, edu_hr_weights_2019 to 2021). If the education
caches are absent it runs the two occupation variants and says in the
summary that the SOURCE comparison could not be made.

Safe to run beside lanes 8 and 10.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("62_reconcile_gradients.py", "output_62/62_summary.txt", 150),
]

if __name__ == "__main__":
    _lane.run("9", STAGES)
