#!/usr/bin/env python3
"""
run_lane34.py -- LANE 34. Submit this file to BatchClient.

  89  the off-diagonal and the two timing gradients
  90  the margin split, and whether adoption can replace exposure

WHY LANE 34 EXISTS. A referee asks how much of the employment result is
working from home rather than AI, citing Lambert and Schindler, who show
that the two exposures are strongly correlated and that entering them
together leaves remote work standing while the AI term attenuates. The
paper answers this today on the POSTING margin only, in Online Appendix
II.3, where the decline sits entirely in the NON-teleworkable half,
-0.233 against -0.005. This lane brings the employment margin to the
same question.

IT DOES NOT LEAD WITH A HORSE RACE, AND THAT IS THE POINT. Teleworkability
and DAIOE correlate at +0.75 across occupations, and firm-level averages
of each correlate higher still, because averaging over a firm's
incumbents strips idiosyncratic occupation variation and leaves the
shared component. A joint regression on two scores that close would give
wide intervals on both, and an imprecise AI coefficient reads as a failed
defence rather than as an underpowered test. Script 46 ran that race on
the WITHDRAWN design and is not the answer.

WHAT RUNS HERE.
  89 Part A, the off-diagonal: four fits of Equation (2), the AI step
     inside the low- and high-teleworkability halves and the
     teleworkability step inside the low- and high-AI halves. The two
     LOW cells discriminate. No joint model, so no collinearity.
  89 Part B, timing: both scores standardised and interacted with the
     quarterly path from 2019Q1, 2022Q1 omitted, no calendar terms.
     NOT a test of whether remote work is a spent 2020-21 shock: the
     return-to-office backlash means teleworkability is still moving,
     and a design that fixed its effect after 2021 would assume the
     answer. The question is whether the two gradients SEPARATE.
  90 Part C, the margin split: hires and separations with both scores
     entered together. The one place a joint model is right, because
     the question is which margin each carries, not which survives.
  90 Part D, adoption: reuse 71's catalogue discovery to look for a
     remote-work item in the ICT and BITA surveys. An existence check.

BOTH SCORES COME FROM ONE BUILDER. 82's build_exposure() takes the
occupation score book as an argument, so the teleworkability score is
the same function on the same employers, freeze year, incumbents aged 31
to 69 and floor, with Dingel and Neiman's book in place of DAIOE's. 90
imports 89's wfh_book() rather than writing its own, so the lane cannot
end with two definitions of teleworkability.

SQL. 89 performs none. **90 PART D DOES**: it reads the survey tables
through 71's discovery, so this lane must run in a slot that may pull.
If it is submitted to a no-SQL slot, Part D fails and Part C still
reports.

WHAT IS PRE-COMMITTED, AND WHAT DELIBERATELY IS NOT. 89's rule 1 (which
cell must bite for AI to be the operative score) and 90's rule 2 (the
reconciliation holds only if teleworkability leads on hires and AI on
separations) are fixed before the run, because with scores this collinear
a result read after the fact is not falsifiable in either direction. 89's
Part B and 90's Part D pre-commit nothing: one is a descriptive path
whose three shapes are all reportable, the other an existence check.

WHAT THE LANE DOES NOT CLAIM. The 50-and-over gain. Online Appendix III.2
already reports that on the continuous route the oldest band's gain loads
as strongly on teleworkability and is not specific to AI exposure, and
Section 3 says the relative change is concentrated in that gain. The lane
concedes the oldest band and keeps the young-worker margin. Fixed now,
not after the results land.

TIMING. 87 took 64 minutes for six fits of the sex panel and 88 took 31
for twelve of the three-band panel. 89 runs four fits plus one wide
quarterly model, 90 two joint fits plus a catalogue read, so budget two
to three hours for the lane.
"""

import os
import sys
from pathlib import Path

os.environ["CANARIES_89_OUT"] = "output_89"
os.environ["CANARIES_90_OUT"] = "output_90"
# 82 is imported for its score builder; its own OUT is pointed at 89's
# folder so a run of this lane leaves no stray output_82 behind.
os.environ["CANARIES_82_OUT"] = "output_89"
os.environ["CANARIES_RWORK_TAG"] = "_34"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("89_wfh_offdiagonal.py", "output_89/89_summary.txt", 150),
    ("90_wfh_margins_adoption.py", "output_90/90_summary.txt", 90),
]

if __name__ == "__main__":
    _lane.run("34", STAGES)
