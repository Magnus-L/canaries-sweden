#!/usr/bin/env python3
"""
run_lane29d.py -- LANE 29d. Submit this file to BatchClient.

  83  part D: the firm-size robustness of the occupation-route score

WHY LANE 29 EXISTS, AND WHY THE SCORE IS NOT REBUILT: see run_lane29a.py.
Every fit takes its quartile from 82_occupation_route.build_exposure(),
the primary arm, and nothing here rebuilds it.

WHY PART D EXISTS. A firm's exposure is a MEAN over its incumbents aged
31 to 69, so its sampling variance falls with their number. A firm with
one incumbent is classified by one worker's occupation, and that
classification then assigns treatment for all of its hiring and
separations for years. The floor never protected against this: five
person-months is one worker employed five months, and lane 28a shows the
floor barely binds at all, 271,047 employers at a floor of one against
262,089 at five. Misclassified binary treatment attenuates towards zero,
so the error is conservative for the SIZE of the step; but a noisier
score has fatter tails, which over-represents small employers at BOTH
extremes of the score distribution, and that is a composition distortion
in the treatment group which count-weighting does not remove. This part
measures it and then tests whether it matters.

WHAT RUNS HERE, in three pieces.
  (i) THE RELIABILITY OF THE SCORE as a function of the number of
      incumbents behind it. NO FIT. The employment-weighted variance of
      the worker-level score is decomposed into between-firm and
      within-firm; the between component is netted of the sampling noise
      it itself contains, since the variance of observed firm means is
      the variance of the true means plus the within variance over n; and
      the implied reliability, between over between plus within over n,
      is reported at the quantiles of the incumbent-count distribution,
      by employers and by employment, together with the share of
      employers and the share of incumbent employment whose score falls
      below a reliability of one half. One table, and it tells a referee
      how much of the sample is well measured. It runs first and costs
      under a minute, so it lands even if the fits later die.
  (ii) THE ADOPTION STEP AT A FLOOR OF SIXTY INCUMBENT PERSON-MONTHS,
      which is five workers employed all year, at both young bands and
      beside the reported floor of five. This is the check the
      econometrics vet asked for and it has never been run. Four fits.
  (iii) THE ADOPTION STEP BY FIRM-SIZE TERCILE at both young bands,
      terciles cut on the number of incumbents aged 31 to 69 in 2019, so
      the split is pre-treatment and fixed by construction. Six fits.

THE QUARTILE IS THE NATIONAL ONE IN EVERY ARM AND IS NEVER RECUT.
Recomputing the cut points inside a subsample changes the treatment
definition with the subsample, and the comparison then confounds a
different treatment with a different population. That is obvious for the
terciles and it is equally true of the floor arm, whose question is
whether the noisiest scores distort the treatment group: the test is to
drop those employers while leaving every surviving employer's treatment
exactly as it was. Script 82's own floor sensitivity recuts, because it
asks a different question -- how many employers a floor reaches -- and
this part is not that. What a recut WOULD have moved is reported as a
number and is not fitted, so a reader can see how much of any difference
would have been the moved cut points.

Ten fits. The two floor-of-five fits are full-size and the rest are
subsets, the terciles about a third each. Budget two and a half to three
hours.

READ RULE 9, fixed before the run and printed by the script at the start
and in the summary.

  THE SIZE ROBUSTNESS PASSES if the adoption step at 22-25 at the
  sixty-person-month floor keeps its sign and sits within ONE standard
  error of the reported floor-of-five step, AND no size tercile carries
  the whole step. Both halves are reported whichever way they fall. If
  the step is carried by the SMALLEST tercile, whose scores are the noisy
  ones, the summary says so in those words and that the concern is NOT
  ANSWERED.

  The reliability table is DESCRIPTIVE and carries no verdict. It is
  reported and NEVER used to correct an estimate: the measurement error
  here is not classical, it is larger for exactly the employers that
  crowd the tails, and the answer to a low reliability is a design that
  raises it -- which is what the floor arm is -- not a correction that
  manufactures precision the data do not have.

  What to expect, so it is not read as a bug. A higher floor scores fewer
  employers, so its standard error is larger than the reported one for
  that reason before anything else is said; the rule therefore reads the
  point estimate against the REPORTED standard error and not against its
  own. And a score's reliability falls with the number of incumbents
  behind it by construction, so a low figure at one incumbent is
  arithmetic and not a finding: what the table is for is the share of the
  sample sitting there.

  THE FIRST STAGE IS IN PART A AND IT GOVERNS THIS PART. If lane 29a's
  summary does not say THE FIRST STAGE REPRODUCES, nothing here is
  quoted, however it falls. Run 29a whether or not you run this one.

SQL. None once the caches are on the share: the occupation cascade and
the 2019 monthly counts are pulled only if lane 28a has not cached them,
which it has. No survey table, no balance sheet, no industry key. Safe
beside any job.

Export: output_83d/83_summary.txt, occ_rest_reliability.csv,
occ_rest_size.csv (every term of every arm with the employer count, the
number of exposed employers in it and the floor or tercile it belongs to)
and the ten vcov_*.csv files. Employer counts below five are suppressed
before anything leaves MONA.
"""

import os
import sys
from pathlib import Path

os.environ["CANARIES_83_PARTS"] = "D"
os.environ["CANARIES_83_OUT"] = "output_83d"
os.environ["CANARIES_82_OUT"] = "output_83d"
os.environ["CANARIES_RWORK_TAG"] = "_29d"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("83_occupation_route_rest.py", "output_83d/83_summary.txt", 180),
]

if __name__ == "__main__":
    _lane.run("29d", STAGES)
