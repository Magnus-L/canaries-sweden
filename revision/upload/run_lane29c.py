#!/usr/bin/env python3
"""
run_lane29c.py -- LANE 29c. Submit this file to BatchClient.

  83  part C: the two robustness tests Results names, on the occupation
      route

WHY LANE 29 EXISTS, AND WHY THE SCORE IS NOT REBUILT: see run_lane29a.py.
Every fit takes its quartile from 82_occupation_route.build_exposure(),
the primary arm, and nothing here rebuilds it.

WHAT RUNS HERE.
  C  the two tests the Results section names, each beside a baseline on
     its own sample.
     (i) INDUSTRY BY AGE BAND BY MONTH at both young bands, which is
         script 80's Part C: the stock specification of Equation (2)
         fitted twice on exactly the same employers, once with the
         paper's three effects and once with the month-by-age effect
         replaced by three-digit industry by age band by month, which
         nests it. An employer with no industry code leaves BOTH fits, so
         the difference between them is the specification and not the
         sample. Four fits.
     (ii) THE CREDIT TEST on the employers carrying a 2019 balance sheet,
         which is script 73's Part B, run through 73's own run_band: a
         median leverage split, the adoption step interacted with it, and
         a BASELINE RE-ESTIMATED ON THAT SAMPLE FIRST, so the comparison
         is a specification change and not a sample change. Six fits,
         because run_band also fits the full-panel baseline at each band
         before it reaches the credit arm. Those two are worth having:
         on this quartile the full-panel adoption step is the quantity
         lane 28 estimated, so the two are a check that this lane and
         that one are fitting the same panel.
     Ten fits. Budget two and a half to three hours.

WHY 80'S INDUSTRY TEST AND NOT 73'S. Script 80's Part C IS script 73's
industry test redone on the completed industry key and beside a
same-sample baseline, and the retained shares the paper quotes are 80's,
not 73's. Running 73's own version here would change the industry key as
well as the exposure route, and the comparison would then carry two
differences at once. The credit test has no such successor and is 73's,
through 73's own leverage builder and its own coverage gate.

READ RULES, fixed before the run and printed by the script at the start
and in the summary, with the education-route figure beside every
estimate. There is no coefficient gate in this part.

  7. THE INDUSTRY TEST. No gate, as script 80 has none. The retained
     share of the adoption step against the same-sample baseline is
     reported whatever it is, beside the education route's, which is 85
     per cent at 22-25 and 47 per cent at 26-30, and beside the share of
     firms coded from a source other than Ftg_2019, since a
     carried-forward code is noisier, absorbs less and so flatters the
     test. Fixed in advance: the step SURVIVES at a band if it keeps its
     sign and retains at least half the same-sample baseline. At 26-30
     the education route itself retains 47 per cent and so would not
     pass, which is said here rather than discovered afterwards.
  8. THE CREDIT TEST. THE STEP IS NOT A CREDIT EFFECT if it keeps its
     sign and at least 80 per cent of the baseline re-estimated on the
     balance-sheet sample, at both bands. The education route keeps 102
     and 105 per cent of -0.0695 and -0.0516. Script 73's own coverage
     gate travels with 73's code -- 500 panel employers and 30 per cent
     of them carrying the covariate -- and BELOW THRESHOLD is an accepted
     outcome reported as NO VERDICT, never as a failure and never as a
     pass. The sample is the employers with a 2019 balance sheet, which
     in this delivery means limited companies: the public sector and the
     unincorporated are not in it, and the paper must say so rather than
     leave it implicit.

  THE FIRST STAGE IS IN PART A AND IT GOVERNS THIS PART. If lane 29a's
  summary does not say THE FIRST STAGE REPRODUCES, nothing here is
  quoted, however it falls. Run 29a whether or not you run this one.

SQL. One INFORMATION_SCHEMA probe of the firm registers and one read of
the 2019 balance sheets (FEK if the delivery holds an FE table, Serrano's
bokslut table otherwise; 73's own builder decides and says which). Plus
the completed industry key if I_industry_key.parquet is not on the share,
and the occupation cascade and the 2019 monthly counts if lane 28a has
not already cached them, which it has. No monthly declarations beyond
that, so this is safe beside any job.

Export: output_83c/83_summary.txt, occ_rest_industry.csv (both
specifications at both bands, the retained share, the number of firms in
each fit and the share coded from a later source) and
occ_rest_credit.csv, with the ten vcov_*.csv files.
"""

import os
import sys
from pathlib import Path

os.environ["CANARIES_83_PARTS"] = "C"
os.environ["CANARIES_83_OUT"] = "output_83c"
os.environ["CANARIES_82_OUT"] = "output_83c"
os.environ["CANARIES_RWORK_TAG"] = "_29c"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("83_occupation_route_rest.py", "output_83c/83_summary.txt", 180),
]

if __name__ == "__main__":
    _lane.run("29c", STAGES)
