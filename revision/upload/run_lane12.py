#!/usr/bin/env python3
"""
run_lane12.py -- LANE 12. Submit this file to BatchClient.

  65  the same design, classified by occupation instead      ~25-35 min

61 classifies a firm by the education mix of its incumbents aged 31 and
over in 2019. Almost every worker has an occupation code in 2019 and
that year's register is final, so the obvious question is why the
education detour. This answers it by running the identical design on the
identical firms with the identical outcome, changing only which register
assigns the quartile.

If the two agree in sign the paper can say the result does not depend on
the register, and keep education as primary for the coverage reason: the
occupation register samples about half the workforce and about two per
cent of the smallest firms, with the rest imputed, and SCB says the
imputed part is not built for analysing transitions.

Script 62 already ran both measures at firm level on the employment
stock and found the occupational one larger, -0.0278 at 22-25 against
-0.0067. If that carries over, the education route is the conservative
one.

READS ONLY CACHES: 47L's L_baseline_2019 and L_counts_2021 to 2025. No
SQL, no cache writes, safe beside anything. Four fits on two panels.

If output_61 is present the summary prints both registers side by side.

  66  what an exposed firm actually experienced              ~2-3 min

Descriptive, no regressions, on the SAME firm classification as the
headline so the two can be read side by side. Every coefficient in this
project is the exposed quartile minus the rest, net of what the fixed
effects remove, which is the right thing to estimate and the wrong thing
to say out loud. This produces the raw counterpart: what happened to the
young-to-older ratio in each exposure quartile, on employment, hiring
and separations, and the quartile's employment share, which is what
converts a coefficient into what a firm experienced.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("65_occupation_arm.py",     "output_65/65_summary.txt", 30),
    ("66_plain_magnitudes.py",   "output_66/66_summary.txt",  3),
]

if __name__ == "__main__":
    _lane.run("12", STAGES)
