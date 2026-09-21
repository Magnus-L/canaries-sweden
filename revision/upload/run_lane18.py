#!/usr/bin/env python3
"""
run_lane18.py -- LANE 18. Submit this file to BatchClient.

  72  is the 31+ incumbent mix a good proxy for the young?     ~2-3 h

ML's question, and a fair one. Every headline scores a firm from the
2019 education mix of its incumbents aged 31 and over. If a firm's young
workers do different work from its older ones, trainees and assistants
rather than junior versions of the professionals who employ them, that
mix is a noisy proxy for whether the YOUNG in that firm are exposed.

The floor itself is forced rather than chosen: 31 is the lowest floor
that keeps both 22-25 and 26-30 out of their own treatment, and letting
a young worker's own record into exposure rebuilds exactly the
circularity the as-of backtest killed. So the question is not whether to
drop the floor, it is what the floor costs.

Three parts, all on cached frames, no SQL.

  A  reliability. Young-band exposure against incumbent exposure for
     every firm: correlation, quartile agreement, top-to-bottom flips,
     broken out by firm size, plus how many firms the young band cannot
     score at all.
  B  leave one band out. For 22-25, exposure built from 26-30 and
     everyone older, so representation improves while the estimated band
     still never enters its own treatment. Headline re-estimated on it.
  C  the artefact on B, because admitting 26-30's records admits staler
     ones and better representation does not buy a pass on staleness.

The read rule is fixed in the script before the run. Note what it does
NOT do: no attenuation correction anywhere. The measurement error here
is not classical, so a correction would be a fiction; the reliability is
reported and the reader sees how much room it leaves. If the proxy turns
out weak, the direction is still knowable, because classical error
attenuates toward zero and our estimates would be conservative.

Reads L_counts and 47h's education caches, writes neither. Safe beside
any other lane.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("72_incumbent_floor.py", "output_72/72_summary.txt", 200),
]

if __name__ == "__main__":
    _lane.run("18", STAGES)
