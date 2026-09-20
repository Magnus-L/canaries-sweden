#!/usr/bin/env python3
"""
run_lane8.py -- LANE 8. Submit this file to BatchClient.

  61  the within-employer design, dated where adoption happened   ~3-4 h

The cleanest design we have puts the treatment at the ChatGPT launch,
and SCB's own survey says firm AI use went 10.4 per cent in 2023, 25.2
in 2024, 35.0 in 2025. Pooling thirteen untreated months with eighteen
treated ones attenuates the estimate for reasons that have nothing to do
with the world. Script 60 measured that: the coefficient grows from
-0.002 to -0.018 as the date moves from the launch to January 2024, with
the standard error flat.

This re-runs 47j's triple difference with the treatment dated properly.
The three windows were fixed by script 60 BEFORE any of this was
estimated, and they enter as one disjoint step function, so a single fit
returns the whole profile instead of four overlapping averages.

READS ONLY CACHES. It needs 47h's (edu_hr_2019, edu_hr_weights_2019 to
2021) and 47L's (L_counts_2021 to L_counts_2025). No SQL at all, so it
cannot collide with anything else running.

Note the panel: the outcome comes from 47L's monthly counts, not from
47h's year frames, because those stop in 2023 and the window we care
about opens in January 2024. The script refuses to run if the counts it
finds do not reach the adoption window.

Safe to run beside lanes 9 and 10.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("61_redated_triple.py", "output_61/61_summary.txt", 210),
]

if __name__ == "__main__":
    _lane.run("8", STAGES)
