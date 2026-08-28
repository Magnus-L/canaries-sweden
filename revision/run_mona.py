#!/usr/bin/env python3
"""
run_mona.py -- master runner for the MONA side. Upload the whole
revision/mona/ folder (scripts as .txt, rename to .py in MONA), then:

    python run_mona.py

Order and gating:
  39 canary gate  -- MUST PASS first; also writes the shared panel cache.
                     A FAIL stops the whole run (sys.exit propagates).
  40..46          -- independent of each other; run sequentially to keep
                     memory bounded (the 35b lesson: Python does not
                     return freed memory to the OS -- separate processes).
  47              -- only when Erik's education-exposure port has landed.

Every script logs to its own output_NN/ and can be re-run alone;
caches make re-runs cheap.
"""

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent / "mona"
ORDER = [
    "39_canary_gate.py",
    "40_coverage_diagnostics.py",
    "41_vintage_event_studies.py",
    "42_frozen_cohort.py",
    "43_poisson_primary.py",
    "44_decile_gradient.py",
    "45_asof_backtest.py",
    "46_wfh_horserace.py",
    # "47_edu_exposure.py",   # enable when the port has landed
]


def main():
    for name in ORDER:
        script = HERE / name
        print(f"\n{'='*70}\n{name}\n{'='*70}")
        t0 = time.time()
        r = subprocess.run([sys.executable, str(script)])
        print(f"{name}: exit {r.returncode} ({(time.time()-t0)/60:.1f} min)")
        if name.startswith("39") and r.returncode != 0:
            print("CANARY GATE FAILED -- aborting the run.")
            sys.exit(1)


if __name__ == "__main__":
    main()
