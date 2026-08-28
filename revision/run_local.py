#!/usr/bin/env python3
"""
run_local.py -- master runner for the local side of the revision package.

    python revision/run_local.py             # all implemented local steps
    python revision/run_local.py --sample    # L1 on the 1% files (test)
    python revision/run_local.py --steps 1 3 # selected steps

Steps (see revision/README.md):
  L1 postings accounting (Ed.1) + monthly coverage by source (Ed.2)  [~20 min full]
  L2 coverage diagnostics on processed data (Ed.2)                    [seconds]
  L3 posting decile gradient (R1.4)                                   [~1 min]
  L4 public YREG check (R2.2)              -- NOT YET IMPLEMENTED
  L5 posting estimators, Poisson (E6/R1.8) -- NOT YET IMPLEMENTED
  L6 seasonality variant (R1.9)            -- NOT YET IMPLEMENTED
  L7 figures rebuild (E8)                  -- NOT YET IMPLEMENTED
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = {
    1: "local/l01_postings_accounting.py",
    2: "local/l02_coverage_diagnostics.py",
    3: "local/l03_decile_gradient_postings.py",
    4: "local/l04_public_yreg_check.py",
    5: "local/l05_posting_estimators.py",
    6: "local/l06_seasonality_variant.py",
    7: "local/l07_figures_rebuild.py",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="*", default=sorted(STEPS))
    ap.add_argument("--sample", action="store_true")
    args = ap.parse_args()

    results = {}
    for n in args.steps:
        script = HERE / STEPS[n]
        if not script.exists():
            print(f"L{n}: {STEPS[n]} not implemented yet -- skipping")
            results[n] = "missing"
            continue
        print(f"\n{'='*70}\nL{n}: {script.name}\n{'='*70}")
        cmd = [sys.executable, str(script)]
        if args.sample and n == 1:
            cmd.append("--sample")
        t0 = time.time()
        r = subprocess.run(cmd)
        results[n] = "ok" if r.returncode == 0 else f"exit {r.returncode}"
        print(f"L{n}: {results[n]} ({time.time()-t0:.0f}s)")

    print("\nSummary:", results)
    if any(v not in ("ok", "missing") for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
