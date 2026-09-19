#!/usr/bin/env python3
"""
run_tonight.py -- ONE submission for the evening of 19 September 2026.

======================================================================
  Submit THIS FILE to BatchClient. Nothing else. It runs whatever is
  still missing, in dependency order, and skips whatever is done.
======================================================================

Order and why:
  47h  the horse race. Skipped if output_47h/horserace_estimates.csv is
       complete; resumed from its caches otherwise. Everything below
       reads its cached year frames, so it must come first.
  47i  firm-mix exposure. ~10 min off the caches.
  47j  within-employer triple difference. ~10 min off the caches.
  48   the gender split, with the char/int fix. ~30 min off its cached
       panel.
  50   the calibration and validation moments, if they did not finish.

A stage that fails does not stop the ones after it: they are independent
and a lost stage should cost a stage, not a night. Every stage's own log
is written by the stage itself; this wrapper writes run_tonight_log.txt
with the sequence, the exit codes and the timings.
"""

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = HERE / "run_tonight_log.txt"

# (script, the file that proves it finished, expected minutes)
# Order is deliberate. 50 first: it is short and it carries the predictive
# validation table the whole local analysis waits on. Then 47L, the design
# the cross-vendor review identified as the one we had missed, which has its
# own SQL and does not depend on 47h. Then 47h, the long one, and the three
# that read its caches.
STAGES = [
    ("50_sim_moments.py",             "output_50/50_summary.txt",     20),
    ("47L_age_baseline_exposure.py",  "output_47L/47L_summary.txt",   75),
    ("47h_edu_horserace.py",          "output_47h/47h_summary.txt",  330),
    ("47k_settled_sample.py",         "output_47k/47k_summary.txt",  120),
    ("47i_firmmix.py",                "output_47i/47i_summary.txt",   12),
    ("47j_within_employer_triple.py", "output_47j/47j_summary.txt",   12),
    ("48_gender_poisson.py",          "output_48/48_summary.txt",     40),
]


class Tee:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8", errors="replace")
        self.out = sys.stdout
        sys.stdout = self
        self.echoed = 0

    def write(self, s):
        self.f.write(s)
        self.f.flush()
        if self.echoed < 2048:            # the 4 KB pipe rule
            self.echoed += len(s)
            try:
                self.out.write(s)
            except Exception:
                pass

    def flush(self):
        self.f.flush()


def main():
    Tee(LOG)
    print("=" * 70)
    print("run_tonight  " + time.strftime("%Y-%m-%d %H:%M"))
    print("=" * 70)
    plan = []
    for script, done, mins in STAGES:
        if (HERE / done).exists():
            print(f"  SKIP  {script:<34} already has {done}")
        else:
            plan.append((script, mins))
            print(f"  RUN   {script:<34} about {mins} min")
    print(f"\n  total about {sum(m for _, m in plan)} min\n")
    results = []
    for script, mins in plan:
        t0 = time.time()
        print("=" * 70)
        print(f"{script}  (expect ~{mins} min)")
        print("=" * 70)
        r = subprocess.run([sys.executable, str(HERE / script)], cwd=str(HERE))
        el = (time.time() - t0) / 60
        results.append((script, r.returncode, el))
        print(f"  {script}: exit {r.returncode} ({el:.1f} min)")
        if r.returncode != 0:
            print(f"  continuing: the later stages do not depend on {script}")
    print("\n" + "=" * 70)
    print("SUMMARY")
    for script, rc, el in results:
        print(f"  {script:<34} exit {rc}  {el:6.1f} min")
    print("\nEXPORT THESE FOLDERS: output_50, output_47L, output_47h, output_47k,")
    print("                     output_47i, output_47j, output_48, and the ones")
    print("                     never fetched: output_41, output_44, output_46")
    print("Then on the Mac:  python3 revision/assemble.py <the export folder>")


if __name__ == "__main__":
    main()
