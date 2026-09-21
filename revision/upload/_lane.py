#!/usr/bin/env python3
"""
_lane.py -- the runner behind run_lane1/2/3. Not submitted on its own.

WHAT IT GUARANTEES

1. A stage that fails does not stop the lane. Stages in a lane are
   independent by construction (the lane plan says so), so a lost stage
   costs that stage.
2. A stage that has already finished is skipped, so resubmitting a killed
   lane is cheap and safe.
3. The lane's log is written before anything is echoed, and the echo is
   capped at 2 KB. BatchClient's stdout is an unread pipe that blocks
   forever at about 4 KB, and a blocked write loses the log as well.
4. Nothing is written to the project share while a stage runs; the lane log
   is small and appended between stages.

WHAT IT DELIBERATELY DOES NOT DO

It does not swallow failures INSIDE a stage. Each script decides for itself
what is essential: an estimate or a primary export is allowed to abort that
script, a diagnostic or a side table is wrapped in its own `opt()` and only
warns. That is the `capture noisily` distinction, and it belongs next to the
code that knows which is which, not in the runner.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


class _Tee:
    LIMIT = 2048

    def __init__(self, path):
        self._f = open(path, "a", encoding="utf-8", errors="replace")
        self._out = sys.stdout
        self._n = 0
        self._stopped = False
        sys.stdout = self
        sys.stderr = self

    def write(self, s):
        try:
            self._f.write(s)
            self._f.flush()
        except BaseException:
            pass
        if self._stopped:
            return
        try:
            self._n += len(s)
            if self._n > self.LIMIT:
                self._stopped = True
                self._out.write("\n[echo capped; the full log is in the lane file]\n")
            else:
                self._out.write(s)
        except BaseException:
            self._stopped = True

    def flush(self):
        try:
            self._out.flush()
        except BaseException:
            pass
        self._f.flush()



# Shared modules every stage imports. A change to any of these makes an
# existing result stale, exactly as a change to the stage script does.
# Discovered the hard way on 21 September 2026: see the note in run().
_DEP_NAMES = ("mona_common.py", "r_fepois.R", "r_fepois_es.R",
              "r_fepois_multi.R")


def _deps():
    return [HERE / n for n in _DEP_NAMES]


def run(lane: str, stages: list):
    """stages: (script, the file that proves it finished, expected minutes)."""
    _Tee(HERE / f"run_lane{lane}_log.txt")
    t0 = time.time()
    print("=" * 70)
    print(f"LANE {lane}  started {time.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    plan = []
    for script, done, mins in stages:
        marker, src = HERE / done, HERE / script
        # A finished stage is skipped so that resubmitting a killed lane is
        # cheap. But "finished" has to mean "finished with THIS code": on
        # 20 Sep a re-uploaded script was skipped because the result of the
        # previous version was still sitting there, and the lane reported
        # success having done nothing. So the marker only counts if it is
        # newer than the script that produced it.
        # ...and "THIS code" means the script AND everything it imports.
        # On 21 Sep a fix to mona_common.py was uploaded, lane 19 was
        # resubmitted twice, and both runs SKIPPED, because the stage
        # script itself had not changed. Two submissions did nothing and
        # reported success. A shared module is part of the code that
        # produced a result, so its timestamp counts too.
        newest = max([src.stat().st_mtime] +
                     [d.stat().st_mtime for d in _deps() if d.exists()]) \
            if src.exists() else 0
        if marker.exists() and newest > marker.stat().st_mtime:
            plan.append((script, mins))
            why = ("the script" if src.exists()
                   and src.stat().st_mtime == newest else "a module it imports")
            print(f"  RUN   {script:<34} about {mins} min  "
                  f"({why} is NEWER than {done}, so the old result "
                  f"is stale and will be overwritten)")
        elif marker.exists():
            print(f"  SKIP  {script:<34} {done} exists")
        elif not src.exists():
            print(f"  ABSENT {script:<33} NOT UPLOADED -- this lane will skip it")
        else:
            plan.append((script, mins))
            print(f"  RUN   {script:<34} about {mins} min")
    print(f"\n  lane {lane}: {len(plan)} stages, about {sum(m for _, m in plan)} min\n")
    results = []
    for script, mins in plan:
        t = time.time()
        print("=" * 70)
        print(f"{script}   (expect ~{mins} min, started {time.strftime('%H:%M')})")
        print("=" * 70)
        try:
            # The child must NEVER inherit BatchClient's stdout. That is an
            # unread OS pipe: it blocks forever at about 4 KB, with no
            # traceback and no exit code, and a lane that hangs there is
            # indistinguishable from a slow one. Capping what the LANE prints
            # does not help, because the child writes past the lane. So the
            # child is handed a FILE, which cannot block. Each stage also
            # writes its own log; this is the backstop that makes a chatty
            # stage impossible to hang on.
            with open(HERE / f"run_lane{lane}_stages.txt", "ab") as fh:
                fh.write(f"\n===== {script} {time.strftime('%H:%M')} =====\n"
                         .encode())
                fh.flush()
                # The stage's own Tee caps its echo at 2 KB, which is right
                # when stdout is BatchClient's blocking pipe and wrong here:
                # this lane has already replaced stdout with a file, which
                # cannot block, and the cap then hides the very detail the
                # file exists to capture. On 19 September all three failures
                # in lanes 1 and 3 were invisible for exactly this reason.
                # The lane knows the destination is safe, so it lifts the cap.
                env = dict(os.environ, CANARIES_ECHO_LIMIT="100000000")
                rc = subprocess.run([sys.executable, str(HERE / script)],
                                    cwd=str(HERE), stdout=fh,
                                    stderr=subprocess.STDOUT, env=env).returncode
        except BaseException as ex:
            rc = -1
            print(f"  could not start {script}: {type(ex).__name__}: {ex}")
        el = (time.time() - t) / 60
        results.append((script, rc, el))
        print(f"  {script}: exit {rc} ({el:.1f} min)")
        if rc != 0:
            print(f"  lane {lane} CONTINUES: the remaining stages do not depend "
                  f"on {script}. Its own log says why it failed.")
    print("\n" + "=" * 70)
    print(f"LANE {lane} SUMMARY   total {(time.time()-t0)/60:.1f} min")
    for script, rc, el in results:
        print(f"  {script:<34} exit {rc}  {el:6.1f} min")
    bad = [s for s, rc, _ in results if rc != 0]
    print(f"  failed: {', '.join(bad) if bad else 'none'}")
