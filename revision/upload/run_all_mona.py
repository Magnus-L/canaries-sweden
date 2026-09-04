#!/usr/bin/env python3
"""
run_all_mona.py -- master runner for the EL67898 revision, MONA side.

FLAT LAYOUT. Every script, every .R helper and this runner sit in the same
folder on the Lydia P1207 share; each script writes its own output_NN/.

    python run_all_mona.py                # lane A, the default single-console run
    python run_all_mona.py --lane b       # only 45, for a second console
    python run_all_mona.py --lane all     # everything in one console, strictly sequential
    python run_all_mona.py --from 43      # resume at a stage
    python run_all_mona.py --only 45      # one stage
    python run_all_mona.py --dry-run      # print the plan, run nothing

ORDER IS DELIBERATE. It is NOT the file-number order.
  39 gates: it pulls 2019-2025 once and writes the shared panel cache. A FAIL
     aborts everything, because every later number would be wrong in the same way.
  43 and 45 come next because the response letter cannot be written without them:
     43 is the Poisson headline, 45 is the coverage defence. Learning that either
     breaks is worth more on hour two than on hour six.
  40, 41, 42 are the Tier-1 diagnostics; 44 and 46 are Tier 2 and may be dropped
     if the trip runs short.

MEMORY. Each stage is a separate process on purpose: Python does not return
freed memory to the OS, so a single long-lived process accumulates (the 35b
lesson). The node ceiling is 100 GB and over-runs are killed without warning.
Do not run lane A stages concurrently with each other.

47 (education exposure) is deliberately absent. It waits on Erik Engberg's
measure and must not hold this trip; see notes/erik-delivery-vs-T12_2026-09-03.md.
"""

import argparse
import hashlib
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------
# BATCH REALITY (data-notes/mona-runtime-conventions.md, learned 4 Sep the
# hard way): BatchClient DISCARDS stderr and keeps no stdout for Python, and
# it cannot pass command-line arguments. So: open our own log before doing
# anything else -- including before importing mona_common, whose import can
# itself fail (pyodbc, pandas) -- mirror both streams into it, and write any
# uncaught traceback there. A crash must never again be invisible.
# ----------------------------------------------------------------------
_LOG = open(HERE / "run_all_mona_log.txt", "a", encoding="utf-8", errors="replace")


class _Mirror:
    def __init__(self, stream):
        self._s = stream
    def write(self, s):
        try:
            self._s.write(s)
        except Exception:
            pass
        _LOG.write(s)
        _LOG.flush()
    def flush(self):
        try:
            self._s.flush()
        except Exception:
            pass
        _LOG.flush()


sys.stdout = _Mirror(sys.stdout)
sys.stderr = _Mirror(sys.stderr)
sys.excepthook = lambda et, ev, tb: (
    print("\nUNCAUGHT EXCEPTION\n" + "".join(traceback.format_exception(et, ev, tb))))
try:
    import getpass
    print("\n" + "=" * 74)
    print("run_all_mona started %s | user %s | python %s"
          % (datetime.now().strftime("%Y-%m-%d %H:%M"), getpass.getuser(),
             sys.version.split()[0]))
    print("=" * 74)
except Exception:
    pass

# (script, lane, tier, what it delivers)
STAGES = [
    ("39_canary_gate.py",           "gate", "gate", "Reproduce g2=-0.010, Poisson -0.174, N=11,970,426; write panel_vintage.parquet"),
    ("39b_panel_diff.py",           "a", "diag", "Localise any N drift: v2 pull vs v1 archive cache, by year/quartile/pair"),
    ("43_poisson_primary.py",       "a", "T4/E6",  "Poisson headline: pooled + ES per age group, bridge inputs, extensive-margin LPM"),
    ("42_frozen_cohort.py",         "a", "T3/E5",  "Coverage-immune cohort: coded by Dec 2023, followed forward"),
    ("45_asof_backtest.py",         "b", "T3/E5",  "CENTREPIECE: as-of backtest, missingness x misclassification frontier, backdated variant"),
    ("40_coverage_diagnostics.py",  "a", "T1/E3",  "Match rates by month x age x quartile; incumbents/hires/entrants; vintage shares"),
    ("41_vintage_event_studies.py", "a", "T2/E4",  "ES by code vintage 2023/2022/2021; same-employer vs job-changer; imputed vs reported"),
    ("44_decile_gradient.py",       "a", "T11",    "Employment deciles, pre-committed monotonicity read"),
    ("46_wfh_horserace.py",         "a", "T10",    "Telework x period interactions, both margins"),
]

REQUIRED_INPUTS = [
    ("daioe_quartiles.dta",     "in input/, uploads directly; hash-checked"),
    ("dingel_neiman_ssyk4.dta", "in input/, uploads directly; 46 dies without it"),
    ("r_fepois.R",              "beside the scripts; upload as .txt, RENAME to .R"),
    ("r_fepois_es.R",           "beside the scripts; upload as .txt, RENAME to .R"),
    ("r_fepois_multi.R",        "beside the scripts; upload as .txt, RENAME to .R"),
]


def preflight() -> bool:
    """
    Check the inputs that are not .py, because those are the ones renamed by hand
    and therefore the ones that get forgotten, and hash the one input that came
    across from the v1 tree so that "the same file" is provable, not assumed.

    Also hashes the scripts against MANIFEST.sha256 when it is present, which
    closes the chain from "this is the code we tested" to "this is the code that
    ran". Shell escapes are banned in MONA; this is ordinary Python.
    """
    import mona_common as mc
    ok = True
    print("PRE-FLIGHT")
    for name, note in REQUIRED_INPUTS:
        path = Path(mc.SHARE) / name if name.endswith((".csv", ".dta")) else HERE / name
        exists = path.exists()
        mark = "ok " if exists else "MISSING"
        extra = ""
        if exists and name == "daioe_quartiles.dta":
            got = hashlib.sha256(path.read_bytes()).hexdigest()
            if got == mc.DAIOE_SHA256:
                extra = "  sha256 matches the repo copy"
            else:
                mark, ok, extra = "BAD HASH", False, f"  got {got[:16]}..., expected {mc.DAIOE_SHA256[:16]}..."
        if not exists:
            ok = False
        print(f"  [{mark:>8}] {name:<26} {note}{extra}")

    # MANIFEST.txt, not .sha256: the portal only accepts a fixed format list and
    # .sha256 is not on it, so a manifest named that way could never be uploaded.
    # The 4 Sep lesson, 65 minutes of it: every ENVIRONMENT dependency is a
    # pre-flight row. The gate crashed at the R step half an hour after the
    # SQL was already spent; this check would have failed in one second.
    try:
        rpath = mc._rscript()
        rv = subprocess.run([rpath, "--version"], capture_output=True,
                            text=True, timeout=60)
        ver = (rv.stdout + rv.stderr).strip().splitlines()[0][:40]
        print(f"  [      ok] Rscript                  {rpath}  ({ver})")
    except Exception as ex:
        ok = False
        print(f"  [ MISSING] Rscript                  {ex}")

    man = HERE / "MANIFEST.txt"
    if man.exists():
        bad = []
        for line in man.read_text().splitlines():
            want, _, fname = line.partition("  ")
            f = HERE / fname.strip()
            if f.exists() and hashlib.sha256(f.read_bytes()).hexdigest() != want:
                bad.append(fname.strip())
        if bad:
            ok = False
            print(f"  [BAD HASH] scripts differ from the tested versions: {', '.join(bad)}")
        else:
            print("  [      ok] MANIFEST.txt             every uploaded script matches what was tested")
    else:
        print("  [    note] MANIFEST.txt             absent; script integrity not verified")
    print()
    return ok


def done_marker(script: str) -> Path:
    # key on the full stage id ("39b", not "39") so 39b and 39 do not share
    # a marker; the stage id is everything before the first underscore
    return HERE / f"output_{script.split('_')[0]}" / "_DONE"


def run(script: str, dry: bool) -> int:
    if dry:
        print(f"  would run {script}")
        return 0
    t0 = time.time()
    # Stream the stage's stdout AND stderr through our mirrored log: a
    # stage's own uncaught crash otherwise dies to discarded stderr too.
    proc = subprocess.Popen([sys.executable, str(HERE / script)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace", cwd=HERE)
    for line in proc.stdout:
        print(line, end="")
    r = proc
    proc.wait()
    mins = (time.time() - t0) / 60
    print(f"  {script}: exit {r.returncode} ({mins:.1f} min)")
    import mona_common as mc
    mc.runlog(script, r.returncode, mins)   # who-ran-what, at the project root
    if r.returncode == 0:
        m = done_marker(script)
        m.parent.mkdir(exist_ok=True)
        m.write_text(f"{datetime.now():%Y-%m-%d %H:%M}  {mins:.1f} min\n")
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    # Default is "all", not "a": batch cannot pass arguments, so the default IS
    # the batch behaviour -- and a lane-a default would silently leave 45, the
    # as-of backtest and centrepiece of the coverage defence, out of every
    # batch run. Two-console interactive use can still pick a lane explicitly.
    ap.add_argument("--lane", default="all", choices=["a", "b", "all"])
    ap.add_argument("--from", dest="start", default=None, help="stage number to resume at, e.g. 43")
    ap.add_argument("--only", default=None, help="run exactly one stage number")
    ap.add_argument("--force", action="store_true",
                    help="rerun stages even when output_NN/_DONE exists (interactive only; "
                         "batch cannot pass arguments, so batch ALWAYS resumes)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retire-caches", action="store_true",
                    help="delete cache/ (measured and listed first). Only at close "
                         "of round, after exports are out and verified -- never mid-round.")
    a = ap.parse_args()

    if a.retire_caches:
        import mona_common as mc
        files = sorted(mc.CACHE_DIR.glob("*")) if mc.CACHE_DIR.exists() else []
        total = sum(f.stat().st_size for f in files if f.is_file())
        print(f"RETIRING cache/ -- {len(files)} files, {total/1e6:.1f} MB")
        for f in files:
            print(f"  {f.stat().st_size/1e6:8.1f} MB  {f.name}")
            f.unlink()
        mc.runlog("--retire-caches", 0, 0.0)
        return

    if a.only:
        plan = [s for s in STAGES if s[0].startswith(a.only)]
    else:
        plan = [s for s in STAGES if a.lane == "all" or s[1] in ("gate", a.lane)]
        if a.lane == "b":
            plan = [s for s in STAGES if s[1] == "b"]     # lane B never re-runs the gate
        if a.start:
            idx = next((i for i, s in enumerate(plan) if s[0].startswith(a.start)), 0)
            plan = plan[idx:]
    # Batch cannot pass arguments (runtime conventions, section 2), so
    # resubmit-after-crash must work with none: a stage with a _DONE marker is
    # skipped unless --force. The gate is the exception -- it always re-runs,
    # because reproducing the pinned numbers is the point of it.
    if not a.force:
        skipped = [s[0] for s in plan if s[1] != "gate" and done_marker(s[0]).exists()]
        if skipped:
            print("resuming: skipping stages already _DONE:", ", ".join(skipped))
        plan = [s for s in plan if s[1] == "gate" or not done_marker(s[0]).exists()]

    print("=" * 74)
    print(f"EL67898 revision -- MONA run  |  lane {a.lane}  |  {datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 74)
    for i, (script, lane, tier, what) in enumerate(plan, 1):
        print(f"  {i}. {script:<28} [{tier:<6}] {what}")
    print()

    if not preflight() and not a.dry_run:
        print("Pre-flight failed. Fix the missing inputs before running.")
        sys.exit(1)
    if a.dry_run:
        return

    t_all = time.time()
    for script, lane, tier, _ in plan:
        print(f"\n{'=' * 74}\n{script}  [{tier}]\n{'=' * 74}")
        rc = run(script, a.dry_run)
        if lane == "gate" and rc != 0:
            print("\nCANARY GATE FAILED. Nothing downstream is trustworthy. Aborting.")
            sys.exit(1)
        if rc != 0:
            print(f"\n{script} failed. Continuing -- later stages do not depend on it. "
                  f"Re-run alone with:  python run_all_mona.py --only {script[:2]}")
    print(f"\n{'=' * 74}\nTOTAL {(time.time() - t_all) / 60:.1f} min\n{'=' * 74}")
    import mona_common as mc
    mc.storage_report()


if __name__ == "__main__":
    main()
