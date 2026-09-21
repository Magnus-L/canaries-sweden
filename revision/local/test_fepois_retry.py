#!/usr/bin/env python3
"""
The low-thread retry in mona_common._run_r.

Reproduces the MECHANISM, not the fix: the first call dies exactly the
way an over-threaded fixest died on 21 September (rc=3221225477 with
"*** recursive gc invocation" and the node reporting plenty free), and
the retry must lower the thread count and succeed.
"""
import os
import sys
import types
from pathlib import Path

os.environ["CANARIES_DRYRUN"] = "1"      # mona_common skips pyodbc
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mona"))
import mona_common as mc

FAILS = 0
CALLS = []


def fake_run(cmd, **kw):
    global FAILS
    CALLS.append(list(cmd))
    if FAILS > 0:
        FAILS -= 1
        return types.SimpleNamespace(
            returncode=mc.R_MEMORY_DEATH, stdout="fixest version: 0.13.2\n",
            stderr="*** recursive gc invocation\n")
    return types.SimpleNamespace(returncode=0, stdout="", stderr="")


mc.subprocess = types.SimpleNamespace(run=fake_run)
ok = True


def check(label, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(("PASS " if cond else "FAIL ") + label + (f"  [{detail}]" if detail else ""))


# 1. a fit with no thread flag: retried at 2, and it succeeds
FAILS, CALLS[:] = 1, []
r = mc._run_r(["Rscript", "x.R", "--input", "a"], Path("."), "t1", "fepois")
check("a memory death with no --nthreads is retried", len(CALLS) == 2)
check("and the retry adds --nthreads 2",
      CALLS[1][-2:] == ["--nthreads", "2"], " ".join(CALLS[1][-2:]))
check("and the caller sees the successful retry", r.returncode == 0)

# 2. already at 8: lowered to 2
FAILS, CALLS[:] = 1, []
mc._run_r(["Rscript", "x.R", "--nthreads", "8"], Path("."), "t2", "fepois")
check("a fit already at 8 threads is lowered to 2",
      CALLS[1][CALLS[1].index("--nthreads") + 1] == "2")

# 3. already at 2: go once more at 1. r73_ind_26-30 died at exactly 2.
FAILS, CALLS[:] = 1, []
r = mc._run_r(["Rscript", "x.R", "--nthreads", "2"], Path("."), "t3", "fepois")
check("a fit that dies at 2 threads is retried at 1", len(CALLS) == 2)
check("and the retry says --nthreads 1",
      CALLS[1][CALLS[1].index("--nthreads") + 1] == "1")

# 3b. already at 1: nothing lower exists
FAILS, CALLS[:] = 1, []
r = mc._run_r(["Rscript", "x.R", "--nthreads", "1"], Path("."), "t3b", "fepois")
check("a fit already at 1 thread is NOT retried", len(CALLS) == 1)
check("and the failure is returned, not swallowed",
      r.returncode == mc.R_MEMORY_DEATH)

# 3c. a fit that needs BOTH steps: 8 -> 2 -> 1
FAILS, CALLS[:] = 1, []
mc._run_r(["Rscript", "x.R", "--nthreads", "8"], Path("."), "t3c", "fepois")
check("8 threads falls to 2 first, not straight to 1",
      CALLS[1][CALLS[1].index("--nthreads") + 1] == "2")

# 4. a non-memory failure is not retried: a bad formula must stay failed
FAILS, CALLS[:] = 0, []


def other_fail(cmd, **kw):
    CALLS.append(list(cmd))
    return types.SimpleNamespace(returncode=1, stdout="",
                                 stderr="Error: object 'foo' not found")


mc.subprocess = types.SimpleNamespace(run=other_fail)
r = mc._run_r(["Rscript", "x.R"], Path("."), "t4", "fepois")
check("a non-memory failure is NOT retried", len(CALLS) == 1)
check("and it is still reported as a failure", r.returncode == 1)

# 5. a fit that works is not touched
CALLS[:] = []
mc.subprocess = types.SimpleNamespace(
    run=lambda cmd, **kw: (CALLS.append(list(cmd)) or
                           types.SimpleNamespace(returncode=0, stdout="",
                                                 stderr="")))
mc._run_r(["Rscript", "x.R"], Path("."), "t5", "fepois")
check("a successful fit runs exactly once, no overhead", len(CALLS) == 1)

print("\n" + "=" * 62)
print("all checks passed" if ok else "FAILED")
sys.exit(0 if ok else 1)
