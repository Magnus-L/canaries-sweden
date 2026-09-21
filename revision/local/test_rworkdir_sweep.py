#!/usr/bin/env python3
"""
The exchange sweep must not delete a concurrent run's live input.

Reproduces 21 September 16:20: lane 19 was submitted twice, and the
second job's startup sweep deleted the first job's exchange files.
"""
import os
import sys
import time
import types
import tempfile
from pathlib import Path

os.environ["CANARIES_DRYRUN"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mona"))
import mona_common as mc

OK = True


def check(label, cond, detail=""):
    global OK
    OK = OK and bool(cond)
    print(("PASS " if cond else "FAIL ") + label + (f"  [{detail}]" if detail else ""))


with tempfile.TemporaryDirectory() as t:
    d = Path(t)
    stale = d / "_rin_multi_old.csv"      # a crashed run's leftover
    live = d / "_rin_multi_live.csv"      # a concurrent run's input
    for f in (stale, live):
        f.write_text("x" * 1000, encoding="utf-8")
    old = time.time() - 7200
    os.utime(stale, (old, old))           # two hours ago

    mc._R_WORKDIR_SWEPT = False
    mc.tempfile = types.SimpleNamespace(gettempdir=lambda: str(d.parent))
    # point the helper at our directory by faking argv's stem
    real_argv = sys.argv[0]
    sys.argv[0] = d.name
    (d.parent / "canaries_rwork").mkdir(exist_ok=True)
    target = d.parent / "canaries_rwork" / d.name
    target.mkdir(exist_ok=True)
    for f in (stale, live):
        (target / f.name).write_text("x" * 1000, encoding="utf-8")
    os.utime(target / stale.name, (old, old))

    mc._r_workdir(d)
    sys.argv[0] = real_argv

    check("a crashed run's stale file IS swept",
          not (target / stale.name).exists())
    check("a concurrent run's LIVE file is NOT deleted",
          (target / live.name).exists(),
          "this is the 16:20 failure")

print("\n" + "=" * 58)
print("all checks passed" if OK else "FAILED")
sys.exit(0 if OK else 1)
