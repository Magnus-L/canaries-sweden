#!/usr/bin/env python3
"""
run_console1.py -- one BatchClient job. Submit THIS FILE, not run_all_mona.py.

Console 1, the heavy lane: 45 (the as-of backtest, the coverage defence) then 46 (the telework horse race, Tier 2).

BatchClient cannot pass command-line arguments (data-notes/
mona-runtime-conventions.md, section 2), so the arguments are set here and
run_all_mona is called as a library. Submitting this file is the whole
procedure: no typing, no options, nothing to remember.

Stages run one after another, each in its own process, and a stage that has
already finished is skipped on a resubmission. So if this job is killed, just
submit it again: it picks up where it stopped.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

STAGES = ["45", "46"]
CONSOLE = "1"

for stage in STAGES:
    sys.argv = ["run_all_mona.py", "--console", CONSOLE, "--only", stage]
    for mod in ("run_all_mona",):
        sys.modules.pop(mod, None)
    import run_all_mona
    try:
        run_all_mona.main()
    except SystemExit as ex:
        if ex.code:
            print(f"stage {stage} exited {ex.code}; continuing to the next")


