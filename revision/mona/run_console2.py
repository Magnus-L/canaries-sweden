#!/usr/bin/env python3
"""
run_console2.py -- one BatchClient job. Submit THIS FILE, not run_all_mona.py.

Console 2, the editor's two named demands: 40 (coverage diagnostics) then 41 (event studies by code vintage).

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

STAGES = ["40", "41"]
CONSOLE = "2"

# Import ONCE. Re-importing per stage re-wrapped sys.stdout in the log
# mirror each time, so the 18 September logs printed every line twice.
sys.argv = ["run_all_mona.py", "--console", CONSOLE]
import run_all_mona

for stage in STAGES:
    sys.argv = ["run_all_mona.py", "--console", CONSOLE, "--only", stage]
    try:
        run_all_mona.main()
    except SystemExit as ex:
        if ex.code:
            print(f"stage {stage} exited {ex.code}; continuing to the next")


