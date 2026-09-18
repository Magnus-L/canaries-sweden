#!/usr/bin/env python3
"""
run_console3.py -- one BatchClient job. Submit THIS FILE, not run_all_mona.py.

Console 3: 44 (employment deciles), then 48 (the gender split in Poisson), then 47 (education-based exposure), which is standalone and runs as its own process.

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

STAGES = ["44", "48"]
CONSOLE = "3"

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

print("\n=== 47_edu_exposure.py (standalone) ===")
r = subprocess.run([sys.executable, str(HERE / "47_edu_exposure.py")],
                   cwd=str(HERE))
print(f"47_edu_exposure.py exited {r.returncode}")
