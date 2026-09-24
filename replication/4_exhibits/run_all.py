#!/usr/bin/env python3
"""
run_all.py: build every register table and figure of the paper and its online
appendix from the exports in 3_register_mona/exports/, in paper order.

Runs each builder of this pack in turn with the Python that runs this script,
stops at the first failure, and lists what each builder wrote. Nothing here
needs register access; the whole pack runs in well under a minute.

    python 4_exhibits/run_all.py
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent

# (builder, extra arguments). Figure 3's builder runs twice: once for the
# quarterly figure of the paper and once for the monthly diagnostic of the
# online appendix.
STEPS = [(p.name, []) for p in sorted(HERE.glob("[0-9][0-9]_*.py"))]
STEPS.insert([s for s, _ in STEPS].index("03_figure3_quarterly_path.py") + 1,
             ("03_figure3_quarterly_path.py", ["--monthly"]))


def main() -> int:
    t_all = time.time()
    for name, args in STEPS:
        label = " ".join([name] + args)
        print(f"\n== {label}")
        t0 = time.time()
        r = subprocess.run([sys.executable, str(HERE / name), *args],
                           cwd=PACKAGE)
        if r.returncode != 0:
            print(f"FAILED: {label} (exit {r.returncode})")
            return r.returncode
        print(f"   done in {time.time() - t0:.1f}s")
    print(f"\nAll {len(STEPS)} builds done in {time.time() - t_all:.1f}s; "
          f"outputs in output/tables/ and output/figures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
