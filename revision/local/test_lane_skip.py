#!/usr/bin/env python3
"""
_lane.py must not skip a stage whose own summary records failed fits.

Reproduces the 21 September case: lanes 14 and 15 were resubmitted to
recover crashed fits, the previous partial summaries were still on the
share and newer than everything, and both lanes SKIPPED and reported
success in seconds.
"""
import importlib.util
import sys
import tempfile
from pathlib import Path

UP = Path(__file__).resolve().parents[1] / "upload"
spec = importlib.util.spec_from_file_location("lane", UP / "_lane.py")
lane = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lane)

OK = True


def check(label, cond, detail=""):
    global OK
    OK = OK and bool(cond)
    print(("PASS " if cond else "FAIL ") + label + (f"  [{detail}]" if detail else ""))


REAL_68 = """POOLED from 2024-01, seasonal removed:
  22-25  stock  true  -0.0408 (0.0150) t -2.72

FITS THAT FAILED: path/year/22-25; 26-30/stock/true
A missing row is a missing fit, never a zero.
"""
REAL_73 = """22-25: baseline -0.0509 (0.0122)

FAILED:
  ind_26-30
"""
CLEAN = """POOLED from 2024-01, seasonal removed:
  22-25  stock  true  -0.0408 (0.0150) t -2.72

Runtime 127.3 min.
"""

with tempfile.TemporaryDirectory() as t:
    d = Path(t)
    for name, txt, want in (("s68.txt", REAL_68, True),
                            ("s73.txt", REAL_73, True),
                            ("clean.txt", CLEAN, False)):
        p = d / name
        p.write_text(txt, encoding="utf-8")
        got = lane._recorded_failures(p)
        check(f"{name}: failures {'detected' if want else 'not claimed'}",
              bool(got) == want, got or "(none)")

    check("the 68 case names the actual failures",
          "path/year/22-25" in lane._recorded_failures(d / "s68.txt"))
    check("the 73 case reads the list on the NEXT line",
          "ind_26-30" in lane._recorded_failures(d / "s73.txt"))

    # a summary that says the failure list is empty must not trip it
    (d / "none.txt").write_text("FAILED:\nnone\n", encoding="utf-8")
    check("an explicit 'none' is not a failure",
          not lane._recorded_failures(d / "none.txt"))
    # a missing marker must not raise
    check("a missing marker returns empty, never raises",
          lane._recorded_failures(d / "nope.txt") == "")

print("\n" + "=" * 60)
print("all checks passed" if OK else "FAILED")
sys.exit(0 if OK else 1)
