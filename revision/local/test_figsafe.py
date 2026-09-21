#!/usr/bin/env python3
"""
_figsafe must refuse to destroy a figure nothing can regenerate.

Reproduces 21 September: figA2_first_stage.pdf existed, no script in the
tree named it, and it was overwritten and lost.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _figsafe

OK = True


def check(label, cond, detail=""):
    global OK
    OK = OK and bool(cond)
    print(("PASS " if cond else "FAIL ") + label + (f"  [{detail}]" if detail else ""))


def attempt(stem, generator, figdir):
    fig = plt.figure()
    try:
        _figsafe.save(fig, stem, generator, figdir=figdir)
        return "wrote"
    except SystemExit as e:
        return f"refused({e.code})"
    finally:
        plt.close(fig)


with tempfile.TemporaryDirectory() as t:
    d = Path(t)

    # 1. a figure that does not exist yet is always fine to create
    check("a new figure is created without complaint",
          attempt("brand_new_thing", HERE / "l15_fig_backtest.py", d)
          == "wrote")

    # 2. THE ACCIDENT: the file exists and NO script in the tree names it
    orphan = "totally_unclaimed_orphan_figure"
    (d / f"{orphan}.pdf").write_bytes(b"%PDF-1.4 irreplaceable\n")
    r = attempt(orphan, HERE / "l15_fig_backtest.py", d)
    check("an ORPHAN is refused, not overwritten", r.startswith("refused"), r)
    check("and the orphan file is untouched",
          (d / f"{orphan}.pdf").read_bytes().endswith(b"irreplaceable\n"))

    # 3. a figure claimed by a DIFFERENT script belongs to that script
    (d / "figA1_asof_backtest.pdf").write_bytes(b"%PDF-1.4 owned\n")
    r = attempt("figA1_asof_backtest", HERE / "l16_fig_firststage.py", d)
    check("a figure owned by another script is refused",
          r.startswith("refused"), r)
    check("and that file is untouched too",
          (d / "figA1_asof_backtest.pdf").read_bytes().endswith(b"owned\n"))

    # 4. a script may always overwrite its OWN figure
    r = attempt("figA1_asof_backtest", HERE / "l15_fig_backtest.py", d)
    check("a script overwrites its own figure freely", r == "wrote", r)
    check("and it really rewrote it",
          not (d / "figA1_asof_backtest.pdf").read_bytes().endswith(b"owned\n"))

    # 5. the override works, once and deliberately
    (d / f"{orphan}.pdf").write_bytes(b"%PDF-1.4 irreplaceable\n")
    os.environ["CANARIES_FIG_FORCE"] = "1"
    r = attempt(orphan, HERE / "l15_fig_backtest.py", d)
    del os.environ["CANARIES_FIG_FORCE"]
    check("CANARIES_FIG_FORCE=1 overrides deliberately", r == "wrote", r)

print("\n" + "=" * 58)
print("all checks passed" if OK else "FAILED")
sys.exit(0 if OK else 1)
