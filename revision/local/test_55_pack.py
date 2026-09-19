#!/usr/bin/env python3
"""
test_55_pack.py -- the export pack must be small, complete and safe.

The three ways this script could hurt us, each tested:
  1 it drops something the write-up needs (a summary, a coefficient table)
  2 it silently omits a large file, so a missing result looks like a null
  3 it exports a cell of fewer than five people

    CANARIES_DRYRUN=1 python3 revision/local/test_55_pack.py
"""
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["CANARIES_DRYRUN"] = "1"
HERE = Path(__file__).resolve().parent
MONA = HERE.parent / "mona"
TMP = Path(tempfile.mkdtemp(prefix="canaries55_"))
sys.path.insert(0, str(MONA))
import mona_common as mc  # noqa: E402
spec = importlib.util.spec_from_file_location("s55", MONA / "55_export_pack.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
mod.HERE = TMP
mod.PACK = TMP / "export_pack"
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


# a realistic output tree
(TMP / "output_43").mkdir(parents=True)
(TMP / "output_43" / "43_summary.txt").write_text("the headline\n")
pd.DataFrame({"age_group": ["22-25", "50+"], "coef": [-0.174, 0.024],
              "n_obs": [11970426, 9000000]}).to_csv(
    TMP / "output_43" / "poisson_pooled.csv", index=False)
(TMP / "output_54").mkdir()
(TMP / "output_54" / "54_log.txt").write_text("a log\n")
# a table carrying a disclosure violation the writing script missed
pd.DataFrame({"age_group": list("abcde"), "n_emp": [3, 0, 7, 12, 4],
              "coef": np.arange(5.0)}).to_csv(
    TMP / "output_54" / "flow_support.csv", index=False)
# a big non-priority file, and a big PRIORITY one
big = pd.DataFrame({"x": np.arange(300_000), "y": np.arange(300_000)})
big.to_csv(TMP / "output_54" / "huge_support.csv", index=False)
big.to_csv(TMP / "output_54" / "flow_gradient.csv", index=False)

mod.main()
pack = sorted(p.name for p in mod.PACK.glob("*"))
man = (mod.PACK / "MANIFEST.txt").read_text()

check("summaries are always packed", "output_43__43_summary.txt" in pack)
check("logs are always packed", "output_54__54_log.txt" in pack)
check("small coefficient tables are packed",
      "output_43__poisson_pooled.csv" in pack)
check("a PRIORITY file is packed even when large",
      "output_54__flow_gradient.csv" in pack)
check("a large non-priority file is NOT packed",
      "output_54__huge_support.csv" not in pack)
check("but it is NAMED in the manifest, so a gap cannot be mistaken "
      "for a null", "huge_support.csv" in man and "300,000 rows" in man)

out = pd.read_csv(mod.PACK / "output_54__flow_support.csv")
check("rows between 1 and the floor are DROPPED on the way out",
      set(out["n_emp"]) == {0, 7, 12}, str(sorted(out["n_emp"])))
check("a zero is kept, because zero discloses nothing",
      (out["n_emp"] == 0).any())
check("the manifest records the row loss", "5 -> 3 rows" in man)
check("every packed file has a sha in the manifest",
      all(n in man for n in pack if n != "MANIFEST.txt"))
total = sum(p.stat().st_size for p in mod.PACK.glob("*")) / 1e6
check("the pack is small", total < 5, f"{total:.2f} MB")

print("\n" + ("ALL PASS" if not FAILS else f"FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)
