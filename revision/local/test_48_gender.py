#!/usr/bin/env python3
"""
test_48_gender.py -- reproduce the 18 Sep failure of 48 and prove the fix.

The panel is built the way the cached MONA pull is: gender as the STRINGS
"1"/"2", because Kon is a char column. Before the fix, panel["gender"] == 1
selected nothing, balance_panel merged an empty float64 month column against
an object one, and the run died with ValueError. Cases:
  1. the old int keys against a string panel select nothing (the bug)
  2. norm_gender accepts a string panel and an int panel, and the split works
  3. a panel missing a gender code raises, rather than estimating on nothing
  4. build() on an empty subset raises with a readable message
No MONA, no R: everything here stops before estimation.
    CANARIES_DRYRUN=1 python3 revision/local/test_48_gender.py
"""
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["CANARIES_DRYRUN"] = "1"
HERE = Path(__file__).resolve().parent
MONA, UPLOAD = HERE.parent / "mona", HERE.parent / "upload"
TMP = Path(tempfile.mkdtemp(prefix="canaries48_"))
SHARE = TMP / "input"; SHARE.mkdir()
shutil.copy(UPLOAD / "daioe_quartiles.dta", SHARE / "daioe_quartiles.dta")
os.environ["CANARIES_SHARE"] = str(SHARE)
sys.path.insert(0, str(MONA))
import mona_common as mc  # noqa: E402
# mona_common builds DAIOE_PATH with a Windows separator, which is right on
# MONA and unopenable here; 48 itself is left untouched.
# (load_daioe binds DAIOE_PATH as a default argument at import, so the
# function itself is what has to be pointed at the local copy.)
_LOCAL_DAIOE = str(SHARE / "daioe_quartiles.dta")
mc.DAIOE_PATH = _LOCAL_DAIOE
_load_daioe = mc.load_daioe
mc.load_daioe = lambda path=_LOCAL_DAIOE: _load_daioe(path)
spec = importlib.util.spec_from_file_location("s48", MONA / "48_gender_poisson.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
mod.OUT = TMP / "output_48"; mod.OUT.mkdir()

RNG = np.random.default_rng(48)
DAIOE = pd.read_stata(SHARE / "daioe_quartiles.dta")
HI = DAIOE.loc[DAIOE["high_exposure"] == 1, "ssyk4"].astype(str).str.zfill(4).to_numpy()
LO = DAIOE.loc[DAIOE["high_exposure"] == 0, "ssyk4"].astype(str).str.zfill(4).to_numpy()
MONTHS = [f"2019-{m:02d}" for m in range(1, 13)]


def panel(gender_values):
    """employer x month x ssyk4 x age x vintage x gender, as the pull gives it."""
    rows = []
    for emp in range(500, 540):
        for ym in MONTHS:
            for code in list(RNG.choice(HI, 2)) + list(RNG.choice(LO, 2)):
                for g in gender_values:
                    rows.append(dict(employer_id=emp, year_month=ym, ssyk4=code,
                                     vintage="own", age_group="22-25", gender=g,
                                     n_emp=int(RNG.integers(1, 6))))
    return pd.DataFrame(rows)


def test_bug_reproduces():
    p = panel(["1", "2"])
    assert p[p["gender"] == 1].empty, "int key should select nothing from a string panel"
    try:
        mod.build(p, "22-25", gender=1)
    except RuntimeError as ex:
        assert "empty panel" in str(ex)
        print("PASS the 18 Sep bug is caught: int key on a string panel raises, "
              "instead of a float64/object merge error deep in balance_panel")
        return
    raise AssertionError("build() accepted an empty gendered subset")


def test_fix_works_both_dtypes():
    for label, vals in (("string panel", ["1", "2"]), ("int panel", [1, 2])):
        p = mod.norm_gender(panel(vals))
        assert set(p["gender"]) == {"1", "2"}, p["gender"].unique()
        for code in mod.GENDERS:
            bal = mod.build(p, "22-25", gender=code)
            # not object: pandas 3 gives string columns a `str` dtype. What
            # matters is that the months are strings, not the float64 an
            # empty month list produced on 18 September.
            assert len(bal) > 0, (label, code)
            assert not pd.api.types.is_numeric_dtype(bal["year_month"]), \
                bal["year_month"].dtype
            assert set(bal.columns) >= {"post_gpt_x_high", "fe_emp_bin", "fe_emp_t"}
        tot = sum(len(mod.build(p, "22-25", gender=c)) for c in mod.GENDERS)
        both = len(mod.build(p, "22-25", gender=None))
        assert tot >= both, (tot, both)
        print(f"PASS fix works on a {label}: both splits non-empty, months not numeric")


def test_missing_code_raises():
    try:
        mod.norm_gender(panel(["1"]))
    except RuntimeError as ex:
        assert "absent from the panel" in str(ex), ex
        print("PASS a panel missing a gender code raises before any fit")
        return
    raise AssertionError("norm_gender accepted a panel with one gender")


if __name__ == "__main__":
    test_bug_reproduces(); test_fix_works_both_dtypes(); test_missing_code_raises()
    print(f"\nALL PASS  (tmp: {TMP})")
