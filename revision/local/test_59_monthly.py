#!/usr/bin/env python3
"""
test_59_monthly.py -- monthly resolution must show a SHAPE that half-years
                      cannot, and must not invent one.

The fixture plants a decline that BUILDS over the first six months of
2025 rather than arriving at once. A half-yearly design sees one average
point. The monthly zoom should show the descent. That difference is the
entire justification for the script, so it is the thing tested.

Also tested: the quarterly scheme rebases on the same quarter rather than
on a mid-2022 base, the term count stays within what R survived, and a
world with no decline produces no descent.

    CANARIES_DRYRUN=1 python3 revision/local/test_59_monthly.py
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
TMP = Path(tempfile.mkdtemp(prefix="canaries59_"))
SHARE = TMP / "input"; SHARE.mkdir()
shutil.copy(UPLOAD / "daioe_quartiles.dta", SHARE / "daioe_quartiles.dta")
os.environ["CANARIES_SHARE"] = str(SHARE)
sys.path.insert(0, str(MONA))
import mona_common as mc  # noqa: E402
mc.SHARE = str(SHARE); mc.CACHE_DIR = TMP / "cache"; mc.CACHE_DIR.mkdir()

def load(name, alias):
    spec = importlib.util.spec_from_file_location(alias, MONA / name)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m
s59 = load("59_monthly_path.py", "s59")
l47 = load("47L_age_baseline_exposure.py", "l47")
s54 = load("54_hiring_flows.py", "s54")
s59.OUT = TMP / "output_59"; s59.OUT.mkdir(); s59.CACHE = mc.CACHE_DIR

D = pd.read_stata(SHARE / "daioe_quartiles.dta")
D["ssyk4"] = D["ssyk4"].astype(str).str.zfill(4)
DAIOE = D.rename(columns={"pctl_rank_genai": "score"})[["ssyk4", "score"]]
HI = D.loc[D.high_exposure == 1, "ssyk4"].to_numpy()
LO = D.loc[D.high_exposure == 0, "ssyk4"].to_numpy()
AGES = s54.AGES
N_EMP = 130
EXPOSED = set(range(1, 66))
MONTHS = [f"{y}-{m:02d}" for y in s54.YEARS for m in range(1, 13 if y < 2025 else 7)]
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def f_baseline(conn=None):
    R = np.random.default_rng(41)
    rows = []
    for emp in range(1, N_EMP + 1):
        for age in AGES:
            young = age in ("22-25", "26-30")
            pool = HI if (emp in EXPOSED and young) else LO
            for c in R.choice(pool, size=3, replace=False):
                rows.append((emp, age, str(c), 15, ""))
    return pd.DataFrame(rows, columns=["employer_id", "age_group", "ssyk4",
                                       "n", "ssyk_status"])


def f_flows(ramp=True, seed=43):
    """A decline that BUILDS through 2025, or no decline at all."""
    R = np.random.default_rng(seed)
    rows = []
    for emp in range(1, N_EMP + 1):
        hit = emp in EXPOSED
        for ym in MONTHS:
            if ym == "2019-01":
                continue
            for age in AGES:
                lam = {"22-25": 7.0, "26-30": 6.0, "31-34": 5.0,
                       "35-40": 5.0, "41-49": 4.0, "50+": 4.0}[age]
                if ramp and hit and age == "22-25" and ym >= "2025-01":
                    step = int(ym[5:7])            # 1 in Jan ... 6 in Jun
                    lam *= float(np.exp(-0.10 * step))
                rows.append((emp, ym, age, int(R.poisson(lam)),
                             int(R.poisson(lam))))
    return pd.DataFrame(rows, columns=["employer_id", "year_month",
                                       "age_group", "n_hire", "n_sep"])


base = f_baseline()
expo = l47.build_exposure(base, DAIOE)
bal = s54.build_panel(f_flows(ramp=True), expo)
panels = {"hires": (bal, "n_hire")}

# ---- the term count must stay within what R survives -----------------
_, pq, yq = s59.build_terms(bal, "quarter", s59.REF_QUARTER)
_, pz, yz = s59.build_terms(bal, "zoom", s59.REF_ZOOM)
check("quarterly stays well under the 154 terms that would not fit",
      len(pq) + len(yq) < 60, f"{len(pq)+len(yq)} terms")
check("the zoom scheme is monthly from 2024 and coarser before",
      len(pz) + len(yz) < 60 and any(t.startswith("x_2025_") for t in pz)
      and any(t.startswith("x_2019H") for t in pz),
      f"{len(pz)+len(yz)} terms")
check("no term name contains a character R cannot use in a formula",
      all(all(c.isalnum() or c == "_" for c in t) for t in pz + yz),
      str([t for t in pz + yz if not all(c.isalnum() or c == "_" for c in t)][:3]))

# ---- the shape, which is the point -----------------------------------
z = s59.run_spec(panels, "zoom", s59.REF_ZOOM)
zh = z[z.outcome == "hires"].set_index("period")["coef"]
m25 = [f"2025-{m:02d}" for m in range(1, 7) if f"2025-{m:02d}" in zh.index]
vals = [zh[m] for m in m25]
check("the monthly zoom recovers a point for every 2025 month",
      len(m25) == 6, str(m25))
check("and shows a DESCENT rather than a flat step",
      vals[-1] < vals[0] - 0.15,
      " ".join(f"{m[-2:]} {v:+.3f}" for m, v in zip(m25, vals)))
check("the descent is monotone enough to read as a build",
      sum(vals[i+1] < vals[i] for i in range(len(vals)-1)) >= 4,
      f"{sum(vals[i+1] < vals[i] for i in range(len(vals)-1))} of 5 steps down")

# ---- quarterly rebasing uses the same quarter ------------------------
q = s59.run_spec(panels, "quarter", s59.REF_QUARTER)
qh = q[q.outcome == "hires"]
check("quarterly reports a rebased column", "rebased" in qh.columns)
check("the rebased 2025Q2 is the steepest quarter",
      qh.set_index("period")["rebased"].idxmin() in ("2025Q2", "2025Q1"),
      str(qh.set_index("period")["rebased"].idxmin()))

# ---- a world with no decline must show no descent --------------------
bal0 = s54.build_panel(f_flows(ramp=False, seed=44), expo)
z0 = s59.run_spec({"hires": (bal0, "n_hire")}, "zoom", s59.REF_ZOOM)
z0h = z0[z0.outcome == "hires"].set_index("period")["coef"]
v0 = [z0h[m] for m in m25 if m in z0h.index]
# A single monthly coefficient on this fixture carries a standard error of
# five to eight points, so a null world wanders by a tenth or two across
# six months. That is the honest cost of the resolution and the reason the
# script says to read a shape rather than a point. The discriminating test
# is therefore comparative: the planted world must descend far more than
# the null world does.
drop_planted = vals[-1] - vals[0]
drop_null = v0[-1] - v0[0]
check("the planted descent is clearly distinguishable from a null world's "
      "wandering",
      drop_planted < drop_null - 0.20,
      f"planted {drop_planted:+.3f} vs null {drop_null:+.3f}")
check("and the null world does not descend at all",
      drop_null > -0.05,
      " ".join(f"{m[-2:]} {v:+.3f}" for m, v in zip(m25, v0)))

# ---- the descriptive series ------------------------------------------
d = s59.describe(bal, "n_hire")
check("the descriptive series covers every month", d.year_month.nunique() > 70,
      str(d.year_month.nunique()))
check("it is floored", ((d.n_cells == 0) | (d.n_cells >= 10)).all())
check("it splits exposure and age", {0, 1} <= set(d.high_exposure)
      and {0, 1} <= set(d.young))

# ---- main(), which the entry-point ratchet flagged as untested --------
# 58 crashed on MONA in a block only main() reaches. Same risk here.
mc.write_cache(base, mc.CACHE_DIR / "L_baseline_2019.parquet")
_F = f_flows()
for _y in s54.YEARS:
    mc.write_cache(_F[_F["year_month"].str[:4] == str(_y)],
                   mc.CACHE_DIR / f"flows_{_y}.parquet")
mc.connect = lambda: (_ for _ in ()).throw(AssertionError("SQL attempted"))
try:
    s59.main(); _ok = True
except BaseException:
    import traceback; traceback.print_exc(); _ok = False
check("main() runs end to end from cache with SQL forbidden", _ok)
if _ok:
    check("it writes the quarterly path", (s59.OUT / "path_quarter.csv").exists())
    check("and the plottable descriptive series",
          (s59.OUT / "descriptive.csv").exists())

print("\n" + ("ALL PASS" if not FAILS else f"FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)
