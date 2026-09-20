#!/usr/bin/env python3
"""
test_61_redated.py -- the step function must locate a late effect that the
                      launch-dated specification averages away.

The fixture plants a decline that begins in JANUARY 2024, so:
  1 the 'adoption' window coefficient must recover it,
  2 the 'launch' window coefficient must be near zero, since nothing
    happened between December 2022 and November 2023,
  3 and a single post-2022-12 dummy, which is what every earlier estimate
    used, must ATTENUATE it. That contrast is the entire argument.

Also tested: the panel is built once and reused, the windows are disjoint
and exhaustive, and a null world produces no step.

    CANARIES_DRYRUN=1 python3 revision/local/test_61_redated.py
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
TMP = Path(tempfile.mkdtemp(prefix="canaries61_"))
SHARE = TMP / "input"; SHARE.mkdir()
for f in ("daioe_quartiles.dta", "utb_grupp2_sun2020_niva3_inr4_nyckel.dta",
          "eloundou_ssyk4.dta"):
    shutil.copy(UPLOAD / f, SHARE / f)
os.environ["CANARIES_SHARE"] = str(SHARE)
sys.path.insert(0, str(MONA))
import mona_common as mc  # noqa: E402
mc.SHARE = str(SHARE); mc.CACHE_DIR = TMP / "cache"; mc.CACHE_DIR.mkdir()

def load(n, a):
    sp = importlib.util.spec_from_file_location(a, MONA / n)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
s61 = load("61_redated_triple.py", "s61")
s61.OUT = TMP / "out"; s61.OUT.mkdir(); s61.CACHE = mc.CACHE_DIR
j47 = s61._j47()
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


# ---- the windows themselves ------------------------------------------
lo = [s for _, s, _ in s61.STEPS]
hi = [e for _, _, e in s61.STEPS]
check("the windows are contiguous and disjoint",
      all(hi[i] == lo[i + 1] for i in range(len(lo) - 1)),
      " ".join(f"{a}..{b}" for a, b in zip(lo, hi)))
check("the first window starts at the ChatGPT launch", lo[0] == mc.CHATGPT_YM)
check("the last window is the adoption window and is open-ended",
      s61.STEPS[-1][0] == "adoption" and s61.STEPS[-1][1] == s61.POOLED_FROM)


# ---- a synthetic panel with a decline that starts in 2024 -------------
MONTHS = [f"{y}-{m:02d}" for y in range(2019, 2026)
          for m in range(1, 13 if y < 2025 else 7)]
AGES = ["22-25", "31-34", "35-40", "41-49", "50+"]
N = 200
R = np.random.default_rng(61)
EXPOSED = set(range(1, 101))


def panel(break_at="2024-01", shock=0.30, seed=5):
    rr = np.random.default_rng(seed)
    rows = []
    for e in range(1, N + 1):
        hi_firm = int(e in EXPOSED)
        for ym in MONTHS:
            for a in AGES:
                lam = 14.0 if a == "22-25" else 20.0
                if break_at and hi_firm and a == "22-25" and ym >= break_at:
                    lam *= float(np.exp(-shock))
                rows.append((e, ym, a, int(rr.poisson(lam)), hi_firm))
    b = pd.DataFrame(rows, columns=["employer_id", "year_month", "age_group",
                                    "n_emp", "high"])
    b["young"] = (b["age_group"] == "22-25").astype(int)
    e = b["employer_id"].astype(str)
    b["fe_emp_t"] = e + "_" + b["year_month"]
    b["fe_emp_age"] = e + "_" + b["age_group"]
    b["fe_t_age"] = b["year_month"] + "_" + b["age_group"]
    return b


bal = panel()
b, step_terms, pool_terms = s61.add_terms(bal)
check("add_terms leaves the panel unmutated", "s_adoption" not in bal.columns)
check("the step terms are disjoint in every row",
      (b[[t for t in step_terms if t.startswith("s_")]] != 0).sum(axis=1).max() <= 1,
      "no row is in two windows at once")

r = mc.run_fepois_multi(b, s61.OUT, tag="t61_step", terms=step_terms,
                        fes=j47.FES)
g = r.set_index("term")["coef"]
check("the adoption window recovers the planted decline",
      g["s_adoption"] < -0.15, f"{g['s_adoption']:+.4f} against -0.30")
check("the launch window is near zero, because nothing happened then",
      abs(g["s_launch"]) < 0.06, f"{g['s_launch']:+.4f}")
check("the step orders the windows correctly",
      g["s_adoption"] < g["s_launch"] - 0.10,
      f"launch {g['s_launch']:+.4f} vs adoption {g['s_adoption']:+.4f}")

# the single launch-dated dummy, which is what every earlier estimate used
b2 = b.copy()
ym = b2["year_month"].astype(str)
b2["post_launch"] = (ym >= mc.CHATGPT_YM).astype(int) * b2["high"] * b2["young"]
r2 = mc.run_fepois_multi(b2, s61.OUT, tag="t61_launch",
                         terms=["post_rb_x_high_x_young", "post_launch"],
                         fes=j47.FES)
launch = float(r2.set_index("term").loc["post_launch", "coef"])
check("a single launch-dated dummy ATTENUATES what the step recovers",
      launch > g["s_adoption"] + 0.05,
      f"launch-dated {launch:+.4f} vs adoption window {g['s_adoption']:+.4f}")

# ---- a null world must produce no step -------------------------------
b0, st0, _ = s61.add_terms(panel(break_at=None, seed=6))
r0 = mc.run_fepois_multi(b0, s61.OUT, tag="t61_null", terms=st0, fes=j47.FES)
g0 = r0.set_index("term")["coef"]
check("with nothing planted, no window shows a decline",
      min(g0[t] for t in st0 if t.startswith("s_")) > -0.06,
      " ".join(f"{t[2:]} {g0[t]:+.3f}" for t in st0 if t.startswith("s_")))


# ======================================================================
# The panel, which is where this script nearly went wrong
# ======================================================================
# 47j takes its panel from 47h's year frames, and those stop in 2023
# because the education register does. Re-dating the treatment to January
# 2024 on that panel would have estimated the headline coefficient on zero
# months of data and returned something that looked like an answer. The
# outcome therefore comes from 47L's monthly counts, and these tests hold
# that in place.
from _fixtures import Fixture  # noqa: E402

h47 = j47._h47()
FIX = Fixture(mc, h47)
book, spec, frames = FIX.install_edu(j47.YEARS)
expo, _ = j47.incumbent_exposure(frames[2019], book, "OL_daioe", spec,
                                 "true", s61.TRUNC)
Q4 = set(expo.loc[expo["fq"] == 4, "employer_id"].astype(int))
BETA24 = float(np.log(0.75))
# the decline starts in JANUARY 2024 and hits the young of the firms the
# education classifier actually calls Q4, which is the world the re-dating
# is built to find
FIX.install_occ(s61.PANEL_YEARS, zmap={(e, "22-25"): 1.0 for e in Q4},
                beta=BETA24, bands=("22-25",), from_ym="2024-01")
counts = pd.concat([pd.read_parquet(mc.CACHE_DIR / f"L_counts_{y}.parquet")
                    for y in s61.PANEL_YEARS], ignore_index=True)

skel = s61.build_skeleton(counts, "22-25", j47)
check("the panel reaches the adoption window",
      not skel.empty and skel["year_month"].max() >= s61.POOLED_FROM,
      f"last month {skel['year_month'].max()}")
check("the skeleton carries no exposure of any kind",
      not ({"fq", "high"} & set(skel.columns)))
# the integer fixed-effect keys must separate exactly what the pasted
# strings separated, or two employers share an effect and the estimate is
# quietly wrong
for fe, pair in (("fe_emp_t", ["employer_id", "year_month"]),
                 ("fe_emp_age", ["employer_id", "age_group"]),
                 ("fe_t_age", ["year_month", "age_group"])):
    check(f"{fe} is one code per distinct {' x '.join(pair)}",
          skel[fe].nunique() == len(skel[pair].drop_duplicates()),
          f"{skel[fe].nunique()} codes, {len(skel[pair].drop_duplicates())} pairs")

b1, _, _ = s61.attach_exposure(skel, expo)
flip = expo.copy(); flip["fq"] = 5 - flip["fq"]
b2, _, _ = s61.attach_exposure(skel, flip)
check("re-attaching a different classification leaves the counts identical",
      b1["n_emp"].equals(b2["n_emp"]) and b1["fe_emp_t"].equals(b2["fe_emp_t"]),
      "so building the skeleton once per band is safe")
check("and it does change who is treated",
      not b1["high"].equals(b2["high"]))


def boom(*a, **k):
    raise AssertionError("SQL attempted although the cache is warm")


mc.connect = boom
s61._j47 = lambda: j47          # main() must use the instance we configured
j47.YOUNG_BANDS = ["22-25"]     # one band; the second runs the same code

# ---- the guard: a panel ending in 2023 must refuse to run ------------
saved = s61.PANEL_YEARS
s61.PANEL_YEARS = [2021, 2022, 2023]
refused = ""
try:
    s61.main()
except RuntimeError as ex:
    refused = str(ex)
except BaseException as ex:                      # noqa: BLE001
    refused = f"WRONG EXCEPTION {type(ex).__name__}: {ex}"
finally:
    s61.PANEL_YEARS = saved
check("a panel that stops before the adoption window is REFUSED",
      "adoption window" in refused, refused[:90])

# ---- end to end on the real entry point ------------------------------
s61.main()
step = pd.read_csv(s61.OUT / "redated_step.csv")
pool = pd.read_csv(s61.OUT / "redated_pooled.csv")
check("every design-arm pair in JOBS produced a step fit",
      len(step) == len(s61.JOBS) * (1 + len(s61.STEPS)),
      f"{len(step)} rows for {len(s61.JOBS)} pairs")
check("the as-of arm is estimated for the primary design, so the",
      ("OL_daioe", "asof") in s61.JOBS,
      "artefact is still measured at the new dating")
check("no fit came back with a status other than ok",
      (step["status"] == "ok").all() and (pool["status"] == "ok").all())

tru = step[(step.design == "OL_daioe") & (step.arm == "true")].set_index("term")
check("main() recovers the 2024 decline in the adoption window",
      abs(tru.loc["s_adoption", "coef"] - BETA24) < 0.12,
      f"{tru.loc['s_adoption', 'coef']:+.4f} against a planted {BETA24:+.4f}")
check("and finds nothing in the launch window, where nothing was planted",
      abs(tru.loc["s_launch", "coef"]) < 0.06,
      f"{tru.loc['s_launch', 'coef']:+.4f}")
ptru = pool[(pool.design == "OL_daioe") & (pool.arm == "true")
            & (pool.term == "post2024_x_high_x_young")]
check("the pooled post-2024 coefficient agrees with the step",
      abs(float(ptru["coef"].iloc[0]) - tru.loc["s_adoption", "coef"]) < 0.03,
      f"pooled {float(ptru['coef'].iloc[0]):+.4f}")

summ = (s61.OUT / "61_summary.txt").read_text()
for must in ("ADOPTION" if False else "adoption", "READ THIS", s61.PANEL_FROM):
    check(f"the summary states {must!r}", must in summ)

print("\n" + ("ALL PASS" if not FAILS else f"FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)
