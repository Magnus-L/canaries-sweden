#!/usr/bin/env python3
"""
test_60_date.py -- re-dating the treatment must find a late break that the
                   launch-date specification misses, and must not invent one.

The fixture plants a decline that begins in JANUARY 2024, not at the
ChatGPT launch. Two things then have to hold, and together they are the
whole argument for the script:

  1. The 2022-12 specification, which is what every earlier estimate uses,
     ATTENUATES it, because it averages thirteen untreated months into the
     post window.
  2. The 2024-01 specification recovers it, and the exploratory profile
     has an INTERIOR MINIMUM at or near the truth rather than drifting to
     an endpoint.

And in a world with no break at all, the profile must not manufacture one.

    CANARIES_DRYRUN=1 python3 revision/local/test_60_date.py
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
TMP = Path(tempfile.mkdtemp(prefix="canaries60_"))
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
s60 = load("60_treatment_date.py", "s60")
l47 = load("47L_age_baseline_exposure.py", "l47")
s54 = load("54_hiring_flows.py", "s54")
s60.OUT = TMP / "output_60"; s60.OUT.mkdir(); s60.CACHE = mc.CACHE_DIR

D = pd.read_stata(SHARE / "daioe_quartiles.dta")
D["ssyk4"] = D["ssyk4"].astype(str).str.zfill(4)
DAIOE = D.rename(columns={"pctl_rank_genai": "score"})[["ssyk4", "score"]]
HI = D.loc[D.high_exposure == 1, "ssyk4"].to_numpy()
LO = D.loc[D.high_exposure == 0, "ssyk4"].to_numpy()
AGES = s54.AGES
N_EMP = 140
EXPOSED = set(range(1, 71))
MONTHS = [f"{y}-{m:02d}" for y in s54.YEARS for m in range(1, 13 if y < 2025 else 7)]
TRUE_BREAK = "2024-01"
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def f_baseline(conn=None):
    R = np.random.default_rng(61)
    rows = []
    for emp in range(1, N_EMP + 1):
        for age in AGES:
            young = age in ("22-25", "26-30")
            pool = HI if (emp in EXPOSED and young) else LO
            for c in R.choice(pool, size=3, replace=False):
                rows.append((emp, age, str(c), 15, ""))
    return pd.DataFrame(rows, columns=["employer_id", "age_group", "ssyk4",
                                       "n", "ssyk_status"])


def f_flows(break_at=TRUE_BREAK, shock=0.35, seed=62):
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
                if break_at and hit and age == "22-25" and ym >= break_at:
                    lam *= float(np.exp(-shock))
                rows.append((emp, ym, age, int(R.poisson(lam)),
                             int(R.poisson(lam))))
    return pd.DataFrame(rows, columns=["employer_id", "year_month",
                                       "age_group", "n_hire", "n_sep"])


base = f_baseline()
expo = l47.build_exposure(base, DAIOE)
bal = s54.build_panel(f_flows(), expo)

# ---- 1. the launch date attenuates what the true date recovers --------
at_launch = s60.fit_at(bal, "n_hire", "2022-12", "t60_launch")
at_truth = s60.fit_at(bal, "n_hire", TRUE_BREAK, "t60_truth")
check("the true date recovers the planted decline",
      at_truth["coef"] < -0.15, f"{at_truth['coef']:+.4f} against -0.35")
check("the ChatGPT-launch date ATTENUATES it, which is the whole point",
      at_launch["coef"] > at_truth["coef"] + 0.05,
      f"launch {at_launch['coef']:+.4f} vs truth {at_truth['coef']:+.4f}")
check("the launch specification reports more post months",
      at_launch["n_post_months"] > at_truth["n_post_months"],
      f"{at_launch['n_post_months']} vs {at_truth['n_post_months']}")

# ---- 2. the profile locates the break ---------------------------------
prof = [s60.fit_at(bal, "n_hire", d, f"t60_g_{d.replace('-','_')}")
        for d in s60.GRID]
P = pd.DataFrame(prof)
P = P[np.isfinite(P["coef"])]
lo = P.loc[P["coef"].idxmin(), "date"]
check("the profile's minimum is at or beside the true break",
      abs(s60.GRID.index(lo) - s60.GRID.index(TRUE_BREAK)) <= 1,
      f"minimum at {lo}, truth {TRUE_BREAK}")
check("and it is an interior minimum, not an endpoint",
      lo not in (P["date"].iloc[0], P["date"].iloc[-1]),
      f"{lo} within {P['date'].iloc[0]}..{P['date'].iloc[-1]}")
check("standard errors grow as the post window shrinks",
      P.sort_values("date")["se"].iloc[-1] > P.sort_values("date")["se"].iloc[0],
      " ".join(f"{r.date[-5:]} {r.se:.3f}" for r in P.itertuples()))

# ---- 3. no break planted, none found ----------------------------------
bal0 = s54.build_panel(f_flows(break_at=None, seed=63), expo)
prof0 = [s60.fit_at(bal0, "n_hire", d, f"t60_n_{d.replace('-','_')}")
         for d in s60.GRID]
P0 = pd.DataFrame(prof0)
P0 = P0[np.isfinite(P0["coef"])]
check("with no break planted, no date shows a material decline",
      P0["coef"].min() > -0.10,
      " ".join(f"{r.date[-5:]} {r.coef:+.3f}" for r in P0.itertuples()))
# 0.10, not 0.15: the observed separation is about 0.14 and a threshold
# set just inside it would fail on a different seed while testing nothing
# extra. Ten log points is already an unmistakable distinction.
check("and the planted world is clearly separated from the null world",
      P["coef"].min() < P0["coef"].min() - 0.10,
      f"planted {P['coef'].min():+.3f} vs null {P0['coef'].min():+.3f}, "
      f"separation {P0['coef'].min() - P['coef'].min():.3f}")

print("\n" + ("ALL PASS" if not FAILS else f"FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)
