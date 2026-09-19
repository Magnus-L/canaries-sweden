#!/usr/bin/env python3
"""
test_5657_dynamics.py -- 56 (event studies by age) and 57 (baseline vintage
and attenuation), on synthetic pulls with real R + fixest.

The two claims that matter, and the fixture is built so a broken script
cannot pass either:

  56  a shock that starts in 2023H1 must appear in 2023H1 and NOT before.
      A script that mislabels half-years, or that picks the young terms by
      row position, would put the jump in the wrong place or mix the two
      coefficient sets, and the pre-period test would silently pass.

  57  attenuation must be MEASURED, not assumed. The fixture degrades the
      2019 assignment year by year by a known amount, so lambda has a
      right answer, and the estimate on a later baseline must be larger in
      magnitude than the one on the stale baseline.

    CANARIES_DRYRUN=1 python3 revision/local/test_5657_dynamics.py
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
TMP = Path(tempfile.mkdtemp(prefix="canaries5657_"))
SHARE = TMP / "input"; SHARE.mkdir()
shutil.copy(UPLOAD / "daioe_quartiles.dta", SHARE / "daioe_quartiles.dta")
os.environ["CANARIES_SHARE"] = str(SHARE)
sys.path.insert(0, str(MONA))
import mona_common as mc  # noqa: E402
mc.SHARE = str(SHARE)
mc.CACHE_DIR = TMP / "cache"; mc.CACHE_DIR.mkdir()

D = pd.read_stata(SHARE / "daioe_quartiles.dta")
D["ssyk4"] = D["ssyk4"].astype(str).str.zfill(4)
DAIOE = D.rename(columns={"pctl_rank_genai": "score"})[["ssyk4", "score"]]
HI = D.loc[D.high_exposure == 1, "ssyk4"].to_numpy()
LO = D.loc[D.high_exposure == 0, "ssyk4"].to_numpy()
AGES = list(mc.AGE_GROUPS)
N_EMP = 130
SHOCK_FROM = "2023-01"          # the truth: nothing before this
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


# A firm's occupation mix DRIFTS. Its exposure in 2023 is not a noisy
# reading of its 2019 exposure; it is a different quantity, because the
# firm's workers changed jobs. That is the model here, and it is the
# reason a 2019 baseline weakens over time.
#
# Two earlier versions of this fixture were wrong in opposite directions.
# The first made 2019 the truth and later years noise, and therefore
# "proved" the stalest baseline was best. The second made a fixed 2023
# truth measured ever better over time, and lambda rose instead of fell.
# Drift is the mechanism: the truth moves, and a fixed baseline falls
# behind it.
EXPOSED_2019 = set(range(1, 66))
SWAPPED = {2019: 0.00, 2021: 0.18, 2022: 0.30, 2023: 0.42,
           2024: 0.52, 2025: 0.58}


def exposed_at(year):
    """Firms exposed in `year`: 2019's set, with a growing share swapped."""
    R = np.random.default_rng(404)
    order = R.permutation(np.arange(1, N_EMP + 1))
    k = int(round(SWAPPED[min(year, 2025)] * N_EMP))
    flip = set(order[:k].tolist())
    return {e for e in range(1, N_EMP + 1)
            if (e in EXPOSED_2019) != (e in flip)}


def baseline_rows(year, seed=17):
    """The occupation mix recorded in `year`."""
    R = np.random.default_rng(seed + year)
    looks = exposed_at(year)
    rows = []
    for emp in range(1, N_EMP + 1):
        for age in AGES:
            young = age in ("22-25", "26-30")
            pool = HI if ((emp in looks) and young) else LO
            for c in R.choice(pool, size=3, replace=False):
                rows.append((emp, age, str(c), 12, ""))
    return pd.DataFrame(rows, columns=["employer_id", "age_group", "ssyk4",
                                       "n", "ssyk_status"])


def counts_rows(years, shock=0.30, young_only=True, seed=21,
                exposure=None):
    """
    `exposure` is a function year -> set of exposed firms, so the two
    scripts can be tested in the worlds they are each responsible for.
    56 places a shock in TIME and gets a world with no drift, where the
    2019 baseline is the right measure. 57 handles DRIFT and gets a world
    where the mix moves, which is precisely what defeats a 2019 baseline.
    Using one world for both would make 56 fail for 57's reason.
    """
    exposure = exposure or (lambda yy: EXPOSED_2019)
    R = np.random.default_rng(seed)
    rows = []
    for y in years:
        for m in range(1, 13 if y < 2025 else 7):
            ym = f"{y}-{m:02d}"
            post = ym >= SHOCK_FROM
            for emp in range(1, N_EMP + 1):
                for age in AGES:
                    lam = {"22-25": 20, "26-30": 22, "31-34": 25,
                           "35-40": 28, "41-49": 30, "50+": 32}[age]
                    hit = (age == "22-25") if young_only else True
                    if post and emp in exposure(min(y, 2025)) and hit:
                        lam *= float(np.exp(-shock))
                    rows.append((emp, ym, age, int(R.poisson(lam)) + 1))
    return pd.DataFrame(rows, columns=["employer_id", "year_month",
                                       "age_group", "n_emp"])


l47 = None
s56 = s57 = None


def load(name, alias):
    spec = importlib.util.spec_from_file_location(alias, MONA / name)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


s56 = load("56_dynamics_by_age.py", "s56")
s57 = load("57_baseline_vintage.py", "s57")
l47 = load("47L_age_baseline_exposure.py", "l47")
s56.OUT = TMP / "output_56"; s56.OUT.mkdir(); s56.CACHE = mc.CACHE_DIR
s57.OUT = TMP / "output_57"; s57.OUT.mkdir(); s57.CACHE = mc.CACHE_DIR

# ---------------- 56: the shock must land where it was planted ---------
base19 = baseline_rows(2019)
mc.write_cache(base19, mc.CACHE_DIR / "L_baseline_2019.parquet")
cnt = counts_rows(l47.YEARS)          # no drift: 56's world
for y in l47.YEARS:
    mc.write_cache(cnt[cnt["year_month"].str[:4] == str(y)],
                   mc.CACHE_DIR / f"L_counts_{y}.parquet")

def no_sql(*a, **k):
    raise AssertionError("SQL attempted although every pull is cached")
mc.connect = no_sql
s56.main()

yd = pd.read_csv(s56.OUT / "dynamics_young.csv")
yd = yd[yd["outcome"] == "stock"].set_index("halfyear")["coef"]
pre = [h for h in yd.index if h < "2022H2"]
post = [h for h in yd.index if h >= "2023H1"]
check("56 estimates a coefficient for every half-year", len(yd) >= 10,
      str(len(yd)))
check("56 puts the reference at 2022H1 with coefficient exactly zero",
      abs(yd.get(mc.REF_HALFYEAR, 9)) < 1e-12)
check("56 finds NOTHING in the pre-period, where nothing was planted",
      max(abs(yd[h]) for h in pre) < 0.06,
      " ".join(f"{h} {yd[h]:+.3f}" for h in pre))
check("56 finds the shock from 2023H1, where it was planted",
      min(yd[h] for h in post) < -0.10,
      " ".join(f"{h} {yd[h]:+.3f}" for h in post))
check("56 separates the young differential from the pooled path",
      (s56.OUT / "dynamics_pooled.csv").exists()
      and len(pd.read_csv(s56.OUT / "dynamics_pooled.csv")) >= 10)

# ---------------- 57: attenuation must be measured ---------------------
# the 2019 assignment decays: by 2022 a third of exposed young cells have
# moved, by 2023 half
for y in (2021, 2022, 2023):
    mc.write_cache(baseline_rows(y), mc.CACHE_DIR / f"L_baseline_{y}.parquet")
# 57's world: the mix DRIFTS and the shock follows current exposure, so a
# 2019 baseline is stale by construction and a 2022 one is less so.
cnt_drift = counts_rows(l47.YEARS, exposure=exposed_at)
for y in l47.YEARS:
    mc.write_cache(cnt_drift[cnt_drift["year_month"].str[:4] == str(y)],
                   mc.CACHE_DIR / f"L_counts_{y}.parquet")
s57.BASE_YEARS = (2019, 2021, 2022, 2023)
s57.main()

rel = pd.read_csv(s57.OUT / "reliability.csv")
lam = rel[rel["age_group"] == "ALL"].set_index("year")["lam"]
check("57 sets lambda(2019) to one by construction", abs(lam[2019] - 1) < 1e-9)
check("57 measures lambda as DECLINING in the distance from the baseline",
      lam[2021] > lam[2022] > lam[2023],
      " ".join(f"{y} {lam[y]:.3f}" for y in (2021, 2022, 2023)))
check("57 measures lambda below one once the mix has drifted",
      lam[2023] < 0.95, f"{lam[2023]:.3f}")

est = pd.read_csv(s57.OUT / "vintage_estimates.csv")
w = est.pivot_table(index="outcome", columns="baseline", values="gamma")
check("57 estimates the same design on both baselines",
      {2019, 2022} <= set(w.columns), str(list(w.columns)))
if {2019, 2022} <= set(w.columns) and "stock" in w.index:
    check("the LESS stale baseline finds a LARGER effect, which is what "
          "attenuation means",
          abs(w.loc["stock", 2022]) > abs(w.loc["stock", 2019]),
          f"2019 {w.loc['stock',2019]:+.4f} vs 2022 {w.loc['stock',2022]:+.4f}")
summ = (s57.OUT / "57_summary.txt").read_text()
for must in ("EXTRAPOLATED", "LOWER BOUND", "Do both or neither"):
    check(f"57's summary states: {must[:22]}", must in summ)

print("\n" + ("ALL PASS" if not FAILS else f"FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)
