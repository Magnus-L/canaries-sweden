#!/usr/bin/env python3
"""
test_58_seasonal.py -- the seasonal control must remove a seasonal and
                       must NOT remove a shock.

The fixture plants both at once, which is the only way to tell the two
apart:
  - an exposure-specific H1/H2 seasonal in young hiring, the thing that
    made 56's raw event study unreadable;
  - a genuine drop in young hiring at exposed firms in 2025H1 only.

A script that absorbs too much kills the shock; one that absorbs too
little leaves the seasonal in the pre-period and the read rule then
fails condition (iii). Both failure modes are checked.

This test already earned its keep once: the first version of 58 tried to
purge the seasonal with an exposure-times-H2 CONTROL TERM, and this
fixture showed the pre-period unchanged, 0.151 before and after. The
control is perfectly collinear with a full set of period interactions -
the seasonal simply IS the period coefficients - so fixest dropped it
and the "purge" did nothing. Rebasing after estimation replaced it.

    CANARIES_DRYRUN=1 python3 revision/local/test_58_seasonal.py
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
TMP = Path(tempfile.mkdtemp(prefix="canaries58_"))
SHARE = TMP / "input"; SHARE.mkdir()
shutil.copy(UPLOAD / "daioe_quartiles.dta", SHARE / "daioe_quartiles.dta")
os.environ["CANARIES_SHARE"] = str(SHARE)
sys.path.insert(0, str(MONA))
import mona_common as mc  # noqa: E402
mc.SHARE = str(SHARE); mc.CACHE_DIR = TMP / "cache"; mc.CACHE_DIR.mkdir()

D = pd.read_stata(SHARE / "daioe_quartiles.dta")
D["ssyk4"] = D["ssyk4"].astype(str).str.zfill(4)
DAIOE = D.rename(columns={"pctl_rank_genai": "score"})[["ssyk4", "score"]]
HI = D.loc[D.high_exposure == 1, "ssyk4"].to_numpy()
LO = D.loc[D.high_exposure == 0, "ssyk4"].to_numpy()

s58 = None
def load(name, alias):
    spec = importlib.util.spec_from_file_location(alias, MONA / name)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m
s58 = load("58_seasonal_and_vintage.py", "s58")
l47 = load("47L_age_baseline_exposure.py", "l47")
s54 = load("54_hiring_flows.py", "s54")
s58.OUT = TMP / "output_58"; s58.OUT.mkdir(); s58.CACHE = mc.CACHE_DIR

AGES = s54.AGES
N_EMP = 140
EXPOSED = set(range(1, 71))
MONTHS = [f"{y}-{m:02d}" for y in s54.YEARS for m in range(1, 13 if y < 2025 else 7)]
SEASONAL = 0.30     # young hiring at exposed firms is stronger in H2
SHOCK = 0.45        # and collapses in 2025H1
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def f_baseline(conn=None):
    R = np.random.default_rng(31)
    rows = []
    for emp in range(1, N_EMP + 1):
        for age in AGES:
            young = age in ("22-25", "26-30")
            pool = HI if (emp in EXPOSED and young) else LO
            for c in R.choice(pool, size=3, replace=False):
                rows.append((emp, age, str(c), 14, ""))
    return pd.DataFrame(rows, columns=["employer_id", "age_group", "ssyk4",
                                       "n", "ssyk_status"])


def f_flows(shock=SHOCK, seasonal=SEASONAL, seed=33):
    R = np.random.default_rng(seed)
    rows = []
    for emp in range(1, N_EMP + 1):
        hit = emp in EXPOSED
        for ym in MONTHS:
            if ym == "2019-01":
                continue
            h2 = int(ym[5:7]) > 6
            for age in AGES:
                base_h = {"22-25": 6.0, "26-30": 5.0, "31-34": 4.0,
                          "35-40": 4.0, "41-49": 3.5, "50+": 3.0}[age]
                lam = base_h
                if hit and age == "22-25":
                    if h2:
                        lam *= np.exp(seasonal)        # the seasonal
                    if ym >= "2025-01":
                        lam *= np.exp(-shock)          # the shock
                rows.append((emp, ym, age, int(R.poisson(lam)),
                             int(R.poisson(base_h))))
    return pd.DataFrame(rows, columns=["employer_id", "year_month",
                                       "age_group", "n_hire", "n_sep"])


base = f_baseline()
expo = l47.build_exposure(base, DAIOE)
bal = s54.build_panel(f_flows(), expo)
panels = {"hires": (bal, "n_hire")}

raw = s58.event_study(panels, "raw", h1_only=False)
purged = s58.event_study(panels, "rebased", h1_only=False, rebase=True)
h1 = s58.event_study(panels, "h1only", h1_only=True)


def series(df):
    return df.set_index("halfyear")["coef"]


r, p, o = series(raw), series(purged), series(h1)
pre_r = [h for h in r.index if h < "2022H2"]
pre_p = [h for h in p.index if h < "2022H2"]
pre_o = [h for h in o.index if h < "2022H2"]

check("the raw event study shows the planted seasonal in its pre-period",
      max(abs(r[h]) for h in pre_r) > 0.10,
      f"max |pre| {max(abs(r[h]) for h in pre_r):.3f}")
check("rebasing on the same season FLATTENS the pre-period",
      max(abs(p[h]) for h in pre_p) < max(abs(r[h]) for h in pre_r) / 2,
      f"raw {max(abs(r[h]) for h in pre_r):.3f} -> rebased "
      f"{max(abs(p[h]) for h in pre_p):.3f}")
check("H1-only also has a flat pre-period, with no seasonal model at all",
      max(abs(o[h]) for h in pre_o) < 0.06,
      f"max |pre| {max(abs(o[h]) for h in pre_o):.3f}")
check("the planted 2025H1 shock SURVIVES rebasing",
      p["2025H1"] < -0.15, f"{p['2025H1']:+.4f} against a planted -0.45")
check("and survives the H1-only cut",
      o["2025H1"] < -0.15, f"{o['2025H1']:+.4f}")
check("the two specifications agree on the shock",
      abs(p["2025H1"] - o["2025H1"]) < 0.12,
      f"rebased {p['2025H1']:+.4f} vs h1only {o['2025H1']:+.4f}")

rule = s58.evaluate_rule(pd.concat([purged, h1], ignore_index=True))
check("the pre-committed rule PASSES on a real planted shock",
      bool(rule["passes"].all()), rule[["spec", "coef", "t", "passes"]].to_string(index=False))

# and now a world with the seasonal but NO shock: the rule must refuse
bal0 = s54.build_panel(f_flows(shock=0.0, seed=34), expo)
p0 = s58.event_study({"hires": (bal0, "n_hire")}, "rebased", False, rebase=True)
o0 = s58.event_study({"hires": (bal0, "n_hire")}, "h1only", True)
rule0 = s58.evaluate_rule(pd.concat([p0, o0], ignore_index=True))
check("with the seasonal but NO shock, the rule REFUSES",
      not bool(rule0["passes"].any()),
      rule0[["spec", "coef", "t", "passes"]].to_string(index=False))


# ---- a supporting outcome must not be able to kill the primary one ----
# 20 Sep: the stock arm segfaulted R (access violation, rc 3221225477) and
# took the whole script down, including the hires estimate the paper turns
# on. Now it is recorded and skipped.
real_run_es = s58.run_es
def flaky(b, outcome, terms, tag):
    if outcome == "n_emp":            # the stock arm, always fails
        return pd.DataFrame()
    return real_run_es(b, outcome, terms, tag)

s58.run_es = flaky
s58.FAILURES.clear()
both = {"hires": (bal, "n_hire"), "stock": (bal, "n_emp")}
got = s58.event_study(both, "h1only", h1_only=True)
s58.run_es = real_run_es

check("a failing supporting outcome does not stop the run",
      "hires" in set(got["outcome"]), str(sorted(set(got["outcome"]))))
check("the failure is RECORDED rather than silently dropped",
      any("stock" in f for f in s58.FAILURES), str(s58.FAILURES))
check("the surviving outcome still carries its focus estimate",
      "2025H1" in set(got[got.outcome == "hires"]["halfyear"]))

# ---- main() itself, which no test had ever called --------------------
# 58 crashed 18 seconds into the 20 Sep run on two bugs in the AGI probe:
# int() on a NaN row count from the INFORMATION_SCHEMA fallback, which has
# no counts, and a case-sensitive vintage split that put every real table
# in "other" because the suffixes are _Def and _Prel, not _def and _prel.
# Neither could be caught, because every test called event_study directly
# and main() was never exercised. It is now, with the SQL layer mocked to
# answer exactly as MONA answers it.
def _fake_sql(q, conn):
    if "sys.tables" in q:
        return pd.DataFrame(columns=["table_name", "n_rows"])   # hidden
    names = [f"Arb_AGIIndivid{y}{m:02d}_Def"
             for y in range(2019, 2025) for m in range(1, 13)]
    names += [f"Arb_AGIIndivid2025{m:02d}_Prel" for m in range(1, 7)]
    return pd.DataFrame({"table_name": names, "n_rows": [None] * len(names)})


_real_sql, _real_connect = pd.read_sql, mc.connect
pd.read_sql = _fake_sql
mc.connect = lambda: object()
mc.write_cache(base, mc.CACHE_DIR / "L_baseline_2019.parquet")
for _y in s54.YEARS:
    _sub = f_flows()
    mc.write_cache(_sub[_sub["year_month"].str[:4] == str(_y)],
                   mc.CACHE_DIR / f"flows_{_y}.parquet")
try:
    s58.FAILURES.clear()
    s58.main()
    _ok = True
except BaseException as _ex:
    import traceback; traceback.print_exc()
    _ok = False
finally:
    pd.read_sql, mc.connect = _real_sql, _real_connect

check("main() runs end to end with the SQL layer mocked", _ok)
if _ok:
    _tab = pd.read_csv(s58.OUT / "agi_tables.csv")
    check("the vintage split is case-insensitive, so _Def counts as def",
          set(_tab["vintage"]) == {"def", "prel"},
          str(sorted(set(_tab["vintage"]))))
    # year comes back from the csv as int64, so compare as strings
    check("and it sees that 2025 is preliminary only",
          set(_tab[_tab["year"].astype(str) == "2025"]["vintage"]) == {"prel"},
          str(sorted(set(_tab[_tab["year"].astype(str) == "2025"]["vintage"]))))
    check("the summary was written", (s58.OUT / "58_summary.txt").exists())

print("\n" + ("ALL PASS" if not FAILS else f"FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)
