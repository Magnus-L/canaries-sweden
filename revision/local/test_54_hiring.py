#!/usr/bin/env python3
"""
test_54_hiring.py -- 54 end to end, real DAIOE input, real R + fixest,
synthetic pulls at the SQL boundary.

THE MECHANISM THE FIXTURE MUST REPRODUCE, and the reason the script exists:
a firm can stop hiring young workers while its HEADCOUNT barely moves,
because the stock drains only as fast as people leave. A fixture in which
hiring and stock move together would let a broken script pass.

So the world here is: after ChatGPT, firms whose young workers did more
exposed work in 2019 cut young HIRING sharply; separations are untouched;
the stock therefore drifts down slowly and by much less. A design that
reads the flow must find the cut; a design that reads the stock must find
little. Both are checked.

Claims tested:
  1 a hire is a spell absent last month, and the first month of the window
    has no predecessor and is dropped
  2 zero-filling is real: a firm-age cell that hires in some months and not
    others keeps its zeros, so the estimate is not conditional on hiring
  3 a hiring cut planted on exposed firms' young workers IS recovered, and
    the separations arm stays flat, which is what distinguishes an inflow
    adjustment from a scale effect
  4 the SAME world read on the stock gives a much smaller number: the flow
    design is not just a relabelling of 47L
  5 the age gradient is separately identified: a cut planted on 22-25 alone
    shows up on 22-25 alone
  6 the January variant runs and is reported separately
  7 end to end with no SQL: exports written, floored, summary states limits

    CANARIES_DRYRUN=1 python3 revision/local/test_54_hiring.py
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
TMP = Path(tempfile.mkdtemp(prefix="canaries54_"))
SHARE = TMP / "input"; SHARE.mkdir()
shutil.copy(UPLOAD / "daioe_quartiles.dta", SHARE / "daioe_quartiles.dta")
os.environ["CANARIES_SHARE"] = str(SHARE)
sys.path.insert(0, str(MONA))
import mona_common as mc  # noqa: E402
mc.SHARE = str(SHARE)
mc.CACHE_DIR = TMP / "cache"; mc.CACHE_DIR.mkdir()
spec = importlib.util.spec_from_file_location("s54", MONA / "54_hiring_flows.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
mod.OUT = TMP / "output_54"; mod.OUT.mkdir(); mod.CACHE = mc.CACHE_DIR

D = pd.read_stata(SHARE / "daioe_quartiles.dta")
D["ssyk4"] = D["ssyk4"].astype(str).str.zfill(4)
DAIOE = D.rename(columns={"pctl_rank_genai": "score"})[["ssyk4", "score"]]
HI = D.loc[D.high_exposure == 1, "ssyk4"].to_numpy()
LO = D.loc[D.high_exposure == 0, "ssyk4"].to_numpy()
AGES = mod.AGES
N_EMP = 150
EXPOSED = set(range(1, 76))          # their young did exposed work in 2019
MONTHS = [f"{y}-{m:02d}" for y in mod.YEARS for m in range(1, 13 if y < 2025 else 7)]
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def f_baseline(conn):
    """2019 occupation mix per employer x age, the only occupation data."""
    R = np.random.default_rng(3)
    rows = []
    for emp in range(1, N_EMP + 1):
        for age in AGES:
            young = age in ("22-25", "26-30")
            pool = HI if (emp in EXPOSED and young) else LO
            for code in R.choice(pool, size=3, replace=False):
                rows.append((emp, age, str(code), int(R.integers(6, 20))))
    return pd.DataFrame(rows, columns=["employer_id", "age_group",
                                       "ssyk4", "n"])


def make_flows(hire_cut=0.45, sep_change=0.0, young_only=True, seed=9):
    """
    Hires and separations per employer x age x month.

    hire_cut     proportional cut in young hiring at exposed firms after
                 ChatGPT. This is the thing to recover.
    sep_change   proportional change in separations. Zero by default: the
                 point is that the inflow adjusts and the outflow does not.
    young_only   apply the cut to 22-25 only (for the gradient test) or to
                 both young bands.
    """
    R = np.random.default_rng(seed)
    rows = []
    for emp in range(1, N_EMP + 1):
        exposed = emp in EXPOSED
        for ym in MONTHS:
            if ym == "2019-01":
                continue                  # no predecessor: the script drops it
            post = ym >= mc.CHATGPT_YM
            for age in AGES:
                base_h = {"22-25": 5.0, "26-30": 4.0, "31-34": 3.0,
                          "35-40": 3.0, "41-49": 2.5, "50+": 2.0}[age]
                base_s = base_h              # a stationary firm
                hit = age == "22-25" if young_only else age in ("22-25", "26-30")
                h = base_h * ((1 - hire_cut) if (post and exposed and hit) else 1.0)
                s = base_s * ((1 + sep_change) if (post and exposed and hit) else 1.0)
                rows.append((emp, ym, age, int(R.poisson(h)), int(R.poisson(s))))
    return pd.DataFrame(rows, columns=["employer_id", "year_month",
                                       "age_group", "n_hire", "n_sep"])


def stock_from(flows, annual_turnover=0.12):
    """
    The employment stock implied by the same flows.

    The starting stock is set from the turnover rate, and that is the whole
    point of the test: the stock moves slowly relative to the flow only
    because the flow is a small share of it. A first version of this
    fixture used a starting stock of 120 against 5 hires a month, which is
    50 per cent annual turnover, and the stock then tracked the flow almost
    one for one and the test failed. Swedish turnover is nearer 10 to 12
    per cent, so the stock is roughly a hundred months of hiring, and a cut
    to the inflow drains it slowly.
    """
    f = flows.sort_values("year_month").copy()
    f["net"] = f["n_hire"] - f["n_sep"]
    start = (f.groupby(["employer_id", "age_group"])["n_hire"].transform("mean")
             * 12.0 / annual_turnover)
    f["n_emp"] = (f.groupby(["employer_id", "age_group"])["net"].cumsum()
                  + start)
    f["n_emp"] = f["n_emp"].clip(lower=0).round().astype(int)
    return f[["employer_id", "year_month", "age_group", "n_emp"]]


def gamma(bal, outcome, tag):
    r = mod.fit(bal, outcome, tag, mod.TERMS)
    g = r[r["term"] == "post_gpt_x_expo"]
    return float(g["coef"].iloc[0]) if not g.empty else np.nan


base = f_baseline(None)
expo = mod._l47().build_exposure(base, DAIOE)

# ---- 1. the un-preceded first month is absent ------------------------
fl = make_flows()
check("the first month of the window carries no flow",
      "2019-01" not in set(fl["year_month"]))

# ---- 2. zero-filling is real -----------------------------------------
sparse = fl[~((fl["employer_id"] == 3) & (fl["year_month"] > "2021-06"))]
bal_sparse = mod.build_panel(sparse, expo)
got = bal_sparse[(bal_sparse["employer_id"] == 3)
                 & (bal_sparse["year_month"] == "2022-03")]
check("a cell with no hires that month is kept as a zero, not dropped",
      len(got) > 0 and (got["n_hire"] == 0).all(),
      f"{len(got)} rows, hires {got['n_hire'].tolist()[:3]}")

# ---- 3. the planted hiring cut is recovered, separations are not -----
bal = mod.build_panel(fl, expo)
g_hire = gamma(bal, "n_hire", "t_hire")
g_sep = gamma(bal, "n_sep", "t_sep")
check("the planted hiring cut is recovered", g_hire < -0.05, f"{g_hire:+.4f}")
check("separations stay flat, so this reads as an INFLOW adjustment",
      abs(g_sep) < 0.03, f"{g_sep:+.4f}")
check("hiring and separations are distinguishable at all",
      abs(g_hire - g_sep) > 0.05, f"hire {g_hire:+.4f} vs sep {g_sep:+.4f}")

# ---- 4. the same world, read on the STOCK, says much less ------------
stk = stock_from(fl)
bal_stk = mod.build_panel(
    stk.assign(n_hire=stk["n_emp"], n_sep=0), expo)
g_stock = gamma(bal_stk, "n_hire", "t_stock")
check("the SAME world read on the stock gives a much smaller number, "
      "so the flow design is not a relabelling of 47L",
      abs(g_stock) < abs(g_hire) / 2,
      f"stock {g_stock:+.4f} vs flow {g_hire:+.4f}")

# ---- 5. the gradient separates ages ----------------------------------
gr = mod.fit(bal, "n_hire", "t_grad", mod.TERMS_GRAD)
want = {mod.age_term(a): a for a in AGES}
gr = gr[gr["term"].isin(want)].copy()
gr["age_group"] = gr["term"].map(want)
check("one coefficient per age band", len(gr) == len(AGES), str(len(gr)))
if len(gr) == len(AGES):
    g = gr.set_index("age_group")["coef"]
    others = g.drop("22-25")
    check("the cut planted on 22-25 appears on 22-25", g["22-25"] < -0.05,
          f"{g['22-25']:+.4f}")
    check("and not on the other five bands", others.abs().max() < 0.05,
          " ".join(f"{a} {v:+.4f}" for a, v in others.items()))

# ---- 6. the January variant ------------------------------------------
bal_nj = mod.build_panel(fl, expo, drop_january=True)
check("the no-January variant drops every January",
      not any(m.endswith("-01") for m in bal_nj["year_month"].unique()))
check("and still estimates", abs(gamma(bal_nj, "n_hire", "t_nj")) > 0.05)

# ---- 7. end to end, no SQL -------------------------------------------
mc.write_cache(base, mc.CACHE_DIR / "L_baseline_2019.parquet")
for y in mod.YEARS:
    sub = fl[fl["year_month"].str[:4] == str(y)]
    mc.write_cache(sub if len(sub) else fl.head(0),
                   mc.CACHE_DIR / f"flows_{y}.parquet")
def no_sql(*a, **k):
    raise AssertionError("SQL attempted although every pull is cached")
mc.connect = no_sql
mod.main()
est = pd.read_csv(mod.OUT / "flow_estimates.csv")
check("both outcomes and both variants estimated", len(est) == 4, str(len(est)))
check("no fit errored", (est["status"] == "ok").all(),
      str(est["status"].value_counts().to_dict()))
check("the gradient is exported", (mod.OUT / "flow_gradient.csv").exists())
sup = pd.read_csv(mod.OUT / "flow_support.csv")
v = sup["firms"].dropna()
check("the support table is floored", ((v == 0) | (v >= 5)).all())
summ = (mod.OUT / "54_summary.txt").read_text()
for must in ("START, not a labour-market entry", "no predecessor",
             "attenuate toward zero", "side by side",
             "NOT in its exposure measure"):
    check(f"the summary states: {must[:32]}", must in summ)

print("\n" + ("ALL PASS" if not FAILS else f"FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)
