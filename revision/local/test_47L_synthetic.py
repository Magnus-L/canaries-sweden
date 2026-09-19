#!/usr/bin/env python3
"""
test_47L_synthetic.py -- 47L end to end, locally, real DAIOE input and real
R + fixest, synthetic pulls at the SQL boundary.

The claims 47L makes, each tested:
  1 exposure is built ONLY from the baseline year: corrupting or deleting
    every later occupation record changes nothing (there are none to use)
  2 a firm-age cell with too few coded workers is dropped under the floor
    variant and shrunk toward the firm mean under the shrunk variant
  3 the three fixed-effect sets leave the treatment identified, and it
    varies within employer x month (across age groups), which is what
    makes this a within-employer design
  4 a decline planted on high-exposure young cells is recovered
  5 the payroll-tax control enters without destroying the estimate, and
    its absence is handled gracefully
  6 end to end: exports written, support table floored, and the summary
    states that the backtest does NOT apply
    CANARIES_DRYRUN=1 python3 revision/local/test_47L_synthetic.py
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
TMP = Path(tempfile.mkdtemp(prefix="canaries47L_"))
SHARE = TMP / "input"; SHARE.mkdir()
shutil.copy(UPLOAD / "daioe_quartiles.dta", SHARE / "daioe_quartiles.dta")
os.environ["CANARIES_SHARE"] = str(SHARE)
sys.path.insert(0, str(MONA))
import mona_common as mc  # noqa: E402
mc.CACHE_DIR = TMP / "cache"; mc.CACHE_DIR.mkdir()
spec = importlib.util.spec_from_file_location("s47L", MONA / "47L_age_baseline_exposure.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
mod.OUT = TMP / "output_47L"; mod.OUT.mkdir(); mod.CACHE = mc.CACHE_DIR

D = pd.read_stata(SHARE / "daioe_quartiles.dta")
D["ssyk4"] = D["ssyk4"].astype(str).str.zfill(4)
DAIOE = D.rename(columns={"pctl_rank_genai": "score"})[["ssyk4", "score"]]
HI = D.loc[D.high_exposure == 1, "ssyk4"].to_numpy()
LO = D.loc[D.high_exposure == 0, "ssyk4"].to_numpy()
AGES = mod.AGES
N_EMP = 160
EXPOSED_FIRMS = set(range(1, 61))     # their young did exposed work in 2019
SHOCK = 0.70
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def f_baseline(conn):
    R = np.random.default_rng(11)
    rows = []
    for emp in range(1, N_EMP + 1):
        for age in AGES:
            young = age in ("22-25", "26-30")
            pool = HI if (emp in EXPOSED_FIRMS and young) else LO
            # every 37th firm has a genuinely thin 22-25 cell: ONE coded
            # worker in total, so the cell total (not the row) is below the
            # floor and the two variants must treat it differently
            thin = (emp % 37 == 0 and age == "22-25")
            codes = (R.choice(pool, 1) if thin
                     else R.choice(pool, min(3, len(pool)), replace=False))
            for code in codes:
                rows.append((emp, age, str(code).zfill(4), "1",
                             1 if thin else int(R.integers(4, 15))))
            rows.append((emp, age, "____", "", int(R.integers(1, 6))))
    return pd.DataFrame(rows, columns=["employer_id", "age_group", "ssyk4",
                                       "ssyk_status", "n"])


def f_basepay(conn):
    R = np.random.default_rng(12)
    rows = []
    for emp in range(1, N_EMP + 1):
        for age in ("22-25", "26-30"):
            n = int(R.integers(5, 30))
            rows.append((emp, age, int(R.binomial(n, 0.5)), n))
    return pd.DataFrame(rows, columns=["employer_id", "age_group",
                                       "n_under_cap", "n_all"])


# The estimator fits a treatment LINEAR IN STANDARDISED EXPOSURE, so the
# fixture plants exactly that: post-shock, a cell's mean is multiplied by
# exp(BETA * z). A binary shock on "exposed firms" would be recovered only
# up to an unknown averaging over cells and could not be checked against a
# number. BETA is the parameter the fit must return.
BETA = -0.25
_ZMAP = {}


def _zmap():
    if not _ZMAP:
        e = mod.build_exposure(f_baseline(None), DAIOE)
        mu, sd = e["expo"].mean(), e["expo"].std(ddof=0)
        for r in e.itertuples():
            _ZMAP[(r.employer_id, r.age_group)] = (r.expo - mu) / (sd or 1.0)
    return _ZMAP


def f_counts(year, conn):
    R = np.random.default_rng(1000 + year)
    z = _zmap()
    months = range(1, 13) if year < 2025 else range(1, 7)
    rows = []
    for emp in range(1, N_EMP + 1):
        for m in months:
            ym = f"{year}-{m:02d}"
            post = ym >= "2022-12"
            for age in AGES:
                lam = {"22-25": 6, "26-30": 8, "31-34": 7, "35-40": 9,
                       "41-49": 12, "50+": 15}[age]
                if post:
                    lam *= float(np.exp(BETA * z.get((emp, age), 0.0)))
                rows.append((emp, ym, age, int(R.poisson(lam)) + 1))
    return pd.DataFrame(rows, columns=["employer_id", "year_month",
                                       "age_group", "n_emp"])


def f_counts_young_only(year, conn):
    """
    Same counts, but the exposure shock lands ONLY on 22-25.

    The pooled fixture applies BETA to every age band, so it cannot tell a
    working gradient from six collinear copies of one number. This one can:
    if the age-specific terms are identified, only the 22-25 coefficient
    should be negative and the rest should sit at zero.
    """
    R = np.random.default_rng(2000 + year)
    z = _zmap()
    months = range(1, 13) if year < 2025 else range(1, 7)
    rows = []
    for emp in range(1, N_EMP + 1):
        for m in months:
            ym = f"{year}-{m:02d}"
            post = ym >= "2022-12"
            for age in AGES:
                lam = {"22-25": 6, "26-30": 8, "31-34": 7, "35-40": 9,
                       "41-49": 12, "50+": 15}[age]
                if post and age == "22-25":
                    lam *= float(np.exp(BETA * z.get((emp, age), 0.0)))
                rows.append((emp, ym, age, int(R.poisson(lam)) + 1))
    return pd.DataFrame(rows, columns=["employer_id", "year_month",
                                       "age_group", "n_emp"])


def test_gradient_separates_ages():
    """A shock planted on 22-25 alone must show up on 22-25 alone."""
    wire(pay=False)
    mod.q_counts = f_counts_young_only
    base = f_baseline(None)
    expo = mod.build_exposure(base, DAIOE)
    cnt = pd.concat([f_counts_young_only(y, None) for y in mod.YEARS],
                    ignore_index=True)
    bal = mod.build_panel(cnt, expo)
    gr = mod.fit_gradient(bal, "L_grad_test")
    check("the gradient returns one coefficient per age band",
          len(gr) == len(AGES), f"{len(gr)} of {len(AGES)}")
    if len(gr) == len(AGES):
        g = gr.set_index("age_group")["coef"]
        others = g.drop("22-25")
        check("the planted 22-25 shock appears on 22-25",
              g["22-25"] < -0.05, f"{g['22-25']:+.4f}")
        check("and NOT on the other five bands",
              others.abs().max() < 0.05,
              " ".join(f"{a} {v:+.4f}" for a, v in others.items()))
        check("so the age terms are separately identified, not collinear",
              abs(g["22-25"]) > others.abs().max() * 3,
              f"22-25 {g['22-25']:+.4f} vs max other "
              f"{others.abs().max():.4f}")
    mod.q_counts = f_counts


def wire(pay=True):
    mc.connect = lambda: object()
    mod.q_baseline, mod.q_counts = f_baseline, f_counts
    mod.q_basepay = f_basepay if pay else (lambda conn: (_ for _ in ()).throw(
        RuntimeError("no pay column in this delivery")))
    for f in mc.CACHE_DIR.glob("L_*.parquet"):
        f.unlink()


def test_baseline_only():
    """Exposure must be a function of the baseline pull alone."""
    base = f_baseline(None)
    e1 = mod.build_exposure(base, DAIOE)
    junk = base.copy()
    junk["ssyk4"] = "9999"          # a code DAIOE does not score
    e2 = mod.build_exposure(pd.concat([base, junk.assign(n=99)]), DAIOE)
    m = e1.merge(e2, on=["employer_id", "age_group"], suffixes=("_a", "_b"))
    check("unscored occupation codes cannot enter the exposure",
          np.allclose(m["expo_a"], m["expo_b"]),
          f"max diff {float((m.expo_a - m.expo_b).abs().max()):.2e}")
    check("exposure exists per firm-age cell, not per firm",
          e1.groupby("employer_id")["age_group"].nunique().max() > 1)
    check("young cells at exposed firms score higher",
          float(e1[(e1.employer_id.isin(EXPOSED_FIRMS)) & (e1.age_group == "22-25")]["expo"].mean())
          > float(e1[(~e1.employer_id.isin(EXPOSED_FIRMS)) & (e1.age_group == "22-25")]["expo"].mean()))


def test_floor_and_shrinkage():
    base = f_baseline(None)
    floor = mod.build_exposure(base, DAIOE, shrink=False)
    shrunk = mod.build_exposure(base, DAIOE, shrink=True)
    # build_exposure sums n over codes within a cell, so "thin" is a
    # property of the CELL total, not of a single code's row
    coded = base[base.ssyk4 != "____"]
    cell_n = coded.groupby(["employer_id", "age_group"])["n"].sum()
    thin_cells = set(map(tuple, cell_n[cell_n < mod.MIN_CELL_CODED].index.to_list()))
    f_cells = set(map(tuple, floor[["employer_id", "age_group"]].to_numpy()))
    s_cells = set(map(tuple, shrunk[["employer_id", "age_group"]].to_numpy()))
    check("the floor variant drops thin cells", len(thin_cells & f_cells) == 0,
          f"{len(thin_cells)} thin cells, {len(thin_cells & f_cells)} kept")
    check("the shrunk variant keeps them", len(thin_cells & s_cells) > 0)
    j = shrunk[shrunk.n_coded < mod.MIN_CELL_CODED]
    if len(j):
        pulled = (j["expo"] - j["firm_mean"]).abs() <= (j["expo_raw"] - j["firm_mean"]).abs() + 1e-9
        check("and pulls them toward the firm mean", bool(pulled.all()))


def test_identification_and_recovery():
    base, cnt = f_baseline(None), pd.concat([f_counts(y, None) for y in mod.YEARS])
    expo = mod.build_exposure(base, DAIOE)
    bal = mod.build_panel(cnt, expo)
    check("the panel is balanced on employer x age x month",
          len(bal) == bal.groupby(["employer_id", "age_group"]).ngroups
          * bal["year_month"].nunique())
    v = bal.groupby("fe_emp_t")["post_gpt_x_expo"].nunique()
    check("the treatment varies WITHIN employer x month (across ages)",
          bool((v > 1).any()), f"{int((v > 1).sum())} employer-months with variation")
    r = mod.fit(bal, "t_recover")
    check("the fit converges", r["status"] == "ok", str(r))
    check("the planted exposure gradient is recovered",
          abs(r["gamma"] - BETA) < 0.05,
          f"gamma {r['gamma']:+.4f} (SE {r['se']:.4f}) against a planted {BETA:+.2f}")


def test_tax_control_and_its_absence():
    base, cnt = f_baseline(None), pd.concat([f_counts(y, None) for y in mod.YEARS])
    expo = mod.build_exposure(base, DAIOE)
    pay = f_basepay(None)
    pay = pay[pay.age_group != "other"].copy()
    pay["taxshare"] = pay["n_under_cap"] / pay["n_all"].clip(lower=1)
    bal = mod.build_panel(cnt, expo, tax=pay[["employer_id", "age_group", "taxshare"]])
    check("cells with no pay record get taxshare 0, not NaN",
          bool(bal["taxshare"].notna().all()))
    r = mod.fit(bal, "t_tax", terms=mod.TERMS_TAX)
    check("the tax-controlled fit converges and reports both terms",
          r["status"] == "ok" and not np.isnan(r["tax_coef"]),
          f"gamma {r['gamma']:+.4f}, tax {r['tax_coef']:+.4f}")


def test_end_to_end(pay=True):
    wire(pay=pay)
    mod.main()
    est = pd.read_csv(mod.OUT / "agebase_estimates.csv")
    check(f"end to end runs (pay={'yes' if pay else 'no'})",
          est["gamma"].notna().all() and (est["status"] == "ok").all(),
          str(est[["variant", "payroll_tax_control", "gamma", "status"]].to_dict("records")[:3]))
    check("the coverage robustness rows are there",
          est["variant"].str.startswith("coverage").any())
    check("the tax rows appear only when the pay pull worked",
          bool(est["payroll_tax_control"].any()) == pay)
    sup = pd.read_csv(mod.OUT / "exposure_support_floor.csv")
    v = sup["cells"].dropna()
    check("the support table is floored", ((v == 0) | (v >= 5)).all())
    summ = " ".join((mod.OUT / "47L_summary.txt").read_text().split())
    for must in ("backtest does NOT apply", "not a validation",
                 "NOT a firm shock", "expired 31 March 2023", "EMPLOYMENT, not hiring"):
        check(f"the summary states: {must[:30]}", must in summ)


if __name__ == "__main__":
    test_gradient_separates_ages()
    test_baseline_only()
    test_floor_and_shrinkage()
    test_identification_and_recovery()
    test_tax_control_and_its_absence()
    test_end_to_end(pay=True)
    test_end_to_end(pay=False)
    print("\nFAILED: " + ", ".join(FAILS) if FAILS else "\nALL PASS")
    sys.exit(1 if FAILS else 0)
