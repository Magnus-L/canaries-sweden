#!/usr/bin/env python3
"""
test_75_reference_window.py -- the window arm must return the LEVEL that
                               68's cumulative arm only implies.

One world: exposed firms' 22-25 employment rises by a known step from
April 2022 (the tightening window) and falls by a known step from January
2024 (adoption), with a Q4 seasonal in every year so the calendar control
has work to do. Then:

  * 68's cumulative terms must return rb ~ +step1 and post ~ step2, the
    step from the tightening level;
  * 75's window terms must return rbw ~ +step1 and post ~ step1 + step2,
    the level after adoption against the pre-hike months;
  * the two must reconcile: 75's post within one SE of 68's rb + post,
    which is the read rule the script applies to the real data;
  * the R wrapper must now write the clustered covariance beside the
    table, and the script must copy it into its output directory;
  * main() must run end to end off the caches with SQL forbidden.

    CANARIES_DRYRUN=1 python3 revision/local/test_75_reference_window.py
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
TMP = Path(tempfile.mkdtemp(prefix="canaries75_"))
SHARE = TMP / "input"; SHARE.mkdir()
for f in ("daioe_quartiles.dta", "eloundou_ssyk4.dta",
          "utb_grupp2_sun2020_niva3_inr4_nyckel.dta"):
    shutil.copy(UPLOAD / f, SHARE / f)
os.environ["CANARIES_SHARE"] = str(SHARE)
sys.path.insert(0, str(MONA)); sys.path.insert(0, str(HERE))
import mona_common as mc  # noqa: E402
mc.SHARE = str(SHARE); mc.CACHE_DIR = TMP / "cache"; mc.CACHE_DIR.mkdir()


def load(n, a):
    sp = importlib.util.spec_from_file_location(a, MONA / n)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


s75 = load("75_reference_window.py", "s75")
s75.OUT = TMP / "out"; s75.OUT.mkdir(); s75.CACHE = mc.CACHE_DIR
s68 = load("68_seasonal_control.py", "s68"); s68.OUT = s75.OUT
s61 = s75._mod("61_redated_triple.py", "s61"); s61.OUT = s75.OUT
j47 = s61._j47(); h47 = j47._h47()

from _fixtures import Fixture  # noqa: E402

FIX = Fixture(mc, h47)
book, spec, frames = FIX.install_edu(j47.YEARS)
expo, _ = j47.incumbent_exposure(frames[2019], book, "OL_daioe", spec, "true",
                                 s61.TRUNC)
Q4F = set(expo.loc[expo["fq"] == 4, "employer_id"].astype(int))
STEP1 = float(np.log(1.12))         # the tightening-window rise
STEP2 = float(np.log(0.78))         # the adoption step, from that level
SEAS = float(np.log(1.25))          # a Q4 bump in every year
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def counts():
    rng = np.random.default_rng(75)
    lam0 = {"22-25": 8, "26-30": 8, "31-34": 7, "35-40": 9, "41-49": 12,
            "50+": 15}
    rows = []
    for emp in range(1, FIX.n_firms + 1):
        hit = emp in Q4F
        for y in s61.PANEL_YEARS:
            for m in range(1, 13 if y < 2025 else 7):
                ym = f"{y}-{m:02d}"
                q = (m - 1) // 3 + 1
                for age, lam in lam0.items():
                    lam_ = lam
                    if hit and age == "22-25":
                        if q == 4:
                            lam_ *= np.exp(SEAS)
                        if ym >= mc.RIKSBANK_YM:
                            lam_ *= np.exp(STEP1)
                        if ym >= s75.POST_FROM:
                            lam_ *= np.exp(STEP2)
                    rows.append((emp, ym, age, int(rng.poisson(lam_)) + 1))
    return pd.DataFrame(rows, columns=["employer_id", "year_month",
                                       "age_group", "n_emp"])


C = counts()


def fit(window: bool):
    skel = s61.build_skeleton(C, "22-25", j47)
    b = skel.merge(expo[["employer_id", "fq"]], on="employer_id", how="inner")
    b["high"] = (b["fq"] == 4).astype(int)
    if window:
        b, terms = s75.add_window_terms(b)
    else:
        b, terms = s68.add_seasonal_terms(b, "post")
    r = mc.run_fepois_multi(b, s75.OUT, tag=f"t75_{'win' if window else 'cum'}",
                            terms=terms, fes=j47.FES)
    return r.set_index("term"), r


gc_, rc = fit(False)
gw, rw = fit(True)
rb_c = float(gc_.loc["rb_x_high_x_young", "coef"])
post_c = float(gc_.loc["post_x_high_x_young", "coef"])
rbw = float(gw.loc["rbw_x_high_x_young", "coef"])
post_w, se_w = (float(gw.loc["post_x_high_x_young", "coef"]),
                float(gw.loc["post_x_high_x_young", "se"]))
int_w = float(gw.loc["interim_x_high_x_young", "coef"])

check("68's cumulative arm recovers the tightening rise",
      abs(rb_c - STEP1) < 0.05, f"rb {rb_c:+.4f} against planted {STEP1:+.4f}")
check("and its post term is the step FROM the tightening level",
      abs(post_c - STEP2) < 0.06, f"post {post_c:+.4f} against planted {STEP2:+.4f}")
check("75's window term recovers the same rise",
      abs(rbw - STEP1) < 0.05, f"rbw {rbw:+.4f}")
check("75's interim reads against the pre-hike months, so it carries the rise",
      abs(int_w - STEP1) < 0.06, f"interim {int_w:+.4f} against {STEP1:+.4f}")
check("75's post term is the LEVEL after adoption against the pre-hike months",
      abs(post_w - (STEP1 + STEP2)) < 0.06,
      f"post {post_w:+.4f} against planted {STEP1 + STEP2:+.4f}")
check("the two arms reconcile within one standard error (the read rule)",
      abs(post_w - (rb_c + post_c)) <= se_w,
      f"window {post_w:+.4f} vs cumulative rb+post {rb_c + post_c:+.4f}, SE {se_w:.4f}")
check("the R wrapper wrote the clustered covariance and it was copied out",
      "vcov" in rw.attrs and Path(rw.attrs["vcov"]).exists(),
      rw.attrs.get("vcov", "missing"))
if "vcov" in rw.attrs:
    V = pd.read_csv(rw.attrs["vcov"]).set_index("term")
    check("the covariance is square on the terms and its diagonal is SE squared",
          set(V.index) == set(V.columns)
          and abs(float(V.loc["post_x_high_x_young", "post_x_high_x_young"])
                  - se_w ** 2) < 1e-6 * max(1.0, se_w ** 2) * 100,
          f"diag {float(V.loc['post_x_high_x_young', 'post_x_high_x_young']):.6f} "
          f"vs se^2 {se_w**2:.6f}")

# ---- end to end -------------------------------------------------------
for y in s61.PANEL_YEARS:
    sub = C[C["year_month"].str.slice(0, 4) == str(y)]
    sub.to_parquet(mc.CACHE_DIR / f"L_counts_{y}.parquet", index=False)
    fl = sub.rename(columns={"n_emp": "n_hire"}).copy()
    fl["n_sep"] = fl["n_hire"]
    fl.to_parquet(mc.CACHE_DIR / f"flows_{y}.parquet", index=False)


def boom(*a, **k):
    raise AssertionError("SQL attempted although the cache is warm")


mc.connect = boom
s75._mod = lambda name, alias: s61 if name.startswith("61") else load(name, alias)
s61._j47 = lambda: j47
s75.JOBS = [("22-25", "stock"), ("22-25", "hires")]
s75.main()
R = pd.read_csv(s75.OUT / "reference_window.csv")
check("main() writes every term for every job it ran",
      {"rbw_x_high_x_young", "interim_x_high_x_young", "post_x_high_x_young",
       "q1_x_high_x_young"} <= set(R["term"])
      and set(R["outcome"]) == {"stock", "hires"})
check("and every fit came back ok", (R["status"] == "ok").all())
check("n_firms is recorded, so the sample is on the record",
      (R["n_firms"] > 0).all())
summ = (s75.OUT / "75_summary.txt").read_text()
for must in ("RECONCILIATION", "ADOPTION STEP", "level after adoption",
             "vcov_s75_"):
    check(f"the summary states {must!r}", must in summ)
check("the covariance files left with the outputs",
      len(list(s75.OUT.glob("vcov_s75_*.csv"))) == 2,
      f"{len(list(s75.OUT.glob('vcov_s75_*.csv')))} files")

print("\n" + ("ALL PASS" if not FAILS else f"FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)
