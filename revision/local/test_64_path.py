#!/usr/bin/env python3
"""
test_64_path.py -- the event study has to tell a step from a trend, and
                   it has to fail when the pre-period is not flat.

Three worlds, each planted and each with a known right answer:

  STEP      nothing until January 2024, then a constant decline. The path
            must be flat before 2024Q1 and negative in every quarter
            after, and the summary must call the pre-period FLAT.
  TREND     a decline that grows month by month from January 2024. The
            path must deepen across the post quarters rather than jump.
  PRE-TREND a decline that starts in 2021, before anything happened. The
            pre-period test must FAIL, because a check that cannot fail
            is not a check.

Also tested: the reference quarter is omitted rather than estimated, the
Riksbank control is absent because the quarter set spans it, and main()
runs end to end off the caches with SQL forbidden.

    CANARIES_DRYRUN=1 python3 revision/local/test_64_path.py
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
TMP = Path(tempfile.mkdtemp(prefix="canaries64_"))
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


s64 = load("64_within_employer_path.py", "s64")
s64.OUT = TMP / "out"; s64.OUT.mkdir(); s64.CACHE = mc.CACHE_DIR
s61 = s64._mod("61_redated_triple.py", "s61")
s61.OUT = s64.OUT
j47 = s61._j47()
h47 = j47._h47()

from _fixtures import Fixture  # noqa: E402

FIX = Fixture(mc, h47)
book, spec, frames = FIX.install_edu(j47.YEARS)
expo, _ = j47.incumbent_exposure(frames[2019], book, "OL_daioe", spec, "true",
                                 s64.TRUNC)
Q4 = set(expo.loc[expo["fq"] == 4, "employer_id"].astype(int))
BETA = float(np.log(0.75))
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def counts_for(kind: str) -> pd.DataFrame:
    """Monthly counts in which the 22-25 decline has a known time shape."""
    rng = np.random.default_rng(64)
    lam0 = {"22-25": 6, "26-30": 8, "31-34": 7, "35-40": 9, "41-49": 12,
            "50+": 15}
    months = [f"{y}-{m:02d}" for y in s61.PANEL_YEARS
              for m in range(1, 13 if y < 2025 else 7)]
    rows = []
    for emp in range(1, FIX.n_firms + 1):
        hi = emp in Q4
        for ym in months:
            since = (int(ym[:4]) - 2024) * 12 + int(ym[5:]) - 1
            for age, lam in lam0.items():
                f = 1.0
                if hi and age == "22-25":
                    if kind == "step" and since >= 0:
                        f = np.exp(BETA)
                    elif kind == "trend" and since >= 0:
                        f = np.exp(BETA * min(1.0, (since + 1) / 18))
                    elif kind == "pretrend":
                        # A decline running from the very first month, which
                        # is what a failing parallel-trends world looks like.
                        # It has to be steep enough to clear the fixture's own
                        # noise: at 140 firms a quarterly coefficient carries
                        # a standard error near 0.06, so a drift of a few log
                        # points per quarter is invisible here and the test
                        # would be measuring the fixture rather than the rule.
                        t = (int(ym[:4]) - 2021) * 12 + int(ym[5:]) - 1
                        f = np.exp(BETA * t / 12)
                rows.append((emp, ym, age, int(rng.poisson(lam * f)) + 1))
    return pd.DataFrame(rows, columns=["employer_id", "year_month",
                                       "age_group", "n_emp"])


def path_for(kind: str, tag: str):
    skel = s61.build_skeleton(counts_for(kind), "22-25", j47)
    bal = skel.merge(expo[["employer_id", "fq"]], on="employer_id", how="inner")
    bal["high"] = (bal["fq"] == 4).astype(int)
    bal, terms = s64.add_quarter_terms(bal)
    r = mc.run_fepois_multi(bal, s64.OUT, tag=tag, terms=terms, fes=j47.FES)
    return r.assign(quarter=r["term"].str.slice(2)).set_index("quarter")


# ---- the term set itself ---------------------------------------------
probe = s61.build_skeleton(counts_for("step"), "22-25", j47)
probe = probe.merge(expo[["employer_id", "fq"]], on="employer_id", how="inner")
probe["high"] = (probe["fq"] == 4).astype(int)
probe, terms = s64.add_quarter_terms(probe)
qs = [t[2:] for t in terms]
check("the reference quarter is omitted, not estimated",
      s64.REF_Q not in qs and f"q_{s64.REF_Q}" not in terms, s64.REF_Q)
check("every other quarter in the panel gets a coefficient",
      len(qs) == probe["year_month"].str.slice(0, 4).nunique() * 4 - 2 - 1,
      f"{len(qs)} quarters, 2021Q1 to 2025Q2 less the reference")
check("the Riksbank control is not carried, the quarters span it",
      not any("post_rb" in t for t in terms))

# ---- world one: a step at 2024Q1 -------------------------------------
st = path_for("step", "t64_step")
pre = st[st.index < s64.REF_Q]["coef"]
pre_t = (st[st.index < s64.REF_Q]["coef"]
         / st[st.index < s64.REF_Q]["se"]).abs()
post = st[st.index >= "2024Q1"]["coef"]
pre_df = st[st.index < s64.REF_Q].reset_index().sort_values("quarter")
_, step_drift_t = s64.pre_slope(pre_df)
check("a planted step leaves the pre-period flat on both criteria",
      pre_t.max() <= s64.PRE_FLAT_T and abs(step_drift_t) <= s64.PRE_SLOPE_T,
      f"largest t {pre_t.max():.2f}, drift t {step_drift_t:+.2f}")
check("and every post quarter is negative",
      (post < 0).all(), " ".join(f"{v:+.3f}" for v in post))
# Two things shrink the recovered step below the planted log(0.75), both
# by design rather than by error. The fixture floors every count at one,
# which for a mean of six turns log(0.75) into log(5.5/7); and the
# month-by-age fixed effect absorbs the average response across firms, so
# what is estimated is the deviation from it and the exposed quartile's
# own share comes out. The test therefore brackets rather than pins.
floor_adj = float(np.log((6 * np.exp(BETA) + 1) / (6 + 1)))
check("a step is recovered, attenuated only by the floor and the "
      "month-by-age normalisation",
      0.4 * floor_adj > post.mean() > 1.15 * floor_adj,
      f"mean {post.mean():+.4f}, floor-adjusted plant {floor_adj:+.4f}")
mid = st.loc["2023Q4", "coef"] if "2023Q4" in st.index else 0.0
check("and nothing appears in the quarter before the break",
      abs(mid) < 0.08, f"2023Q4 {mid:+.4f}")

# ---- world two: a trend from 2024Q1 ----------------------------------
tr = path_for("trend", "t64_trend")
tp = tr[tr.index >= "2024Q1"]["coef"]
check("a planted trend deepens across the post quarters rather than jumping",
      tp.iloc[-1] < tp.iloc[0] - 0.05,
      f"first {tp.iloc[0]:+.4f}, last {tp.iloc[-1]:+.4f}")
check("so the design can tell a trend from a step",
      (tp.iloc[-1] - tp.iloc[0]) < (post.iloc[-1] - post.iloc[0]) - 0.05,
      f"trend spread {tp.iloc[-1]-tp.iloc[0]:+.3f} vs step spread "
      f"{post.iloc[-1]-post.iloc[0]:+.3f}")

# ---- world three: the check must be able to fail ---------------------
pt = path_for("pretrend", "t64_pretrend")
ppre = pt[pt.index < s64.REF_Q]["coef"]
ppre_t = (pt[pt.index < s64.REF_Q]["coef"]
          / pt[pt.index < s64.REF_Q]["se"]).abs()
ppre_df = pt[pt.index < s64.REF_Q].reset_index().sort_values("quarter")
pb, pbt = s64.pre_slope(ppre_df)
check("a decline that starts in 2021 breaks the pre-period test",
      ppre_t.max() > s64.PRE_FLAT_T or abs(pbt) > s64.PRE_SLOPE_T,
      f"largest single t {ppre_t.max():.2f}, drift {pb:+.4f} per quarter "
      f"at t {pbt:+.2f}")
# The drift criterion, tested on the function rather than through a fitted
# world. Note what this does NOT claim: against a clean ramp the
# single-quarter test is the sharper of the two, because the endpoint of a
# ramp carries a larger t than its slope. The drift test earns its place by
# being hard to move with one noisy quarter, which is the failure mode a
# max-t rule has.
ramp = pd.DataFrame({"coef": [0.100, 0.080, 0.060, 0.040, 0.020, 0.000],
                     "se": [0.05] * 6})
rb, rbt = s64.pre_slope(ramp)
check("the drift criterion recovers a known slope",
      abs(rb - (-0.02)) < 1e-9, f"{rb:+.5f} per quarter against -0.02000")
spike = ramp.copy()
spike["coef"] = [0.0, 0.0, 0.13, 0.0, 0.0, 0.0]
sb, sbt = s64.pre_slope(spike)
check("and one noisy quarter moves the drift far less than it moves max-t",
      (spike["coef"] / spike["se"]).abs().max() > 2.5 and abs(sbt) < 1.0,
      f"single-quarter t {(spike['coef']/spike['se']).abs().max():.2f}, "
      f"drift t {sbt:+.2f}: the two criteria catch different things")

# ---- end to end -------------------------------------------------------
FIX.install_occ(s61.PANEL_YEARS)          # placeholder, overwritten below
for y in s61.PANEL_YEARS:
    c = counts_for("step")
    c[c["year_month"].str.slice(0, 4) == str(y)].to_parquet(
        mc.CACHE_DIR / f"L_counts_{y}.parquet", index=False)


def boom(*a, **k):
    raise AssertionError("SQL attempted although the cache is warm")


mc.connect = boom
s64._mod = lambda name, alias: s61 if name.startswith("61") else load(name, alias)
s64.JOBS = (("OL_daioe", "true", "22-25"),)
s64.main()
P = pd.read_csv(s64.OUT / "path.csv")
check("main() writes one row per quarter", len(P) == len(qs), f"{len(P)} rows")
check("and every fit came back ok", (P["status"] == "ok").all())
summ = (s64.OUT / "64_summary.txt").read_text()
check("the summary states the pre-period verdict", "pre-period FLAT" in summ)
check("the summary counts the post quarters", "from 2024Q1:" in summ)
for must in ("READ THIS", "2023Q4", "preliminary AGI file"):
    check(f"the summary states {must!r}", must in summ)

print("\n" + ("ALL PASS" if not FAILS else f"FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)
