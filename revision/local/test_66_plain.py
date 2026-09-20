#!/usr/bin/env python3
"""
test_66_plain.py -- the descriptive translation must recover a planted
                    magnitude, and must not disclose a thin cell.

  1 with a known decline planted on the exposed quartile's young, the
    raw exposed-minus-rest change recovers it,
  2 the pre window excludes the pandemic years, so a 2020 shock planted
    outside it does not move the answer,
  3 no printed cell rests on fewer firms than the floor,
  4 main() runs end to end off the caches with SQL forbidden and states
    the quartile's employment share, which is what converts a
    coefficient into what a firm experienced.

    CANARIES_DRYRUN=1 python3 revision/local/test_66_plain.py
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
TMP = Path(tempfile.mkdtemp(prefix="canaries66_"))
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


s66 = load("66_plain_magnitudes.py", "s66")
s66.OUT = TMP / "out"; s66.OUT.mkdir(); s66.CACHE = mc.CACHE_DIR
s61 = s66._mod("61_redated_triple.py", "s61"); s61.OUT = s66.OUT
j47 = s61._j47(); h47 = j47._h47()

from _fixtures import Fixture  # noqa: E402

FIX = Fixture(mc, h47)
book, spec, frames = FIX.install_edu(j47.YEARS)
expo, _ = j47.incumbent_exposure(frames[2019], book, "OL_daioe", spec, "true",
                                 s61.TRUNC)
Q4 = set(expo.loc[expo["fq"] == 4, "employer_id"].astype(int))
BETA = float(np.log(0.75))
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def _base():
    """
    The unshocked panel, drawn ONCE. Every variant below thins this same
    draw rather than redrawing with a different mean, because
    rng.poisson consumes a variable amount of randomness and redrawing
    desynchronises the whole stream: the 2022 onward months would then
    differ between variants for a reason that has nothing to do with the
    thing being varied. The fixture module carries the same warning and I
    walked into it anyway.
    """
    rng = np.random.default_rng(66)
    lam0 = {"22-25": 6, "26-30": 8, "31-34": 7, "35-40": 9, "41-49": 12,
            "50+": 15}
    rows = []
    for emp in range(1, FIX.n_firms + 1):
        for y in s61.PANEL_YEARS:
            for m in range(1, 13 if y < 2025 else 7):
                ym = f"{y}-{m:02d}"
                for age, lam in lam0.items():
                    rows.append((emp, ym, age, int(rng.poisson(lam))))
    return pd.DataFrame(rows, columns=["employer_id", "year_month",
                                       "age_group", "n_emp"])


BASE = _base()


def counts(shock_from="2024-01", covid=False):
    """Thin the shared draw, so variants differ only where intended."""
    rng = np.random.default_rng(99)
    c = BASE.copy()
    hit = (c["employer_id"].isin(Q4) & (c["age_group"] == "22-25")
           & (c["year_month"] >= shock_from))
    c.loc[hit, "n_emp"] = rng.binomial(c.loc[hit, "n_emp"], np.exp(BETA))
    if covid:
        # a 2021 shock, outside the pre window, which the comparison must
        # not pick up
        pre21 = (c["age_group"] == "22-25") & (c["year_month"] < "2022-01")
        c.loc[pre21, "n_emp"] = rng.binomial(c.loc[pre21, "n_emp"], 0.5)
    c["n_emp"] = c["n_emp"].astype(int) + 1
    return c


def gap(c):
    g = s66.describe(c, expo, "n_emp", j47.YOUNG_BANDS, j47.INCUMBENT_BANDS)
    w = s66.age_ratio(g, "22-25", j47.INCUMBENT_BANDS)
    hi = float(w[w.fq == 4]["log_change"].iloc[0])
    lo = float(w[w.fq < 4]["log_change"].mean())
    return hi - lo, g


d, g = gap(counts())
# the +1 floor on a mean of six turns log(0.75) into log(5.5/7)
floor_adj = float(np.log((6 * np.exp(BETA) + 1) / (6 + 1)))
check("the raw exposed-minus-rest change recovers the planted decline",
      abs(d - floor_adj) < 0.05,
      f"{d:+.4f} against a floor-adjusted plant of {floor_adj:+.4f}")
check("no printed cell rests on fewer firms than the floor",
      (g["n_firms"] >= s66.MIN_FIRMS).all(),
      f"smallest cell {int(g['n_firms'].min())} firms")

d2, _ = gap(counts(covid=True))
check("a 2021 shock outside the pre window does not move the comparison",
      abs(d2 - d) < 0.03,
      f"{d2:+.4f} against {d:+.4f}: the window starts in 2022 for this reason")

# ---- end to end -------------------------------------------------------
c = counts()
for y in s61.PANEL_YEARS:
    c[c["year_month"].str.slice(0, 4) == str(y)].to_parquet(
        mc.CACHE_DIR / f"L_counts_{y}.parquet", index=False)
    f = c[c["year_month"].str.slice(0, 4) == str(y)].rename(
        columns={"n_emp": "n_hire"}).copy()
    f["n_sep"] = f["n_hire"]
    f.to_parquet(mc.CACHE_DIR / f"flows_{y}.parquet", index=False)


def boom(*a, **k):
    raise AssertionError("SQL attempted although the cache is warm")


mc.connect = boom
s66.main()
P = pd.read_csv(s66.OUT / "plain_stock.csv")
check("main() writes the stock table", len(P) > 0 and
      (P["n_firms"] >= s66.MIN_FIRMS).all())
check("and the flows table", (s66.OUT / "plain_flows.csv").exists())
summ = (s66.OUT / "66_summary.txt").read_text()
check("the summary states the exposed quartile's employment share",
      "of incumbent employment" in summ,
      "this is what converts a coefficient into what a firm experienced")
for must in ("Descriptive", "READ THIS", "never present one as though it "
             "were the other".capitalize()[:20]):
    check(f"the summary states {must!r}", must in summ)

print("\n" + ("ALL PASS" if not FAILS else f"FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)
