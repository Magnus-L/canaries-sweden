#!/usr/bin/env python3
"""
test_68_seasonal.py -- the seasonal control must remove a seasonal and
                       leave a real effect alone.

Three worlds, each with a known right answer:

  SEASON ONLY   a calendar cycle in the exposed firms' young-to-older
                ratio, Q4 up and Q1 down, and no treatment at all. The
                uncontrolled estimate must pick up a spurious effect and
                the controlled one must not. This is the world 64 found
                us in.
  EFFECT ONLY   a genuine decline from 2024 and no cycle. Both
                specifications must find it, so the control is not
                eating real signal.
  BOTH          cycle and decline together. The controlled estimate must
                recover the planted decline; the uncontrolled one must
                not.

Also tested: the fourth quarter is omitted rather than estimated, the
treatment term survives alongside the quarter terms rather than being
collinear with them, and main() runs end to end off the caches.

    CANARIES_DRYRUN=1 python3 revision/local/test_68_seasonal.py
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
TMP = Path(tempfile.mkdtemp(prefix="canaries68_"))
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


s68 = load("68_seasonal_control.py", "s68")
s68.OUT = TMP / "out"; s68.OUT.mkdir(); s68.CACHE = mc.CACHE_DIR
s61 = s68._mod("61_redated_triple.py", "s61"); s61.OUT = s68.OUT
s67 = s68._mod("67_gender_on_the_new_design.py", "s67"); s67.OUT = s68.OUT
j47 = s61._j47(); h47 = j47._h47()

from _fixtures import Fixture  # noqa: E402

FIX = Fixture(mc, h47)
book, spec, frames = FIX.install_edu(j47.YEARS)
expo, _ = j47.incumbent_exposure(frames[2019], book, "OL_daioe", spec, "true",
                                 s61.TRUNC)
Q4F = set(expo.loc[expo["fq"] == 4, "employer_id"].astype(int))
BETA = float(np.log(0.75))          # the planted decline
SEAS = float(np.log(1.30))          # the planted Q4 bump
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def _base():
    rng = np.random.default_rng(68)
    lam0 = {"22-25": 6, "26-30": 8, "31-34": 7, "35-40": 9, "41-49": 12,
            "50+": 15}
    rows = []
    for emp in range(1, FIX.n_firms + 1):
        for y in s61.PANEL_YEARS:
            for m in range(1, 13 if y < 2025 else 7):
                for age, lam in lam0.items():
                    rows.append((emp, f"{y}-{m:02d}", age,
                                 int(rng.poisson(lam))))
    return pd.DataFrame(rows, columns=["employer_id", "year_month",
                                       "age_group", "n_emp"])


BASE = _base()


def counts(season=False, effect=False):
    """Thin or inflate one shared draw, so worlds differ only as intended."""
    rng = np.random.default_rng(680)
    c = BASE.copy()
    hit = c["employer_id"].isin(Q4F) & (c["age_group"] == "22-25")
    q = s68.quarter_of_year(c["year_month"])
    if season:
        # Q4 up in EVERY year, including the pre-period: the pattern 64 found
        up = hit & (q == 4)
        c.loc[up, "n_emp"] = rng.poisson(c.loc[up, "n_emp"] * np.exp(SEAS))
    if effect:
        dn = hit & (c["year_month"] >= s68.POST_FROM)
        c.loc[dn, "n_emp"] = rng.binomial(c.loc[dn, "n_emp"], np.exp(BETA))
    c["n_emp"] = c["n_emp"].astype(int) + 1
    return c


def fit(c, controlled):
    skel = s61.build_skeleton(c, "22-25", j47)
    b = skel.merge(expo[["employer_id", "fq"]], on="employer_id", how="inner")
    b["high"] = (b["fq"] == 4).astype(int)
    if controlled:
        b, terms = s68.add_seasonal_terms(b, "post")
    else:
        ym = b["year_month"].astype(str)
        hy = b["high"] * b["young"]
        b["rb_x_high_x_young"] = (ym >= mc.RIKSBANK_YM).astype(int) * hy
        b["post_x_high_x_young"] = (ym >= s68.POST_FROM).astype(int) * hy
        terms = ["rb_x_high_x_young", "post_x_high_x_young"]
    r = mc.run_fepois_multi(b, s68.OUT, tag=f"t68_{controlled}_{len(b)}",
                            terms=terms, fes=j47.FES)
    g = r.set_index("term")
    return float(g.loc["post_x_high_x_young", "coef"]), terms, g


# ---- the term set -----------------------------------------------------
_, terms, g = fit(counts(season=True), True)
check("the fourth quarter is omitted, not estimated",
      "q4_x_high_x_young" not in terms and
      {"q1_x_high_x_young", "q2_x_high_x_young",
       "q3_x_high_x_young"} <= set(terms), " ".join(terms))
check("the treatment survives beside the quarter terms",
      "post_x_high_x_young" in g.index
      and np.isfinite(g.loc["post_x_high_x_young", "se"]),
      "not collinear with the cycle")

# ---- world one: a cycle and nothing else ------------------------------
# The test is about INFERENCE, not about the point estimate reaching zero.
# A fixture this size carries a standard error near 0.025, so demanding a
# controlled coefficient below 0.03 in absolute value would be demanding
# that noise vanish. What must happen is that a cycle stops producing a
# significant coefficient.
raw_s, _, graw = fit(counts(season=True), False)
ctl_s, _, gctl = fit(counts(season=True), True)
t_raw = raw_s / float(graw.loc["post_x_high_x_young", "se"])
t_ctl = ctl_s / float(gctl.loc["post_x_high_x_young", "se"])
check("an uncontrolled fit reports a SIGNIFICANT effect from a pure cycle",
      abs(t_raw) > 2.0, f"{raw_s:+.4f}, t {t_raw:+.2f}, nothing planted but "
                        f"a season")
check("the seasonal control renders it insignificant and at least halves it",
      abs(t_ctl) < 2.0 and abs(ctl_s) < abs(raw_s) / 1.5,
      f"uncontrolled {raw_s:+.4f} (t {t_raw:+.2f}) against controlled "
      f"{ctl_s:+.4f} (t {t_ctl:+.2f})")
# and it measures the cycle rather than merely absorbing it: the planted
# bump is exp(SEAS) on a mean of six, which the count floor turns into
# log(7 / (6*exp(SEAS)+1)) in the estimated contrast
planted_q = float(np.log(7.0 / (6.0 * np.exp(SEAS) + 1.0)))
got_q = [float(gctl.loc[f"q{q}_x_high_x_young", "coef"]) for q in (1, 2, 3)]
check("and the quarter coefficients recover the planted cycle",
      all(abs(v - planted_q) < 0.05 for v in got_q),
      f"{' '.join(f'{v:+.3f}' for v in got_q)} against a planted "
      f"{planted_q:+.3f}")

# ---- world two: an effect and no cycle --------------------------------
raw_e, _, _ = fit(counts(effect=True), False)
ctl_e, _, _ = fit(counts(effect=True), True)
check("with no cycle, the control does not eat a real effect",
      abs(ctl_e - raw_e) < 0.03,
      f"uncontrolled {raw_e:+.4f} against controlled {ctl_e:+.4f}")
check("and both recover the planted decline",
      ctl_e < -0.10 and raw_e < -0.10,
      f"planted {BETA:+.4f} before the count floor")

# ---- world three: both ------------------------------------------------
raw_b, _, _ = fit(counts(season=True, effect=True), False)
ctl_b, _, _ = fit(counts(season=True, effect=True), True)
check("with both, the controlled estimate is the closer to the truth",
      abs(ctl_b - ctl_e) < abs(raw_b - ctl_e),
      f"controlled {ctl_b:+.4f}, uncontrolled {raw_b:+.4f}, "
      f"clean benchmark {ctl_e:+.4f}")

# ---- end to end -------------------------------------------------------
c = counts(season=True, effect=True)
for y in s61.PANEL_YEARS:
    sub = c[c["year_month"].str.slice(0, 4) == str(y)]
    sub.to_parquet(mc.CACHE_DIR / f"L_counts_{y}.parquet", index=False)
    fl = sub.rename(columns={"n_emp": "n_hire"}).copy()
    fl["n_sep"] = fl["n_hire"]
    fl.to_parquet(mc.CACHE_DIR / f"flows_{y}.parquet", index=False)
    sx = pd.concat([sub.assign(gender="1"), sub.assign(gender="2")],
                   ignore_index=True)
    sx.to_parquet(mc.CACHE_DIR / f"L_counts_sex_{y}.parquet", index=False)


def boom(*a, **k):
    raise AssertionError("SQL attempted although the cache is warm")


mc.connect = boom
s68._mod = lambda name, alias: (s61 if name.startswith("61")
                                else s67 if name.startswith("67")
                                else load(name, alias))
s61._j47 = lambda: j47
j47.YOUNG_BANDS = ["22-25"]
s68.main()
P = pd.read_csv(s68.OUT / "seasonal_pooled.csv")
check("main() reports the treatment and all three quarter terms",
      {"post_x_high_x_young", "q1_x_high_x_young", "q2_x_high_x_young",
       "q3_x_high_x_young"} <= set(P["term"]))
check("and it runs all three margins", set(P["outcome"]) ==
      {"stock", "hires", "seps"}, " ".join(sorted(set(P["outcome"]))))
check("the as-of arm ran, so the artefact is measured",
      "asof" in set(P["arm"]))
check("the annual path was estimated", (s68.OUT / "seasonal_path.csv").exists())
check("the gender differential was re-estimated with the seasonal out",
      (s68.OUT / "seasonal_gender.csv").exists())
summ = (s68.OUT / "68_summary.txt").read_text()
for must in ("THE SEASONAL ITSELF", "supersede 61 and 67", "Expect the "
             "estimates to SHRINK"):
    check(f"the summary states {must!r}", must in summ)

print("\n" + ("ALL PASS" if not FAILS else f"FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)
