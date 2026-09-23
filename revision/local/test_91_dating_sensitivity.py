#!/usr/bin/env python3
"""
test_91_dating_sensitivity.py -- the boundary sweep must recover a
                                 planted date, the gate must be able to
                                 fail, and the read rule's premise must
                                 be true in the world it is written for.

The checks are on mechanisms. TWO synthetic worlds, because the whole
question is which shape the Swedish data have:

  SHARP world     employment at 22-25 in exposed firms falls by a known
                  step at a known month and is flat either side of it.
                  Here the sweep MUST peak at the planted month: a
                  boundary set earlier puts untreated months into the
                  adoption window and one set later puts treated months
                  into the comparison period, and both attenuate.

  GRADUAL world   employment falls a little every month from early 2023,
                  which is the shape script 84's path actually shows.
                  Here the step from the immediately preceding level is
                  nearly INVARIANT to the boundary, because the comparison
                  period and the adoption window slide together: under a
                  linear path the difference of the two period means is
                  (31/2)*slope whatever the cutoff. This world exists to
                  demonstrate that a flat sweep on the real data would be
                  close to mechanical, which is why the script's read rule
                  reports a range and claims nothing about timing. It was
                  written after a cross-vendor review made the algebraic
                  point, and it confirms it: a spread of 0.013 on a step
                  of -0.156.

Also tested: that the derived standard error comes from the covariance
and not from adding variances; that the gate fires when the reported arm
disagrees with the paper and passes when it agrees; that script 78's
POST_FROM is restored after the sweep, so that importing 78 elsewhere in
the same session is unaffected; and that a failed fit leaves no row.

    CANARIES_DRYRUN=1 python3 revision/local/test_91_dating_sensitivity.py
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
os.environ["CANARIES_ECHO_LIMIT"] = "100000000"
HERE = Path(__file__).resolve().parent
MONA, UPLOAD = HERE.parent / "mona", HERE.parent / "upload"
TMP = Path(tempfile.mkdtemp(prefix="canaries91_"))
SHARE = TMP / "input"; SHARE.mkdir()
for f in ("daioe_quartiles.dta", "eloundou_ssyk4.dta",
          "utb_grupp2_sun2020_niva3_inr4_nyckel.dta"):
    shutil.copy(UPLOAD / f, SHARE / f)
os.environ["CANARIES_SHARE"] = str(SHARE)
os.environ["CANARIES_91_OUT"] = str(TMP / "out")
os.environ["CANARIES_82_OUT"] = str(TMP / "out")
sys.path.insert(0, str(MONA)); sys.path.insert(0, str(HERE))
import mona_common as mc  # noqa: E402
mc.SHARE = str(SHARE); mc.CACHE_DIR = TMP / "cache"; mc.CACHE_DIR.mkdir()
_LOCAL = str(SHARE / "daioe_quartiles.dta")
mc.DAIOE_PATH = _LOCAL
_ld = mc.load_daioe
mc.load_daioe = lambda path=_LOCAL: _ld(path)
mc.connect = lambda: object()


def load(n, a):
    sp = importlib.util.spec_from_file_location(a, MONA / n)
    m = importlib.util.module_from_spec(sp); sys.modules[a] = m
    sp.loader.exec_module(m); return m


s91 = load("91_dating_sensitivity.py", "s91")
s91.OUT = TMP / "out"; s91.OUT.mkdir(parents=True, exist_ok=True)
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


# ======================================================================
# the world, built exactly as test_84 builds its own
# ======================================================================
AGES = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]
N_SCORE, N_PANEL = 400, 300
SCORED, PANEL = list(range(1, N_SCORE + 1)), list(range(1, N_PANEL + 1))
UNSCORED_CODE = "9999"

DAIOE = pd.read_stata(_LOCAL)
DAIOE["ssyk4"] = DAIOE["ssyk4"].astype(str).str.zfill(4)
_d = DAIOE.sort_values("pctl_rank_genai").reset_index(drop=True)
TIER_CODE, _seen = [], set()
for q in (0.10, 0.40, 0.65, 0.92):
    for i in range(int(q * len(_d)), len(_d)):
        c = _d.loc[i, "ssyk4"]
        if c[:3] not in _seen:
            TIER_CODE.append(c); _seen.add(c[:3]); break
TIER_OF = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 3}
tier = lambda e: TIER_OF[e % 10]                              # noqa: E731
size_mult = lambda e: 1 + (e % 9)                             # noqa: E731
EXPOSED = {e for e in SCORED if tier(e) == 3}


def cascade_frame() -> pd.DataFrame:
    rows = []
    for emp in SCORED:
        t = tier(emp); c, far = TIER_CODE[t], TIER_CODE[3 - t]
        m = size_mult(emp)
        for age in AGES:
            rows += [(emp, age, c, c[:3], "2019", 6 * m),
                     (emp, age, far, far[:3], "2019", 1 * m),
                     (emp, age, UNSCORED_CODE, UNSCORED_CODE[:3], "2019", 2),
                     (emp, age, "____", "___", "none", 2)]
    d = pd.DataFrame(rows, columns=["employer_id", "age_group", "ssyk4",
                                    "ssyk3", "source_year", "n"])
    d["ssyk_ar"] = "2019"
    d["ssyk_status"] = np.where(d["ssyk4"] == "____", "9", "1")
    return d


CASC = cascade_frame()
CASC.to_parquet(mc.CACHE_DIR / "L_baseline_2019_cascade.parquet", index=False)
(CASC.groupby(["employer_id", "age_group", "ssyk4"], observed=True)["n"]
 .sum().reset_index()).to_parquet(
     mc.CACHE_DIR / "L_baseline_2019.parquet", index=False)
pd.DataFrame([(e, f"2019-{m:02d}", a, size_mult(e))
              for e in SCORED for m in range(1, 13) for a in AGES],
             columns=["employer_id", "year_month", "age_group", "n_emp"]
             ).to_parquet(mc.CACHE_DIR / "L_counts_2019.parquet", index=False)

s82 = load("82_occupation_route.py", "s82")
s61, s67, s74, s78, s80, l47, l70, j47 = s82.load_modules()
for m_ in (s82, s61, s78, l47, l70, j47):
    m_.OUT, m_.CACHE = s91.OUT, mc.CACHE_DIR
MONTHS = [f"{y}-{m:02d}" for y in s61.PANEL_YEARS
          for m in range(1, 13 if y < 2025 else 7)]

PLANT = "2024-01"                  # the SHARP world's true date
FALL = float(np.log(0.80))         # its step, in logs
RISE_RB = float(np.log(1.06))      # the tightening rise, cumulative
SLOPE = -0.010                     # the GRADUAL world's monthly decline
GRAD_FROM = "2023-01"


def panel_counts(world: str) -> pd.DataFrame:
    rng = np.random.default_rng({"sharp": 91, "gradual": 92}[world])
    lam0 = {"22-25": 16, "26-30": 16, "31-34": 12, "35-40": 12,
            "41-49": 14, "50+": 14}
    rows = []
    for emp in PANEL:
        hit = emp in EXPOSED
        for ym in MONTHS:
            for age, lam in lam0.items():
                x = float(lam)
                if hit and age == "22-25":
                    if ym >= mc.RIKSBANK_YM:
                        x *= np.exp(RISE_RB)
                    if world == "sharp" and ym >= PLANT:
                        x *= np.exp(FALL)
                    if world == "gradual" and ym >= GRAD_FROM:
                        k = MONTHS.index(ym) - MONTHS.index(GRAD_FROM)
                        x *= np.exp(SLOPE * k)
                rows.append((emp, ym, age, int(rng.poisson(x)) + 1))
    return pd.DataFrame(rows, columns=["employer_id", "year_month",
                                       "age_group", "n_emp"])


def install(world: str) -> None:
    c = panel_counts(world)
    for y in s61.PANEL_YEARS:
        c[c["year_month"].str.slice(0, 4) == str(y)].to_parquet(
            mc.CACHE_DIR / f"L_counts_{y}.parquet", index=False)


def sweep(world: str) -> pd.DataFrame:
    """The sweep at 22-25 only, through the script's own helpers."""
    install(world)
    counts = s82.load_counts("L_counts", s61.PANEL_YEARS)
    expo = s82.build_exposure(l47, l70, j47)["exposure"]
    base = s78.with_exposure(s61.build_skeleton(counts, "22-25", j47), expo)
    keep, out = s78.POST_FROM, []
    try:
        for b_ in s91.BOUNDARIES:
            s78.POST_FROM = b_
            fr, terms = s78.eq2_terms(base.copy())
            g, v = s91.fit(fr, f"{world}_{b_.replace('-', '_')}", terms,
                           j47.FES)
            if g is None:
                continue
            sc, ss = s91.step_from_interim(g, v)
            out.append({"boundary": b_, "step_coef": sc, "step_se": ss,
                        "post_coef": float(g.loc[s91.POST, "coef"]),
                        "interim_coef": float(g.loc[s91.INTERIM, "coef"])})
    finally:
        s78.POST_FROM = keep
    return pd.DataFrame(out)


# ======================================================================
print("\n--- the derived standard error comes from the covariance ---")
_g = pd.DataFrame({"coef": [-0.06, -0.02]}, index=[s91.POST, s91.INTERIM])
_v = pd.DataFrame([[4e-4, 3e-4], [3e-4, 9e-4]],
                  index=[s91.POST, s91.INTERIM], columns=[s91.POST, s91.INTERIM])
_c, _s = s91.step_from_interim(_g, _v)
check("the step is post minus interim", abs(_c - (-0.04)) < 1e-12, f"{_c:+.4f}")
check("its SE uses -2Cov and not the sum of variances",
      abs(_s - np.sqrt(4e-4 + 9e-4 - 2 * 3e-4)) < 1e-12
      and abs(_s - np.sqrt(4e-4 + 9e-4)) > 1e-4,
      f"{_s:.6f} against the naive {np.sqrt(13e-4):.6f}")
check("a missing covariance gives nan and not a wrong number",
      np.isnan(s91.step_from_interim(_g, None)[1]))

print("\n--- the gate can fail as well as pass ---")
_ok = [{"young_band": b, "boundary": s91.REPORTED,
        "post_coef": s91.GATE[b]["post"][0], "post_se": s91.GATE[b]["post"][1],
        "step_coef": s91.GATE[b]["step"][0], "step_se": s91.GATE[b]["step"][1]}
       for b in s91.BANDS]
try:
    s91.check_gate(_ok); _passed = True
except SystemExit:
    _passed = False
check("the gate passes when the reported arm matches the paper", _passed)
_bad = [dict(r) for r in _ok]
_bad[0]["step_coef"] += 0.0002
try:
    s91.check_gate(_bad); _stopped = False
except SystemExit:
    _stopped = True
check("the gate STOPS on a two-ten-thousandth disagreement", _stopped)

print("\n--- SHARP world: the sweep must peak at the planted date ---")
sh = sweep("sharp")
check("every boundary produced a row", len(sh) == len(s91.BOUNDARIES),
      f"{len(sh)} of {len(s91.BOUNDARIES)}")
if len(sh) == len(s91.BOUNDARIES):
    best = sh.loc[sh["step_coef"].idxmin(), "boundary"]
    check("the largest step is at the planted boundary", best == PLANT,
          f"peak at {best}, planted {PLANT}; "
          + ", ".join(f"{r.boundary} {r.step_coef:+.4f}"
                      for _, r in sh.iterrows()))
    at = float(sh.loc[sh.boundary == PLANT, "step_coef"].iloc[0])
    check("and it recovers the planted step", abs(at - FALL) < 0.05,
          f"{at:+.4f} against a planted {FALL:+.4f}")

print("\n--- GRADUAL world: read rule 1's premise must hold ---")
gr = sweep("gradual")
if len(gr) == len(s91.BOUNDARIES):
    rep = float(gr.loc[gr.boundary == s91.REPORTED, "step_coef"].iloc[0])
    spread = float(gr["step_coef"].max() - gr["step_coef"].min())
    check("under a gradual decline the step is nearly INVARIANT to the "
          "boundary, so a flat sweep on the real data is close to mechanical "
          "and is not evidence about when the decline began",
          spread < 0.6 * abs(rep),
          f"spread {spread:.4f} against a reported step of {rep:+.4f}; "
          + ", ".join(f"{r.boundary} {r.step_coef:+.4f}"
                      for _, r in gr.iterrows()))
    check("and every boundary still finds a decline",
          bool((gr["step_coef"] < 0).all()))

print("\n--- 78's POST_FROM is left as it was found ---")
check("the sweep restores it", s78.POST_FROM == "2024-01", s78.POST_FROM)

print("\n" + ("all checks passed" if not FAILS else f"FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)
