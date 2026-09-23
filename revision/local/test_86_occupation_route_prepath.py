#!/usr/bin/env python3
"""
test_86_occupation_route_prepath.py -- the path must carry no calendar
                                       terms, the window must follow the
                                       caches, the drift must recover a
                                       planted trend, and the gate must
                                       be able to fail.

The checks are on mechanisms, not on outputs. One synthetic world, drawn
so that the drift test has something to find:

  400 employers carry a 2019 occupation mix and a 2019 person-month
      count; 300 of them also carry the monthly panel, from 2019 so that
      the extended window has something to run on.
  A PRE-LAUNCH TREND is planted IN THE EXPOSURE CONTRAST: the young band
      of exposed employers rises by a fixed proportion every month from
      January 2021 to November 2022 and then holds. It is a level path
      with a slope in the pre-period, which is what the drift test
      exists to detect, and it stops before the launch so that the test
      window is the only place it lives.
  A TREATMENT is planted on top: the young band of exposed employers
      falls from January 2024.

Also tested: that the score is lane 28's own and is not rebuilt; that
the path terms are quarter dummies with 2022Q1 omitted and hold no
calendar or tightening term; that the window follows the caches and that
removing L_counts_2019 and L_counts_2020 shortens it and is SAID; that
78's two export names do not survive; that the gate passes against a
matching prior and fails against a moved one; and main() end to end.

    CANARIES_DRYRUN=1 python3 revision/local/test_86_occupation_route_prepath.py
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
TMP = Path(tempfile.mkdtemp(prefix="canaries86_"))
SHARE = TMP / "input"; SHARE.mkdir()
for f in ("daioe_quartiles.dta", "eloundou_ssyk4.dta",
          "utb_grupp2_sun2020_niva3_inr4_nyckel.dta"):
    shutil.copy(UPLOAD / f, SHARE / f)
os.environ["CANARIES_SHARE"] = str(SHARE)
os.environ["CANARIES_86_OUT"] = str(TMP / "out")
os.environ["CANARIES_82_OUT"] = str(TMP / "out")
sys.path.insert(0, str(MONA)); sys.path.insert(0, str(HERE))
import mona_common as mc  # noqa: E402
mc.SHARE = str(SHARE); mc.CACHE_DIR = TMP / "cache"; mc.CACHE_DIR.mkdir()
_LOCAL = str(SHARE / "daioe_quartiles.dta")
mc.DAIOE_PATH = _LOCAL
_ld = mc.load_daioe
mc.load_daioe = lambda path=_LOCAL: _ld(path)


def _no_sql():
    raise AssertionError("this lane performs no SQL and must not connect")


mc.connect = _no_sql

FITS = []
_real_multi = mc.run_fepois_multi


def counting_multi(panel, workdir, tag, *a, **kw):
    FITS.append((tag, tuple(kw.get("terms", ()))))
    return _real_multi(panel, workdir, tag, *a, **kw)


mc.run_fepois_multi = counting_multi


def load(n, a):
    sp = importlib.util.spec_from_file_location(a, MONA / n)
    m = importlib.util.module_from_spec(sp); sys.modules[a] = m
    sp.loader.exec_module(m); return m


s86 = load("86_occupation_route_prepath.py", "s86")
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


AGES = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]
N_SCORE, N_PANEL = 400, 300
SCORED, PANEL = list(range(1, N_SCORE + 1)), list(range(1, N_PANEL + 1))
UNSCORED = "9999"
D = pd.read_stata(_LOCAL)
D["ssyk4"] = D["ssyk4"].astype(str).str.zfill(4)
_d = D.sort_values("pctl_rank_genai").reset_index(drop=True)
TIER_CODE, seen = [], set()
for q in (0.10, 0.40, 0.65, 0.92):
    for i in range(int(q * len(_d)), len(_d)):
        c = _d.loc[i, "ssyk4"]
        if c[:3] not in seen:
            TIER_CODE.append(c); seen.add(c[:3]); break
TIER_OF = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 3}
tier = lambda e: TIER_OF[e % 10]
size_mult = lambda e: 1 + (e % 9)
EXPOSED = {e for e in SCORED if tier(e) == 3}

rows = []
for emp in SCORED:
    t_, m_ = tier(emp), size_mult(emp)
    c, far = TIER_CODE[t_], TIER_CODE[3 - t_]
    for age in AGES:
        rows.append((emp, age, c, c[:3], "2019", 6 * m_))
        rows.append((emp, age, far, far[:3], "2019", 1 * m_))
        rows.append((emp, age, UNSCORED, UNSCORED[:3], "2019", 2))
CASC = pd.DataFrame(rows, columns=["employer_id", "age_group", "ssyk4",
                                   "ssyk3", "source_year", "n"])
CASC["ssyk_ar"] = "2019"; CASC["ssyk_status"] = "1"
CASC.to_parquet(mc.CACHE_DIR / "L_baseline_2019_cascade.parquet", index=False)
(CASC.groupby(["employer_id", "age_group", "ssyk4"], observed=True)["n"].sum()
 .reset_index()).to_parquet(mc.CACHE_DIR / "L_baseline_2019.parquet",
                            index=False)

(s82, s61, s78, l47, l70, j47) = s86.load_modules()
for m_ in (s82, s61, s78, l47, l70, j47):
    m_.OUT, m_.CACHE = s86.OUT, mc.CACHE_DIR
s86.OUT.mkdir(parents=True, exist_ok=True)

# The panel runs from 2019 so that the extended window has something to
# run on, and the drift window is 2021-01 to 2022-11 inside it.
YEARS = [2019, 2020] + list(s61.PANEL_YEARS)
MONTHS = [f"{y}-{m:02d}" for y in YEARS
          for m in range(1, 13 if y < 2025 else 7)]
DRIFT_FROM, DRIFT_TO = "2021-01", "2022-11"
PER_MONTH = float(np.log(1.010))          # the planted pre-launch slope
FALL = float(np.log(0.82))                # the planted adoption fall


def panel() -> pd.DataFrame:
    """A pre-launch TREND in the exposure contrast, which is what the
    drift test exists to detect, and an adoption fall on top of it."""
    rng = np.random.default_rng(86)
    lam = {"22-25": 16, "26-30": 15, "31-34": 13, "35-40": 13,
           "41-49": 14, "50+": 14}
    idx = {ym: i for i, ym in enumerate(m for m in MONTHS
                                        if DRIFT_FROM <= m <= DRIFT_TO)}
    held = len(idx) - 1
    out = []
    for emp in PANEL:
        hit = emp in EXPOSED
        for ym in MONTHS:
            for age, l0 in lam.items():
                x = float(l0)
                if hit and age in ("22-25", "26-30"):
                    # rises through the drift window, then holds the level
                    # it reached: a slope in the pre-period and not a bump
                    steps = idx.get(ym, held if ym > DRIFT_TO else 0)
                    x *= np.exp(PER_MONTH * steps)
                    if ym >= "2024-01":
                        x *= np.exp(FALL)
                out.append((emp, ym, age, int(rng.poisson(x)) + 1))
    return pd.DataFrame(out, columns=["employer_id", "year_month",
                                      "age_group", "n_emp"])


C = panel()
for y in YEARS:
    C[C["year_month"].str.slice(0, 4) == str(y)].to_parquet(
        mc.CACHE_DIR / f"L_counts_{y}.parquet", index=False)

print("\n--- the score and the terms ---")
built = s82.build_exposure(l47, l70, j47)
check("the score is 82's primary arm",
      built["arm"] == s82.MAIN_LEVEL and built["floor"] == s82.FLOOR_MAIN,
      f"{built['arm']} floor {built['floor']}")

skel = s78.build_skeleton_bands(C, ["22-25"] + j47.INCUMBENT_BANDS,
                                "22-25", j47, s78.EXTENDED_FROM)
b = s78.with_exposure(skel, built["exposure"])
b, pterms = s78.plain_path_terms(b)
labs = sorted(t.removeprefix("pq_").split("_x_high")[0] for t in pterms)
check("the path is one dummy per quarter with the reference omitted",
      s86.REF_QUARTER not in labs and len(labs) == len(set(labs)),
      f"{len(labs)} quarters, {labs[0]} to {labs[-1]}")
check("the path runs from 2019Q1 when the counts are cached",
      labs[0] == "2019Q1", labs[0])
check("the path carries no calendar or tightening term",
      not any(t.startswith("q1_") or t.startswith("q2_")
              or t.startswith("q3_") or t.startswith("rbw_")
              or t.startswith("trend_") for t in pterms))
_, dterms = s78.drift_terms(b.head(1000).copy())
check("the drift terms are the cycle, the window and the trend",
      set(dterms) == {"q1_x_high_x_young", "q2_x_high_x_young",
                      "q3_x_high_x_young", "rbw_x_high_x_young",
                      s86.TREND},
      str(sorted(dterms)))
check("no path term is a drift term", not (set(pterms) & set(dterms)))
del skel, b

print("\n--- main(), with the extended window ---")
rc = s86.main()
check("main returns 0", rc == 0, str(rc))
P = s86.OUT / "occ_route_prepath.csv"
Q = s86.OUT / "occ_route_predrift.csv"
check("the path export exists under this route's name", P.exists())
check("the drift export exists under this route's name", Q.exists())
check("78's own export names do not survive",
      not (s86.OUT / "prepath_plain.csv").exists()
      and not (s86.OUT / "predrift.csv").exists())

A = pd.read_csv(P) if P.exists() else pd.DataFrame()
check("both young bands are in the path",
      len(A) and set(A["young_band"]) == {"22-25", "26-30"},
      str(sorted(set(A.get("young_band", [])))))
check("the reference quarter is written at zero and not fitted",
      len(A) and (A[A.quarter == s86.REF_QUARTER]["coef"] == 0).all()
      and (A[A.quarter == s86.REF_QUARTER]["status"] == "reference").all())
check("the path starts in 2019 in the export",
      len(A) and A["quarter"].min() == "2019Q1", str(A["quarter"].min()))
check("four fits, a path and a drift for each band",
      sum(1 for t, _ in FITS if t.startswith("s78_prepath_")) == 2
      and sum(1 for t, _ in FITS if t.startswith("s78_predrift_")) == 2,
      str([t for t, _ in FITS]))

B = pd.read_csv(Q) if Q.exists() else pd.DataFrame()
if len(B):
    r = B[(B.young_band == "22-25") & (B.term == s86.TREND)]
    check("the planted pre-launch trend is recovered, positive",
          len(r) and float(r["coef"].iloc[0]) > 0
          and abs(float(r["coef"].iloc[0]) / float(r["se"].iloc[0])) > 2,
          f"{float(r['coef'].iloc[0]):+.5f} "
          f"({float(r['se'].iloc[0]):.5f})" if len(r) else "absent")

summ = (s86.OUT / "86_summary.txt").read_text(encoding="utf-8")
check("the summary reports the gate", "THE GATE:" in summ)
check("the summary reports the drift", "THE DRIFT TEST, REFITTED:" in summ)
check("the summary reports the path", "THE PLAIN PATH:" in summ)
check("the summary prints the read rules",
      "READ RULES, FIXED BEFORE THE RUN" in summ)
check("no prior drift means NO GATE, not a pass",
      "NO PRIOR DRIFT FOUND" in (s86.OUT / "86_log.txt").read_text(
          encoding="utf-8") or "not run" in summ)

print("\n--- the gate can pass and can fail ---")
prior_dir = s86.OUT / "output_83b"
prior_dir.mkdir(parents=True, exist_ok=True)
s86.PRIOR = (str(prior_dir),)
# lane 29b's export carries the standard error and the cell count too,
# which the gate needs to judge a gap in standard errors.
B[["young_band", "term", "coef", "se", "n_obs"]].to_csv(
    prior_dir / s86.PRIOR_FILE, index=False)
s86.FAILURES.clear(); s86.NOTES.clear()
s86.main()
ok = (s86.OUT / "86_summary.txt").read_text(encoding="utf-8")
check("a matching prior passes the gate",
      "THE WIDER FRAME TELLS THE SAME STORY" in ok)

moved = B[["young_band", "term", "coef", "se", "n_obs"]].copy()
trend = moved["term"] == s86.TREND
moved.loc[trend, "coef"] = moved.loc[trend, "coef"] + 0.5
moved.to_csv(prior_dir / s86.PRIOR_FILE, index=False)
s86.FAILURES.clear(); s86.NOTES.clear()
s86.main()
bad = (s86.OUT / "86_summary.txt").read_text(encoding="utf-8")
check("a trend that moves by more than a standard error fails the gate",
      "THE DRIFT DISAGREES" in bad)
check("a failed gate is recorded as a failure", "WHAT FAILED" in bad)

print("\n--- the window follows the caches and is SAID ---")
for y in (2019, 2020):
    (mc.CACHE_DIR / f"L_counts_{y}.parquet").unlink()
s86.FAILURES.clear(); s86.NOTES.clear()
s86.main()
short = (s86.OUT / "86_summary.txt").read_text(encoding="utf-8")
check("without the 2019 and 2020 counts the path is short and says so",
      "are NOT on the share" in short)
A2 = pd.read_csv(s86.OUT / "occ_route_prepath.csv")
check("the short path starts at the panel's own first quarter",
      A2["quarter"].min() > "2019Q4", str(A2["quarter"].min()))

print("\n" + "=" * 60)
print(f"{len(FAILS)} FAILED" if FAILS else "ALL PASS")
for f in FAILS:
    print("  " + f)
shutil.rmtree(TMP, ignore_errors=True)
raise SystemExit(1 if FAILS else 0)
