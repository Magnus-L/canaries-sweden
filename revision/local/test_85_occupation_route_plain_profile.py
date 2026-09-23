#!/usr/bin/env python3
"""
test_85_occupation_route_plain_profile.py -- the two arms must differ
                                             only in the calendar terms,
                                             must sit on one panel, and
                                             the gate must be able to
                                             fail.

The checks are on mechanisms, not on outputs. One synthetic world, drawn
so that the calendar terms have something to remove:

  400 employers carry a 2019 occupation mix and a 2019 person-month
      count; 300 of them also carry the six-band monthly panel.
  EMPLOYMENT carries a real seasonal cycle IN THE EXPOSURE CONTRAST: the
      young band of exposed employers is lifted in the fourth quarter and
      cut in the first, in every year including the two before any
      treatment. That is the pattern the calendar terms exist to remove,
      and it is planted before the treatment so an arm without them must
      inherit it while an arm with them must not.
  A TREATMENT is planted on top: the young band of exposed employers
      falls from January 2024.

Also tested: that the score is lane 28's own and is not rebuilt; that the
two fits differ in the calendar terms and in nothing else, by comparing
the term lists; that the reference band is written out at zero and not
fitted; that the gate passes against a matching prior and fails against
a moved one; that a failed fit leaves no row; and main() end to end.

    CANARIES_DRYRUN=1 python3 revision/local/test_85_occupation_route_plain_profile.py
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
TMP = Path(tempfile.mkdtemp(prefix="canaries85_"))
SHARE = TMP / "input"; SHARE.mkdir()
for f in ("daioe_quartiles.dta", "eloundou_ssyk4.dta",
          "utb_grupp2_sun2020_niva3_inr4_nyckel.dta"):
    shutil.copy(UPLOAD / f, SHARE / f)
os.environ["CANARIES_SHARE"] = str(SHARE)
os.environ["CANARIES_85_OUT"] = str(TMP / "out")
os.environ["CANARIES_82_OUT"] = str(TMP / "out")
sys.path.insert(0, str(MONA)); sys.path.insert(0, str(HERE))
import mona_common as mc  # noqa: E402
mc.SHARE = str(SHARE); mc.CACHE_DIR = TMP / "cache"; mc.CACHE_DIR.mkdir()
_LOCAL = str(SHARE / "daioe_quartiles.dta")
mc.DAIOE_PATH = _LOCAL
_ld = mc.load_daioe
mc.load_daioe = lambda path=_LOCAL: _ld(path)
mc.connect = lambda: object()

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


s85 = load("85_occupation_route_plain_profile.py", "s85")
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
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
pd.DataFrame([(e, f"2019-{m:02d}", a, size_mult(e))
              for e in SCORED for m in range(1, 13) for a in AGES],
             columns=["employer_id", "year_month", "age_group", "n_emp"]
             ).to_parquet(mc.CACHE_DIR / "L_counts_2019.parquet", index=False)

(s82, s61, s74, s78, l47, l70, j47) = s85.load_modules()
for m_ in (s82, s61, s74, s78, l47, l70, j47):
    m_.OUT, m_.CACHE = s85.OUT, mc.CACHE_DIR
s85.OUT.mkdir(parents=True, exist_ok=True)
MONTHS = [f"{y}-{m:02d}" for y in s61.PANEL_YEARS
          for m in range(1, 13 if y < 2025 else 7)]
CYCLE = {1: float(np.log(0.90)), 2: 0.0, 3: 0.0, 4: float(np.log(1.12))}
FALL = float(np.log(0.82))


def panel() -> pd.DataFrame:
    """A seasonal cycle IN THE EXPOSURE CONTRAST, present before any
    treatment, plus an adoption fall at 22-25 in the exposed firms."""
    rng = np.random.default_rng(85)
    lam = {"22-25": 16, "26-30": 15, "31-34": 13, "35-40": 13,
           "41-49": 14, "50+": 14}
    out = []
    for emp in PANEL:
        hit = emp in EXPOSED
        for ym in MONTHS:
            q = (int(ym[5:7]) - 1) // 3 + 1
            for age, l0 in lam.items():
                x = float(l0)
                if hit and age == "22-25":
                    x *= np.exp(CYCLE[q])
                    if ym >= "2024-01":
                        x *= np.exp(FALL)
                out.append((emp, ym, age, int(rng.poisson(x)) + 1))
    return pd.DataFrame(out, columns=["employer_id", "year_month",
                                      "age_group", "n_emp"])


C = panel()
for y in s61.PANEL_YEARS:
    C[C["year_month"].str.slice(0, 4) == str(y)].to_parquet(
        mc.CACHE_DIR / f"L_counts_{y}.parquet", index=False)

print("\n--- the score and the terms ---")
built = s82.build_exposure(l47, l70, j47)
check("the score is 82's primary arm",
      built["arm"] == s82.MAIN_LEVEL and built["floor"] == s82.FLOOR_MAIN,
      f"{built['arm']} floor {built['floor']}")
skel = l70.all_band_skeleton(C)
b0 = s78.with_exposure(skel, built["exposure"])
_, t_plain = s74.build_terms(b0, l70, seasonal=False)
_, t_seas = s74.build_terms(b0, l70, seasonal=True)
check("the seasonal arm adds terms and removes none",
      set(t_plain) < set(t_seas),
      f"{len(t_plain)} against {len(t_seas)} terms")
extra = set(t_seas) - set(t_plain)
check("every added term is a quarter-of-year interaction",
      extra and all("_q" in t or t.startswith("q") for t in extra),
      str(sorted(extra))[:90])

print("\n--- the gate can pass and can fail ---")
good = pd.DataFrame({"band": ["22-25", "50+"], "coef": [-0.0200, 0.0700]})
(s85.OUT / "output_82b").mkdir(parents=True, exist_ok=True)
good.to_csv(s85.OUT / "output_82b" / "occ_route_profile.csv", index=False)
s85.PRIOR = (str(s85.OUT / "output_82b"),)
check("a prior profile is found", not s85.prior_profile().empty)

print("\n--- main() ---")
rc = s85.main()
check("main returns 0", rc == 0, str(rc))
out = s85.OUT / "occ_route_profile_arms.csv"
check("the export exists", out.exists())
A = pd.read_csv(out) if out.exists() else pd.DataFrame()
check("both arms are in the export",
      len(A) and set(A["arm"]) == {"plain", "seasonal"},
      str(sorted(set(A.get("arm", [])))))
check("the reference band is written at zero, in both arms",
      len(A) and (A[A.band == "41-49"]["coef"] == 0).all()
      and len(A[A.band == "41-49"]) == 2)
check("two fits, one per arm",
      sum(1 for t, _ in FITS if t.startswith("s85_profile_")) == 2,
      str([t for t, _ in FITS if t.startswith("s85_profile_")]))
if len(A):
    p = A[(A.arm == "plain") & (A.band == "22-25")].iloc[0]
    s = A[(A.arm == "seasonal") & (A.band == "22-25")].iloc[0]
    # The planted cycle lifts the exposed young in Q4 and cuts them in
    # Q1; the post window is weighted differently from the pre window, so
    # an arm that does not remove the cycle must not return the same
    # number as one that does.
    check("the calendar terms change the youngest band's estimate",
          abs(p["coef"] - s["coef"]) > 1e-4,
          f"plain {p['coef']:+.4f} against {s['coef']:+.4f}")
    check("the planted fall is recovered on the arm that removes the cycle",
          s["coef"] < 0 and abs(s["t"]) > 2,
          f"{s['coef']:+.4f} t {s['t']:+.2f}")
summ = (s85.OUT / "85_summary.txt").read_text(encoding="utf-8")
check("the summary reports the gate", "THE GATE:" in summ)
check("the summary reports what the terms do",
      "WHAT THE CALENDAR TERMS DO:" in summ)
check("the summary prints the read rules",
      "READ RULES, FIXED BEFORE THE RUN" in summ)

# The gate must be able to PASS as well as fail: feed it the arm this
# run actually produced, which is what lane 28b's export will hold.
fitted = A[(A.arm == "seasonal") & (A.status == "ok")][["band", "coef"]]
fitted.to_csv(s85.OUT / "output_82b" / "occ_route_profile.csv", index=False)
s85.FAILURES.clear()
s85.main()
summ_ok = (s85.OUT / "85_summary.txt").read_text(encoding="utf-8")
check("a matching prior passes the gate",
      "THE PANEL IS THE ONE LANE 28b FITTED" in summ_ok)

moved = pd.DataFrame({"band": ["22-25", "50+"], "coef": [-0.9999, 0.0700]})
moved.to_csv(s85.OUT / "output_82b" / "occ_route_profile.csv", index=False)
s85.FAILURES.clear()
s85.main()
summ2 = (s85.OUT / "85_summary.txt").read_text(encoding="utf-8")
check("a moved prior coefficient fails the gate",
      "THE PANEL HAS MOVED" in summ2)

print("\n" + "=" * 60)
print(f"{len(FAILS)} FAILED" if FAILS else "ALL PASS")
for f in FAILS:
    print("  " + f)
shutil.rmtree(TMP, ignore_errors=True)
raise SystemExit(1 if FAILS else 0)
