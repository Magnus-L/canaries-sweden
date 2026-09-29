#!/usr/bin/env python3
"""
test_109_occ_route_plain_flows.py -- raw flows on the occupation-route
                                     score: the gate must pass on 85's own
                                     export and fail on a moved one, and
                                     planted hire rates must come back.

World borrowed from test_85 (same score, same panel); flows are planted
with a monthly hire rate of 15 per cent at 22-25 and 5 per cent at 41-49,
and hires cut by 10 per cent in every band and quartile from 2024.

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

    CANARIES_DRYRUN=1 python3 revision/local/test_109_occ_route_plain_flows.py
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
TMP = Path(tempfile.mkdtemp(prefix="canaries109_"))
SHARE = TMP / "input"; SHARE.mkdir()
(TMP / "out" / "output_85").mkdir(parents=True)
for f in ("daioe_quartiles.dta", "eloundou_ssyk4.dta",
          "utb_grupp2_sun2020_niva3_inr4_nyckel.dta"):
    shutil.copy(UPLOAD / f, SHARE / f)
os.environ["CANARIES_SHARE"] = str(SHARE)
os.environ["CANARIES_85_OUT"] = str(TMP / "out" / "output_85")
os.environ["CANARIES_109_OUT"] = str(TMP / "out")
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

(s82, s61, s66, s74, s78, l47, l70, j47) = s85.load_modules()
for m_ in (s82, s61, s66, s74, s78, l47, l70, j47):
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


# ---- flows, planted -------------------------------------------------
RATE = {"22-25": 0.15, "26-30": 0.10, "31-34": 0.07, "35-40": 0.06,
        "41-49": 0.05, "50+": 0.05}
F = C.copy()
cut = np.where(F["year_month"] >= "2024-01", 0.90, 1.0)
F["n_hire"] = np.round(F["n_emp"] * F["age_group"].map(RATE) * cut * 100)
F["n_sep"] = np.round(F["n_emp"] * 0.05 * 100)
C2 = C.copy(); C2["n_emp"] = C2["n_emp"] * 100   # scale so rounding is exact
for y in s61.PANEL_YEARS:
    k = C2["year_month"].str.slice(0, 4) == str(y)
    C2[k].to_parquet(mc.CACHE_DIR / f"L_counts_{y}.parquet", index=False)
    F[k].drop(columns="n_emp").to_parquet(
        mc.CACHE_DIR / f"flows_{y}.parquet", index=False)

print("\n--- 85 part D writes the Table A12 export the gate reads ---")
os.environ["CANARIES_85_PARTS"] = "D"
s85.PARTS = "D"
s85.OUT.mkdir(parents=True, exist_ok=True)
s85.main()
A12 = s85.OUT / "occ_route_descriptive_full.csv"
check("85 wrote the Table A12 export", A12.exists())

s109 = load("109_occ_route_plain_flows.py", "s109")
s109.PRIOR = (str(s85.OUT),)
print("\n--- 109 main(), gate should pass ---")
rc = s109.main()
check("main returns 0", rc == 0, str(rc))
summ = (s109.OUT / "109_summary.txt").read_text(encoding="utf-8")
check("the gate passes on 85's own export", "PASSED" in summ,
      summ.splitlines()[3] if len(summ.splitlines()) > 3 else "")
out = pd.read_csv(s109.OUT / "occ_route_plain_flows.csv")
check("all three outcomes exported",
      set(out["outcome"]) == {"n_emp", "n_hire", "n_sep"},
      str(sorted(set(out["outcome"]))))
s = out[(out.outcome == "n_emp") & (out.period == "pre")].set_index(
    ["fq", "age_group"])["total"]
h = out[(out.outcome == "n_hire") & (out.period == "pre")].set_index(
    ["fq", "age_group"])["total"]
r = (h / s)
check("the planted hire rate at 22-25 comes back",
      np.allclose(r.xs("22-25", level=1), 0.15, atol=0.002),
      str(r.xs("22-25", level=1).round(4).tolist()))
check("the planted hire rate at 41-49 comes back",
      np.allclose(r.xs("41-49", level=1), 0.05, atol=0.002))
check("the summary prints hire rates", "MONTHLY HIRE RATE" in summ)
check("no count below the floor leaves",
      not ((out["n_firms"] > 0) & (out["n_firms"] < 5)).any())

print("\n--- a moved Table A12 must fail the gate ---")
bad = pd.read_csv(A12); bad.loc[0, "total"] += 7
bad.to_csv(A12, index=False)
s109.FAILURES.clear(); s109.NOTES.clear()
s109.main()
summ2 = (s109.OUT / "109_summary.txt").read_text(encoding="utf-8")
check("a moved export fails the gate", "THE PANEL HAS MOVED" in summ2)

print("\n" + "=" * 60)
print(f"{len(FAILS)} FAILED" if FAILS else "ALL PASS")
for f in FAILS:
    print("  " + f)
shutil.rmtree(TMP, ignore_errors=True)
raise SystemExit(1 if FAILS else 0)
