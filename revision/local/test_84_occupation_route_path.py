#!/usr/bin/env python3
"""
test_84_occupation_route_path.py -- the path must be lane 28's score
                                    decomposed by script 68's terms, the
                                    dating rules must be able to fail,
                                    and a missing fit must never be drawn
                                    as a zero.

The checks are on mechanisms, not on outputs. One synthetic world, drawn
so that every rule the script claims to follow has something to catch:

  400 employers carry a 2019 occupation mix and a 2019 person-month
      count, so the score is built over a population the floor can bite
      on; 300 of them also carry the monthly panel, which is the real
      shape, every panel in these lanes being a subset of the scored
      employers.
  each employer's incumbents hold one occupation drawn from four codes
      in four separate three-digit groups at four separated points of
      the DAIOE distribution, so the employment-weighted quartile is the
      planted tier and the test can name the exposed employers before
      anything is estimated.
  EMPLOYMENT is planted with a path and not merely a step: flat at both
      young bands through 2023, a fall at 22-25 in the exposed firms
      from 2024-01, and a fall at 26-30 that begins a YEAR LATER, from
      2025-01. That is the shape the paper's sentence claims, so the two
      dating rules have a world in which they should pass; the planted
      null below gives them one in which they must fail.

Also tested: that the score is lane 28's own, bit for bit, and is not
rebuilt here; that the terms come from script 68's add_seasonal_terms
and carry the Riksbank interaction, the three quarter-of-year terms and
the fourth quarter omitted; that the exported period labels are the ones
the figure generator reads; that the quarterly and monthly shapes can be
fitted on one frame without one disturbing the other; that a fit which
fails is recorded and leaves no row rather than a zero; that both
verdicts can fail as well as pass; and main() end to end.

    CANARIES_DRYRUN=1 python3 revision/local/test_84_occupation_route_path.py
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
os.environ["CANARIES_84_SHAPES"] = "QM"
os.environ["CANARIES_ECHO_LIMIT"] = "100000000"
HERE = Path(__file__).resolve().parent
MONA, UPLOAD = HERE.parent / "mona", HERE.parent / "upload"
TMP = Path(tempfile.mkdtemp(prefix="canaries84_"))
SHARE = TMP / "input"; SHARE.mkdir()
for f in ("daioe_quartiles.dta", "eloundou_ssyk4.dta",
          "utb_grupp2_sun2020_niva3_inr4_nyckel.dta"):
    shutil.copy(UPLOAD / f, SHARE / f)
os.environ["CANARIES_SHARE"] = str(SHARE)
os.environ["CANARIES_84_OUT"] = str(TMP / "out")
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
    """Every Poisson fit goes through one function, so the count is of
    fits and not of call sites, and the terms of each are recorded."""
    FITS.append((tag, tuple(kw.get("terms", a[0] if a else ()))))
    return _real_multi(panel, workdir, tag, *a, **kw)


mc.run_fepois_multi = counting_multi


def load(n, a):
    sp = importlib.util.spec_from_file_location(a, MONA / n)
    m = importlib.util.module_from_spec(sp); sys.modules[a] = m
    sp.loader.exec_module(m); return m


s84 = load("84_occupation_route_path.py", "s84")
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


# ======================================================================
# the world
# ======================================================================
AGES = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]
N_SCORE, N_PANEL = 400, 300
SCORED = list(range(1, N_SCORE + 1))
PANEL = list(range(1, N_PANEL + 1))
UNSCORED_CODE = "9999"

DAIOE = pd.read_stata(_LOCAL)
DAIOE["ssyk4"] = DAIOE["ssyk4"].astype(str).str.zfill(4)
assert UNSCORED_CODE not in set(DAIOE["ssyk4"])
_d = DAIOE.sort_values("pctl_rank_genai").reset_index(drop=True)
TIER_CODE, _seen = [], set()
for q in (0.10, 0.40, 0.65, 0.92):
    for i in range(int(q * len(_d)), len(_d)):
        c = _d.loc[i, "ssyk4"]
        if c[:3] not in _seen:
            TIER_CODE.append(c); _seen.add(c[:3]); break
assert len(TIER_CODE) == 4, TIER_CODE
# Uneven tiers, as in test_83: the quartile cuts are weighted by
# incumbent employment, and with four equal tiers a cut lands on a
# planted score and the top quartile stops being the top tier.
TIER_OF = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 3}


def tier(emp: int) -> int:
    return TIER_OF[emp % 10]


def size_mult(emp: int) -> int:
    return 1 + (emp % 9)


EXPOSED = {e for e in SCORED if tier(e) == 3}


def cascade_frame() -> pd.DataFrame:
    """The 2019 occupation mix, as 82's cascade pull returns it."""
    rows = []
    for emp in SCORED:
        t = tier(emp)
        c, far = TIER_CODE[t], TIER_CODE[3 - t]
        m = size_mult(emp)
        for age in AGES:
            rows.append((emp, age, c, c[:3], "2019", 6 * m))
            rows.append((emp, age, far, far[:3], "2019", 1 * m))
            rows.append((emp, age, UNSCORED_CODE, UNSCORED_CODE[:3],
                         "2019", 2))
            rows.append((emp, age, "____", "___", "none", 2))
    d = pd.DataFrame(rows, columns=["employer_id", "age_group", "ssyk4",
                                    "ssyk3", "source_year", "n"])
    d["ssyk_ar"] = "2019"
    d["ssyk_status"] = np.where(d["ssyk4"] == "____", "9", "1")
    return d


def counts_2019() -> pd.DataFrame:
    rows = []
    for emp in SCORED:
        k = size_mult(emp)
        rows += [(emp, f"2019-{m:02d}", age, k)
                 for m in range(1, 13) for age in AGES]
    return pd.DataFrame(rows, columns=["employer_id", "year_month",
                                       "age_group", "n_emp"])


CASC = cascade_frame()
CASC.to_parquet(mc.CACHE_DIR / "L_baseline_2019_cascade.parquet", index=False)
(CASC.groupby(["employer_id", "age_group", "ssyk4"], observed=True)["n"]
 .sum().reset_index()).to_parquet(
     mc.CACHE_DIR / "L_baseline_2019.parquet", index=False)
counts_2019().to_parquet(mc.CACHE_DIR / "L_counts_2019.parquet", index=False)

# ---- the monthly panel, with a PATH and not merely a step ------------
FALL_22 = float(np.log(0.80))   # from 2024-01 at 22-25, exposed
FALL_26 = float(np.log(0.85))   # from 2025-01 at 26-30, exposed: a year later
RISE_RB = float(np.log(1.06))   # the tightening rise at 22-25, exposed
OPEN_22, OPEN_26 = "2024-01", "2025-01"

(s82, s61, s68, s78, l47, l70, j47) = s84.load_modules()
for m_ in (s82, s61, s68, s78, l47, l70, j47):
    m_.OUT, m_.CACHE = s84.OUT, mc.CACHE_DIR
s84.OUT.mkdir(parents=True, exist_ok=True)
MONTHS = [f"{y}-{m:02d}" for y in s61.PANEL_YEARS
          for m in range(1, 13 if y < 2025 else 7)]
RB_FROM, RB_TO = mc.RIKSBANK_YM, mc.CHATGPT_YM


def panel_counts() -> pd.DataFrame:
    """
    Monthly employment with the planted path in it: flat at both young
    bands through 2023, a fall at 22-25 from 2024-01 and a fall at 26-30
    from 2025-01, both in the exposed firms only, plus the tightening
    rise at 22-25. No pre-launch trend anywhere, so a quarter before 2024
    that comes back significant is the estimator's doing and not the
    fixture's.
    """
    rng = np.random.default_rng(84)
    lam0 = {"22-25": 16, "26-30": 16, "31-34": 12, "35-40": 12,
            "41-49": 14, "50+": 14}
    rows = []
    for emp in PANEL:
        hit = emp in EXPOSED
        for ym in MONTHS:
            for age, lam in lam0.items():
                x = float(lam)
                if hit:
                    if age == "22-25":
                        # The Riksbank term is cumulative, so the level
                        # reached during the tightening months PERSISTS;
                        # a bump that ended in November 2022 would make
                        # every later quarter negative against it, which
                        # is what the specification says and not what the
                        # paper's gamma_1 describes.
                        if ym >= RB_FROM:
                            x *= np.exp(RISE_RB)
                        if ym >= OPEN_22:
                            x *= np.exp(FALL_22)
                    elif age == "26-30" and ym >= OPEN_26:
                        x *= np.exp(FALL_26)
                rows.append((emp, ym, age, int(rng.poisson(x)) + 1))
    return pd.DataFrame(rows, columns=["employer_id", "year_month",
                                       "age_group", "n_emp"])


COUNTS = panel_counts()
for y in s61.PANEL_YEARS:
    COUNTS[COUNTS["year_month"].str.slice(0, 4) == str(y)].to_parquet(
        mc.CACHE_DIR / f"L_counts_{y}.parquet", index=False)


# ======================================================================
# mechanisms, before main()
# ======================================================================
print("\n--- the score is lane 28's, and is not rebuilt ---")
built = s82.build_exposure(l47, l70, j47)
OCC = built["exposure"]
check("the score is 82's primary arm",
      built["arm"] == s82.MAIN_LEVEL and built["floor"] == s82.FLOOR_MAIN,
      f"{built['arm']} floor {built['floor']}")
hi = set(OCC.loc[OCC["fq"] == 4, "employer_id"].astype(int))
check("the top quartile is the planted top tier",
      hi and hi <= EXPOSED and len(hi) >= 0.8 * len(EXPOSED & set(OCC["employer_id"].astype(int))),
      f"{len(hi)} exposed of {len(OCC)} scored")
check("84 builds no score of its own",
      not any(n.startswith("occ_route_exposure") for n in dir(s84)),
      "no rebuild helper on the module")

print("\n--- the terms are 68's ---")
skel = s61.build_skeleton(COUNTS, "22-25", j47)
b = s78.with_exposure(skel, OCC)
b, qterms = s68.add_seasonal_terms(b, "quarter")
check("the Riksbank interaction is kept on",
      "rb_x_high_x_young" in qterms)
check("three quarter-of-year terms, the fourth omitted",
      sum(t.startswith("q") and "_x_high_x_young" in t for t in qterms) == 3
      and "q4_x_high_x_young" not in qterms)
check("the post quarters start at the launch quarter",
      min(t for t in qterms if t.startswith("pq_")) == "pq_2022Q4_x_high_x_young",
      min((t for t in qterms if t.startswith("pq_")), default="none"))
b, mterms = s68.add_seasonal_terms(b, "month")
check("the monthly shape omits December and keeps eleven",
      sum(t.startswith("m") and t[1:3].isdigit() for t in mterms) == 11
      and "m12_x_high_x_young" not in mterms)
check("one frame carries both shapes without either losing a column",
      all(t in b.columns for t in qterms) and all(t in b.columns
                                                  for t in mterms))

print("\n--- the labels the figure reads ---")
fake = pd.DataFrame({"coef": [0.1, -0.2], "se": [0.01, 0.02],
                     "n_obs": [10, 10]},
                    index=["pq_2024Q1_x_high_x_young",
                           "pq_2024Q2_x_high_x_young"])
rows = s84.path_rows(fake, list(fake.index), "quarter", "22-25", 7)
check("quarter labels are YYYYQn",
      [r["period"] for r in rows] == ["2024Q1", "2024Q2"],
      str([r["period"] for r in rows]))
fakem = pd.DataFrame({"coef": [-0.1], "se": [0.01], "n_obs": [10]},
                     index=["pm_2024_03_x_high_x_young"])
rowm = s84.path_rows(fakem, list(fakem.index), "month", "22-25", 7)
check("month labels are YYYY-MM", rowm[0]["period"] == "2024-03",
      rowm[0]["period"])

print("\n--- the dating rules can pass and can fail ---")
def synth(band, first_sig, n=8, start=2023):
    out, k = [], 0
    for y in range(start, start + 3):
        for q in range(1, 5):
            p = f"{y}Q{q}"
            neg = p >= first_sig
            out.append({"young_band": band, "shape": "quarter", "period": p,
                        "coef": -0.05 if neg else 0.001,
                        "se": 0.01, "t": -5.0 if neg else 0.1,
                        "n_firms": 100, "n_obs": 10, "status": "ok"})
            k += 1
            if k >= n * 3:
                break
    return out


good = synth("22-25", "2024Q3") + synth("26-30", "2025Q3")
v1, lines = s84.verdicts(good)
check("the dating reproduces on a planted path", v1 == "THE DATING REPRODUCES",
      v1)
check("the lag reproduces when 26-30 opens later",
      any("THE LAG REPRODUCES" in l for l in lines))
early = synth("22-25", "2023Q1") + synth("26-30", "2025Q3")
v1b, _ = s84.verdicts(early)
check("a fall before 2024 fails the dating rule",
      v1b == "THE DATING DOES NOT REPRODUCE", v1b)
same = synth("22-25", "2024Q3") + synth("26-30", "2024Q3")
_, lines_same = s84.verdicts(same)
check("simultaneous openings fail the lag rule",
      any("THE LAG DOES NOT REPRODUCE" in l for l in lines_same))
none_rows = [r for r in good if r["young_band"] == "22-25"]
_, lines_none = s84.verdicts(none_rows)
check("a band with no path gives NO VERDICT on the lag, not a pass",
      any("NO VERDICT" in l for l in lines_none))

print("\n--- a failed fit leaves no row ---")
before = len(s84.FAILURES)
bad = s84.fit(b.head(50), "deliberate_failure", ["not_a_column"], j47.FES)
check("a failed fit returns None and is recorded",
      bad is None and len(s84.FAILURES) > before)

# ======================================================================
# main(), end to end
# ======================================================================
print("\n--- main() ---")
s84.FAILURES.clear()
rc = s84.main()
check("main returns 0", rc == 0, str(rc))
out = s84.OUT / "occ_route_path.csv"
check("the path export exists", out.exists())
P = pd.read_csv(out) if out.exists() else pd.DataFrame()
check("both bands and both shapes are in the export",
      set(P["young_band"]) == {"22-25", "26-30"}
      and set(P["shape"]) == {"quarter", "month"} if len(P) else False,
      f"{sorted(set(P.get('shape', [])))} x {sorted(set(P.get('young_band', [])))}")
check("the export carries the columns the figure needs",
      set(["young_band", "shape", "period", "coef", "se", "status"])
      <= set(P.columns) if len(P) else False)
q = P[P["shape"] == "quarter"] if len(P) else pd.DataFrame()
check("the quarterly path runs to the end of the panel",
      len(q) and q["period"].max() >= "2025Q2", 
      str(q["period"].max()) if len(q) else "none")
check("four fits, two shapes by two bands",
      sum(1 for t, _ in FITS if t.startswith("s84_path_")) == 4,
      str([t for t, _ in FITS if t.startswith("s84_path_")]))
check("no fit was run on a rebuilt score",
      not any("occ_route_exposure" in t for t, _ in FITS))
summ = (s84.OUT / "84_summary.txt")
check("the summary exists and states the dating", summ.exists()
      and "THE DATING" in summ.read_text(encoding="utf-8"))
check("the consistency line can say the two fits are not one panel",
      "THE TWO DIFFER BY MORE THAN THE ESTIMATE ITSELF"
      in s84.consistency(P.to_dict("records")) if len(P) else False,
      "the planted fall is far from the real pooled step, so it fires here")
check("the summary prints the read rules",
      "READ RULES, FIXED BEFORE THE RUN" in summ.read_text(encoding="utf-8"))

# The planted world: 22-25 must open in 2024 and 26-30 a year later. The
# fixture is drawn so that this is true; if the script cannot recover it
# on data this clean, it will not recover it on the register.
if len(q):
    a = s84.first_fall(P.to_dict("records"), "22-25")
    c = s84.first_fall(P.to_dict("records"), "26-30")
    check("the planted opening at 22-25 is recovered in 2024",
          a is not None and a[:4] == "2024", str(a))
    check("the planted lag at 26-30 is recovered, and is later",
          c is not None and a is not None and c > a, f"{a} then {c}")

print("\n" + "=" * 60)
print(f"{len(FAILS)} FAILED" if FAILS else "ALL PASS")
for f in FAILS:
    print("  " + f)
shutil.rmtree(TMP, ignore_errors=True)
raise SystemExit(1 if FAILS else 0)
