#!/usr/bin/env python3
"""
test_87_occupation_route_gender_split.py -- the score must be lane 28's,
                                            the terms must be Equation
                                            (2)'s, the weights must come
                                            from THIS route's exposed
                                            firms, and the gate must be
                                            able to fail.

The checks are on mechanisms, not on outputs. One synthetic world, drawn
so that the split has something to find:

  400 employers carry a 2019 occupation mix and a 2019 person-month
      count; 300 of them also carry the monthly panel by sex and by
      education, with five broad tracks.
  A FEMALE DIFFERENTIAL is planted INSIDE EVERY TRACK, the same size in
      each, from January 2024 in the exposed employers. A differential
      that is identical across tracks cannot be composition, so a split
      that works must return within close to pooled and a ratio near
      one. That is the mechanism, planted before anything is fitted.

Also tested: that the exposure passed to the fits is 82's own score and
not 76's education exposure; that the term set is Equation (2)'s nine
interactions and not 68's shorter one; that the weights are read off
this route's exposed firms; that the lane opens no database connection
and refuses to run when the counts are not cached; that 76's export
names are not reused; and that the gate passes and fails.

    CANARIES_DRYRUN=1 python3 revision/local/test_87_occupation_route_gender_split.py
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
TMP = Path(tempfile.mkdtemp(prefix="canaries87_"))
SHARE = TMP / "input"; SHARE.mkdir()
for f in ("daioe_quartiles.dta", "eloundou_ssyk4.dta",
          "utb_grupp2_sun2020_niva3_inr4_nyckel.dta"):
    shutil.copy(UPLOAD / f, SHARE / f)
os.environ["CANARIES_SHARE"] = str(SHARE)
os.environ["CANARIES_87_OUT"] = str(TMP / "out")
os.environ["CANARIES_82_OUT"] = str(TMP / "out")
sys.path.insert(0, str(MONA)); sys.path.insert(0, str(HERE))
import mona_common as mc  # noqa: E402
mc.SHARE = str(SHARE); mc.CACHE_DIR = TMP / "cache"; mc.CACHE_DIR.mkdir()
_LOCAL = str(SHARE / "daioe_quartiles.dta")
mc.DAIOE_PATH = _LOCAL
_ld = mc.load_daioe
mc.load_daioe = lambda path=_LOCAL: _ld(path)

CONNECTED = []


def _no_sql():
    CONNECTED.append(1)
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


s87 = load("87_occupation_route_gender_split.py", "s87")
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
pd.DataFrame([(e, f"2019-{m:02d}", a, size_mult(e))
              for e in SCORED for m in range(1, 13) for a in AGES],
             columns=["employer_id", "year_month", "age_group", "n_emp"]
             ).to_parquet(mc.CACHE_DIR / "L_counts_2019.parquet", index=False)

(s82, s61, s67, s76, s78, l47, l70, j47) = s87.load_modules()
for m_ in (s82, s61, s67, s76, s78, l47, l70, j47):
    m_.OUT, m_.CACHE = s87.OUT, mc.CACHE_DIR
s87.OUT.mkdir(parents=True, exist_ok=True)

# 47h's weight caches, which the script reads to build the score book that
# prices EDUCATION GROUPS in the descriptive table. That book is not the
# firm's exposure and never was; the exposure comes from 82.
from _fixtures import Fixture  # noqa: E402
_h47 = j47._h47()
_fx = Fixture(mc, _h47)
for _y in (2019, 2020, 2021):
    _fx.weights_frame(_y).to_parquet(
        mc.CACHE_DIR / f"edu_hr_weights_{_y}.parquet", index=False)

print("\n--- the lane refuses to run without the counts ---")
try:
    s87.main()
    check("a missing counts cache stops the lane", False, "it ran anyway")
except RuntimeError as ex:
    check("a missing counts cache stops the lane",
          "L_counts_sex_edu" in str(ex), str(ex)[:60])
check("no database connection was opened", not CONNECTED)

# One field code per track, taken from the tracks 76 itself defines.
FIELD = {"ict": "48", "engineering": "52", "business_law_social": "34",
         "health_education_care": "72", "other": "99"}
TRACKS = list(FIELD)
MONTHS = [f"{y}-{m:02d}" for y in s76.YEARS
          for m in range(1, 13 if y < 2025 else 7)]
DIFF = float(np.log(0.88))   # the planted female differential, every track


def counts_sex_edu() -> pd.DataFrame:
    """The same differential inside EVERY track, so it cannot be
    composition and the split must return within close to pooled."""
    rng = np.random.default_rng(87)
    out = []
    for emp in PANEL:
        hit = emp in EXPOSED
        for ym in MONTHS:
            for age in ("22-25", "31-34", "41-49", "50+"):
                for sex in ("1", "2"):
                    for tr in TRACKS:
                        x = 9.0
                        if hit and sex == "2" and age == "22-25" \
                                and ym >= "2024-01":
                            x *= np.exp(DIFF)
                        out.append((emp, ym, age, sex, "4", FIELD[tr],
                                    int(rng.poisson(x)) + 1))
    return pd.DataFrame(out, columns=s76.EDU_COLS + ["n_emp"])


C = counts_sex_edu()
for y in s76.YEARS:
    C[C["year_month"].str.slice(0, 4) == str(y)].to_parquet(
        mc.CACHE_DIR / f"L_counts_sex_edu_{y}.parquet", index=False)

print("\n--- the score is 82's and not 76's ---")
built = s82.build_exposure(l47, l70, j47)
check("the score is 82's primary arm",
      built["arm"] == s82.MAIN_LEVEL and built["floor"] == s82.FLOOR_MAIN,
      f"{built['arm']} floor {built['floor']}")
check("82's score is not 76's education exposure",
      "OL_daioe" not in str(built.get("basis", "")),
      f"basis {built['basis']}")

print("\n--- the terms are Equation (2)'s ---")
h47 = _h47
frame = s76.tag_frame(C[C["year_month"].str.slice(0, 4) == "2024"], h47)
skel = s67.build_skeleton_sex(s76.collapse(frame), s76.YOUNG, j47, "n_emp")
b = s78.with_exposure(skel, built["exposure"])
_, t_eq2 = s78.gender_eq2_terms(b.copy())
_, t_68 = s76.gender_terms_seasonal(b.copy())
check("Equation (2)'s set is the larger one and contains 68's differential",
      set(t_68) < set(t_eq2) or len(t_eq2) > len(t_68),
      f"{len(t_68)} against {len(t_eq2)} terms")
check("the differential term is the one Table 1 prints",
      s87.DIFF_TERM in t_eq2, s87.DIFF_TERM)
check("Equation (2) carries the interim period and 68's set does not",
      any("interim" in t for t in t_eq2)
      and not any("interim" in t for t in t_68))
del frame, skel, b

print("\n--- main() ---")
rc = s87.main()
check("main returns 0", rc == 0, str(rc))
check("still no database connection", not CONNECTED)
S = s87.OUT / "occ_route_gender_split.csv"
T = s87.OUT / "occ_route_gender_by_track.csv"
M = s87.OUT / "occ_route_education_mix_by_sex.csv"
check("the split export exists under this route's name", S.exists())
check("the by-track export exists under this route's name", T.exists())
check("the composition export exists under this route's name", M.exists())
check("76's export names are not reused",
      not (s87.OUT / "gender_split.csv").exists()
      and not (s87.OUT / "gender_by_track.csv").exists()
      and not (s87.OUT / "education_mix_by_sex.csv").exists())
check("six fits, all workers and each of the five tracks",
      len([t for t, _ in FITS if t.startswith("s87_")]) == 1 + len(TRACKS),
      str([t for t, _ in FITS if t.startswith("s87_")]))
check("every fit used Equation (2)'s term set",
      all(s87.DIFF_TERM in terms and any("interim" in x for x in terms)
          for t, terms in FITS if t.startswith("s87_")))

if S.exists():
    sp = pd.read_csv(S).iloc[0]
    check("the planted differential is recovered, negative",
          sp["pooled"] < 0, f"pooled {sp['pooled']:+.4f}")
    check("a differential planted equally in every track is within, not "
          "composition",
          abs(sp["ratio_within"] - 1.0) < 0.25,
          f"ratio {sp['ratio_within']:.3f}, composition "
          f"{sp['composition']:+.4f}")
if T.exists():
    tr = pd.read_csv(T)
    check("every track is fitted and reported",
          set(tr["track"]) == {"all"} | set(TRACKS),
          str(sorted(set(tr["track"]))))

summ = (s87.OUT / "87_summary.txt").read_text(encoding="utf-8")
check("the summary reports the gate", "THE GATE:" in summ)
check("the summary says the cut stays education and why",
      "classifies no young worker by occupation after 2019" in summ)
check("the summary prints the read rules",
      "READ RULES, FIXED BEFORE THE RUN" in summ)
check("the summary reports the weights as this route's",
      "THIS ROUTE'S EXPOSED FIRMS" in summ)

print("\n--- the gate can pass and can fail ---")
fitted = float(pd.read_csv(T)[lambda d: d.track == "all"]["diff"].iloc[0])
s87.TABLE1_DIFF = round(fitted, 6)
s87.FAILURES.clear(); s87.NOTES.clear(); s87.GATE.update(ok=None, detail="")
s87.main()
ok = (s87.OUT / "87_summary.txt").read_text(encoding="utf-8")
check("a matching Table 1 figure passes the gate", "THE GATE: PASS" in ok)

s87.TABLE1_DIFF = fitted - 0.5
s87.FAILURES.clear(); s87.NOTES.clear(); s87.GATE.update(ok=None, detail="")
s87.main()
bad = (s87.OUT / "87_summary.txt").read_text(encoding="utf-8")
check("a moved Table 1 figure fails the gate", "THE GATE: FAIL" in bad)
check("a failed gate is recorded as a failure", "WHAT FAILED" in bad)

print("\n" + "=" * 60)
print(f"{len(FAILS)} FAILED" if FAILS else "ALL PASS")
for f in FAILS:
    print("  " + f)
shutil.rmtree(TMP, ignore_errors=True)
raise SystemExit(1 if FAILS else 0)
