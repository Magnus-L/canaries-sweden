#!/usr/bin/env python3
"""
test_69_cohort.py -- the fixture must GENERATE the January turnover, not
                     assert it.

69 exists to decide whether the Q4/Q1 cycle script 64 found is a real
seasonal or an accounting artefact of a moving age window. A fixture that
simply stamped a seasonal onto both bases could not tell the two apart, so
this one builds counts at BIRTH-YEAR level and aggregates them two ways:

  ageband   band = year - birth year, so everyone crosses on 1 January
  cohort    band = birth year, fixed once from ages in REF_YEAR

Nothing in the generator knows about quarters. The only thing planted is
that exposed firms hold their young workers at the OLD end of the band,
which is what a professional-services firm recruiting graduates at 23-25
rather than 22 looks like. If the moving-window aggregation then shows an
exposure-differential January step and the cohort one does not, the
artefact is real and 69's read rule will see it.

    CANARIES_DRYRUN=1 python3 revision/local/test_69_cohort.py
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
TMP = Path(tempfile.mkdtemp(prefix="canaries69_"))
SHARE = TMP / "input"; SHARE.mkdir()
for f in ("daioe_quartiles.dta", "utb_grupp2_sun2020_niva3_inr4_nyckel.dta",
          "eloundou_ssyk4.dta"):
    shutil.copy(UPLOAD / f, SHARE / f)
os.environ["CANARIES_SHARE"] = str(SHARE)
sys.path.insert(0, str(MONA))
import mona_common as mc  # noqa: E402
mc.SHARE = str(SHARE); mc.CACHE_DIR = TMP / "cache"; mc.CACHE_DIR.mkdir()
sys.path.insert(0, str(HERE))
from _fixtures import Fixture  # noqa: E402


def load(n, a):
    sp = importlib.util.spec_from_file_location(a, MONA / n)
    m = importlib.util.module_from_spec(sp); sys.modules[a] = m
    sp.loader.exec_module(m); return m


s69 = load("69_cohort_basis.py", "s69")
j47 = s69._mod("47j_within_employer_triple.py", "j47")
h47 = j47._h47()
s69.OUT = TMP / "out"; s69.OUT.mkdir(); s69.CACHE = mc.CACHE_DIR
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


# ---- 1. the cohort arithmetic ----------------------------------------
lo, hi = s69.cohort_years("22-25")
check("22-25 in 2022 means born 1997-2000", (lo, hi) == (1997, 2000),
      f"{lo}-{hi}")
spans = [s69.cohort_years(b) for b in s69.YOUNG_BANDS + s69.INCUMBENT_BANDS]
years = [y for a, b in spans for y in range(a, b + 1)]
check("the cohort bands do not overlap", len(years) == len(set(years)),
      f"{len(years)} years, {len(set(years))} distinct")
check("the cohort bands leave no gap",
      sorted(set(years)) == list(range(min(years), max(years) + 1)))
check("a cohort band never moves with the calendar",
      s69.cohort_years("22-25") == s69.cohort_years("22-25"))


# ---- 2. a generator that knows nothing about quarters ----------------
YEARS = [2021, 2022, 2023, 2024, 2025]
N_FIRMS, N_EXPOSED = 120, 50
REF = s69.REF_YEAR


def person_counts(seed=7, treat_beta=0.0):
    """
    Counts by employer x BIRTH YEAR x month.

    Exposed firms put their young workers at the old end of the 22-25
    band; unexposed firms spread them evenly. No quarter, month or
    seasonal term appears anywhere in this function.
    """
    rng = np.random.default_rng(seed)
    y0, y1 = s69.cohort_years("22-25")
    young_years = list(range(y0 - 4, y1 + 1))     # a little wider than the band
    older = {b: s69.cohort_years(b) for b in s69.INCUMBENT_BANDS}
    rows = []
    for emp in range(1, N_FIRMS + 1):
        exposed = emp <= N_EXPOSED
        # the planted asymmetry: exposed firms skew OLD inside the band
        w = np.array([1.0 + (3.0 if exposed else 0.0) * (i / len(young_years))
                      for i in range(len(young_years))])
        w = w / w.sum()
        for yr in YEARS:
            months = range(1, 13) if yr < 2025 else range(1, 7)
            for m in months:
                ym = f"{yr}-{m:02d}"
                for i, by in enumerate(young_years):
                    lam = 40.0 * w[i]
                    if treat_beta and exposed and ym >= "2024-01":
                        lam *= float(np.exp(treat_beta))
                    rows.append((emp, ym, by, int(rng.poisson(lam)) + 1))
                for b, (a0, a1) in older.items():
                    for by in range(a0, a1 + 1):
                        lam = 60.0 / (a1 - a0 + 1)
                        rows.append((emp, ym, by, int(rng.poisson(lam)) + 1))
    return pd.DataFrame(rows, columns=["employer_id", "year_month",
                                       "fodelse", "n_emp"])


def as_ageband(pc):
    """band = calendar year minus birth year: the moving window."""
    d = pc.copy()
    yr = d["year_month"].str[:4].astype(int)
    age = yr - d["fodelse"]
    band = pd.Series(pd.NA, index=d.index, dtype="object")
    for b, (a0, a1) in s69.BAND_AGES.items():
        band = band.mask(age.between(a0, a1), b)
    d["age_group"] = band
    d = d[d["age_group"].notna()]
    return (d.groupby(["employer_id", "year_month", "age_group"],
                      observed=True)["n_emp"].sum().reset_index())


def as_cohort(pc):
    """band = birth year, fixed: 69's own definition."""
    d = pc.copy()
    band = pd.Series(pd.NA, index=d.index, dtype="object")
    for b in s69.YOUNG_BANDS + s69.INCUMBENT_BANDS:
        a0, a1 = s69.cohort_years(b)
        band = band.mask(d["fodelse"].between(a0, a1), b)
    d["age_group"] = band
    d = d[d["age_group"].notna()]
    return (d.groupby(["employer_id", "year_month", "age_group"],
                      observed=True)["n_emp"].sum().reset_index())


pc = person_counts()
ab, co = as_ageband(pc), as_cohort(pc)


def jan_step(counts):
    """
    Exposure-differential January step in the young-to-older ratio.

    The statistic 64 reacts to, computed directly: how much the exposed
    firms' young share moves at the turn of the year relative to the
    unexposed firms'. No estimator involved, so this measures the fixture
    rather than the code.
    """
    d = counts.copy()
    d["exposed"] = (d["employer_id"] <= N_EXPOSED).astype(int)
    d["young"] = d["age_group"].isin(s69.YOUNG_BANDS).astype(int)
    g = (d.groupby(["year_month", "exposed", "young"], observed=True)["n_emp"]
         .sum().unstack("young").fillna(0.0))
    ratio = np.log((g[1] + 1) / (g[0] + 1)).unstack("exposed")
    diff = (ratio[1] - ratio[0]).sort_index()
    dec = [m for m in diff.index if m.endswith("-12")]
    steps = []
    for m in dec:
        nxt = f"{int(m[:4]) + 1}-01"
        if nxt in diff.index:
            steps.append(abs(diff[nxt] - diff[m]))
    return float(np.mean(steps)) if steps else float("nan")


s_ab, s_co = jan_step(ab), jan_step(co)
check("the fixture GENERATES a January step on the moving window",
      s_ab > 0.01, f"mean |step| {s_ab:.4f}")
check("the same data shows a much smaller step on fixed cohorts",
      s_co < 0.5 * s_ab, f"ageband {s_ab:.4f} vs cohort {s_co:.4f}")
check("no month is missing from either aggregation",
      set(ab['year_month']) == set(co['year_month']))
check("both aggregations conserve the same people",
      abs(ab['n_emp'].sum() - co['n_emp'].sum()) <= 0.02 * ab['n_emp'].sum(),
      f"{ab['n_emp'].sum():,} vs {co['n_emp'].sum():,}")


# ---- 3. the read rule, exercised at its own boundaries ---------------
def verdict_for(ratio):
    season = pd.DataFrame([
        {"basis": "ageband", "young_band": "22-25", "term": "hy_q4",
         "coef": 0.10, "se": 0.01},
        {"basis": "cohort", "young_band": "22-25", "term": "hy_q4",
         "coef": 0.10 * ratio, "se": 0.01}])
    pooled = pd.DataFrame([
        {"basis": "ageband", "young_band": "22-25", "coef": -0.05,
         "se": 0.01, "n_firms": 100},
        {"basis": "cohort", "young_band": "22-25", "coef": -0.048,
         "se": 0.01, "n_firms": 100}])
    return " ".join(s69.verdict(pooled, season))


check("a collapsed seasonal reads MECHANICAL",
      "SEASONAL MECHANICAL" in verdict_for(0.2))
check("an unchanged seasonal reads REAL", "SEASONAL REAL" in verdict_for(0.95))
check("an intermediate seasonal reads AMBIGUOUS",
      "AMBIGUOUS" in verdict_for(0.6))
check("a cohort estimate inside the interval reads ROBUST",
      "HEADLINE ROBUST" in verdict_for(0.2))

far = pd.DataFrame([
    {"basis": "ageband", "young_band": "22-25", "coef": -0.05, "se": 0.01,
     "n_firms": 100},
    {"basis": "cohort", "young_band": "22-25", "coef": -0.005, "se": 0.01,
     "n_firms": 100}])
seas = pd.DataFrame([
    {"basis": "ageband", "young_band": "22-25", "term": "hy_q4",
     "coef": 0.10, "se": 0.01},
    {"basis": "cohort", "young_band": "22-25", "term": "hy_q4",
     "coef": 0.02, "se": 0.01}])
check("a cohort estimate outside the interval reads HEADLINE MOVES",
      "HEADLINE MOVES" in " ".join(s69.verdict(far, seas)))
check("a missing basis does not crash the verdict",
      "UNAVAILABLE" in " ".join(s69.verdict(
          far[far.basis == "ageband"], seas[seas.basis == "ageband"])))


# ---- 4. the seasonal is fitted on the PRE window only ----------------
probe = ab.copy()
probe["high"] = (probe["employer_id"] <= N_EXPOSED).astype(int)
probe["young"] = probe["age_group"].isin(["22-25"]).astype(int)
bs, terms = s69.season_terms(probe)
check("the seasonal fit uses only pre-launch months",
      len(bs) and bs["year_month"].max() <= s69.PRE_TO,
      f"max {bs['year_month'].max()}")
check("Q3 is the omitted quarter",
      sorted(terms) == ["hy_q1", "hy_q2", "hy_q4"], str(sorted(terms)))


# ---- 5. end to end, with both caches installed -----------------------
fx = Fixture(mc, h47, n_firms=N_FIRMS, n_exposed=N_EXPOSED)
fx.install_edu([2019], cache=mc.CACHE_DIR)
tr = person_counts(seed=11, treat_beta=-0.20)
ab_t, co_t = as_ageband(tr), as_cohort(tr)
for y in YEARS:
    ab_t[ab_t["year_month"].str[:4] == str(y)].to_parquet(
        mc.CACHE_DIR / f"L_counts_{y}.parquet", index=False)
    co_t[co_t["year_month"].str[:4] == str(y)].to_parquet(
        mc.CACHE_DIR / f"L_cohort_{y}.parquet", index=False)

s69.PANEL_YEARS = YEARS
_real_stdout = sys.stdout
try:
    s69.main()
    ran = True
except SystemExit as ex:
    ran = False; print(f"      main() exited: {ex}")
except Exception as ex:  # noqa: BLE001
    ran = False; print(f"      main() raised: {type(ex).__name__}: {ex}")
# main() installs a Tee, which caps console echo. Without restoring stdout
# every check below this line would run and report into a temporary file
# that the test then deletes, which is indistinguishable from not running.
sys.stdout = _real_stdout
check("main() runs end to end with both caches present", ran)
check("a summary is written", (s69.OUT / "69_summary.txt").exists())
if (s69.OUT / "69_summary.txt").exists():
    txt = (s69.OUT / "69_summary.txt").read_text()
    check("the summary states the cost of the design honestly",
          "cannot speak about whoever is" in txt)
    check("the summary reaches one of the three seasonal verdicts",
          any(v in txt for v in ("SEASONAL MECHANICAL", "SEASONAL REAL",
                                 "AMBIGUOUS", "UNAVAILABLE")))


# ---- 6. it refuses rather than races on a missing shared cache -------
(mc.CACHE_DIR / f"L_counts_{YEARS[-1]}.parquet").unlink()
refused = False
try:
    s69.main()
except SystemExit as ex:
    refused = "does not build it" in str(ex)
except Exception:  # noqa: BLE001
    refused = False
sys.stdout = _real_stdout
check("a missing L_counts is refused, not rebuilt from another slot",
      refused)

print("\n" + "=" * 62)
print(f"{'FAILED: ' + ', '.join(FAILS) if FAILS else 'all checks passed'}")
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if FAILS else 0)
