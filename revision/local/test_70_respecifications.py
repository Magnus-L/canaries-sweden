#!/usr/bin/env python3
"""
test_70_respecifications.py -- Part A must SEPARATE two worlds that the
                               paper's current reporting cannot.

The whole reason 70 exists is that the paper reads two separately
significant coefficients as evidence the young are distinctively hit, when
the point estimates actually put 22-25 above 41-49. Part A answers that by
dropping 41-49 deliberately, so what comes back is the contrast itself.

A test that only checked "it produces a number" would not establish that.
So two worlds are planted and the estimator must tell them apart:

  EQUAL   22-25 and 41-49 both decline by the same beta. The contrast
          must be indistinguishable from zero, which is the case the
          paper would currently misreport as canaries.
  YOUNG   only 22-25 declines. The contrast must be clearly negative.

If Part A cannot separate those two it is worthless, whatever it prints.

    CANARIES_DRYRUN=1 python3 revision/local/test_70_respecifications.py
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
TMP = Path(tempfile.mkdtemp(prefix="canaries70_"))
SHARE = TMP / "input"; SHARE.mkdir()
for f in ("daioe_quartiles.dta", "utb_grupp2_sun2020_niva3_inr4_nyckel.dta",
          "eloundou_ssyk4.dta"):
    shutil.copy(UPLOAD / f, SHARE / f)
os.environ["CANARIES_SHARE"] = str(SHARE)
sys.path.insert(0, str(MONA))
import mona_common as mc  # noqa: E402
mc.SHARE = str(SHARE); mc.CACHE_DIR = TMP / "cache"; mc.CACHE_DIR.mkdir()
# mona_common builds DAIOE_PATH with a Windows separator, right on MONA and
# unopenable here. load_daioe binds it as a default argument at import, so
# the function is what must be repointed. The scripts stay untouched.
_LOCAL_DAIOE = str(SHARE / "daioe_quartiles.dta")
mc.DAIOE_PATH = _LOCAL_DAIOE
_load_daioe = mc.load_daioe
mc.load_daioe = lambda path=_LOCAL_DAIOE: _load_daioe(path)
sys.path.insert(0, str(HERE))
from _fixtures import Fixture, AGES  # noqa: E402


def load(n, a):
    sp = importlib.util.spec_from_file_location(a, MONA / n)
    m = importlib.util.module_from_spec(sp); sys.modules[a] = m
    sp.loader.exec_module(m); return m


s70 = load("70_respecifications.py", "s70")
s70.OUT = TMP / "out"; s70.OUT.mkdir(); s70.CACHE = mc.CACHE_DIR
j47 = s70._mod("47j_within_employer_triple.py", "j47")
h47 = j47._h47()
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


# ---- the exposure the script will actually use -----------------------
YEARS = [2021, 2022, 2023, 2024, 2025]
fx = Fixture(mc, h47, n_firms=140, n_exposed=60)
fx.install_edu([2019], cache=mc.CACHE_DIR)
# The occupation route must score FEWER firms than the education route,
# because that asymmetry is the whole subject of Part C: in the register it
# is 65,146 against 311,227. A fixture where both routes see every firm
# would make the coverage rungs vacuous, so a third of the employers get
# their incumbent occupation codes blanked, which is what an uncoded firm
# looks like to 65.
_base = fx.baseline_frame()
_blank = sorted(_base["employer_id"].unique())[::3]
_mask = (_base["employer_id"].isin(_blank)
         & _base["age_group"].isin(j47.INCUMBENT_BANDS))
_base.loc[_mask, "ssyk4"] = "____"
_base.to_parquet(mc.CACHE_DIR / "L_baseline_2019.parquet", index=False)
expo = s70.edu_exposure(j47, s70.DESIGN, s70.ARM)
HIGH = set(expo[expo["fq"] == 4]["employer_id"])
check("the education route classifies the fixture", len(expo) > 50,
      f"{len(expo)} firms, {len(HIGH)} in the top quartile")
check("the top quartile is neither empty nor everything",
      0 < len(HIGH) < len(expo))


def counts(decline_bands, beta=-0.30, seed=5):
    """
    Monthly counts where exactly `decline_bands` fall in high-exposure
    firms after the adoption date, by exactly `beta`. Nothing else moves,
    so any contrast the estimator reports is one the generator put there.
    """
    rng = np.random.default_rng(seed)
    lam0 = {"22-25": 9, "26-30": 10, "31-34": 9, "35-40": 11,
            "41-49": 13, "50+": 15}
    rows = []
    for emp in expo["employer_id"]:
        hi = emp in HIGH
        for y in YEARS:
            for m in (range(1, 13) if y < 2025 else range(1, 7)):
                ym = f"{y}-{m:02d}"
                for age in AGES:
                    lam = float(lam0[age])
                    if hi and age in decline_bands and ym >= s70.POOLED_FROM:
                        lam *= float(np.exp(beta))
                    rows.append((emp, ym, age, int(rng.poisson(lam)) + 1))
    return pd.DataFrame(rows, columns=["employer_id", "year_month",
                                       "age_group", "n_emp"])


# ---- the skeleton ----------------------------------------------------
c_equal = counts(("22-25", "41-49"))
skel = s70.all_band_skeleton(c_equal)
check("the skeleton carries only the contrast bands, not all six",
      sorted(skel["age_group"].unique()) == sorted(s70.CONTRAST_BANDS),
      str(sorted(skel["age_group"].unique())))
check("the reference band is among them, or nothing is identified",
      s70.REF_BAND in s70.CONTRAST_BANDS)
check("both young bands are kept",
      {"22-25", "26-30"} <= set(s70.CONTRAST_BANDS))
n_emp = skel["employer_id"].nunique()
n_ym = skel["year_month"].nunique()
check("the skeleton is balanced",
      len(skel) == n_emp * n_ym * len(s70.CONTRAST_BANDS),
      f"{len(skel):,} vs {n_emp * n_ym * len(s70.CONTRAST_BANDS):,}")
check("the fixed effects are integers, not pasted strings",
      all(str(skel[c].dtype).startswith("int")
          for c in ("fe_emp_t", "fe_emp_age", "fe_t_age")))
check("the panel starts where the script says it does",
      skel["year_month"].min() >= s70.PANEL_FROM)
# The 21 September failure: six bands x 172,396 firms x 54 months is
# 55.9M rows and about 9.3M employer-month levels, and fepois took an
# access violation. Three bands is the fix, so the panel must actually
# be smaller rather than merely relabelled.
check("three bands really halve the panel",
      len(skel) < 0.6 * n_emp * n_ym * len(s70.ALL_BANDS),
      f"{len(skel):,} vs six-band {n_emp * n_ym * len(s70.ALL_BANDS):,}")


no_ref = c_equal[~((c_equal["employer_id"] == c_equal["employer_id"].iloc[0])
                   & (c_equal["age_group"] == s70.REF_BAND))]
sk2 = s70.all_band_skeleton(no_ref)
check("a firm without the reference band is dropped",
      c_equal["employer_id"].iloc[0] not in set(sk2["employer_id"]))


# ---- Part A must separate the two worlds -----------------------------
def contrast(cnt, tag):
    sink = []
    s70.part_a(cnt, expo, j47, sink)
    df = pd.DataFrame(sink)
    if df.empty:
        return None, None
    df.to_csv(TMP / f"a_{tag}.csv", index=False)
    r = df[df.band_vs_ref == "22_25"]
    if r.empty:
        return None, None
    return float(r.iloc[0]["coef"]), float(r.iloc[0]["t"])


c_eq, t_eq = contrast(c_equal, "equal")
c_yo, t_yo = contrast(counts(("22-25",), seed=6), "young")

check("EQUAL world: the contrast is near zero",
      c_eq is not None and abs(c_eq) < 0.08, f"coef {c_eq}")
check("EQUAL world: the contrast is not significant",
      t_eq is not None and abs(t_eq) < 2.5, f"t {t_eq}")
check("YOUNG world: the contrast is clearly negative",
      c_yo is not None and c_yo < -0.15, f"coef {c_yo}")
check("YOUNG world: the contrast is significant",
      t_yo is not None and t_yo < -2.5, f"t {t_yo}")
check("the two worlds are separated, which is the point of Part A",
      None not in (c_eq, c_yo) and c_yo < c_eq - 0.10,
      f"equal {c_eq:.4f} vs young {c_yo:.4f}"
      if None not in (c_eq, c_yo) else "")

adf = pd.read_csv(TMP / "a_equal.csv")
check(f"{s70.REF_BAND} is the omitted band and reports no coefficient",
      s70.REF_BAND.replace("-", "_") not in set(adf["band_vs_ref"]))
check("every other band reports a contrast",
      len(adf) == len(s70.CONTRAST_BANDS) - 1, f"{len(adf)} rows")
check("the reference is recorded in the output",
      set(adf["reference"]) == {s70.REF_BAND})


# ---- Part C builds the skeleton once, not once per rung --------------
calls = {"n": 0}
_real = s70.all_band_skeleton


def counted(cnt):
    calls["n"] += 1
    return _real(cnt)


s70.all_band_skeleton = counted
l65 = s70._mod("65_occupation_arm.py", "l65")
csink = []
s70.part_c(c_equal, j47, l65, csink)
s70.all_band_skeleton = _real
check("Part C builds the panel once for all four rungs", calls["n"] == 1,
      f"{calls['n']} builds")
if csink:
    cdf = pd.DataFrame(csink)
    rungs = set(cdf["rung"])
    n_of = {r: int(cdf[cdf.rung == r]["n_firms"].iloc[0]) for r in rungs
            if len(cdf[cdf.rung == r])}
    check("Part C reports the intersection rungs as a subset of the full one",
          all(v <= n_of.get("A_edu_full", 0) for v in n_of.values()),
          ", ".join(f"{r}:{n_of[r]}" for r in sorted(n_of)))
    check("the duplicate C rung is gone, it was B re-run",
          "C_edu_joint" not in n_of, str(sorted(n_of)))
    check("the coverage rung is STRICTLY smaller, so A-vs-B is not vacuous",
          n_of.get("B_edu_intersect", 0) < n_of.get("A_edu_full", 0),
          f"A {n_of.get('A_edu_full')} vs B {n_of.get('B_edu_intersect')}")
    check("the occupation rung sits on the same firms as its education twin",
          n_of.get("D_occ_joint") == n_of.get("B_edu_intersect"),
          f"B {n_of.get('B_edu_intersect')} vs D {n_of.get('D_occ_joint')}")


# ---- end to end ------------------------------------------------------
for y in YEARS:
    c_equal[c_equal["year_month"].str[:4] == str(y)].to_parquet(
        mc.CACHE_DIR / f"L_counts_{y}.parquet", index=False)
s70.PANEL_YEARS = YEARS
_real_stdout = sys.stdout
try:
    s70.main()
    ran = True
except BaseException as ex:  # noqa: BLE001
    ran = False
    sys.stdout = _real_stdout
    print(f"      main() raised: {type(ex).__name__}: {ex}")
sys.stdout = _real_stdout
check("main() runs end to end", ran)
check("a summary is written", (s70.OUT / "70_summary.txt").exists())
if (s70.OUT / "70_summary.txt").exists():
    txt = (s70.OUT / "70_summary.txt").read_text()
    check("the summary reaches a read on the age contrast",
          "READ:" in txt)
    check("the summary states the limit of the C-versus-D comparison",
          "not the register alone" in txt)

# ---- it refuses rather than doing SQL it promised not to do ----------
(mc.CACHE_DIR / f"L_counts_{YEARS[0]}.parquet").unlink()
refused = False
try:
    s70.main()
except SystemExit as ex:
    refused = "does no SQL by design" in str(ex)
except BaseException:  # noqa: BLE001
    refused = False
sys.stdout = _real_stdout
check("a missing counts cache is refused, not silently pulled", refused)

print("\n" + "=" * 62)
print(f"{'FAILED: ' + ', '.join(FAILS) if FAILS else 'all checks passed'}")
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if FAILS else 0)
