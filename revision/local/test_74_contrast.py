#!/usr/bin/env python3
"""
test_74_contrast.py -- the seasonal control must CHANGE a contrast that
                       the calendar created, and spare one it did not.

74 exists to decide whether the 22-25 versus 41-49 contrast is partly a
calendar artefact. A test that only checked it produces two numbers would
not establish that it can tell those cases apart, so two worlds are
planted:

  REAL      the young decline for real, evenly across quarters. The
            seasonally adjusted contrast must keep it.
  CALENDAR  nothing happens on average, but the exposure-differential
            young-to-older ratio has a Q4 spike, and the post window is
            weighted differently across quarters from the pre window. The
            unadjusted contrast must see a spurious effect and the
            adjusted one must remove it.

If the adjusted arm cannot kill the second, the control is decoration.

    CANARIES_DRYRUN=1 python3 revision/local/test_74_contrast.py
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
TMP = Path(tempfile.mkdtemp(prefix="canaries74_"))
SHARE = TMP / "input"; SHARE.mkdir()
for f in ("daioe_quartiles.dta", "utb_grupp2_sun2020_niva3_inr4_nyckel.dta",
          "eloundou_ssyk4.dta"):
    shutil.copy(UPLOAD / f, SHARE / f)
os.environ["CANARIES_SHARE"] = str(SHARE)
sys.path.insert(0, str(MONA))
import mona_common as mc  # noqa: E402
mc.SHARE = str(SHARE); mc.CACHE_DIR = TMP / "cache"; mc.CACHE_DIR.mkdir()
_LOCAL = str(SHARE / "daioe_quartiles.dta")
mc.DAIOE_PATH = _LOCAL
_ld = mc.load_daioe
mc.load_daioe = lambda path=_LOCAL: _ld(path)
sys.path.insert(0, str(HERE))
from _fixtures import Fixture  # noqa: E402


def load(n, a):
    sp = importlib.util.spec_from_file_location(a, MONA / n)
    m = importlib.util.module_from_spec(sp); sys.modules[a] = m
    sp.loader.exec_module(m); return m


s74 = load("74_contrast_seasonal.py", "s74")
s74.OUT = TMP / "out"; s74.OUT.mkdir(); s74.CACHE = mc.CACHE_DIR
l70 = s74._mod("70_respecifications.py", "l70")
j47 = s74._mod("47j_within_employer_triple.py", "j47")
h47 = j47._h47()
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


YEARS = [2021, 2022, 2023, 2024, 2025]
fx = Fixture(mc, h47, n_firms=200, n_exposed=85)
fx.install_edu([2019], cache=mc.CACHE_DIR)
EXPO = l70.edu_exposure(j47, l70.DESIGN, l70.ARM)
HIGH = set(EXPO[EXPO["fq"] == 4]["employer_id"])
check("the fixture classifies firms", len(EXPO) > 100 and 0 < len(HIGH) < len(EXPO),
      f"{len(EXPO)} firms, {len(HIGH)} high")


def counts(world, seed=3):
    """
    REAL: a flat decline for 22-25 in exposed firms after the date.
    CALENDAR: no treatment at all, but exposed firms' young workers have
    a recurring Q4 surplus. Because the post window contains a different
    mix of quarters from the pre window, an uncontrolled fit reads that
    as an effect.
    """
    rng = np.random.default_rng(seed)
    # one intensity per band in l70.CONTRAST_BANDS, which went from
    # three to six on 21 September; a missing key used to raise here
    lam0 = {"22-25": 10, "26-30": 11, "31-34": 12, "35-40": 12,
            "41-49": 13, "50+": 12}
    rows = []
    for emp in EXPO["employer_id"]:
        hi = emp in HIGH
        for y in YEARS:
            for m in (range(1, 13) if y < 2025 else range(1, 7)):
                ym = f"{y}-{m:02d}"
                q = (m - 1) // 3 + 1
                for age in l70.CONTRAST_BANDS:
                    lam = float(lam0.get(age, 12))
                    if hi and age == "22-25":
                        if world == "real" and ym >= s74.POOLED_FROM:
                            lam *= float(np.exp(-0.30))
                        if world == "calendar":
                            lam *= float(np.exp(0.30 if q == 4 else -0.10))
                    rows.append((emp, ym, age, int(rng.poisson(lam)) + 1))
    return pd.DataFrame(rows, columns=["employer_id", "year_month",
                                       "age_group", "n_emp"])


def both_arms(world):
    skel = l70.all_band_skeleton(counts(world))
    sink = []
    for seasonal in (False, True):
        s74.run_arm(skel, EXPO, l70, j47, seasonal, sink)
    d = pd.DataFrame(sink)
    g = lambda arm: (float(d[(d.arm == arm) & (d.band_vs_ref == "22_25")]
                           .iloc[0]["coef"])
                     if len(d[(d.arm == arm) & (d.band_vs_ref == "22_25")])
                     else None)
    return g("plain"), g("seasonal")


# ---- the terms themselves ---------------------------------------------
probe = l70.all_band_skeleton(counts("real")).merge(
    EXPO[["employer_id", "fq"]], on="employer_id", how="inner")
probe["high"] = (probe["fq"] == 4).astype(int)
_, t_plain = s74.build_terms(probe.copy(), l70, seasonal=False)
_, t_seas = s74.build_terms(probe.copy(), l70, seasonal=True)
check("the plain arm has no calendar terms",
      not any(x.startswith("q") for x in t_plain), str(t_plain))
check("the seasonal arm adds three quarters per non-reference band",
      len(t_seas) - len(t_plain) == 3 * (len(l70.CONTRAST_BANDS) - 1),
      f"{len(t_plain)} -> {len(t_seas)} terms")
check("Q4 is omitted, matching 68",
      not any("q4_" in x for x in t_seas))
check("the reference band gets no calendar term of its own, so every "
      "coefficient stays a difference from it",
      not any(l70.REF_BAND.replace("-", "_") in x for x in t_seas))


# ---- the two worlds ----------------------------------------------------
p_real, s_real = both_arms("real")
p_cal, s_cal = both_arms("calendar")
check("REAL world: the plain arm sees the planted decline",
      p_real is not None and p_real < -0.15, f"{p_real}")
check("REAL world: the seasonal control SPARES it",
      None not in (p_real, s_real) and abs(s_real) >= 0.5 * abs(p_real),
      f"plain {p_real:.4f} -> seasonal {s_real:.4f}"
      if None not in (p_real, s_real) else "")
check("CALENDAR world: the plain arm is FOOLED by the cycle",
      p_cal is not None and abs(p_cal) > 0.02, f"{p_cal}")
check("CALENDAR world: the seasonal control REMOVES it, so the control "
      "is not decoration",
      None not in (p_cal, s_cal) and abs(s_cal) < 0.5 * abs(p_cal),
      f"plain {p_cal:.4f} -> seasonal {s_cal:.4f}"
      if None not in (p_cal, s_cal) else "")


# ---- the read rule, including its void condition -----------------------
def words(plain, seas):
    d = pd.DataFrame([
        {"arm": "plain", "band_vs_ref": "22_25", "coef": plain, "se": 0.0135,
         "n_firms": 100},
        {"arm": "seasonal", "band_vs_ref": "22_25", "coef": seas, "se": 0.0140,
         "n_firms": 100}])
    return " ".join(s74.verdict(d))


check("a surviving contrast reads STABLE",
      "STABLE" in words(-0.0357, -0.0330))
check("a contrast the cycle explains reads SEASONAL",
      "SEASONAL" in words(-0.0357, -0.0080))
check("a plain arm that does NOT reproduce lane 16 voids the comparison",
      "VOID" in words(-0.0900, -0.0850))
check("and a void verdict reports no reading",
      "STABLE" not in words(-0.0900, -0.0850)
      and "SEASONAL." not in words(-0.0900, -0.0850))
check("an empty frame does not crash the verdict",
      isinstance(s74.verdict(pd.DataFrame()), list))

# ---- end to end -------------------------------------------------------
CNT = counts("real")
for y in YEARS:
    CNT[CNT["year_month"].str[:4] == str(y)].to_parquet(
        mc.CACHE_DIR / f"L_counts_{y}.parquet", index=False)
s74.PANEL_YEARS = YEARS
_real = sys.stdout
try:
    s74.main()
    ran = True
except BaseException as ex:  # noqa: BLE001
    ran = False; sys.stdout = _real
    print(f"      main() raised: {type(ex).__name__}: {ex}")
sys.stdout = _real
check("main() runs end to end", ran)
check("a summary is written", (s74.OUT / "74_summary.txt").exists())
if (s74.OUT / "74_summary.txt").exists():
    txt = (s74.OUT / "74_summary.txt").read_text()
    check("the summary explains why the seasonal terms are per band",
          "constant within employer-month" in txt)
    check("it reaches a verdict or says why it cannot",
          any(w in txt for w in ("STABLE", "SEASONAL.", "VOID", "NO VERDICT")))

(mc.CACHE_DIR / f"L_counts_{YEARS[0]}.parquet").unlink()
refused = False
try:
    s74.main()
except SystemExit as ex:
    refused = "performs no SQL" in str(ex)
except BaseException:
    refused = False
sys.stdout = _real
check("a missing counts cache is refused, not pulled", refused)

print("\n" + "=" * 62)
print(f"{'FAILED: ' + ', '.join(FAILS) if FAILS else 'all checks passed'}")
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if FAILS else 0)
