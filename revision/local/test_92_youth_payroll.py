#!/usr/bin/env python3
"""
test_92_youth_payroll.py -- the split must recompose, and the verdict must
                            be able to go either way.

The rival says the decline at 22-25 is the withdrawal of the reduced youth
payroll contribution on 31 March 2023, which reached workers up to age 22
or 23 and never reached 24-25. Script 92 splits the band at that line. The
test plants both worlds, so the read rule has one in which it must reject
the rival and one in which it must let it stand:

  AI world       the adoption step is planted on BOTH halves of the young
                 band in exposed firms. 24-25 were never eligible, so a
                 decline there cannot be the payroll withdrawal: rule 1
                 must fire and the rival must be rejected.

  PAYROLL world  the step is planted on 22-23 ONLY, and from April 2023
                 rather than January 2024, which is the shape the rival
                 actually predicts. Rule 2 must fire and the script must
                 say the rival survives. A test that could only reject
                 would be worthless.

Also tested: the recomposition gate stops the run when the two sub-bands
do not sum to the 22-25 counts the paper estimates on; the comparison
bands are byte-identical to the paper's four; and no SQL is issued when
the fine caches are already present, so a rerun costs nothing.

    CANARIES_DRYRUN=1 python3 revision/local/test_92_youth_payroll.py
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
TMP = Path(tempfile.mkdtemp(prefix="canaries92_"))
SHARE = TMP / "input"; SHARE.mkdir()
for f in ("daioe_quartiles.dta", "eloundou_ssyk4.dta",
          "utb_grupp2_sun2020_niva3_inr4_nyckel.dta"):
    shutil.copy(UPLOAD / f, SHARE / f)
os.environ["CANARIES_SHARE"] = str(SHARE)
os.environ["CANARIES_92_OUT"] = str(TMP / "out")
os.environ["CANARIES_82_OUT"] = str(TMP / "out")
sys.path.insert(0, str(MONA)); sys.path.insert(0, str(HERE))
import mona_common as mc  # noqa: E402
mc.SHARE = str(SHARE); mc.CACHE_DIR = TMP / "cache"; mc.CACHE_DIR.mkdir()
_LOCAL = str(SHARE / "daioe_quartiles.dta")
mc.DAIOE_PATH = _LOCAL
_ld = mc.load_daioe
mc.load_daioe = lambda path=_LOCAL: _ld(path)

SQL_CALLS = []
mc.connect = lambda: object()


def load(n, a):
    sp = importlib.util.spec_from_file_location(a, MONA / n)
    m = importlib.util.module_from_spec(sp); sys.modules[a] = m
    sp.loader.exec_module(m); return m


s92 = load("92_youth_payroll_rival.py", "s92")
s92.OUT = TMP / "out"; s92.OUT.mkdir(parents=True, exist_ok=True)
s92.q_counts_fine = lambda year, conn: SQL_CALLS.append(year) or pd.DataFrame()
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


# ---------------------------------------------------------------- world
FINE = ["22-23", "24-25", "26-30", "31-34", "35-40", "41-49", "50+"]
N_SCORE, N_PANEL = 400, 300
SCORED, PANEL = list(range(1, N_SCORE + 1)), list(range(1, N_PANEL + 1))
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
AGES82 = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]

rows = []
for emp in SCORED:
    t = tier(emp); c, far = TIER_CODE[t], TIER_CODE[3 - t]; m = size_mult(emp)
    for age in AGES82:
        rows += [(emp, age, c, c[:3], "2019", 6 * m),
                 (emp, age, far, far[:3], "2019", 1 * m),
                 (emp, age, "____", "___", "none", 2)]
CASC = pd.DataFrame(rows, columns=["employer_id", "age_group", "ssyk4",
                                   "ssyk3", "source_year", "n"])
CASC["ssyk_ar"] = "2019"
CASC["ssyk_status"] = np.where(CASC["ssyk4"] == "____", "9", "1")
CASC.to_parquet(mc.CACHE_DIR / "L_baseline_2019_cascade.parquet", index=False)
(CASC.groupby(["employer_id", "age_group", "ssyk4"], observed=True)["n"]
 .sum().reset_index()).to_parquet(
    mc.CACHE_DIR / "L_baseline_2019.parquet", index=False)
pd.DataFrame([(e, f"2019-{m:02d}", a, size_mult(e))
              for e in SCORED for m in range(1, 13) for a in AGES82],
             columns=["employer_id", "year_month", "age_group", "n_emp"]
             ).to_parquet(mc.CACHE_DIR / "L_counts_2019.parquet", index=False)

s82 = load("82_occupation_route.py", "s82")
s61, s67, s74, s78, s80, l47, l70, j47 = s82.load_modules()
for m_ in (s82, s61, s78, l47, l70, j47):
    m_.OUT, m_.CACHE = s92.OUT, mc.CACHE_DIR
MONTHS = [f"{y}-{m:02d}" for y in s61.PANEL_YEARS
          for m in range(1, 13 if y < 2025 else 7)]
AI_FALL = float(np.log(0.82))       # from 2024-01, both halves
PAY_FALL = float(np.log(0.82))      # from 2023-04, 22-23 only
RISE_RB = float(np.log(1.06))


def counts_fine(world: str) -> pd.DataFrame:
    rng = np.random.default_rng({"ai": 92, "payroll": 93}[world])
    lam = {"22-23": 9, "24-25": 9, "26-30": 16, "31-34": 12,
           "35-40": 12, "41-49": 14, "50+": 14}
    out = []
    for emp in PANEL:
        hit = emp in EXPOSED
        for ym in MONTHS:
            for age, l0 in lam.items():
                x = float(l0)
                if hit:
                    if ym >= mc.RIKSBANK_YM and age in ("22-23", "24-25"):
                        x *= np.exp(RISE_RB)
                    if world == "ai" and age in ("22-23", "24-25") \
                            and ym >= "2024-01":
                        x *= np.exp(AI_FALL)
                    if world == "payroll" and age == "22-23" \
                            and ym >= "2023-04":
                        x *= np.exp(PAY_FALL)
                out.append((emp, ym, age, int(rng.poisson(x)) + 1))
    return pd.DataFrame(out, columns=["employer_id", "year_month",
                                      "age_group", "n_emp"])


def install(world: str) -> pd.DataFrame:
    fine = counts_fine(world)
    for y in s61.PANEL_YEARS:
        fine[fine["year_month"].str.slice(0, 4) == str(y)].to_parquet(
            mc.CACHE_DIR / f"L_counts_fine_{y}.parquet", index=False)
    coarse = fine.copy()
    coarse["age_group"] = coarse["age_group"].replace(
        {"22-23": "22-25", "24-25": "22-25"})
    coarse = (coarse.groupby(["employer_id", "year_month", "age_group"],
                             observed=True)["n_emp"].sum().reset_index())
    for y in s61.PANEL_YEARS:
        coarse[coarse["year_month"].str.slice(0, 4) == str(y)].to_parquet(
            mc.CACHE_DIR / f"L_counts_{y}.parquet", index=False)
    return fine


print("\n--- the comparison bands are the paper's own ---")
check("92 compares against 47j's four incumbent bands, unchanged",
      list(j47.INCUMBENT_BANDS) == ["31-34", "35-40", "41-49", "50+"],
      str(list(j47.INCUMBENT_BANDS)))
check("the split is at the eligibility line and 24-25 is never eligible",
      [b for b, _, _ in s92.SPLIT] == ["22-23", "24-25"])

print("\n--- the recomposition gate stops a split that does not sum ---")
fine = install("ai")
bad = fine.copy()
m = (bad.age_group == "24-25") & (bad.year_month == MONTHS[5])
bad.loc[m, "n_emp"] = bad.loc[m, "n_emp"] + 3
coarse_ok = s82.load_counts("L_counts", s61.PANEL_YEARS)
try:
    s92.check_recomposes(bad, coarse_ok); stopped = False
except SystemExit:
    stopped = True
check("a sub-band that does not recompose stops the run", stopped)
try:
    s92.check_recomposes(fine, coarse_ok); ok = True
except SystemExit:
    ok = False
check("the true split passes the gate", ok)

print("\n--- no SQL when the fine caches are present ---")
n0 = len(SQL_CALLS)
s92.load_fine(s61.PANEL_YEARS, object())
check("load_fine reads the caches and issues no pull",
      len(SQL_CALLS) == n0, f"{len(SQL_CALLS)-n0} pulls")


def verdict(world: str) -> str:
    install(world)
    s92.FAILURES.clear()
    rc = s92.main()
    txt = (s92.OUT / "92_summary.txt").read_text(encoding="utf-8")
    print(txt)
    return txt


print("\n=== AI world: the rival must be REJECTED ===")
t_ai = verdict("ai")
check("rule 1 fires and the rival is rejected in the AI world",
      "THE RIVAL IS REJECTED" in t_ai)
check("the summary records that 24-25 were never eligible",
      "NEVER eligible" in t_ai)

print("\n=== PAYROLL world: the rival must SURVIVE ===")
t_pay = verdict("payroll")
check("rule 2 fires and the rival survives in the payroll world",
      "THE RIVAL SURVIVES" in t_pay)
check("the script refuses to test the difference of two correlated fits",
      "Do not test it." in t_pay)

print("\n" + ("all checks passed" if not FAILS else f"FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)
