#!/usr/bin/env python3
"""
test_102_month_of_year.py -- the dry run of script 102 (lane 38c).

THE MECHANISM PLANTED. On 97's employer x age band x sex x month grid
(240 employers, four DAIOE tiers, tier 3 the top quartile), young workers
in exposed employers fall by FALL from January 2024 and young women there
by FALL_F more, as in the 97 test. On top of that, a WITHIN-QUARTER
calendar cycle in the young cells of exposed employers, the same in every
year: October and November +A/2, December -A (so every quarter's mean is
zero and the three quarter terms cannot see it), and the same pattern
with amplitude B on young women alone.

Why that pattern is the mechanism the letter names: the interim window,
December 2022 to December 2023, holds two Decembers in thirteen months,
so its mean of the cycle is -A/13 while the later window, six full
quarters, averages to zero. A specification with quarter terms cannot
remove it and reads tau as FALL + A/13; eleven month-of-year terms remove
it and read FALL. The same holds for the female differential with B.

Checked by hand first (the raw window contrast carries the bias), then:
  (i)  the month-of-year tau reproduces the planted FALL, the quarter tau
       is off by about A/13, and the difference between the two fits,
       which share every draw, is A/13 to within a small tolerance; the
       same for the female differential with B;
  (ii) the summary prints both taus, both clusterings and the differences,
       and says the difference's SE is not available;
also: the gate stops on Table 1's numbers, the industry-clustered fit has
the same coefficient and its own SE, main() runs end to end with no SQL
and no identifier out, and no employer count under the floor leaves.

    python3 revision/local/test_102_month_of_year.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _fixtures_trip37 as fx  # noqa: E402

mc, TMP = fx.sandbox("102", ("CANARIES_102_OUT",))
s102 = fx.load("102_month_of_year.py", "s102")
s102.OUT = TMP / "out"
s102.CACHE = mc.CACHE_DIR
check = fx.Check()

EMPS = list(range(1, 241))
tier = lambda e: e % 4                                          # noqa: E731
size = lambda e: 4.2 if tier(e) == 3 else 1 + (e // 4) % 5     # noqa: E731
fx.install_score(mc, EMPS, tier, size, Path(mc.SHARE))
HIGH = {e for e in EMPS if tier(e) == 3}
FALL, FALL_F = float(np.log(0.85)), float(np.log(0.88))
A, B = 0.26, 0.26                  # the within-quarter cycle's amplitude
MONTHS = fx.months()
LAM = {"22-25": 8, "26-30": 9, "31-34": 7, "35-40": 8, "41-49": 10, "50+": 12}
INTERIM_MONTHS = [m for m in MONTHS if "2022-12" <= m <= "2023-12"]
LATER_MONTHS = [m for m in MONTHS if m >= "2024-01"]


def cycle(month: pd.Series) -> np.ndarray:
    """October and November +1/2, December -1, other months 0: a
    within-quarter cycle whose quarterly mean is zero."""
    mo = month.str.slice(5, 7).astype(int).to_numpy()
    return np.where(mo == 12, -1.0, np.where(mo >= 10, 0.5, 0.0))


def expected_bias(amp: float) -> float:
    """Later-window mean of the cycle minus interim-window mean, times the
    amplitude: what a quarter-term specification reads into tau for a
    cell whose log count carries amp x cycle exactly (each sex alone)."""
    c_i = cycle(pd.Series(INTERIM_MONTHS)).mean()
    c_l = cycle(pd.Series(LATER_MONTHS)).mean()
    return amp * (c_l - c_i)


def pooled_dev(c: np.ndarray) -> np.ndarray:
    """The POOLED 22-25 cell at exposed employers is men plus women with
    equal rates: men carry A x cycle, women (A + B) x cycle, so the pooled
    log count moves by log((e^{Ac} + e^{(A+B)c}) / 2), not by A x cycle."""
    return np.log((np.exp(A * c) + np.exp((A + B) * c)) / 2.0)


def pooled_bias() -> float:
    d_i = pooled_dev(cycle(pd.Series(INTERIM_MONTHS))).mean()
    d_l = pooled_dev(cycle(pd.Series(LATER_MONTHS))).mean()
    return float(d_l - d_i)


# the pooled 22-25 fall: men fall by FALL, women by FALL + FALL_F
FALL_POOLED = float(np.log((np.exp(FALL) + np.exp(FALL + FALL_F)) / 2.0))


E, Bd, S, M = np.meshgrid(EMPS, list(LAM), ["1", "2"], MONTHS, indexing="ij")
GRID = pd.DataFrame({"employer_id": E.ravel(), "age_group": Bd.ravel(),
                     "gender": S.ravel(), "year_month": M.ravel()})


def world(with_cycle: bool) -> pd.DataFrame:
    d = GRID
    lam = (d["age_group"].map(LAM) * d["employer_id"].map(size)).to_numpy(float)
    hi = d["employer_id"].isin(HIGH).to_numpy()
    young = d["age_group"].isin(["22-25", "26-30"]).to_numpy()
    y22 = (d["age_group"] == "22-25").to_numpy()
    fem = (d["gender"] == "2").to_numpy()
    later = (d["year_month"] >= "2024-01").to_numpy()
    lam = lam * np.exp(FALL * (hi & young & later))
    lam = lam * np.exp(FALL_F * (hi & y22 & fem & later))
    if with_cycle:
        cyc = cycle(d["year_month"])
        lam = lam * np.exp(A * cyc * (hi & young))
        lam = lam * np.exp(B * cyc * (hi & y22 & fem))
    out = d.copy()
    out["n_emp"] = fx.poisson_same_noise(lam, 102)
    return out


def install(with_cycle: bool) -> pd.DataFrame:
    sx = world(with_cycle)
    fx.write_by_year(sx, mc.CACHE_DIR, "L_counts_sex")
    st = (sx.groupby(["employer_id", "year_month", "age_group"])["n_emp"]
          .sum().reset_index())
    fx.write_by_year(st, mc.CACHE_DIR, "L_counts")
    pd.DataFrame({"employer_id": EMPS,
                  "ind3": ["100" if (e * 7) % 3 == 0 else "200" for e in EMPS],
                  "source": "Ftg_2019"}).to_parquet(
        mc.CACHE_DIR / "I_industry_key.parquet", index=False)
    return sx


def hand_female(sx: pd.DataFrame, high: set) -> float:
    out = {}
    for g in ("1", "2"):
        c = sx[sx["gender"] == g]
        out[g] = fx.triple_diff(c, high, ["22-25"],
                                ["31-34", "35-40", "41-49", "50+"])
    return out["2"] - out["1"]


s82, s61, s67, s73, s78, s80, l47, l70, j47 = s102.load_modules()
for m_ in (s82, s61, s67, s73, s78, s80, l47, l70, j47):
    m_.OUT, m_.CACHE = s102.OUT, mc.CACHE_DIR
EXPO = s82.build_exposure(l47, l70, j47)["exposure"]
check("the planted tier is the top quartile",
      set(EXPO.loc[EXPO["fq"] == 4, "employer_id"]) == HIGH)
BIAS_A, BIAS_B = expected_bias(A), expected_bias(B)
BIAS_P = pooled_bias()
OCT_P = float(pooled_dev(np.array([0.5]))[0] - pooled_dev(np.array([-1.0]))[0])
check("the planted cycle biases a window contrast by A/13 (two Decembers "
      "in thirteen interim months, none net in the later window)",
      abs(BIAS_A - A / 13) < 1e-9, f"{BIAS_A:+.4f} against {A / 13:+.4f}")
print(f"  pooled 22-25: planted fall {FALL_POOLED:+.4f}, quarter-spec bias "
      f"{BIAS_P:+.4f}, October against December {OCT_P:+.4f}")

print("\n--- the stock gate stops on Table 1's numbers ---")
install(True)
counts = s102.load_counts("L_counts", s61.PANEL_YEARS, s102.COUNT_COLS)
s102.ROWS.clear(); s102.FAILURES.clear(); s102.NOTES.clear()
try:
    s102.stock(counts, EXPO, s61, s73, s78, s80, j47); stopped = False
except SystemExit:
    stopped = True
check("the stock gate STOPS against Table 1", stopped)

R = {}
for with_cycle in (False, True):
    kind = "CYCLE" if with_cycle else "NO CYCLE"
    print(f"\n=== the {kind} world ===")
    sx = install(with_cycle)
    st = (sx.groupby(["employer_id", "year_month", "age_group"])["n_emp"]
          .sum().reset_index())
    hand = fx.triple_diff(st, HIGH, ["22-25"], ["31-34", "35-40", "41-49", "50+"])
    hand_f = hand_female(sx, HIGH)
    if with_cycle:
        check("by hand (cycle): the raw window contrast carries the bias",
              abs((hand - FALL_POOLED) - BIAS_P) < 0.03,
              f"{hand:+.4f} against pooled fall {FALL_POOLED:+.4f} + {BIAS_P:+.4f}")
        check("by hand (cycle): the raw female contrast carries B/13",
              abs((hand_f - FALL_F) - BIAS_B) < 0.04,
              f"{hand_f:+.4f} against {FALL_F:+.4f} + {BIAS_B:+.4f}")
    s102.check = lambda label, got, want: []      # gates pointed at this world
    s102.ROWS.clear(); s102.FAILURES.clear(); s102.NOTES.clear()
    counts = s102.load_counts("L_counts", s61.PANEL_YEARS, s102.COUNT_COLS)
    s102.stock(counts, EXPO, s61, s73, s78, s80, j47)
    sexc = s102.load_counts("L_counts_sex", s61.PANEL_YEARS, s102.SEX_COLS)
    s102.sex(sexc, EXPO, s67, s73, s78, s80, j47)
    check(f"{kind}: no fit failed", not s102.FAILURES, "; ".join(s102.FAILURES))
    g = lambda *k: s102.get(*k)                                  # noqa: E731
    R[with_cycle] = {
        "q": g("G", "gate", "22-25", "hy_tau"),
        "m": g("M", "month_of_year", "22-25", "hy_tau"),
        "qf": g("G", "sex_gate", "22-25", "hyf_tau"),
        "mf": g("M", "sex_month_of_year", "22-25", "hyf_tau"),
        "m_ind": g("M", "month_of_year_indcl", "22-25", "hy_tau"),
        "q_ind": g("G", "gate_indcl", "22-25", "hy_tau"),
    }
    r = R[with_cycle]
    print(f"  quarter tau {r['q'][0]:+.4f} ({r['q'][1]:.4f}), month tau "
          f"{r['m'][0]:+.4f} ({r['m'][1]:.4f}); female quarter "
          f"{r['qf'][0]:+.4f}, month {r['mf'][0]:+.4f}")
    check(f"{kind}: month-of-year tau reproduces the planted pooled fall",
          abs(r["m"][0] - FALL_POOLED) < 0.04,
          f"{r['m'][0]:+.4f} against {FALL_POOLED:+.4f}")
    check(f"{kind}: month-of-year female differential reproduces FALL_F",
          abs(r["mf"][0] - FALL_F) < 0.05,
          f"{r['mf'][0]:+.4f} against {FALL_F:+.4f}")
    check(f"{kind}: the industry-clustered month fit has the same coefficient "
          "and its own SE",
          abs(r["m"][0] - r["m_ind"][0]) < 1e-9 and r["m"][1] != r["m_ind"][1],
          f"SE employer {r['m'][1]:.5f}, industry {r['m_ind'][1]:.5f}")
    if with_cycle:
        d = r["q"][0] - r["m"][0]
        check("cycle: quarter tau exceeds month tau by the planted pooled bias",
              abs(d - BIAS_P) < 0.012, f"{d:+.4f} against {BIAS_P:+.4f}")
        df = r["qf"][0] - r["mf"][0]
        check("cycle: quarter female differential exceeds the month one by B/13",
              abs(df - BIAS_B) < 0.015, f"{df:+.4f} against {BIAS_B:+.4f}")
        dec = [x for x in s102.ROWS if x["part"] == "M"
               and x["spec"] == "month_of_year" and x["term"].startswith("m")]
        check("cycle: eleven month-of-year coefficients are exported",
              len(dec) == 11, str(len(dec)))
        oct_c = g("M", "month_of_year", "22-25", "m10_x_high_x_young")[0]
        check("cycle: October against December reads the planted pooled cycle",
              abs(oct_c - OCT_P) < 0.05, f"{oct_c:+.4f} against {OCT_P:+.4f}")
    else:
        d = r["q"][0] - r["m"][0]
        check("no cycle: quarter and month taus agree",
              abs(d) < 0.012, f"{d:+.4f}")
    if with_cycle:
        # the two worlds share every uniform: the movement of the quarter
        # tau between them is the planted bias with the noise cancelled
        dq = r["q"][0] - R[False]["q"][0]
        dm = r["m"][0] - R[False]["m"][0]
        check("across worlds: the cycle moves the quarter tau by the planted "
              "bias and the month tau not at all",
              abs(dq - BIAS_P) < 0.008 and abs(dm) < 0.008,
              f"quarter {dq:+.4f}, month {dm:+.4f}, planted {BIAS_P:+.4f}")
        s102.GATE = {"post": g("G", "gate", "22-25", "hy_post"),
                     "tau": g("G", "gate", "22-25", "hy_tau")}
        s102.SEX_GATE = {"post": g("G", "sex_gate", "22-25", "hyf_post"),
                         "tau": g("G", "sex_gate", "22-25", "hyf_tau")}
        W_GATE, W_SEX = s102.GATE, s102.SEX_GATE

print("\n--- main(), end to end, in the CYCLE world ---")
install(True)
s102.check = fx.load("102_month_of_year.py", "s102_fresh").check
s102.GATE, s102.SEX_GATE = W_GATE, W_SEX
s102.ROWS.clear(); s102.FAILURES.clear(); s102.NOTES.clear()
s102.DONE = s102.PLANNED = 0
_stdout = sys.stdout
rc = s102.main()
sys.stdout = _stdout
out = pd.read_csv(s102.OUT / "month_of_year.csv")
summ = (s102.OUT / "102_summary.txt").read_text()
check("main() returns 0", rc == 0, f"rc {rc}; {s102.FAILURES}")
check("every attempted fit came back (eight)", s102.DONE == s102.PLANNED == 8,
      f"{s102.DONE} of {s102.PLANNED}")
check("the summary prints both taus with both clusterings",
      "quarter terms (Table 1)" in summ and "month-of-year terms" in summ
      and summ.count("by industry") >= 4)
check("the summary prints the differences and says their SE is unavailable",
      summ.count("difference (month minus quarter)") == 2
      and "not available from two separate covariances" in summ)
check("the summary prints the month-of-year cycle for both panels",
      "THE CYCLE THE QUARTER TERMS LEAVE" in summ and summ.count("m10 ") >= 2)
check("no employer count under the floor",
      bool(((out["n_firms"].isna()) | (out["n_firms"] >= 5)).all()))
check("no identifier column is exported",
      not any(c in out.columns for c in ("employer_id", "ind3", "cl_ind",
                                         "fe_emp_t")))
check.done()
