#!/usr/bin/env python3
"""
test_96_payroll_cohorts.py -- the dry run of script 96 (lane 37c, second
                              stage): the payroll reduction on FIXED,
                              DISJOINT birth cohorts.

THE MECHANISM THE FIXTURE MUST REPRODUCE: cohort turnover across age
bands. Every employer holds workers of each birth year 1950-2005, and the
cohorts differ in size (later cohorts larger), so an AGE-BAND group
changes composition every January as one cohort ages in and another ages
out, while a group defined by birth year does not. The first checks show
exactly that on the fixture's own national totals: the paper's 31-69 band
jumps from December to January, the fixed reference born 1956-1990 does
not. That is why the script builds its groups from birth years with no
age filter.

TWO WORLDS sharing every uniform:
  AI        in top-quartile employers, workers AGED 22-30 fall by FALL
            from January 2024 (age, not cohort).
  PAYROLL   in top-quartile employers, workers BORN 1998-2004 (covered)
            fall by FALL from January 2024 (cohort, not age).
The read rule must say "the pattern extends to cohorts the reduction never
covered" in the first and "the payroll rival is not excluded" in the
second. The per-class dose gradients must be flat for the never covered
and at FALL for every covered class in the payroll world (all classes,
2003 included, are in the panel throughout because there is no age
filter). The gate stops on Table 1's numbers; main() runs end to end with
no SQL and no identifier in the export.

    python3 revision/local/test_96_payroll_cohorts.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _fixtures_trip37 as fx  # noqa: E402

mc, TMP = fx.sandbox("96", ("CANARIES_96_OUT",))
s96 = fx.load("96_payroll_cohorts.py", "s96")
s96.OUT = TMP / "out"
s96.CACHE = mc.CACHE_DIR
check = fx.Check()

EMPS = list(range(1, 161))
tier = lambda e: e % 4                                          # noqa: E731
size = lambda e: 4.2 if tier(e) == 3 else 1 + (e // 4) % 5     # noqa: E731
fx.install_score(mc, EMPS, tier, size, Path(mc.SHARE))
HIGH = {e for e in EMPS if tier(e) == 3}
FALL = float(np.log(0.85))
MONTHS = fx.months()
COHORTS = list(range(1950, 2006))

E, BY, M = np.meshgrid(EMPS, COHORTS, MONTHS, indexing="ij")
G = pd.DataFrame({"employer_id": E.ravel(), "by": BY.ravel(),
                  "year_month": M.ravel()})
G["age"] = G["year_month"].str.slice(0, 4).astype(int) - G["by"]
G = G[G["age"].between(16, 75)].reset_index(drop=True)


def world(kind: str) -> pd.DataFrame:
    # later cohorts larger: the turnover an age band inherits
    lam = (0.5 + 0.03 * (G["by"] - 1950)) * G["employer_id"].map(size)
    hi = G["employer_id"].isin(HIGH).to_numpy()
    later = (G["year_month"] >= "2024-01").to_numpy()
    if kind == "ai":
        hit = hi & later & G["age"].between(22, 30).to_numpy()
    else:
        hit = hi & later & G["by"].between(1998, 2004).to_numpy()
    d = G.copy()
    d["n_emp"] = fx.poisson_same_noise(lam.to_numpy() * np.exp(FALL * hit), 961)
    return d


def cell(by: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=by.index, dtype="object")
    for k, (lo, hi) in s96.CELLS.items():
        out[(by >= lo) & (by <= hi)] = k
    return out


def install(kind: str) -> pd.DataFrame:
    d = world(kind)
    p = d.assign(cell=cell(d["by"]), gender="1")
    p = p[p["cell"].notna()]
    pull = (p.groupby(["employer_id", "year_month", "cell", "gender"])
            ["n_emp"].sum().reset_index())
    fx.write_by_year(pull, mc.CACHE_DIR, s96.PREFIX)
    band = pd.Series(pd.NA, index=d.index, dtype="object")
    for k, (lo, hi) in {"22-25": (22, 25), "26-30": (26, 30),
                        "31-34": (31, 34), "35-40": (35, 40),
                        "41-49": (41, 49), "50+": (50, 69)}.items():
        band[d["age"].between(lo, hi)] = k
    lc = (d.assign(age_group=band).dropna(subset=["age_group"])
          .groupby(["employer_id", "year_month", "age_group"])["n_emp"]
          .sum().reset_index())
    fx.write_by_year(lc, mc.CACHE_DIR, "L_counts")
    return d


print("\n--- the mechanism: cohort turnover moves an age band, not a cohort ---")
D0 = world("ai")
nat = D0[D0["year_month"].isin(["2023-12", "2024-01"])]
band31 = nat[nat["age"].between(31, 69)].groupby("year_month")["n_emp"].sum()
fixed = nat[nat["by"].between(1956, 1990)].groupby("year_month")["n_emp"].sum()
jb = band31["2024-01"] / band31["2023-12"] - 1
jf = fixed["2024-01"] / fixed["2023-12"] - 1
check("the 31-69 age band jumps in January (a cohort ages in, one out)",
      abs(jb) > 0.005, f"{jb:+.2%}")
check("the fixed reference born 1956-1990 does not", abs(jf) < 0.003,
      f"{jf:+.2%}")
nc_ages = sorted({2021 - b for b in range(1994, 1998)} |
                 {2025 - b for b in range(1994, 1998)})
check("the never-covered cohorts span 24-27 in 2021 and 28-31 in 2025",
      nc_ages == [24, 25, 26, 27, 28, 29, 30, 31], str(nc_ages))

print("\n--- the statute ---")
want = {1997: 0, 1998: 12, 1999: 24, 2000: 27, 2002: 27, 2003: 15, 2004: 3}
check("statutory months by birth year",
      {b: s96.eligible_months(b) for b in want} == want)

s82, s61, s78, l47, l70, j47 = s96.load_modules()
for m_ in (s82, s61, s78, l47, l70, j47):
    m_.OUT, m_.CACHE = s96.OUT, mc.CACHE_DIR
EXPO = s82.build_exposure(l47, l70, j47)["exposure"]


def hand(d: pd.DataFrame, young_bys) -> float:
    c = d[d["by"].isin(list(young_bys)) | d["by"].between(1956, 1990)]
    c = c.assign(age_group=np.where(c["by"].isin(list(young_bys)), "y", "o"))
    return fx.triple_diff(c, HIGH, ["y"], ["o"])


GATE_WORLD = None
for kind in ("ai", "payroll"):
    print(f"\n=== the {kind.upper()} world ===")
    D = install(kind)
    hn, he = hand(D, range(1994, 1998)), hand(D, range(1998, 2004))
    print(f"  by hand: never covered {hn:+.4f}, ever covered {he:+.4f}")
    if kind == "payroll":
        check("payroll by hand: never covered flat", abs(hn) < 0.03)
        check("payroll by hand: ever covered fall", abs(he - FALL) < 0.03)
    s96.ROWS.clear(); s96.FAILURES.clear(); s96.NOTES.clear()
    if GATE_WORLD is None:
        try:
            s96.stock_gate(EXPO, s61, s78, j47); stopped = False
        except SystemExit:
            stopped = True
        check("the gate STOPS against Table 1's numbers", stopped)
        GATE_WORLD = {"post": s96.get("G", "gate_22_25", "post"),
                      "tau": s96.get("G", "gate_22_25", "tau")}
    s96.GATE = GATE_WORLD if kind == "ai" else s96.GATE
    s96.ROWS.clear(); s96.FAILURES.clear(); s96.NOTES.clear()
    pull = s96.load_pull(s61.PANEL_YEARS)
    s96.part_c(pull, EXPO, s61, s78, j47)
    s96.part_d(pull, EXPO, s61, s78, j47)
    check(f"{kind}: no fit failed", not s96.FAILURES, "; ".join(s96.FAILURES))
    a, sa = s96.get("C", "gradient_nc", "gradient")
    b, sb = s96.get("C", "gradient_ec", "gradient")
    d_, sd = s96.get("C", "difference", "ec_minus_nc")
    check(f"{kind}: the never-covered gradient matches the hand contrast",
          abs(a - hn) < 0.03, f"{a:+.4f} vs {hn:+.4f}")
    check(f"{kind}: the difference is ec - nc with its own SE",
          abs(d_ - (b - a)) < 1e-9 and sd == sd and sd > 0,
          f"{d_:+.4f} ({sd:.4f})")
    s96.write_summary()
    summ = (s96.OUT / "96_summary.txt").read_text()
    if kind == "ai":
        check("AI: never covered decline distinguishably", a < -1.96 * sa,
              f"{a:+.4f} ({sa:.4f})")
        check("AI: read rule says the pattern extends to the never covered",
              "EXTENDS TO COHORTS THE REDUCTION NEVER COVERED" in summ)
    else:
        check("payroll: never covered not distinguishably negative",
              not (a < -1.96 * sa), f"{a:+.4f} ({sa:.4f})")
        check("payroll: ever covered decline", b < -1.96 * sb
              and abs(b - he) < 0.03, f"{b:+.4f}")
        check("payroll: read rule says the rival is not excluded",
              "THE PAYROLL RIVAL IS NOT EXCLUDED" in summ)
        for k in ("e1998", "e1999", "e2000_02", "e2003"):
            c, s = s96.get("D", f"class_{k}", "gradient")
            check(f"payroll: class {k} falls by the planted amount",
                  abs(c - FALL) < 0.06, f"{c:+.4f} ({s:.4f})")
        c0, _ = s96.get("D", "class_nc", "gradient")
        check("payroll: the never-covered class is flat", abs(c0) < 0.03,
              f"{c0:+.4f}")

print("\n--- main(), end to end, AI world ---")
install("ai")
s96.GATE = GATE_WORLD
s96.ROWS.clear(); s96.FAILURES.clear(); s96.NOTES.clear()
s96.DONE = s96.PLANNED = 0
_stdout = sys.stdout
rc = s96.main()
sys.stdout = _stdout
out = pd.read_csv(s96.OUT / "payroll_cohorts.csv")
summ = (s96.OUT / "96_summary.txt").read_text()
check("main() returns 0", rc == 0, f"rc {rc}; {s96.FAILURES}")
check("every attempted fit came back", s96.DONE == s96.PLANNED,
      f"{s96.DONE} of {s96.PLANNED}")
check("the groups' national paths are exported",
      set(out.loc[out["part"] == "paths", "spec"]) == set(s96.CELLS))
check("the labels say gradient, not tau",
      "COHORT-SPECIFIC EXPOSURE GRADIENTS" in summ
      and not (out.loc[out["part"] == "C", "term"] == "tau").any())
check("no identifier column is exported",
      not any(c in out.columns for c in ("employer_id", "cell", "fe_emp_t")))
check.done()
