#!/usr/bin/env python3
"""
test_96_payroll_cohorts.py -- the dry run of script 96 (lane 37b).

THE MECHANISM PLANTED. The rival is about BIRTH COHORTS (who was covered
by the reduced contribution), the paper's claim about AGE. The fixture is
therefore built person-cohort by person-cohort: every employer holds
workers of each birth year 1952-2003 and of each sex, a worker is in the
panel in a year only while aged 22-69 (age = year minus birth year, the
panel's own definition), and the pull, 47L's L_counts and 67's
L_counts_sex are all sums of the same cohort counts, as they are on MONA.
Two worlds share every uniform draw and differ only in whom the decline
falls on:

  AI world       in top-quartile employers, workers AGED 22-30 fall by
                 FALL from January 2024 (young women by FALL_F more).
  PAYROLL world  in top-quartile employers, workers BORN 1998-2004 (ever
                 covered) fall by FALL from January 2024; age plays no
                 part.

The read rule must say "not the payroll reduction" in the first world and
"the payroll rival survives" in the second; a test that could only reject
the rival would be worthless. Each is first checked by hand as a raw
triple difference on the cohort cells.

Also: the statute's eligibility table; the cell probe refuses a pull
carrying a 22-25 band cell; the pull re-banded by age equals L_counts and
L_counts_sex exactly (and a sex-0 worker is in the first and not the
second); the generalised skeleton equals 78's on one young cell; the gate
stops on a miss; main() runs end to end with no SQL and no identifier in
the export.

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

EMPS = list(range(1, 201))
tier = lambda e: e % 4                                          # noqa: E731
size = lambda e: 4.2 if tier(e) == 3 else 1 + (e // 4) % 5     # noqa: E731
fx.install_score(mc, EMPS, tier, size, Path(mc.SHARE))
HIGH = {e for e in EMPS if tier(e) == 3}
FALL, FALL_F = float(np.log(0.85)), float(np.log(0.90))
MONTHS = fx.months()
COHORTS = list(range(1952, 2004))


def grid() -> pd.DataFrame:
    """employer x birth year x sex x month, rows only where the cohort is
    aged 22-69 that year; plus a handful of sex-0 workers."""
    e, by, g, ym = np.meshgrid(EMPS, COHORTS, ["1", "2"], MONTHS,
                               indexing="ij")
    d = pd.DataFrame({"employer_id": e.ravel(), "by": by.ravel(),
                      "gender": g.ravel(), "year_month": ym.ravel()})
    d["age"] = d["year_month"].str.slice(0, 4).astype(int) - d["by"]
    d = d[(d["age"] >= 22) & (d["age"] <= 69)].reset_index(drop=True)
    z = d[(d["employer_id"] % 13 == 0) & (d["by"] == 1970)
          & (d["gender"] == "1")].assign(gender="0")
    return pd.concat([d, z], ignore_index=True)


G = grid()


def world(kind: str) -> pd.DataFrame:
    lam = 0.9 * G["employer_id"].map(size).to_numpy()
    lam = np.where(G["gender"] == "0", 0.4, lam)
    hi = G["employer_id"].isin(HIGH).to_numpy()
    later = (G["year_month"] >= "2024-01").to_numpy()
    if kind == "ai":
        hit = hi & later & (G["age"] <= 30).to_numpy()
        fem = hit & (G["gender"] == "2").to_numpy() & (G["age"] <= 25).to_numpy()
        lam = lam * np.exp(FALL * hit) * np.exp(FALL_F * fem)
    else:
        hit = hi & later & G["by"].between(1998, 2004).to_numpy()
        lam = lam * np.exp(FALL * hit)
    d = G.copy()
    d["n_emp"] = fx.poisson_same_noise(lam, 96)
    return d


def cell_of(d: pd.DataFrame) -> pd.Series:
    band = pd.Series(pd.NA, index=d.index, dtype="object")
    for k, (lo, hi) in s96.BANDS6.items():
        band[(d["age"] >= lo) & (d["age"] <= hi)] = k
    return band.where(d["by"] < 1994, "b" + d["by"].astype(str))


def install(kind: str) -> tuple:
    d = world(kind)
    d["cell"] = cell_of(d)
    pull = (d.groupby(["employer_id", "year_month", "cell", "gender"])
            ["n_emp"].sum().reset_index())
    fx.write_by_year(pull, mc.CACHE_DIR, s96.PREFIX)
    band = pd.Series(pd.NA, index=d.index, dtype="object")
    for k, (lo, hi) in s96.BANDS6.items():
        band[(d["age"] >= lo) & (d["age"] <= hi)] = k
    d["age_group"] = band
    lc = (d.groupby(["employer_id", "year_month", "age_group"])["n_emp"]
          .sum().reset_index())
    fx.write_by_year(lc, mc.CACHE_DIR, "L_counts")
    ls = (d[d["gender"].isin(["1", "2"])]
          .groupby(["employer_id", "year_month", "age_group", "gender"])
          ["n_emp"].sum().reset_index())
    fx.write_by_year(ls, mc.CACHE_DIR, "L_counts_sex")
    return d, lc


def by_hand(d: pd.DataFrame, young_bys) -> float:
    c = d[d["by"].isin(list(young_bys)) | ((d["by"] < 1994) & (d["age"] >= 31))]
    c = c.assign(age_group=np.where(c["by"].isin(list(young_bys)), "y", "o"))
    return fx.triple_diff(c, HIGH, ["y"], ["o"])


print("\n--- the statute's eligibility table ---")
want = {1996: 0, 1997: 0, 1998: 12, 1999: 24, 2000: 27, 2001: 27, 2002: 27,
        2003: 15, 2004: 3, 2005: 0}
got = {b: s96.eligible_months(b) for b in want}
check("months eligible by birth year match the statute", got == want, str(got))

s82, s61, s67, s78, l47, l70, j47 = s96.load_modules()
for m_ in (s82, s61, s67, s78, l47, l70, j47):
    m_.OUT, m_.CACHE = s96.OUT, mc.CACHE_DIR
EXPO = s82.build_exposure(l47, l70, j47)["exposure"]
check("the planted tier is the top quartile",
      set(EXPO.loc[EXPO["fq"] == 4, "employer_id"]) == HIGH)

RESULTS = {}
for kind in ("ai", "payroll"):
    print(f"\n=== the {kind.upper()} world ===")
    D, LC = install(kind)
    ha = by_hand(D, range(1994, 1998))
    hb = by_hand(D, range(1998, 2004))
    print(f"  by hand: never covered {ha:+.4f}, ever covered {hb:+.4f}")
    if kind == "ai":
        check("by hand (AI): the never covered fall", ha < -0.08, f"{ha:+.4f}")
    else:
        check("by hand (payroll): the never covered do not fall",
              abs(ha) < 0.03, f"{ha:+.4f}")
        check("by hand (payroll): the ever covered fall",
              abs(hb - FALL) < 0.04, f"{hb:+.4f}")
    pull = s96.load_pull(s61.PANEL_YEARS)
    s96.NOTES.clear()
    six = s96.rebanded(pull, sex=False)
    s96.compare(six, "L_counts", False, s61.PANEL_YEARS)
    s96.compare(s96.rebanded(pull, sex=True), "L_counts_sex", True,
                s61.PANEL_YEARS)
    check("re-banded pull equals L_counts exactly",
          any("from L_counts in 0 of" in n for n in s96.NOTES), s96.NOTES[0])
    check("re-banded sex pull equals L_counts_sex exactly",
          any("from L_counts_sex in 0 of" in n for n in s96.NOTES),
          s96.NOTES[-1])
    if kind == "ai":
        # the generalised skeleton equals 78's on one young cell
        fr = s96.cohort_frame(pull, {"young": range(1994, 1998)}, sex=False)
        a = s78.build_skeleton_bands(fr, ["young"] + s96.REF, "young", j47,
                                     s61.PANEL_FROM)
        b = s96.skeleton_multi(fr, ["young"], j47, s61.PANEL_FROM)
        k = ["employer_id", "age_group", "year_month"]
        a = a.sort_values(k).reset_index(drop=True)
        b = b.sort_values(k).reset_index(drop=True)
        check("skeleton_multi on one young cell is 78's skeleton",
              a[k + ["n_emp", "young"]].equals(b[k + ["n_emp", "young"]]),
              f"{len(a)} and {len(b)} rows")
        # the gate: a probe run fails against Table 1 and stops
        s96.ROWS.clear(); s96.FAILURES.clear()
        try:
            s96.stock_gate(six, EXPO, s61, s78, j47); stopped = False
        except SystemExit:
            stopped = True
        check("the stock gate STOPS against Table 1's numbers", stopped)
        s96.GATE = {"post": s96.get("G", "gate_22_25", "post"),
                    "tau": s96.get("G", "gate_22_25", "tau")}
        s96.ROWS.clear(); s96.FAILURES.clear()
        try:
            s96.stock_gate(six, EXPO, s61, s78, j47); ok = True
        except SystemExit:
            ok = False
        check("and passes on the world's own", ok)
        s96.ROWS.clear(); s96.FAILURES.clear()
        s96.part_s(pull, EXPO, s67, s78, j47)
        check("the sex gate refuses Table 1's numbers and stops Part S",
              any("sex gate failed" in f for f in s96.FAILURES)
              and not any(r["spec"] == "sex_never_1994_1997"
                          for r in s96.ROWS))
        s96.SEX_GATE = {"post": s96.get("S", "sex_gate_22_25",
                                        "female_diff_post"),
                        "tau": s96.get("S", "sex_gate_22_25",
                                       "female_diff_tau")}
    s96.ROWS.clear(); s96.FAILURES.clear(); s96.NOTES.clear()
    s96.part_c(pull, EXPO, s61, s78, j47)
    s96.part_d(pull, EXPO, s61, s78, j47)
    if kind == "ai":
        s96.part_s(pull, EXPO, s67, s78, j47)
    check(f"{kind}: no fit failed", not s96.FAILURES, "; ".join(s96.FAILURES))
    RESULTS[kind] = {"rows": list(s96.ROWS), "ha": ha, "hb": hb}
    g = lambda *k: s96.get(*k)                                  # noqa: E731
    a, sa = g("C", "a_never", "tau")
    b, sb = g("C", "b_ever", "tau")
    b2, _ = g("C", "b2_ever_both_windows", "tau")
    check(f"{kind}: tau (a) matches the hand contrast", abs(a - ha) < 0.03,
          f"{a:+.4f} ({sa:.4f}) vs {ha:+.4f}")
    if kind == "ai":
        check("AI: the never covered decline distinguishably",
              a < -1.96 * sa, f"{a:+.4f} ({sa:.4f})")
        fd, sfd = g("S", "sex_never_1994_1997", "female_diff_tau")
        check("AI: the never-covered female differential is about nil "
              "(women were planted only at 22-25)", abs(fd) < 0.05,
              f"{fd:+.4f} ({sfd:.4f})")
    else:
        check("payroll: the never covered do not decline distinguishably",
              not (a < -1.96 * sa), f"{a:+.4f} ({sa:.4f})")
        check("payroll: the ever covered do", b < -1.96 * sb
              and abs(b - hb) < 0.03, f"{b:+.4f} ({sb:.4f})")
        check("payroll: the both-windows ever covered do too",
              abs(b2 - FALL) < 0.05, f"{b2:+.4f}")
        d00, _ = g("D", "class_d00", "tau")
        d24, _ = g("D", "class_d24", "tau")
        check("payroll: per class, dose 0 flat and dose 24 falls",
              abs(d00) < 0.03 and abs(d24 - FALL) < 0.05,
              f"d00 {d00:+.4f}, d24 {d24:+.4f}")
        sl, ssl = g("D", "dose_linear", "tau_per_year_eligible")
        check("payroll: the dose slope is negative", sl < -1.96 * ssl,
              f"{sl:+.4f} ({ssl:.4f}) per year")
    s96.ROWS[:] = RESULTS[kind]["rows"]
    s96.write_summary()
    summ = (s96.OUT / "96_summary.txt").read_text()
    want_v = ("NOT THE PAYROLL REDUCTION" if kind == "ai"
              else "THE PAYROLL RIVAL SURVIVES")
    check(f"{kind}: the read rule gives the planted verdict", want_v in summ)

print("\n--- the cell probe refuses a band cell for the young ---")
bad = pd.read_parquet(mc.CACHE_DIR / f"{s96.PREFIX}_2022.parquet")
keep = bad.copy()
bad.loc[bad.index[0], "cell"] = "22-25"
bad.to_parquet(mc.CACHE_DIR / f"{s96.PREFIX}_2022.parquet", index=False)
try:
    s96.load_pull([2022]); refused = False
except RuntimeError:
    refused = True
keep.to_parquet(mc.CACHE_DIR / f"{s96.PREFIX}_2022.parquet", index=False)
check("a 22-25 band cell in the pull is refused", refused)

print("\n--- main(), end to end, in the AI world ---")
install("ai")
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
check("the summary gives the verdict", "NOT THE PAYROLL REDUCTION" in summ)
check("the summary lists the cohorts' ages", "born 1997:  0 months" in summ
      and "born 2003: 15 months" in summ)
check("young person-months are reported for the cohort contrasts",
      bool((out.loc[(out["part"] == "C") & (out["term"] == "tau"),
                    "young_person_months_later"] > 0).all()))
check("no employer count under the floor",
      bool(((out["n_firms"].isna()) | (out["n_firms"] >= 5)).all()))
check("no identifier column is exported",
      not any(c in out.columns for c in ("employer_id", "cell", "fe_emp_t")))
check.done()
