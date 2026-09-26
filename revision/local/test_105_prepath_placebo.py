#!/usr/bin/env python3
"""
test_105_prepath_placebo.py -- the dry run of script 105 (lane 39a).

THE MECHANISM PLANTED. 97's employer x age band x month grid extended to
January 2019 (240 employers, four DAIOE tiers, tier 3 the top quartile),
in two worlds that share every uniform draw:

  A  young workers in top employers fall by FALL from January 2024 and
     nothing else moves: the drift from 2019 is FLAT, every placebo tau
     is near zero and reads its hand triple difference.
  B  world A plus a relative decline of the young at top employers over
     2019 and 2020: their level starts DROP log points above the rest in
     January 2019 and reaches zero in December 2020, flat thereafter. The
     drift from 2019 is negative and NOT FLAT; the S = 36 placebo, whose
     interim is 2020 and later 2021-22, reads the drop by hand; the
     S = 24 placebo (2021 against 2022) reads nearly nothing. This is the
     shape of Table A (prepath): high in 2019-20, flat in 2021-22.

Flows (world A): separations at top x young rise by RISE from January
2024, hires do not move; the industry-clustered fit reads both by hand
and the flows gate stops on a wrong reference value.

Also: shift_ym and the placebo windows by hand; the stock gate stops on
Table 1's numbers; main() end to end with no SQL, nine fits, both export
files, the summary's blocks, no identifier out, no count under the floor.

    python3 revision/local/test_105_prepath_placebo.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _fixtures_trip37 as fx  # noqa: E402

mc, TMP = fx.sandbox("105", ("CANARIES_105_OUT",))
s105 = fx.load("105_prepath_placebo.py", "s105")
s105.OUT = TMP / "out"
s105.CACHE = mc.CACHE_DIR
check = fx.Check()
SHARE = Path(mc.SHARE)

EMPS = list(range(1, 241))
tier = lambda e: e % 4                                          # noqa: E731
size = lambda e: 4.2 if tier(e) == 3 else 1 + (e // 4) % 5     # noqa: E731
fx.install_score(mc, EMPS, tier, size, SHARE)
HIGH = {e for e in EMPS if tier(e) == 3}
FALL, DROP, RISE = float(np.log(0.85)), 0.10, float(np.log(1.4))
MONTHS = [f"{y}-{m:02d}" for y in range(2019, 2026) for m in range(1, 13 if y < 2025 else 7)]
LAM = {"22-25": 8, "26-30": 9, "31-34": 7, "35-40": 8, "41-49": 10, "50+": 12}
OLD = ["31-34", "35-40", "41-49", "50+"]

E, Bd, M = np.meshgrid(EMPS, list(LAM), MONTHS, indexing="ij")
GRID = pd.DataFrame({"employer_id": E.ravel(), "age_group": Bd.ravel(), "year_month": M.ravel()})


def world(pre_decline: bool) -> pd.DataFrame:
    d = GRID
    lam = (d["age_group"].map(LAM) * d["employer_id"].map(size)).to_numpy(float)
    hi = d["employer_id"].isin(HIGH).to_numpy()
    young = (d["age_group"] == "22-25").to_numpy()
    later = (d["year_month"] >= "2024-01").to_numpy()
    lam = lam * np.exp(FALL * (hi & young & later))
    if pre_decline:
        y = d["year_month"].str.slice(0, 4).astype(int).to_numpy()
        m = d["year_month"].str.slice(5, 7).astype(int).to_numpy()
        t = (y - 2019) * 12 + (m - 1)
        level = np.where(t < 24, DROP * (1 - t / 24.0), 0.0)
        lam = lam * np.exp(level * (hi & young))
    out = d.copy()
    out["n_emp"] = fx.poisson_same_noise(lam, 105)
    return out


def flows_world(st: pd.DataFrame) -> pd.DataFrame:
    d = st[st["year_month"] >= "2021-01"].copy()
    base = 0.12 * (d["age_group"].map(LAM) * d["employer_id"].map(size)).to_numpy(float)
    hi = d["employer_id"].isin(HIGH).to_numpy()
    young = (d["age_group"] == "22-25").to_numpy()
    later = (d["year_month"] >= "2024-01").to_numpy()
    d["n_hire"] = fx.poisson_same_noise(base, 1051)
    d["n_sep"] = fx.poisson_same_noise(base * np.exp(RISE * (hi & young & later)), 1052)
    return d[["employer_id", "year_month", "age_group", "n_hire", "n_sep"]]


def write_world(st: pd.DataFrame) -> None:
    for y in range(2019, 2026):
        st[st["year_month"].str.slice(0, 4) == str(y)][
            ["employer_id", "year_month", "age_group", "n_emp"]].to_parquet(
            mc.CACHE_DIR / f"L_counts_{y}.parquet", index=False)


def install(pre_decline: bool) -> pd.DataFrame:
    st = world(pre_decline)
    write_world(st)
    fl = flows_world(st)
    for y in range(2021, 2026):
        fl[fl["year_month"].str.slice(0, 4) == str(y)].to_parquet(
            mc.CACHE_DIR / f"flows_{y}.parquet", index=False)
    pd.DataFrame({"employer_id": EMPS,
                  "ind3": ["100" if (e * 7) % 3 == 0 else "200" for e in EMPS],
                  "source": "Ftg_2019"}).to_parquet(
        mc.CACHE_DIR / "I_industry_key.parquet", index=False)
    return st


def hand_placebo(st: pd.DataFrame, S: int, young="22-25") -> float:
    """The shifted triple difference on the truncated counts."""
    c = st[st["year_month"] < "2022-12"]
    return fx.triple_diff(c, HIGH, [young], OLD,
                          post_from=s105.shift_ym("2024-01", S),
                          interim_from=s105.shift_ym("2022-12", S))


def reset() -> None:
    s105.ROWS.clear(); s105.FROWS.clear()
    s105.FAILURES.clear(); s105.NOTES.clear()
    s105.DONE = s105.PLANNED = 0


print("\n--- the arithmetic ---")
check("shift_ym moves YYYY-MM back by whole months",
      s105.shift_ym("2024-01", 24) == "2022-01" and s105.shift_ym("2022-12", 36) == "2019-12"
      and s105.shift_ym("2022-04", 24) == "2020-04" and s105.shift_ym("2021-01", 36) == "2018-01"
      and s105.shift_ym("2022-11", 1) == "2022-10")
toy = pd.DataFrame({"year_month": MONTHS[:47], "high": 1, "young": 1})
toy, terms, (rb_s, launch_s, post_s) = s105.placebo_terms(toy, 24)
check("placebo windows at S = 24: tightening 2020-04, interim 2020-12 to 2021-12, later from 2022-01",
      (rb_s, launch_s, post_s) == ("2020-04", "2020-12", "2022-01")
      and toy.loc[toy["year_month"] == "2021-06", s105.INTERIM].iloc[0] == 1
      and toy.loc[toy["year_month"] == "2022-01", s105.POST].iloc[0] == 1
      and toy.loc[toy["year_month"] == "2020-11", s105.INTERIM].iloc[0] == 0
      and toy.loc[toy["year_month"] == "2020-03", "rb_x_high_x_young"].iloc[0] == 0
      and len(terms) == 6)
toy = pd.DataFrame({"year_month": MONTHS[:47], "high": 1, "young": 1})
_, _, (rb_s, launch_s, post_s) = s105.placebo_terms(toy, 36)
check("placebo windows at S = 36: tightening 2019-04, interim 2019-12 to 2020-12, later from 2021-01",
      (rb_s, launch_s, post_s) == ("2019-04", "2019-12", "2021-01"))

s82, s61, s73, s78, s80, l47, l70, j47 = s105.load_modules()
for m_ in (s82, s61, s73, s78, s80, l47, l70, j47):
    m_.OUT, m_.CACHE = s105.OUT, mc.CACHE_DIR

R = {}
for name, pre in (("A", False), ("B", True)):
    print(f"\n=== world {name} ===")
    st = install(pre)
    reset()
    EXPO = s82.build_exposure(l47, l70, j47, audit=False)["exposure"]
    check(f"{name}: the planted tier is the top quartile",
          set(EXPO.loc[EXPO["fq"] == 4, "employer_id"]) == HIGH)
    counts = s105.load_counts("L_counts", s61.PANEL_YEARS, s105.COUNT_COLS)
    if name == "A":
        try:
            s105.gate(counts, EXPO, s61, s78, j47)
            stopped = False
        except SystemExit:
            stopped = True
        check("the stock gate STOPS against Table 1", stopped)
        # the gate's own numbers in this world, for main() at the end
        W_GATE = {"post": s105.get("G", "gate", "22-25", "hy_post"),
                  "tau": s105.get("G", "gate", "22-25", "hy_tau")}
        reset()
    s105.check = lambda *a, **k: []               # the gate is pointed at this world
    early = s105.load_counts("L_counts", [2019, 2020], s105.COUNT_COLS)
    extended = pd.concat([early, counts], ignore_index=True)
    s105.pre_period(extended, EXPO, s78, j47)
    check(f"{name}: no fit failed in the pre-period parts", not s105.FAILURES, "; ".join(s105.FAILURES))
    g = lambda *k: s105.get(*k)                                  # noqa: E731
    r = {"d19": g("D", "drift_from_2019", "22-25", s105.TREND),
         "d19_26": g("D", "drift_from_2019", "26-30", s105.TREND),
         "d20": g("D", "drift_from_2020", "22-25", s105.TREND),
         "carry": g("D", "drift_from_2019", "22-25", "trend_carried_15_5_months"),
         "p24": g("B", "placebo_shift_24", "22-25", "hy_tau"),
         "p36": g("B", "placebo_shift_36", "22-25", "hy_tau"),
         "p24_26": g("B", "placebo_shift_24", "26-30", "hy_tau"),
         "h24": hand_placebo(st, 24), "h36": hand_placebo(st, 36),
         "h24_26": hand_placebo(st, 24, "26-30")}
    R[name] = r
    print(f"  drift 2019 {r['d19'][0]:+.6f} ({r['d19'][1]:.6f}); 2020 {r['d20'][0]:+.6f}; "
          f"26-30 {r['d19_26'][0]:+.6f}; placebo S24 {r['p24'][0]:+.4f} ({r['p24'][1]:.4f}) "
          f"hand {r['h24']:+.4f}; S36 {r['p36'][0]:+.4f} ({r['p36'][1]:.4f}) hand {r['h36']:+.4f}; "
          f"26-30 S24 {r['p24_26'][0]:+.4f} hand {r['h24_26']:+.4f}")
    check(f"{name}: the carried drift is the trend times 15.5",
          abs(r["carry"][0] - r["d19"][0] * 15.5) < 1e-9 and abs(r["carry"][1] - r["d19"][1] * 15.5) < 1e-9)
    check(f"{name}: every placebo tau reads its hand triple difference (within 0.03)",
          abs(r["p24"][0] - r["h24"]) < 0.03 and abs(r["p36"][0] - r["h36"]) < 0.03
          and abs(r["p24_26"][0] - r["h24_26"]) < 0.03)
    if name == "A":
        # A 2-SE rule on three noisy tests has about a one-in-seven chance of
        # a false NOT FLAT in a world with no drift, so the mechanism check is
        # world B against world A, and world A is only required to be small.
        check("A: no drift trend exceeds 3 SE at either band or start (nothing planted)",
              all(abs(c) < 3 * s for c, s in (r["d19"], r["d19_26"], r["d20"])),
              "; ".join(f"t {c / s:+.2f}" for c, s in (r["d19"], r["d19_26"], r["d20"])))
        check("A: every placebo tau is near zero (under 2.5 SE and under 0.4 of the planted fall)",
              all(abs(c) < 2.5 * s and abs(c) < 0.4 * abs(FALL) for c, s in (r["p24"], r["p36"], r["p24_26"])))
        # the flows, on world A
        flows = s105.load_counts("flows", s61.PANEL_YEARS, s105.FLOW_COLS)
        s105.flows_industry(flows, EXPO, s61, s73, s78, s80, j47)
        check("A: no flow fit failed", not s105.FAILURES, "; ".join(s105.FAILURES))
        fs = flows.rename(columns={"n_sep": "n_emp"})
        fh = flows.rename(columns={"n_hire": "n_emp"})
        hs, hh = fx.triple_diff(fs, HIGH, ["22-25"], OLD), fx.triple_diff(fh, HIGH, ["22-25"], OLD)
        ts = s105.get("F", "seps_indcl", "22-25", "hy_tau", rows=s105.FROWS)
        th = s105.get("F", "hires_indcl", "22-25", "hy_tau", rows=s105.FROWS)
        print(f"  flows: seps tau {ts[0]:+.4f} ({ts[1]:.4f}) hand {hs:+.4f}; hires {th[0]:+.4f} ({th[1]:.4f}) hand {hh:+.4f}")
        check("A: the industry-clustered separations tau reads the planted rise by hand (within 0.05)",
              abs(ts[0] - hs) < 0.05 and ts[0] > 0.5 * RISE)
        check("A: hires read their hand value and are near zero",
              abs(th[0] - hh) < 0.05 and abs(th[0]) < 0.4 * RISE)
        check("A: the flows rows carry the outcome and the cluster is not exported",
              set(pd.DataFrame(s105.FROWS)["outcome"]) == {"hires", "seps"}
              and "cl_ind" not in pd.DataFrame(s105.FROWS).columns)
        W_FLOW = {"hires": (th[0], th[1]), "seps": (ts[0], ts[1])}
        st_A = st
    else:
        check("B: the drift from 2019 at 22-25 is negative and NOT FLAT, and far beyond world A's",
              r["d19"][0] < 0 and abs(r["d19"][0]) > 2 * r["d19"][1]
              and abs(r["d19"][0]) > 3 * abs(R["A"]["d19"][0]), f"t {r['d19'][0] / r['d19'][1]:+.2f}")
        check("B: the drift from 2019 at 26-30 stays FLAT (nothing planted there)",
              abs(r["d19_26"][0]) < 2 * r["d19_26"][1])
        check("B: the S = 36 placebo reads the 2020-to-2021 drop (below -0.015)", r["p36"][0] < -0.015,
              f"{r['p36'][0]:+.4f}")
        check("B: the S = 24 placebo reads far less than S = 36 (2021 against 2022 is nearly flat)",
              abs(r["p24"][0]) < 0.5 * abs(r["p36"][0]), f"{r['p24'][0]:+.4f} vs {r['p36'][0]:+.4f}")
        check("B: world A and world B share their noise: the 26-30 placebo agrees to 0.005",
              abs(r["p24_26"][0] - R["A"]["p24_26"][0]) < 0.005)

print("\n--- the flows gate stops on a wrong reference ---")
install(False)
reset()
s105.check = fx.load("105_prepath_placebo.py", "s105_fresh").check
s105.FLOW_GATE = {"hires": (+0.5, 0.03), "seps": (+0.5, 0.01)}
flows = s105.load_counts("flows", s61.PANEL_YEARS, s105.FLOW_COLS)
try:
    s105.flows_industry(flows, EXPO, s61, s73, s78, s80, j47)
    stopped = False
except SystemExit:
    stopped = True
check("the flows gate STOPS when the coefficient does not reproduce the reference", stopped)

print("\n--- main(), end to end, in world A ---")
reset()
s105.GATE, s105.FLOW_GATE = W_GATE, W_FLOW
_stdout = sys.stdout
rc = s105.main()
sys.stdout = _stdout
out = pd.read_csv(s105.OUT / "prepath_placebo.csv")
fl = pd.read_csv(s105.OUT / "flows_industry.csv")
summ = (s105.OUT / "105_summary.txt").read_text()
check("main() returns 0", rc == 0, f"rc {rc}; {s105.FAILURES}")
check("every attempted fit came back (nine)", s105.DONE == s105.PLANNED == 9,
      f"{s105.DONE} of {s105.PLANNED}")
check("the summary prints the gate, D, B and F blocks with their read rules",
      "stock 22-25: tau" in summ and "D. THE DRIFT TEST" in summ
      and "B. THE BACKDATED PLACEBO" in summ and "F. THE FLOWS" in summ
      and summ.count(": placebo tau ") == 3 and summ.count("carried over 15.5 months") == 3
      and summ.count("windows: reference") == 3 and "NO FIT" not in summ)
check("the export holds D, B and G rows and the flows file both outcomes",
      set(out["part"]) == {"G", "D", "B"} and set(fl["outcome"]) == {"hires", "seps"}
      and (out["spec"] == "placebo_shift_36").sum() > 0)
check("no employer count under the floor",
      bool(((out["n_firms"].isna()) | (out["n_firms"] >= 5)).all())
      and bool(((fl["n_firms"].isna()) | (fl["n_firms"] >= 5)).all()))
check("no identifier column is exported",
      not any(c in out.columns or c in fl.columns
              for c in ("employer_id", "ind3", "cl_ind", "fe_emp_t")))
check.done()
