#!/usr/bin/env python3
"""
test_95_pension_reference.py -- the dry run of script 95 (lane 37a).

THE MECHANISM PLANTED. The review's worry is that a pension-driven rise
at 60 and over, in exposed employers, makes the young look worse against
a reference that is itself moving. So the world plants exactly that, and
nothing at 50-59:

  young (22-25 and 26-30) in top-quartile employers: x exp(RB) from April
      2022 and x exp(FALL) from January 2024 (the adoption step; the
      interim window is untouched, so tau = FALL);
  60-64 and 65-69 in top-quartile employers: x exp(OLD) from January 2024;
  50-59 and everything else: nothing.

The estimator must then find: tau against 31-69 MORE negative than FALL
(the reference moved), tau against 31-59 and 31-49 near FALL, and in the
eight-band profile a tau of about zero at 50-59 and about OLD at 60-64 and
65-69. Each is checked against a hand-computed triple difference first,
so a failure can be placed in the fixture or in the code. Some employers
hold no worker aged 31-49, so the common sample E49 is a strict subset of
the headline panel.

Also: the data check counts zero differences when L_counts is the
collapse of the pull and a positive number when it is not; the Table 1
gate stops the script on a 0.0006 miss; Part P stops before the eight-band
fits when the seven-band profile does not reproduce its target; and main()
runs end to end, issues no SQL, and exports no identifier.

    python3 revision/local/test_95_pension_reference.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _fixtures_trip37 as fx  # noqa: E402

mc, TMP = fx.sandbox("95", ("CANARIES_95_OUT",))
s95 = fx.load("95_pension_reference.py", "s95")
s95.OUT = TMP / "out"
s95.CACHE = mc.CACHE_DIR
check = fx.Check()

EMPS = list(range(1, 321))
tier = lambda e: e % 4                                          # noqa: E731
# Tier 3 has ONE size, so its employers share one firm score to the last
# bit: the quartile cut falls inside tier 3 (it holds about a third of
# incumbent employment) and float noise in a tied score would otherwise
# split the tier across Q3 and Q4. Real scores are not tied.
size = lambda e: 4.2 if tier(e) == 3 else 1 + (e // 4) % 5    # noqa: E731
NO_MID = {e for e in EMPS if e % 7 == 0}      # no worker aged 31-49
fx.install_score(mc, EMPS, tier, size, Path(mc.SHARE))

FALL, OLD, RB = float(np.log(0.85)), float(np.log(1.20)), float(np.log(1.05))
LAM = {"22-25": 8, "26-30": 9, "31-34": 7, "35-40": 8, "41-49": 10,
       "50-59": 9, "60-64": 6, "65-69": 4}
MONTHS = fx.months()


def world() -> pd.DataFrame:
    rows = []
    for e in EMPS:
        hi = tier(e) == 3
        for band, lam in LAM.items():
            if e in NO_MID and band in ("31-34", "35-40", "41-49"):
                continue
            for ym in MONTHS:
                x = lam * size(e)
                if hi and band in ("22-25", "26-30"):
                    if ym >= "2022-04":
                        x *= np.exp(RB)
                    if ym >= "2024-01":
                        x *= np.exp(FALL)
                if hi and band in ("60-64", "65-69") and ym >= "2024-01":
                    x *= np.exp(OLD)
                rows.append((e, ym, band, x))
    d = pd.DataFrame(rows, columns=["employer_id", "year_month", "age_group",
                                    "lam"])
    d["n_emp"] = fx.poisson_same_noise(d["lam"].to_numpy(), 95)
    return d.drop(columns="lam")


AGE8 = world()
fx.write_by_year(AGE8, mc.CACHE_DIR, "L_counts_age8")
SIX = s95.relabel(AGE8, s95.TO_SIX)
fx.write_by_year(SIX, mc.CACHE_DIR, "L_counts")
HIGH = {e for e in EMPS if tier(e) == 3}

s82, s61, s78, l47, l70, j47 = s95.load_modules()
for m_ in (s82, s61, s78, l47, l70, j47):
    m_.OUT, m_.CACHE = s95.OUT, mc.CACHE_DIR
built = s82.build_exposure(l47, l70, j47)
EXPO = built["exposure"]
check("the planted tier is the top quartile",
      set(EXPO.loc[EXPO["fq"] == 4, "employer_id"]) == HIGH,
      f"{(EXPO['fq'] == 4).sum()} employers in Q4")

print("\n--- by hand: the fixture carries the mechanism ---")
old69 = ["31-34", "35-40", "41-49", "50-59", "60-64", "65-69"]
h69 = fx.triple_diff(AGE8, HIGH, ["22-25"], old69)
h59 = fx.triple_diff(AGE8, HIGH, ["22-25"], old69[:4])
h49 = fx.triple_diff(AGE8, HIGH, ["22-25"], old69[:3])
check("by hand, against 31-49 the contrast is the planted fall",
      abs(h49 - FALL) < 0.03, f"{h49:+.4f} against {FALL:+.4f}")
check("by hand, against 31-69 it is more negative (the reference moved)",
      h69 < h49 - 0.02, f"{h69:+.4f} against {h49:+.4f}")

print("\n--- the data check ---")
s95.NOTES.clear()
s95.compare_with_lcounts(SIX, s82, s61)
check("zero differences when L_counts is the collapse",
      any("differs from 47L's L_counts in 0 of" in n for n in s95.NOTES),
      s95.NOTES[-1] if s95.NOTES else "")
bad = SIX[SIX["year_month"].str.startswith("2023")].copy()
bad.loc[bad.index[:3], "n_emp"] += 1
bad.to_parquet(mc.CACHE_DIR / "L_counts_2023.parquet", index=False)
s95.NOTES.clear()
s95.compare_with_lcounts(SIX, s82, s61)
check("a perturbed L_counts is reported",
      any("in 3 of" in n for n in s95.NOTES), s95.NOTES[-1])
fx.write_by_year(SIX, mc.CACHE_DIR, "L_counts")

print("\n--- the gate and Part R, run through the script's own functions ---")
keep_gate, keep_parts = s95.check_gate, s95.PARTS
s95.check_gate = lambda band: None
s95.ROWS.clear(); s95.FAILURES.clear(); s95.NOTES.clear()
s95.PARTS = "R"
s95.gate_and_r(SIX, AGE8, EXPO, s61, s78, j47)
s95.check_gate = keep_gate
g = lambda *k: s95.get(*k)                                      # noqa: E731
t69 = g("G", "gate", "full", "22-25", "31-69", "tau")
t59 = g("R", "ref_31-59", "own", "22-25", "31-59", "tau")
t49 = g("R", "ref_31-49", "own", "22-25", "31-49", "tau")
e69 = g("R", "ref_31-69", "E49", "22-25", "31-69", "tau")
e59 = g("R", "ref_31-59", "E49", "22-25", "31-59", "tau")
e49 = g("R", "ref_31-49", "E49", "22-25", "31-49", "tau")
check("no fit failed", not s95.FAILURES, "; ".join(s95.FAILURES))
check("tau against 31-69 matches the hand contrast",
      abs(t69[0] - h69) < 0.03, f"{t69[0]:+.4f} ({t69[1]:.4f}) vs {h69:+.4f}")
check("tau against 31-49 recovers the planted fall",
      abs(t49[0] - FALL) < 0.03 and t49[0] < -1.96 * t49[1],
      f"{t49[0]:+.4f} ({t49[1]:.4f})")
check("tau against 31-59 recovers it too (50-59 unmoved)",
      abs(t59[0] - FALL) < 0.03, f"{t59[0]:+.4f}")
check("on E49 the three references differ in the reference alone",
      all(x[0] == x[0] for x in (e69, e59, e49)) and e69[0] < e49[0] - 0.02,
      f"31-69 {e69[0]:+.4f}, 31-59 {e59[0]:+.4f}, 31-49 {e49[0]:+.4f}")
n69 = [r["n_firms"] for r in s95.ROWS if r["sample"] == "full"
       and r["young_band"] == "22-25"][0]
n49 = [r["n_firms"] for r in s95.ROWS if r["sample"] == "E49"
       and r["young_band"] == "22-25"]
check("E49 is a strict subset of the headline panel",
      len(set(n49)) == 1 and n49[0] < n69, f"{n49[0]} of {n69}")
tr = [r for r in s95.ROWS if r["term"] == "tau"][0]
check("tau rows carry the covariance pieces and the SE follows from them",
      abs(np.sqrt(tr["var_post"] + tr["var_interim"]
                  - 2 * tr["cov_post_interim"]) - tr["se"]) < 1e-12)
GATE_WORLD = {b: {"post": g("G", "gate", "full", b, "31-69", "post"),
                  "tau": g("G", "gate", "full", b, "31-69", "tau")}
              for b in s95.BANDS}

print("\n--- the gate can fail as well as pass ---")
s95.GATE = {b: dict(v) for b, v in GATE_WORLD.items()}
try:
    s95.check_gate("22-25"); ok = True
except SystemExit:
    ok = False
check("the gate passes on the world's own numbers", ok)
s95.GATE["22-25"]["tau"] = (GATE_WORLD["22-25"]["tau"][0] + 0.0006,
                            GATE_WORLD["22-25"]["tau"][1])
try:
    s95.check_gate("22-25"); stopped = False
except SystemExit:
    stopped = True
check("the gate STOPS on a 0.0006 miss", stopped)
s95.GATE = {b: dict(v) for b, v in GATE_WORLD.items()}

print("\n--- Part P: refuses the eight bands when seven do not reproduce ---")
s95.ROWS.clear(); s95.FAILURES.clear()
s95.part_p(AGE8, EXPO, s61, s78, j47)
check("a wrong split-at-65 target stops Part P before the eight bands",
      any("does NOT reproduce" in f for f in s95.FAILURES)
      and not any(r["spec"].startswith("p8") for r in s95.ROWS))
p7 = {r["young_band"]: (r["coef"], r["se"]) for r in s95.ROWS
      if r["spec"] == "p7_gamma" and r["term"] == "gamma2_style"}
n7 = [r["n_firms"] for r in s95.ROWS if r["spec"] == "p7_gamma"][0]
s95.SPLIT65, s95.SPLIT65_FIRMS = p7, n7
s95.ROWS.clear(); s95.FAILURES.clear(); s95.NOTES.clear()
s95.part_p(AGE8, EXPO, s61, s78, j47)
pt = lambda b: g("P", "p8_tau", "profile", b, "41-49", "tau")  # noqa: E731
check("Part P runs on one sample", not s95.FAILURES
      and any("same" in n for n in s95.NOTES), "; ".join(s95.FAILURES))
check("50-59 tau is about zero", abs(pt("50-59")[0]) < 0.03,
      f"{pt('50-59')[0]:+.4f} ({pt('50-59')[1]:.4f})")
for b in ("60-64", "65-69"):
    check(f"{b} tau recovers the planted rise", abs(pt(b)[0] - OLD) < 0.04
          and pt(b)[0] > 1.96 * pt(b)[1], f"{pt(b)[0]:+.4f} ({pt(b)[1]:.4f})")
check("22-25 tau against 41-49 recovers the fall",
      abs(pt("22-25")[0] - FALL) < 0.03, f"{pt('22-25')[0]:+.4f}")
g8 = g("P", "p8_gamma", "profile", "60-64", "41-49", "gamma2_style")
check("the gamma_2-style eight-band row exists", g8[0] == g8[0],
      f"{g8[0]:+.4f}")

print("\n--- main(), end to end ---")
s95.ROWS.clear(); s95.FAILURES.clear(); s95.NOTES.clear()
s95.PARTS = "RP"
s95.DONE = s95.PLANNED = 0
_stdout = sys.stdout
rc = s95.main()
sys.stdout = _stdout
out = pd.read_csv(s95.OUT / "pension_reference.csv")
summ = (s95.OUT / "95_summary.txt").read_text()
check("main() returns 0", rc == 0, f"rc {rc}; {s95.FAILURES}")
check("every attempted fit came back", s95.DONE == s95.PLANNED,
      f"{s95.DONE} of {s95.PLANNED}")
check("the summary states both verdicts", "R1:" in summ and "P1:" in summ)
check("the summary reads R1 as passed in this world",
      "DOES NOT REST ON THE PENSION AGES" in summ)
check("the summary reads P1 as NOT met (the gain is only at 60+)",
      "P1 NOT MET" in summ)
check("no employer count under the floor",
      bool(((out["n_firms"].isna()) | (out["n_firms"] >= 5)).all()))
check("no identifier column is exported",
      not any(c in out.columns for c in ("employer_id", "ssyk4", "fe_emp_t")))
check("the log exists", (s95.OUT / "95_log.txt").exists())
check.done()
