#!/usr/bin/env python3
"""
test_103_eloundou_classification.py -- the dry run of script 103 (lane 38d).

THE MECHANISM PLANTED. On 97's employer x age band x sex x month grid
(240 employers, four DAIOE tiers, tier 3 the top quartile), young workers
in DAIOE-top employers fall by FALL from January 2024 and young women
there by FALL_F more, as in the 97 and 102 tests. The Eloundou rating is
a fixture file with one score per tier code, in two worlds:

  SAME     the Eloundou order equals the DAIOE order, so the two
           classifications coincide employer for employer: the agreement
           statistics must be 1.0 and the Eloundou fits must return the
           DAIOE fits to the last decimal (same employers, same high).
  SHIFTED  tier 2 outranks tier 3 on the Eloundou rating, so Eloundou's
           top quartile (a quarter of incumbent EMPLOYMENT, cut at the
           75th percentile of the weighted distribution) takes tiers 2
           and 3 together (120 employers) while DAIOE's takes tier 3 (60).
           By hand: 75 per cent of employers and 77.3 per cent of
           incumbent employment sit in the same quartile; every DAIOE-top
           employer is Eloundou-top and half of Eloundou's top is
           DAIOE-top (Jaccard 0.5). The planted fall sits at tier 3 only,
           so the Eloundou classification reads an ATTENUATED tau: the
           raw triple difference on its own top set, computed by hand,
           and well inside (0.3, 0.85) of the planted pooled fall.

Also: the gate stops on Table 1's numbers; the industry-clustered
Eloundou fit has the same coefficient and its own SE; the common-sample
DAIOE fit equals the gate when every employer is scored on both; main()
runs end to end with no SQL, ten fits, both export files, the summary's
lines, no employer count under the floor, no identifier out.

    python3 revision/local/test_103_eloundou_classification.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _fixtures_trip37 as fx  # noqa: E402

mc, TMP = fx.sandbox("103", ("CANARIES_103_OUT",))
s103 = fx.load("103_eloundou_classification.py", "s103")
s103.OUT = TMP / "out"
s103.CACHE = mc.CACHE_DIR
check = fx.Check()
SHARE = Path(mc.SHARE)

EMPS = list(range(1, 241))
tier = lambda e: e % 4                                          # noqa: E731
size = lambda e: 4.2 if tier(e) == 3 else 1 + (e // 4) % 5     # noqa: E731
fx.install_score(mc, EMPS, tier, size, SHARE)
CODE = fx.tier_codes(SHARE)                       # tier -> real four-digit code
HIGH_D = {e for e in EMPS if tier(e) == 3}
HIGH_E_SHIFTED = {e for e in EMPS if tier(e) in (2, 3)}
FALL, FALL_F = float(np.log(0.85)), float(np.log(0.88))
MONTHS = fx.months()
LAM = {"22-25": 8, "26-30": 9, "31-34": 7, "35-40": 8, "41-49": 10, "50+": 12}
OLD = ["31-34", "35-40", "41-49", "50+"]
FALL_POOLED = float(np.log((np.exp(FALL) + np.exp(FALL + FALL_F)) / 2.0))


def write_eloundou(order: dict) -> None:
    """The fixture rating: one row per tier code, scores by `order`
    (tier -> score), the real file's columns and dtypes. Codes outside the
    four tiers are dropped, so every scored incumbent is a tier code and
    an employer is scored on Eloundou exactly when it is scored on DAIOE."""
    rows = []
    for t, code in CODE.items():
        s = order[t]
        rows.append({"ssyk4": int(code), "eloundou_score": float(s),
                     "eloundou_quartile": int(sorted(order.values()).index(s) + 1),
                     "high_exposure_eloundou": int(s == max(order.values()))})
    d = pd.DataFrame(rows)
    # 82's loader asserts a few hundred codes: pad with unused real codes
    # carrying the lowest score, none of which any fixture incumbent holds
    real = pd.read_stata(fx.UPLOAD / "eloundou_ssyk4.dta")
    real["ssyk4"] = real["ssyk4"].astype(int)
    pad = real[~real["ssyk4"].isin(d["ssyk4"])].copy()
    pad["eloundou_score"] = 0.0
    pad["eloundou_quartile"] = 1
    pad["high_exposure_eloundou"] = 0
    out = pd.concat([d, pad[d.columns]], ignore_index=True)
    out.to_stata(SHARE / "eloundou_ssyk4.dta", write_index=False)


SAME = {0: 0.10, 1: 0.30, 2: 0.55, 3: 0.80}
SHIFTED = {0: 0.10, 1: 0.30, 2: 0.80, 3: 0.55}

E, Bd, S, M = np.meshgrid(EMPS, list(LAM), ["1", "2"], MONTHS, indexing="ij")
GRID = pd.DataFrame({"employer_id": E.ravel(), "age_group": Bd.ravel(),
                     "gender": S.ravel(), "year_month": M.ravel()})


def world() -> pd.DataFrame:
    d = GRID
    lam = (d["age_group"].map(LAM) * d["employer_id"].map(size)).to_numpy(float)
    hi = d["employer_id"].isin(HIGH_D).to_numpy()
    young = d["age_group"].isin(["22-25", "26-30"]).to_numpy()
    y22 = (d["age_group"] == "22-25").to_numpy()
    fem = (d["gender"] == "2").to_numpy()
    later = (d["year_month"] >= "2024-01").to_numpy()
    lam = lam * np.exp(FALL * (hi & young & later))
    lam = lam * np.exp(FALL_F * (hi & y22 & fem & later))
    out = d.copy()
    out["n_emp"] = fx.poisson_same_noise(lam, 103)
    return out


def install() -> pd.DataFrame:
    sx = world()
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
        out[g] = fx.triple_diff(c, high, ["22-25"], OLD)
    return out["2"] - out["1"]


def stat(pop: str, key: str) -> float:
    for r in s103.AGREE:
        if r["population"] == pop and r["statistic"] == key:
            return r["value"]
    return float("nan")


def reset() -> None:
    s103.ROWS.clear(); s103.AGREE.clear()
    s103.FAILURES.clear(); s103.NOTES.clear()
    s103.DONE = s103.PLANNED = 0


s82, s61, s67, s73, s78, s80, l47, l70, j47 = s103.load_modules()
for m_ in (s82, s61, s67, s73, s78, s80, l47, l70, j47):
    m_.OUT, m_.CACHE = s103.OUT, mc.CACHE_DIR
sx = install()
st = (sx.groupby(["employer_id", "year_month", "age_group"])["n_emp"]
      .sum().reset_index())

print("\n=== the two classifications, by hand and by the script ===")
write_eloundou(SAME)
reset()
EXPO_D, EXPO_E_SAME = s103.build_both(s82, l47, l70, j47)
check("DAIOE: the planted tier is the top quartile",
      set(EXPO_D.loc[EXPO_D["fq"] == 4, "employer_id"]) == HIGH_D)
check("SAME: the Eloundou top quartile is the DAIOE top quartile",
      set(EXPO_E_SAME.loc[EXPO_E_SAME["fq"] == 4, "employer_id"]) == HIGH_D)
check("SAME: every employer scored on DAIOE is scored on Eloundou",
      set(EXPO_E_SAME["employer_id"]) == set(EXPO_D["employer_id"]))
s103.agreement(EXPO_D, EXPO_E_SAME, "all_scored_employers")
check("SAME: agreement 1.0 on employers, employment and the top quartiles",
      all(abs(stat("all_scored_employers", k) - 1.0) < 1e-12 for k in
          ("share_employers_same_quartile", "share_employment_same_quartile",
           "share_top_daioe_also_top_eloundou", "jaccard_top_quartiles")))

write_eloundou(SHIFTED)
reset()
_, EXPO_E_SHIFT = s103.build_both(s82, l47, l70, j47)
check("SHIFTED: the Eloundou top quartile is tiers 2 and 3 (120 employers)",
      set(EXPO_E_SHIFT.loc[EXPO_E_SHIFT["fq"] == 4, "employer_id"]) == HIGH_E_SHIFTED,
      f"{int((EXPO_E_SHIFT['fq'] == 4).sum())} employers")
s103.agreement(EXPO_D, EXPO_E_SHIFT, "all_scored_employers")
n_emp = EXPO_D.set_index("employer_id")["n"]
same_emp = {e for e in EMPS if tier(e) in (0, 1, 3)}
w_same = float(n_emp[list(same_emp)].sum() / n_emp.sum())
check("SHIFTED: 75% of employers in the same quartile (by hand: tiers 0, 1, 3)",
      abs(stat("all_scored_employers", "share_employers_same_quartile") - 0.75) < 1e-12,
      f"{stat('all_scored_employers', 'share_employers_same_quartile'):.4f}")
check("SHIFTED: the employment share in the same quartile matches the hand weight",
      abs(stat("all_scored_employers", "share_employment_same_quartile") - w_same) < 1e-9,
      f"{stat('all_scored_employers', 'share_employment_same_quartile'):.4f} vs {w_same:.4f}")
check("SHIFTED: all of DAIOE's top is Eloundou's; half of Eloundou's top is DAIOE's; "
      "Jaccard 0.5",
      abs(stat("all_scored_employers", "share_top_daioe_also_top_eloundou") - 1.0) < 1e-12
      and abs(stat("all_scored_employers", "share_top_eloundou_also_top_daioe") - 0.5) < 1e-12
      and abs(stat("all_scored_employers", "jaccard_top_quartiles") - 0.5) < 1e-12)
check("SHIFTED: the cross-tabulation puts tier 2 (60 employers) in DAIOE Q3 x Eloundou Q4",
      stat("all_scored_employers", "n_daioe_q3_eloundou_q4") == 60)
check("SHIFTED: the export holds no cell between 1 and 4",
      all((v == 0) or (v >= 5) or v != v for v in
          [r["value"] for r in s103.AGREE if r["statistic"].startswith("n_daioe_q")]))

print("\n--- the stock gate stops on Table 1's numbers ---")
reset()
counts = s103.load_counts("L_counts", s61.PANEL_YEARS, s103.COUNT_COLS)
try:
    s103.stock(counts, EXPO_D, EXPO_E_SHIFT, "22-25", s61, s73, s78, s80, j47,
               gate=True)
    stopped = False
except SystemExit:
    stopped = True
check("the stock gate STOPS against Table 1", stopped)

HAND_D = fx.triple_diff(st, HIGH_D, ["22-25"], OLD)
HAND_E = fx.triple_diff(st, HIGH_E_SHIFTED, ["22-25"], OLD)
HAND_D26 = fx.triple_diff(st, HIGH_D, ["26-30"], OLD)
HAND_E26 = fx.triple_diff(st, HIGH_E_SHIFTED, ["26-30"], OLD)
HAND_F_D, HAND_F_E = hand_female(sx, HIGH_D), hand_female(sx, HIGH_E_SHIFTED)
print(f"  by hand: DAIOE tau {HAND_D:+.4f} (planted pooled {FALL_POOLED:+.4f}); "
      f"Eloundou-shifted tau {HAND_E:+.4f}; female DAIOE {HAND_F_D:+.4f}, "
      f"Eloundou {HAND_F_E:+.4f}")
check("by hand: the shifted top set attenuates tau to between 0.3 and 0.85 of the fall",
      0.3 < HAND_E / FALL_POOLED < 0.85, f"{HAND_E / FALL_POOLED:.2f}")

R = {}
for name, expo_e, high_e in (("SAME", EXPO_E_SAME, HIGH_D),
                             ("SHIFTED", EXPO_E_SHIFT, HIGH_E_SHIFTED)):
    print(f"\n=== fits, the {name} world ===")
    s103.check = lambda label, got, want: []      # the gate is pointed at this world
    reset()
    counts = s103.load_counts("L_counts", s61.PANEL_YEARS, s103.COUNT_COLS)
    s103.stock(counts, EXPO_D, expo_e, "22-25", s61, s73, s78, s80, j47, gate=True)
    s103.stock(counts, EXPO_D, expo_e, "26-30", s61, s73, s78, s80, j47, gate=False)
    sexc = s103.load_counts("L_counts_sex", s61.PANEL_YEARS, s103.SEX_COLS)
    s103.sex(sexc, EXPO_D, expo_e, s67, s73, s78, s80, j47)
    check(f"{name}: no fit failed", not s103.FAILURES, "; ".join(s103.FAILURES))
    g = lambda *k: s103.get(*k)                                  # noqa: E731
    r = {"gate": g("G", "gate", "22-25", "hy_tau"),
         "d": g("D", "daioe_common", "22-25", "hy_tau"),
         "e": g("E", "eloundou", "22-25", "hy_tau"),
         "e_ind": g("E", "eloundou_indcl", "22-25", "hy_tau"),
         "d26": g("D", "daioe_common", "26-30", "hy_tau"),
         "e26": g("E", "eloundou", "26-30", "hy_tau"),
         "fgate": g("G", "sex_gate", "22-25", "hyf_tau"),
         "fd": g("D", "sex_daioe_common", "22-25", "hyf_tau"),
         "fe": g("E", "sex_eloundou", "22-25", "hyf_tau"),
         "fe_ind": g("E", "sex_eloundou_indcl", "22-25", "hyf_tau")}
    R[name] = r
    print(f"  22-25: gate {r['gate'][0]:+.4f}, DAIOE common {r['d'][0]:+.4f}, "
          f"Eloundou {r['e'][0]:+.4f} ({r['e'][1]:.4f}; ind {r['e_ind'][1]:.4f}); "
          f"26-30: DAIOE {r['d26'][0]:+.4f}, Eloundou {r['e26'][0]:+.4f}; female: "
          f"gate {r['fgate'][0]:+.4f}, DAIOE {r['fd'][0]:+.4f}, Eloundou {r['fe'][0]:+.4f}")
    check(f"{name}: the common-sample DAIOE fit equals the gate (every employer "
          "scored on both)",
          abs(r["d"][0] - r["gate"][0]) < 1e-9 and abs(r["d"][1] - r["gate"][1]) < 1e-9)
    check(f"{name}: the DAIOE tau reproduces the planted pooled fall",
          abs(r["d"][0] - FALL_POOLED) < 0.04, f"{r['d'][0]:+.4f} vs {FALL_POOLED:+.4f}")
    check(f"{name}: the industry-clustered Eloundou fit has the same coefficient "
          "and its own SE",
          abs(r["e"][0] - r["e_ind"][0]) < 1e-9 and r["e"][1] != r["e_ind"][1])
    check(f"{name}: the female DAIOE tau reproduces FALL_F",
          abs(r["fd"][0] - FALL_F) < 0.05, f"{r['fd'][0]:+.4f} vs {FALL_F:+.4f}")
    if name == "SAME":
        check("SAME: the Eloundou fits return the DAIOE fits to the last decimal "
              "(22-25, 26-30, female)",
              all(abs(r[a][0] - r[b][0]) < 1e-9 and abs(r[a][1] - r[b][1]) < 1e-9
                  for a, b in (("e", "d"), ("e26", "d26"), ("fe", "fd"))))
    else:
        check("SHIFTED: the Eloundou tau reads the hand triple difference on its own "
              "top set", abs(r["e"][0] - HAND_E) < 0.03,
              f"{r['e'][0]:+.4f} vs hand {HAND_E:+.4f}")
        check("SHIFTED: the Eloundou tau is attenuated, not the planted fall",
              0.3 < r["e"][0] / r["d"][0] < 0.85, f"ratio {r['e'][0] / r['d'][0]:.2f}")
        check("SHIFTED: the female Eloundou differential reads its hand value",
              abs(r["fe"][0] - HAND_F_E) < 0.04, f"{r['fe'][0]:+.4f} vs {HAND_F_E:+.4f}")
        check("SHIFTED: 26-30 on Eloundou reads its hand value",
              abs(r["e26"][0] - HAND_E26) < 0.03, f"{r['e26'][0]:+.4f} vs {HAND_E26:+.4f}")
        W_GATE = {"post": g("G", "gate", "22-25", "hy_post"),
                  "tau": g("G", "gate", "22-25", "hy_tau")}
        W_SEX = {"post": g("G", "sex_gate", "22-25", "hyf_post"),
                 "tau": g("G", "sex_gate", "22-25", "hyf_tau")}

print("\n--- main(), end to end, in the SHIFTED world ---")
s103.check = fx.load("103_eloundou_classification.py", "s103_fresh").check
s103.GATE, s103.SEX_GATE = W_GATE, W_SEX
reset()
_stdout = sys.stdout
rc = s103.main()
sys.stdout = _stdout
out = pd.read_csv(s103.OUT / "eloundou_classification.csv")
agr = pd.read_csv(s103.OUT / "classification_agreement.csv")
summ = (s103.OUT / "103_summary.txt").read_text()
check("main() returns 0", rc == 0, f"rc {rc}; {s103.FAILURES}")
check("every attempted fit came back (ten)", s103.DONE == s103.PLANNED == 10,
      f"{s103.DONE} of {s103.PLANNED}")
check("the summary prints gate, DAIOE-common and Eloundou for all three objects",
      summ.count("DAIOE, common employers") == 3
      and summ.count("Eloundou, same employers") == 3
      and "Table 1 (printed)" in summ)
check("the summary prints the differences and says their SE is unavailable",
      summ.count("difference (Eloundou minus DAIOE") == 3
      and "not available from two separate covariances" in summ)
check("the summary prints the agreement for both populations",
      "all_scored_employers:" in summ and "estimation_panel_22_25:" in summ
      and summ.count("same quartile:") == 2)
check("the agreement file holds both populations with the cross-tabulation",
      set(agr["population"]) == {"all_scored_employers", "estimation_panel_22_25"}
      and (agr["statistic"].str.startswith("n_daioe_q")).sum() == 32)
check("no employer count under the floor",
      bool(((out["n_firms"].isna()) | (out["n_firms"] >= 5)).all()))
check("no identifier column is exported",
      not any(c in out.columns for c in ("employer_id", "ind3", "cl_ind",
                                         "fe_emp_t"))
      and not any(c in agr.columns for c in ("employer_id", "ind3")))
check.done()
