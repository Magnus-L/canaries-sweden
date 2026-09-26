#!/usr/bin/env python3
"""
test_106_wfh_headline.py -- the dry run of script 106 (lane 39b).

THE MECHANISM PLANTED. 240 employers on the four DAIOE tiers, but each
employer's 2019 incumbents hold TWO codes: the tier's own and the code two
tiers away, the second with a weight of 0, 20 or 40 per cent by employer,
so both employer scores vary within a tier and the AI and teleworkability
orderings differ. The teleworkability fixture scores the four codes
0.95, 0.35, 0.65, 0.50 (tiers 0 to 3), so the AI-top tier sits in the
lower teleworkability half and the teleworkability top quartile holds
tier 0 plus part of the most mixed tier-2 employers. Two worlds share every draw:

  A  young workers in the AI top quartile fall by FALL from January 2024,
     and young women there by FALL_F more. The joint fit must return the
     fall on the AI indicator and nothing on z_wfh; the AI step among the
     low-teleworkability employers must read the fall by hand; the
     teleworkability step among the low-AI employers must read nothing.
  B  the same fall planted on the TELEWORKABILITY top quartile instead.
     The joint fit must load on z_wfh; the teleworkability step among the
     low-AI employers must read the fall by hand; the AI step among the
     low-teleworkability employers must read nothing.

Two of the four cells are degenerate in this world (the indicator does
not vary in the half); the script must say NOT ESTIMABLE and not fit.

    python3 revision/local/test_106_wfh_headline.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _fixtures_trip37 as fx  # noqa: E402

mc, TMP = fx.sandbox("106", ("CANARIES_106_OUT",))
s106 = fx.load("106_wfh_headline.py", "s106")
s106.OUT = TMP / "out"
s106.CACHE = mc.CACHE_DIR
check = fx.Check()
SHARE = Path(mc.SHARE)

EMPS = list(range(1, 241))
tier = lambda e: e % 4                                          # noqa: E731
size = lambda e: 4.2 if tier(e) == 3 else 1 + (e // 4) % 5     # noqa: E731
frac = lambda e: (0.0, 0.2, 0.4)[(e // 4) % 3]                  # noqa: E731
CODE = fx.tier_codes(SHARE)
WFH_SCORE = {0: 0.95, 1: 0.35, 2: 0.65, 3: 0.50}
FALL, FALL_F = float(np.log(0.85)), float(np.log(0.88))
MONTHS = fx.months()
LAM = {"22-25": 8, "26-30": 9, "31-34": 7, "35-40": 8, "41-49": 10, "50+": 12}
OLD = ["31-34", "35-40", "41-49", "50+"]


def install_two_codes() -> None:
    """82's three caches for a world in which every employer's incumbents
    hold two codes, so the two employer scores vary within a tier."""
    rows = []
    for e in EMPS:
        c1, c2 = CODE[tier(e)], CODE[(tier(e) + 2) % 4]
        n1 = int(round(6 * size(e) * (1 - frac(e))))
        n2 = int(round(6 * size(e) * frac(e)))
        for age in fx.AGES6:
            rows.append((e, age, c1, c1[:3], "2019", n1))
            if n2 > 0:
                rows.append((e, age, c2, c2[:3], "2019", n2))
            rows.append((e, age, "____", "___", "none", 1))
    casc = pd.DataFrame(rows, columns=["employer_id", "age_group", "ssyk4",
                                       "ssyk3", "source_year", "n"])
    casc["ssyk_ar"] = "2019"
    casc["ssyk_status"] = np.where(casc["ssyk4"] == "____", "9", "1")
    casc.to_parquet(mc.CACHE_DIR / "L_baseline_2019_cascade.parquet", index=False)
    (casc.groupby(["employer_id", "age_group", "ssyk4"], observed=True)["n"]
     .sum().reset_index()).to_parquet(mc.CACHE_DIR / "L_baseline_2019.parquet", index=False)
    pd.DataFrame([(e, f"2019-{m:02d}", a, int(round(10 * size(e))))
                  for e in EMPS for m in range(1, 13) for a in fx.AGES6],
                 columns=["employer_id", "year_month", "age_group", "n_emp"]
                 ).to_parquet(mc.CACHE_DIR / "L_counts_2019.parquet", index=False)


def write_wfh() -> None:
    """The teleworkability fixture: the real file with the four tier codes
    re-scored, so every other code keeps a real value."""
    real = pd.read_stata(fx.UPLOAD / "dingel_neiman_ssyk4.dta")
    real["ssyk4"] = real["ssyk4"].astype(int)
    for t, code in CODE.items():
        real.loc[real["ssyk4"] == int(code), "teleworkable"] = WFH_SCORE[t]
        if int(code) not in set(real["ssyk4"]):
            real = pd.concat([real, pd.DataFrame({"ssyk4": [int(code)],
                                                  "teleworkable": [WFH_SCORE[t]]})])
    real.to_stata(SHARE / "dingel_neiman_ssyk4.dta", write_index=False)


install_two_codes()
write_wfh()
s82, s61, s67, s78, l47, l70, j47 = s106.load_modules()
for m_ in (s82, s61, s67, s78, l47, l70, j47):
    m_.OUT, m_.CACHE = s106.OUT, mc.CACHE_DIR

print("\n=== the two scores ===")
EXPO_A, EXPO_W = s106.build_both(s82, l47, l70, j47)
HIGH_A = set(EXPO_A.loc[EXPO_A["fq"] == 4, "employer_id"])
HIGH_W = set(EXPO_W.loc[EXPO_W["fq"] == 4, "employer_id"])
check("the AI top quartile is the AI-top tier (all its mixes stay above the cut)",
      HIGH_A == {e for e in EMPS if tier(e) == 3}, f"{len(HIGH_A)} employers")
# The cut falls inside tier 2's most mixed employers (their mixes differ
# slightly with the rounding of the two code weights), so the set is tier 0
# plus part of that group; what matters is that it is a different set.
check("the teleworkability top quartile is a different set: all of tier 0 plus some of the most mixed tier 2",
      {e for e in EMPS if tier(e) == 0} <= HIGH_W
      <= {e for e in EMPS if tier(e) == 0 or (tier(e) == 2 and frac(e) == 0.4)}
      and not (HIGH_W & HIGH_A), f"{len(HIGH_W)} employers")
BOTH = s106.overlap(EXPO_A, EXPO_W)
ov = {r["item"]: r["value"] for r in s106.OVERLAP}
check("every employer carries both scores; the halves are 120 and 120",
      ov["n_employers_both_scores"] == 240 and BOTH["hi_ai"].sum() == 120 and BOTH["hi_wfh"].sum() == 120)
check("z_wfh has incumbent-weighted mean zero and SD one",
      abs(np.average(BOTH["z_wfh"], weights=EXPO_A.set_index("employer_id").loc[BOTH["employer_id"], "n"])) < 1e-9
      and abs(np.sqrt(np.average(BOTH["z_wfh"] ** 2,
                                 weights=EXPO_A.set_index("employer_id").loc[BOTH["employer_id"], "n"])) - 1) < 1e-9)
check("the scores are not collinear (|Spearman| under 0.6) and the top quartiles do not coincide",
      abs(ov["spearman"]) < 0.6 and ov["n_top_both"] == 0, f"Spearman {ov['spearman']:+.3f}")
LO_WFH = set(BOTH.loc[BOTH["hi_wfh"] == 0, "employer_id"])
LO_AI = set(BOTH.loc[BOTH["hi_ai"] == 0, "employer_id"])
check("the AI-top tier lies in the low-teleworkability half; the teleworkability top has members in the low-AI half",
      HIGH_A <= LO_WFH and len(HIGH_W & LO_AI) > 0 and len(LO_AI - HIGH_W) > 0)

E, Bd, S, M = np.meshgrid(EMPS, list(LAM), ["1", "2"], MONTHS, indexing="ij")
GRID = pd.DataFrame({"employer_id": E.ravel(), "age_group": Bd.ravel(),
                     "gender": S.ravel(), "year_month": M.ravel()})


def world(high: set) -> pd.DataFrame:
    d = GRID
    lam = (d["age_group"].map(LAM) * d["employer_id"].map(size)).to_numpy(float)
    hi = d["employer_id"].isin(high).to_numpy()
    y22 = (d["age_group"] == "22-25").to_numpy()
    fem = (d["gender"] == "2").to_numpy()
    later = (d["year_month"] >= "2024-01").to_numpy()
    lam = lam * np.exp(FALL * (hi & y22 & later))
    lam = lam * np.exp(FALL_F * (hi & y22 & fem & later))
    out = d.copy()
    out["n_emp"] = fx.poisson_same_noise(lam, 106)
    return out


def install(high: set) -> tuple:
    sx = world(high)
    fx.write_by_year(sx, mc.CACHE_DIR, "L_counts_sex")
    st = (sx.groupby(["employer_id", "year_month", "age_group"])["n_emp"].sum().reset_index())
    fx.write_by_year(st, mc.CACHE_DIR, "L_counts")
    return sx, st


def hand(st, high, employers=None) -> float:
    c = st if employers is None else st[st["employer_id"].isin(employers)]
    return fx.triple_diff(c, high, ["22-25"], OLD)


def hand_female(sx, high) -> float:
    out = {}
    for g in ("1", "2"):
        out[g] = fx.triple_diff(sx[sx["gender"] == g], high, ["22-25"], OLD)
    return out["2"] - out["1"]


def reset() -> None:
    s106.ROWS.clear(); s106.FAILURES.clear(); s106.NOTES.clear()
    s106.DONE = s106.PLANNED = 0


def status(tag) -> str:
    return next((r["status"] for r in s106.ROWS if r["part"] == "O" and r["spec"] == tag
                 and r["term"] == "hy_tau"), "missing")


R = {}
for name, high in (("A", HIGH_A), ("B", HIGH_W)):
    print(f"\n=== world {name}: the fall on the {'AI' if name == 'A' else 'teleworkability'} top ===")
    sx, st = install(high)
    reset()
    counts = s106.load_counts("L_counts", s61.PANEL_YEARS, s106.COUNT_COLS)
    if name == "A":
        try:
            s106.stock_gate(counts, EXPO_A, s61, s78, j47)
            stopped = False
        except SystemExit:
            stopped = True
        check("the stock gate STOPS against Table 1", stopped)
        W_GATE = {"post": s106.get("G", "gate", "22-25", "hy_post"),
                  "tau": s106.get("G", "gate", "22-25", "hy_tau")}
        reset()
    s106.check = lambda *a, **k: []               # the gates are pointed at this world
    b0 = s106.stock_gate(counts, EXPO_A, s61, s78, j47)
    s106.joint(b0, BOTH, j47, s78)
    s106.offdiagonal(b0, BOTH, j47, s78)
    sexc = s106.load_counts("L_counts_sex", s61.PANEL_YEARS, s106.SEX_COLS)
    s106.sex(sexc, EXPO_A, BOTH, s67, s78, j47)
    check(f"{name}: no fit failed", not s106.FAILURES, "; ".join(s106.FAILURES))
    g = lambda *k: s106.get(*k)                                  # noqa: E731
    r = {"gate": g("G", "gate", "22-25", "hy_tau"), "ai": g("J", "ai_only", "22-25", "hy_tau"),
         "j_ai": g("J", "joint", "22-25", "hy_tau"), "j_w": g("J", "joint", "22-25", "wfh_tau"),
         "w": g("J", "wfh_only", "22-25", "wfh_tau"),
         "o1": g("O", "ai_in_low_wfh", "22-25", "hy_tau"), "o3": g("O", "wfh_in_low_ai", "22-25", "hy_tau"),
         "f": g("S", "sex_ai_only", "22-25", "hyf_tau"), "fj_ai": g("S", "sex_joint", "22-25", "hyf_tau"),
         "fj_w": g("S", "sex_joint", "22-25", "wfhf_tau"),
         "h_all": hand(st, HIGH_A), "h_o1": hand(st, HIGH_A, LO_WFH), "h_o3": hand(st, HIGH_W, LO_AI),
         "h_f": hand_female(sx, HIGH_A)}
    R[name] = r
    print(f"  ai_only {r['ai'][0]:+.4f} ({r['ai'][1]:.4f}) hand {r['h_all']:+.4f}; joint AI {r['j_ai'][0]:+.4f} "
          f"({r['j_ai'][1]:.4f}), WFH/SD {r['j_w'][0]:+.4f} ({r['j_w'][1]:.4f}); wfh_only {r['w'][0]:+.4f}; "
          f"cells: ai|lowWFH {r['o1'][0]:+.4f} hand {r['h_o1']:+.4f}; wfh|lowAI {r['o3'][0]:+.4f} hand {r['h_o3']:+.4f}; "
          f"female: {r['f'][0]:+.4f} hand {r['h_f']:+.4f}, joint {r['fj_ai'][0]:+.4f}, WFH {r['fj_w'][0]:+.4f}")
    check(f"{name}: the common-sample AI-only fit equals the gate (every employer carries both scores)",
          abs(r["ai"][0] - r["gate"][0]) < 1e-9 and abs(r["ai"][1] - r["gate"][1]) < 1e-9)
    check(f"{name}: the AI-only tau reads the hand triple difference on the AI top (within 0.03)",
          abs(r["ai"][0] - r["h_all"]) < 0.03)
    check(f"{name}: the discriminating cells read their hand triple differences on the half (within 0.04)",
          abs(r["o1"][0] - r["h_o1"]) < 0.04 and abs(r["o3"][0] - r["h_o3"]) < 0.04)
    check(f"{name}: the two degenerate cells are recorded NOT ESTIMABLE and not fitted",
          status("ai_in_high_wfh") == "not_estimable" and status("wfh_in_high_ai") == "not_estimable")
    if name == "A":
        check("A: the joint fit keeps the fall on the AI indicator (within 0.05 of AI only) and loads little on z_wfh",
              abs(r["j_ai"][0] - r["ai"][0]) < 0.05 and abs(r["j_w"][0]) < 0.4 * abs(FALL)
              and abs(r["j_ai"][0]) > 2.5 * abs(r["j_w"][0]), f"AI {r['j_ai'][0]:+.4f} WFH {r['j_w'][0]:+.4f}")
        check("A: the AI step among low-teleworkability employers reads the fall; the teleworkability step among low-AI employers reads nothing",
              r["o1"][0] < 0.5 * FALL and abs(r["o3"][0]) < 0.4 * abs(FALL))
        check("A: the female differential reads FALL_F by hand, and the joint sex fit keeps it on the AI indicator",
              abs(r["f"][0] - r["h_f"]) < 0.05 and abs(r["fj_ai"][0] - r["f"][0]) < 0.06
              and abs(r["fj_w"][0]) < 0.5 * abs(FALL_F), f"{r['f'][0]:+.4f} vs hand {r['h_f']:+.4f}; joint {r['fj_ai'][0]:+.4f}, WFH {r['fj_w'][0]:+.4f}")
        W_SEX = {"post": g("G", "sex_gate", "22-25", "hyf_post"), "tau": g("G", "sex_gate", "22-25", "hyf_tau")}
    else:
        check("B: the joint fit loads the fall on z_wfh (negative, beyond 2 SE) and the teleworkability-only fit agrees in sign",
              r["j_w"][0] < 0 and abs(r["j_w"][0]) > 2 * r["j_w"][1] and r["w"][0] < 0,
              f"WFH/SD {r['j_w'][0]:+.4f} ({r['j_w'][1]:.4f}); AI {r['j_ai'][0]:+.4f}")
        check("B: the teleworkability step among low-AI employers reads the fall; the AI step among low-teleworkability employers reads nothing",
              r["o3"][0] < 0.5 * FALL and abs(r["o1"][0]) < 0.4 * abs(FALL))
        check("B: the joint AI tau is smaller in magnitude than in world A",
              abs(r["j_ai"][0]) < abs(R["A"]["j_ai"][0]))

print("\n--- main(), end to end, in world A ---")
install(HIGH_A)
reset()
s106.check = fx.load("106_wfh_headline.py", "s106_fresh").check
s106.GATE, s106.SEX_GATE = W_GATE, W_SEX
_stdout = sys.stdout
rc = s106.main()
sys.stdout = _stdout
out = pd.read_csv(s106.OUT / "wfh_headline.csv")
ovl = pd.read_csv(s106.OUT / "wfh_scores_overlap.csv")
summ = (s106.OUT / "106_summary.txt").read_text()
check("main() returns 0", rc == 0, f"rc {rc}; {s106.FAILURES}")
check("nine fits attempted and returned (eleven planned less the two degenerate cells)",
      s106.DONE == s106.PLANNED == 9, f"{s106.DONE} of {s106.PLANNED}")
check("the summary prints the gates, the scores, J, O with 89's rule, and S",
      "stock 22-25: tau" in summ and "sex 22-25: female differential" in summ
      and "THE TWO SCORES:" in summ and "J. THE JOINT SPECIFICATION" in summ
      and "O. THE SPLIT-SAMPLE CELLS" in summ and summ.count("NOT ESTIMABLE") == 2
      and "89's rule:" in summ and "S. THE FEMALE DIFFERENTIAL" in summ and "NO FIT" not in summ)
check("the export holds G, J, O and S rows and the overlap file its cells",
      set(out["part"]) == {"G", "J", "O", "S"}
      and set(ovl["item"]) >= {"spearman", "cell_ai0_wfh0", "n_top_both"})
check("no employer count under the floor",
      bool(((out["n_firms"].isna()) | (out["n_firms"] >= 5) | (out["n_firms"] == 0)).all()))
check("no identifier column is exported",
      not any(c in out.columns for c in ("employer_id", "z_wfh", "hi_ai", "hi_wfh", "high_w", "fe_emp_t")))
check.done()
