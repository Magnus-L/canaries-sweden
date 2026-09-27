#!/usr/bin/env python3
"""
test_108_adoption_by_index.py -- dry run of script 108 (lane 39d).

THE MECHANISM PLANTED. 2,400 employers on the four DAIOE tiers (tier 3 the
DAIOE top quartile). The Eloundou fixture is the SHIFTED world of
test_103: tier 2 outranks tier 3, so the Eloundou top quartile is tiers 2
and 3. A planted firm survey (ITFtg_Stora_2023) records AI use with
probability 0.60 in tier 3 and 0.20 elsewhere, and language generation
with 0.30 against 0.08. So DAIOE is the better predictor by construction:
  - daioe_common reads about +40 points, eloundou_common less;
  - daioe_only (tier 3 firms against tiers 0-1, the Eloundou top excluded
    from the sample... i.e. no firm here, see below) and eloundou_only
    (tier 2 against tiers 0-1) reads about zero.
Because every DAIOE-top firm is also Eloundou-top in this world, the
daioe_only route has NO top firms and must be reported below threshold,
while eloundou_only must read about zero. The joint fit (both dummies) must
put the planted gap on DAIOE (tier 3 = DAIOE top AND Eloundou top, tier 2 =
Eloundou top only, so the DAIOE dummy picks up tier 3 against tier 2) and
about zero on Eloundou, with a significant positive difference. A second
world swaps the planted use to tier 2 (the Eloundou-only tier): the joint
fit must then put it on Eloundou and a negative difference. No verdict is
computed: the script reports, it does not adjudicate (ML, 27 Sep).

Checked: the routes are built as specified; 71's arms run on them
through a planted catalogue and read_sql; the gate reads the daioe_all
row; the thin routes use the lowered threshold and 71's threshold is
restored afterwards; the summary and both export files.

    python3 revision/local/test_108_adoption_by_index.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _fixtures_trip37 as fx  # noqa: E402

mc, TMP = fx.sandbox("108", ("CANARIES_108_OUT", "CANARIES_103_OUT"))
s108 = fx.load("108_adoption_by_index.py", "s108")
s108.OUT = TMP / "out"
s108.CACHE = mc.CACHE_DIR
check = fx.Check()
SHARE = Path(mc.SHARE)

EMPS = list(range(1, 2401))
tier = lambda e: e % 4                                          # noqa: E731
size = lambda e: 4.2 if tier(e) == 3 else 1 + (e // 4) % 5     # noqa: E731
fx.install_score(mc, EMPS, tier, size, SHARE)
CODE = fx.tier_codes(SHARE)
SHIFTED = {0: 0.10, 1: 0.30, 2: 0.80, 3: 0.55}


def write_eloundou(order):
    rows = [{"ssyk4": int(c), "eloundou_score": float(order[t]), "eloundou_quartile": 1,
             "high_exposure_eloundou": 0} for t, c in CODE.items()]
    d = pd.DataFrame(rows)
    real = pd.read_stata(fx.UPLOAD / "eloundou_ssyk4.dta")
    real["ssyk4"] = real["ssyk4"].astype(int)
    pad = real[~real["ssyk4"].isin(d["ssyk4"])].copy()
    pad["eloundou_score"] = 0.0
    pad["eloundou_quartile"] = 1
    pad["high_exposure_eloundou"] = 0
    pd.concat([d, pad[d.columns]], ignore_index=True).to_stata(SHARE / "eloundou_ssyk4.dta", write_index=False)


write_eloundou(SHIFTED)
s103, s82, s73, s71, l47, l70, j47 = s108.load_modules()
for m_ in (s82, l47, l70, j47):
    m_.OUT, m_.CACHE = s108.OUT, mc.CACHE_DIR
EXPO_D, EXPO_E = s103.build_both(s82, l47, l70, j47)
check("DAIOE top quartile is tier 3; Eloundou top is tiers 2 and 3",
      set(EXPO_D.loc[EXPO_D.fq == 4, "employer_id"]) == {e for e in EMPS if tier(e) == 3}
      and set(EXPO_E.loc[EXPO_E.fq == 4, "employer_id"]) == {e for e in EMPS if tier(e) in (2, 3)})

R = s108.build_routes(EXPO_D, EXPO_E, s73)
nid = lambda s: set(s73.norm_id(pd.Series(sorted(s))))  # noqa: E731
check("routes: daioe_only excludes every Eloundou-top firm and has no top firm here",
      int((R["daioe_only"].fq == 4).sum()) == 0
      and not (set(R["daioe_only"].employer_id) & nid({e for e in EMPS if tier(e) in (2, 3)})))
check("routes: eloundou_only is tiers 0-2, top = tier 2",
      set(R["eloundou_only"].employer_id) == nid({e for e in EMPS if tier(e) in (0, 1, 2)})
      and set(R["eloundou_only"].loc[R["eloundou_only"].fq == 4, "employer_id"]) == nid({e for e in EMPS if tier(e) == 2}))


def plant_survey(use_tier):
    rng = np.random.default_rng(108)
    ids = s73.norm_id(pd.Series(EMPS))
    u1, u2 = rng.random(len(EMPS)), rng.random(len(EMPS))
    hi = np.array([tier(e) == use_tier for e in EMPS])
    df = pd.DataFrame({"PeOrgNr": ids.values,
                       "E_AI_TML": (u1 < np.where(hi, 0.60, 0.20)).astype(int),
                       "E_AI_TNLG": (u2 < np.where(hi, 0.30, 0.08)).astype(int)})
    return df


SURVEY = {}
s71.discover = lambda conn: pd.DataFrame({"TABLE_NAME": ["ITFtg_Stora_2023"] * 3,
                                          "COLUMN_NAME": ["PeOrgNr", "E_AI_TML", "E_AI_TNLG"]})
_real_read_sql = pd.read_sql
def fake_read_sql(q, conn, *a, **k):
    if "ITFtg_Stora_2023" in q:
        return SURVEY["df"].copy()
    raise RuntimeError(f"unexpected query in the dry run: {q[:80]}")
s71.pd.read_sql = fake_read_sql
s108.open_conn = lambda: type("C", (), {"close": lambda self: None})()

results = {}
for name, use_tier in (("DAIOE world", 3), ("ELOUNDOU world", 2)):
    SURVEY["df"] = plant_survey(use_tier)
    s108.NOTES.clear(); s108.FAILURES.clear()
    thr0 = s71.MIN_ITFTG_HIGH
    rows, counts = s108.run_arms(R, s71, s73)
    t = s108.tidy(rows)
    results[name] = t
    check(f"{name}: 71's threshold restored after the thin routes", s71.MIN_ITFTG_HIGH == thr0)
    g = lambda r, o: t[(t.route == r) & (t.outcome == o)]  # noqa: E731
    print(t.to_string())
    check(f"{name}: daioe_all, daioe_common and eloundou_common estimated for both outcomes",
          all(len(g(r, o)) == 1 for r in ("daioe_all", "daioe_common", "eloundou_common") for o in ("ai_any", "ai_genai")))
    J = lambda r: float(g(r, "ai_any").coef_points.iloc[0]) if len(g(r, "ai_any")) == 1 else None  # noqa: E731
    jd, je, jdiff = J("joint_daioe"), J("joint_eloundou"), J("joint_diff")
    jse = float(g("joint_diff", "ai_any").se_points.iloc[0]) if jdiff is not None else None
    check(f"{name}: the joint fit reports DAIOE, Eloundou and their difference",
          None not in (jd, je, jdiff) and abs(jdiff - (jd - je)) < 1e-6)
    if name == "DAIOE world":
        check("DAIOE world: daioe_common reads the planted any-AI gap (1 - 0.4*0.7 against 1 - 0.8*0.92 = 45.6 points) on any AI, and eloundou_common less",
              abs(float(g("daioe_common", "ai_any").coef_points.iloc[0]) - 45.6) < 6
              and float(g("eloundou_common", "ai_any").coef_points.iloc[0]) < float(g("daioe_common", "ai_any").coef_points.iloc[0]),
              f'{float(g("daioe_common", "ai_any").coef_points.iloc[0]):.1f}')
        check("DAIOE world: eloundou_only reads about zero",
              len(g("eloundou_only", "ai_any")) == 1 and abs(float(g("eloundou_only", "ai_any").coef_points.iloc[0])) < 6)
        check("DAIOE world: daioe_only has no top firm, so it is not estimated",
              len(g("daioe_only", "ai_any")) == 0)
        check("DAIOE world: joint fit puts the planted 45.6-point gap on DAIOE, Eloundou about zero, diff significant",
              abs(jd - 45.6) < 6 and abs(je) < 6 and jdiff > 1.96 * jse, f"{jd:.1f} {je:.1f} {jdiff:.1f} ({jse:.1f})")
    else:
        check("ELOUNDOU world: eloundou_only reads the planted any-AI gap (1 - 0.4*0.7 against 1 - 0.8*0.92 = 45.6 points) (daioe_only has no top firm in this world)",
              abs(float(g("eloundou_only", "ai_any").coef_points.iloc[0]) - 45.6) < 6)
        check("ELOUNDOU world: joint fit puts the gap on Eloundou (tier 2 against 0-1) and DAIOE negative (tier 3 against 2), diff significant negative",
              abs(je - 45.6) < 6 and abs(jd + 45.6) < 6 and jdiff < -1.96 * jse, f"{jd:.1f} {je:.1f} {jdiff:.1f} ({jse:.1f})")

print("\n--- the gate and main() ---")
SURVEY["df"] = plant_survey(3)
t = results["DAIOE world"]
r = t[(t.route == "daioe_all") & (t.source == "ITFtg_Stora_2023") & (t.outcome == "ai_any")]
s108.GATE_POINTS, s108.GATE_N = float(r.coef_points.iloc[0]), int(r.n.iloc[0])
check("the gate passes on the world's own daioe_all row", s108.gate(t) == [])
s108.GATE_POINTS += 1.0
check("the gate fails when the reference moves by a point", len(s108.gate(t)) == 1)
s108.GATE_POINTS -= 1.0
_mods = (s103, s82, s73, s71, l47, l70, j47)
s108.load_modules = lambda: _mods
s108.NOTES.clear(); s108.FAILURES.clear()
_stdout = sys.stdout
rc = s108.main()
sys.stdout = _stdout
summ = (s108.OUT / "108_summary.txt").read_text()
out = pd.read_csv(s108.OUT / "adoption_by_index.csv")
check("main() returns 0 and writes both files", rc == 0 and (s108.OUT / "adoption_counts.csv").exists(), f"rc {rc} {s108.FAILURES}")
check("the summary prints the gate, A1, A2 and A3, and no verdict", "GATE: PASSES" in summ and "A1. SAME FIRMS" in summ
      and "A2. THE DISAGREEMENT" in summ and "A3. JOINT FIT" in summ and "better predictor" not in summ)
check("no identifier column is exported", "employer_id" not in out.columns)
check.done()
