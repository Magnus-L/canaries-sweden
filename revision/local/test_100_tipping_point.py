#!/usr/bin/env python3
"""
test_100_tipping_point.py -- the dry run of script 100 (lane 38a).

THE MECHANISM PLANTED. The threat is that workers with no birth year or
sex (the unlinked) are disproportionately young at exposed employers in
the later period, so that the panel undercounts young employment there
exactly when it is measured to fall. One employer x band x sex x month
world with a planted young decline FALL at top-quartile employers from
January 2024 (young women FALL_F more), and two linkage worlds on the
same draws:

  SMALL  the unlinked are 0.2 per cent of every employer-month, flat
         across groups and periods. The tipping point needs far more
         young person-months than there are unlinked: T1 and T3 pass,
         and allocating every unlinked person-month to the young leaves
         tau negative (T2, T4).
  BIG    at exposed employers in the later period the unlinked equal 45
         per cent of the TRUE young count of the employer-month (the
         missing young of the threat, planted at the size that would
         undo the decline). T1 and T2 must fail.

Hand checks before the estimator: the planted contrast; Y, U_HL and the
own-rate excess recomputed from the fixture; the allocation totals. The
tipping point must be verified (tau at k* within a tenth of an SE of
zero) and m* close to exp(-tau) - 1. The gate is pointed at each world
(as test_99 does) and stops on Table 1's numbers.

    python3 revision/local/test_100_tipping_point.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _fixtures_trip37 as fx  # noqa: E402

mc, TMP = fx.sandbox("100", ("CANARIES_100_OUT",))
s100 = fx.load("100_tipping_point.py", "s100")
s100.OUT = TMP / "out"
s100.CACHE = mc.CACHE_DIR
check = fx.Check()

EMPS = list(range(1, 241))
tier = lambda e: e % 4                                          # noqa: E731
size = lambda e: 4.2 if tier(e) == 3 else 1 + (e // 4) % 5     # noqa: E731
fx.install_score(mc, EMPS, tier, size, Path(mc.SHARE))
HIGH = {e for e in EMPS if tier(e) == 3}
FALL, FALL_F = float(np.log(0.85)), float(np.log(0.90))
MONTHS = fx.months()
LAM = {"22-25": 8, "26-30": 9, "31-34": 7, "35-40": 8, "41-49": 10, "50+": 12}
E, B, S, M = np.meshgrid(EMPS, list(LAM), ["1", "2"], MONTHS, indexing="ij")
G = pd.DataFrame({"employer_id": E.ravel(), "age_group": B.ravel(),
                  "gender": S.ravel(), "year_month": M.ravel()})
hi = G["employer_id"].isin(HIGH).to_numpy()
young = (G["age_group"] == "22-25").to_numpy()
fem = (G["gender"] == "2").to_numpy()
later = (G["year_month"] >= "2024-01").to_numpy()
lam = (G["age_group"].map(LAM) * G["employer_id"].map(size)).to_numpy(float)
lam = lam * np.where(hi & young & later, np.exp(FALL), 1.0) \
    * np.where(hi & young & later & fem, np.exp(FALL_F), 1.0)
G["n_emp"] = fx.poisson_same_noise(lam, 1001)
# the TRUE young count had nothing gone missing (what BIG takes away)
G["n_true"] = fx.poisson_same_noise(
    (G["age_group"].map(LAM) * G["employer_id"].map(size)).to_numpy(float),
    1001)
keys = ["employer_id", "year_month", "age_group"]
POOL = G.groupby(keys)["n_emp"].sum().reset_index()
fx.write_by_year(POOL, mc.CACHE_DIR, "L_counts")
fx.write_by_year(G[keys + ["gender", "n_emp"]], mc.CACHE_DIR, "L_counts_sex")

old = ["31-34", "35-40", "41-49", "50+"]
h = fx.triple_diff(POOL, HIGH, ["22-25"], old)
# pooled over the two sexes, half of whom fall by FALL_F more
POOLED_FALL = FALL + float(np.log((1 + np.exp(FALL_F)) / 2))
check("by hand: the pooled counts carry the planted fall",
      abs(h - POOLED_FALL) < 0.03, f"{h:+.4f} against {POOLED_FALL:+.4f}")

# employer-month totals of every age (the unlinked pull's n_all)
EM = G.groupby(["employer_id", "year_month"])["n_emp"].sum().rename(
    "n_panel").reset_index()
YT = (G[young].groupby(["employer_id", "year_month"])["n_true"].sum()
      .rename("young_true").reset_index())
EM = EM.merge(YT, on=["employer_id", "year_month"])
EM["n_all"] = np.round(EM["n_panel"] * 1.3).astype(int)
EMhi = EM["employer_id"].isin(HIGH)
EMlater = EM["year_month"] >= "2024-01"


def install_unlinked(world: str) -> pd.DataFrame:
    u = EM.copy()
    base = np.round(0.002 * u["n_all"]).astype(int)
    if world == "big":
        base = np.where(EMhi & EMlater,
                        np.round(0.45 * u["young_true"]).astype(int), base)
    u["n_noreg"] = base
    u["n_nobirth"] = 0
    u["n_nosex"] = np.round(0.001 * u["n_all"]).astype(int)
    fx.write_by_year(u[s100.UNL_COLS], mc.CACHE_DIR, "T_unlinked")
    return u


s82, s61, s67, s78, l47, l70, j47 = s100.load_modules()
for m_ in (s82, s61, s67, s78, l47, l70, j47):
    m_.OUT, m_.CACHE = s100.OUT, mc.CACHE_DIR
EXPO = s82.build_exposure(l47, l70, j47)["exposure"]


def world_gates() -> tuple:
    """The two gates pointed at this fixture's own panels."""
    s100.EST.clear()
    b = s78.with_exposure(s61.build_skeleton(POOL, "22-25", j47), EXPO)
    b, t = s78.eq2_terms(b)
    s100.fit_tau(b, "probe_p", t, j47.FES, "Z", "p", 1, s100.POST,
                 s100.INTERIM)
    bs = s78.with_exposure(s67.build_skeleton_sex(
        G[keys + ["gender", "n_emp"]], "22-25", j47, "n_emp"), EXPO)
    bs, ts = s78.gender_eq2_terms(bs)
    s100.fit_tau(bs, "probe_s", ts, j47.FES, "Z", "s", 1, s100.FPOST,
                 s100.FINTERIM)
    g = {"post": s100.get("Z", "p", "post"), "tau": s100.get("Z", "p", "tau")}
    gs = {"post": s100.get("Z", "s", "post"), "tau": s100.get("Z", "s", "tau")}
    return g, gs


GP, GS = world_gates()
print(f"  world gates: pooled {GP}, sex {GS}")
SUMM = {}
for world in ("small", "big"):
    print(f"\n=== the {world.upper()} world ===")
    U = install_unlinked(world)
    s100.GATE, s100.SEX_GATE = GP, GS
    s100.EST.clear(); s100.FAILURES.clear(); s100.NOTES.clear()
    s100.DONE = s100.PLANNED = 0
    _stdout = sys.stdout
    rc = s100.main()
    sys.stdout = _stdout
    SUMM[world] = (s100.OUT / "100_summary.txt").read_text()
    check(f"{world}: main() returns 0", rc == 0, f"{rc}; {s100.FAILURES}")
    check(f"{world}: every attempted fit came back",
          s100.DONE == s100.PLANNED == 10, f"{s100.DONE} of {s100.PLANNED}")
    # hand checks
    Y = float(POOL[POOL["employer_id"].isin(HIGH)
                   & (POOL["age_group"] == "22-25")
                   & (POOL["year_month"] >= "2024-01")]["n_emp"].sum())
    Yg, _ = s100.get("P", "tipping", "young_pm_high_later_Y")
    check(f"{world}: Y is the young person-months at exposed employers, later",
          Yg == Y, f"{Yg:,.0f} vs {Y:,.0f}")
    uh = U[EMhi]
    U_HL = float(uh.loc[uh["year_month"] >= "2024-01", "n_noreg"].sum())
    A_HL = float(uh.loc[uh["year_month"] >= "2024-01", "n_all"].sum())
    ui = uh[uh["year_month"].between("2022-12", "2023-12")]
    r_HI = float(ui["n_noreg"].sum() / ui["n_all"].sum())
    check(f"{world}: U_HL from the accounting equals the fixture",
          s100.get("P", "tipping", "U_high_later")[0] == U_HL,
          f"{s100.get('P', 'tipping', 'U_high_later')[0]:,.0f} vs {U_HL:,.0f}")
    exc = s100.get("P", "tipping", "excess_own_rate")[0]
    check(f"{world}: the own-rate excess by hand",
          abs(exc - (U_HL - r_HI * A_HL)) < 1e-6, f"{exc:,.1f}")
    t0, se0 = s100.get("P", "gate", "tau")
    m, _ = s100.get("P", "tipping", "m_star")
    tk, _ = s100.get("P", "tipping", "tau_at_kstar")
    check(f"{world}: the tipping point is verified (tau at k* ~ 0)",
          abs(tk) <= 0.1 * se0, f"tau(k*) {tk:+.5f}, 0.1 SE {0.1 * se0:.5f}")
    check(f"{world}: m* is close to exp(-tau) - 1",
          abs(m - (np.exp(-t0) - 1)) < 0.01, f"{m:.4f} vs "
          f"{np.exp(-t0) - 1:.4f}")
    got, _ = s100.get("P", "x_all", "allocated_pm")
    want = float(uh.loc[uh["year_month"] >= "2022-12", "n_noreg"].sum())
    check(f"{world}: X_all allocates every unlinked person-month (interim, "
          f"later)", got == want, f"{got:,.0f} vs {want:,.0f}")
    ts, ss = s100.get("S", "tipping", "tau_at_kstar")
    sg = s100.get("S", "gate", "tau")[1]
    check(f"{world}: the sex tipping point is verified",
          abs(ts) <= 0.1 * sg, f"{ts:+.5f}")
    for f in ("tipping_point.csv", "unlinked_accounting.csv",
              "unlinked_by_month.csv", "unlinked_accounting_sex.csv"):
        d = pd.read_csv(s100.OUT / f)
        check(f"{world}: {f} carries no identifier",
              not any(c in d.columns for c in ("employer_id", "person_id")))
    acc = pd.read_csv(s100.OUT / "unlinked_accounting.csv")
    row = acc[(acc["group"] == "high") & (acc["period"] == "later")].iloc[0]
    check(f"{world}: the accounting share is unlinked over all person-months",
          abs(row["share_u_pooled"] - U_HL / A_HL) < 1e-12)
    if world == "small":
        check("small: T1 passes", "MISSING DEMOGRAPHICS CANNOT TIP IT"
              in SUMM[world].split("T3")[0])
        check("small: T2 passes", "THE EXTREMAL ALLOCATION LEAVES IT NEGATIVE"
              in SUMM[world].split("T3")[0])
        check("small: T3 and T4 pass",
              SUMM[world].count("CANNOT TIP IT") == 2
              and SUMM[world].count("LEAVES IT NEGATIVE") == 2)
    else:
        check("big: T1 refused", "T1 NOT MET" in SUMM[world])
        check("big: T2 refused", "T2 NOT MET" in SUMM[world])
        tx, _ = s100.get("P", "x_all", "tau")
        check("big: allocating the unlinked to the young undoes the decline",
              tx > t0 + 0.1, f"{tx:+.4f} vs gate {t0:+.4f}")

print("\n--- a gate that misses stops its design ---")
s100.GATE = {"post": (-0.0578, 0.0155), "tau": (-0.0399, 0.0102)}
s100.EST.clear(); s100.FAILURES.clear(); s100.NOTES.clear()
s100.DONE = s100.PLANNED = 0
_stdout = sys.stdout
rc = s100.main()
sys.stdout = _stdout
check("the pooled gate refuses Table 1's numbers in the fixture world",
      any("THE POOLED GATE FAILED" in f for f in s100.FAILURES)
      and not any(r["part"] == "P" and r["spec"] == "tipping"
                  for r in s100.EST), "; ".join(s100.FAILURES)[:200])
check("a failed gate makes the run non-zero (the lane will not skip it)",
      rc == 1)
check.done()
