#!/usr/bin/env python3
"""
test_97_headline_checks.py -- the dry run of script 97 (lane 37b).

THREE WORLDS on one employer x age band x sex x month grid, sharing every
uniform draw, so each read rule is shown to pass in one and fail in
another:

  AI        young workers in top-quartile employers fall by FALL from
            January 2024, young women there by FALL_F more; industry and
            leverage are assigned independently of exposure.
            K1, P1 and I1 must all pass.
  DRIFT     the AI world plus a pre-launch linear decline of young women
            in exposed employers (DRIFT per month, January 2021 to November
            2022). P1 must fail; nothing else changes.
  CONFOUND  NO exposure effect at all. Young women in industry 100 fall by
            FALL_F from January 2024, and young workers in high-leverage
            employers by FALL; exposed employers are mostly in industry 100
            and mostly high-leverage, so the naive contrasts are negative.
            I1 and K1 must fail: the industry and leverage terms absorb
            what the baseline attributes to exposure.

Each planted contrast is checked by hand before the estimator. Also: the
stock gate stops on Table 1's numbers; the leverage and industry inputs
are injected (their SQL is script 73's and 80's, run on MONA); 73's
coverage gate of 500 employers is lowered to 50 for a 240-employer world
and said so; main() runs end to end with no SQL and no identifier out.

    python3 revision/local/test_97_female_diagnostics.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _fixtures_trip37 as fx  # noqa: E402

mc, TMP = fx.sandbox("97", ("CANARIES_97_OUT",))
s97 = fx.load("97_headline_checks.py", "s97")
s97.OUT = TMP / "out"
s97.CACHE = mc.CACHE_DIR
check = fx.Check()

EMPS = list(range(1, 241))
tier = lambda e: e % 4                                          # noqa: E731
size = lambda e: 4.2 if tier(e) == 3 else 1 + (e // 4) % 5     # noqa: E731
fx.install_score(mc, EMPS, tier, size, Path(mc.SHARE))
HIGH = {e for e in EMPS if tier(e) == 3}
FALL, FALL_F, DRIFT = float(np.log(0.85)), float(np.log(0.88)), -0.02
MONTHS = fx.months()
LAM = {"22-25": 8, "26-30": 9, "31-34": 7, "35-40": 8, "41-49": 10, "50+": 12}


def industry(e: int, kind: str) -> str:
    if kind == "confound":
        return "100" if (e in HIGH) == (e % 5 != 0) else "200"
    return "100" if (e * 7) % 3 == 0 else "200"


def levhi(e: int, kind: str) -> bool:
    if kind == "confound":
        return (e in HIGH) == (e % 6 != 0)
    return (e * 11) % 4 < 2


E, B, S, M = np.meshgrid(EMPS, list(LAM), ["1", "2"], MONTHS, indexing="ij")
GRID = pd.DataFrame({"employer_id": E.ravel(), "age_group": B.ravel(),
                     "gender": S.ravel(), "year_month": M.ravel()})


def world(kind: str) -> pd.DataFrame:
    d = GRID
    lam = (d["age_group"].map(LAM) * d["employer_id"].map(size)).to_numpy(float)
    hi = d["employer_id"].isin(HIGH).to_numpy()
    young = d["age_group"].isin(["22-25", "26-30"]).to_numpy()
    y22 = (d["age_group"] == "22-25").to_numpy()
    fem = (d["gender"] == "2").to_numpy()
    later = (d["year_month"] >= "2024-01").to_numpy()
    ym = d["year_month"]
    t = ((ym.str.slice(0, 4).astype(int) - 2021) * 12
         + ym.str.slice(5, 7).astype(int) - 1).to_numpy()
    pre = (ym <= "2022-11").to_numpy()
    if kind in ("ai", "drift"):
        lam = lam * np.exp(FALL * (hi & young & later))
        lam = lam * np.exp(FALL_F * (hi & y22 & fem & later))
        if kind == "drift":
            lam = lam * np.exp(DRIFT * (t - 12) * (hi & y22 & fem & pre))
    else:
        ind = d["employer_id"].map(lambda e: industry(e, kind)).to_numpy()
        lev = d["employer_id"].map(lambda e: levhi(e, kind)).to_numpy()
        lam = lam * np.exp(FALL_F * ((ind == "100") & y22 & fem & later))
        lam = lam * np.exp(FALL * (lev & young & later))
    out = d.copy()
    out["n_emp"] = fx.poisson_same_noise(lam, 97)
    return out


def install(kind: str) -> pd.DataFrame:
    sx = world(kind)
    fx.write_by_year(sx, mc.CACHE_DIR, "L_counts_sex")
    st = (sx.groupby(["employer_id", "year_month", "age_group"])["n_emp"]
          .sum().reset_index())
    fx.write_by_year(st, mc.CACHE_DIR, "L_counts")
    pd.DataFrame({"employer_id": EMPS,
                  "ind3": [industry(e, kind) for e in EMPS],
                  "source": "Ftg_2019"}).to_parquet(
        mc.CACHE_DIR / "I_industry_key.parquet", index=False)
    lev = pd.DataFrame({"employer_id": [str(e) for e in EMPS],
                        "lev": [0.9 if levhi(e, kind) else 0.3 for e in EMPS]})
    s97.leverage = lambda s73, _l=lev: _l.copy()
    return sx


def hand_female(sx: pd.DataFrame, high: set) -> float:
    """The female differential's contrast by hand: the young-to-old log
    ratio of women minus men, high minus low, later minus interim."""
    out = {}
    for g in ("1", "2"):
        c = sx[sx["gender"] == g]
        out[g] = fx.triple_diff(c, high, ["22-25"], ["31-34", "35-40",
                                                     "41-49", "50+"])
    return out["2"] - out["1"]


_load_modules = s97.load_modules


def _lm():
    """73's coverage gate is 500 employers; this world has 240."""
    r = _load_modules()
    r[3].MIN_FIRMS = 50
    return r


s97.load_modules = _lm
s82, s61, s67, s73, s78, s80, l47, l70, j47 = s97.load_modules()
for m_ in (s82, s61, s67, s73, s78, s80, l47, l70, j47):
    m_.OUT, m_.CACHE = s97.OUT, mc.CACHE_DIR
EXPO = s82.build_exposure(l47, l70, j47)["exposure"]
check("the planted tier is the top quartile",
      set(EXPO.loc[EXPO["fq"] == 4, "employer_id"]) == HIGH)

print("\n--- the stock gate stops on Table 1's numbers ---")
install("ai")
counts = s97.load_counts("L_counts", s61.PANEL_YEARS, s97.COUNT_COLS)
s97.ROWS.clear(); s97.FAILURES.clear()
try:
    s97.stock_gate(counts, "22-25", EXPO, s61, s78, j47); stopped = False
except SystemExit:
    stopped = True
check("the stock gate STOPS against Table 1", stopped)
GATE_KEEP, SEX_KEEP = s97.GATE, s97.SEX_GATE

V = {}
for kind in ("ai", "drift", "confound"):
    print(f"\n=== the {kind.upper()} world ===")
    sx = install(kind)
    if kind == "ai":
        hf = hand_female(sx, HIGH)
        check("by hand (AI): the female differential is the planted one",
              abs(hf - FALL_F) < 0.04, f"{hf:+.4f} against {FALL_F:+.4f}")
    if kind == "confound":
        hf = hand_female(sx, HIGH)
        check("by hand (confound): the naive female contrast is negative",
              hf < -0.04, f"{hf:+.4f}")
    s97.check = lambda label, got, want: []       # gates pointed at this world
    s97.ROWS.clear(); s97.FAILURES.clear(); s97.NOTES.clear()
    counts = s97.load_counts("L_counts", s61.PANEL_YEARS, s97.COUNT_COLS)
    lev = s97.leverage(s73)
    lmap = dict(zip(s73.norm_id(lev["employer_id"]), lev["lev"]))
    for band in s97.BANDS:
        b0 = s97.stock_gate(counts, band, EXPO, s61, s78, j47)
        if band == "22-25":
            s97.part_q(b0, EXPO, s78, j47)
        s97.part_k(b0, band, lmap, s73, s78, j47)
    sex = s97.load_counts("L_counts_sex", s61.PANEL_YEARS, s97.SEX_COLS)
    bsex = s97.sex_gate(sex, EXPO, s67, s78, j47)
    s97.part_p(bsex, j47, s73, s80)
    s97.part_i(bsex, s73, s80, s78, j47)
    check(f"{kind}: no fit failed", not s97.FAILURES, "; ".join(s97.FAILURES))
    V[kind] = "\n".join(s97.verdicts())
    print(V[kind])
    g = lambda *k: s97.get(*k)                                  # noqa: E731
    if kind == "ai":
        fd, sfd = g("G", "sex_gate", "22-25", "hyf_tau")
        check("AI: the sex gate fit recovers the planted differential",
              abs(fd - FALL_F) < 0.04, f"{fd:+.4f} ({sfd:.4f})")
        # the stock step at 22-25 is FALL for men and FALL + FALL_F for
        # women, so the planted stock contrast is computed by hand
        st = (sx.groupby(["employer_id", "year_month", "age_group"])
              ["n_emp"].sum().reset_index())
        hk = fx.triple_diff(st, HIGH, ["22-25"],
                            ["31-34", "35-40", "41-49", "50+"])
        k1, _ = g("K", "leverage", "22-25", "hy_tau")
        check("AI: tau with leverage in matches the hand contrast",
              abs(k1 - hk) < 0.02, f"{k1:+.4f} against {hk:+.4f}")
        TREND_AI = g("P", "drift", "22-25", "trend_x_hyf")[0]
        for k in (2, 3):
            qk, sq = g("Q", "quartiles_vs_q1", "22-25", f"q{k}_vs_q1_tau")
            check(f"AI: Q{k} against Q1 is about zero", abs(qk) < 0.03,
                  f"{qk:+.4f} ({sq:.4f})")
        q4, _ = g("Q", "quartiles_vs_q1", "22-25", "q4_vs_q1_tau")
        check("AI: Q4 against Q1 is the planted contrast",
              abs(q4 - hk) < 0.03, f"{q4:+.4f} against {hk:+.4f}")
        zc, zs = g("Q", "continuous_per_sd", "22-25", "z_tau")
        check("AI: the continuous score per SD is negative and significant",
              zc < -1.96 * zs, f"{zc:+.4f} ({zs:.4f})")
        de, se_e = g("P", "drift", "22-25", "trend_x_hyf")
        di, se_i = g("P", "drift_indcl", "22-25", "trend_x_hyf")
        check("AI: the industry-clustered drift has the same coefficient "
              "and its own SE", abs(de - di) < 1e-9 and se_i != se_e,
              f"SE employer {se_e:.5f}, industry {se_i:.5f}")
        check("AI: K1, P1 and I1 all pass",
              "CREDIT DOES NOT CARRY TAU" in V[kind]
              and "FLAT BEFORE THE LAUNCH" in V[kind]
              and "NOT AN INDUSTRY SHOCK" in V[kind], V[kind])
        pq = [r for r in s97.ROWS if r["part"] == "P"
              and r["spec"].startswith("path")
              and r["term"].endswith("_x_hyf")]
        check("AI: the quarterly path covers 2021Q1 to 2022Q4 less the "
              "reference, under both clusterings", len(pq) == 14, str(len(pq)))
        tr = [r for r in s97.ROWS if r["term"] == "hyf_tau"][0]
        check("tau rows carry the covariance pieces and the SE follows",
              abs(np.sqrt(tr["var_post"] + tr["var_interim"]
                          - 2 * tr["cov_post_interim"]) - tr["se"]) < 1e-12)
        s97.GATE = {b: {"post": g("G", "gate", b, "hy_post"),
                        "tau": g("G", "gate", b, "hy_tau")}
                    for b in s97.BANDS}
        s97.SEX_GATE = {"post": g("G", "sex_gate", "22-25", "hyf_post"),
                        "tau": g("G", "sex_gate", "22-25", "hyf_tau")}
        AI_GATE, AI_SEX = s97.GATE, s97.SEX_GATE
    if kind == "drift":
        tr, st = g("P", "drift", "22-25", "trend_x_hyf")
        # the two worlds share every uniform, so the difference of the two
        # trends is the planted drift with the noise cancelled
        check("drift: the trend moves by the planted drift",
              abs((tr - TREND_AI) - DRIFT) < 0.002,
              f"{tr:+.5f} against the AI world's {TREND_AI:+.5f}")
        check("drift: P1 fails", "P1 NOT MET" in V[kind])
    if kind == "confound":
        check("confound: I1 fails", "I1 NOT MET" in V[kind], V[kind])
        check("confound: K1 fails", "K1 NOT MET" in V[kind], V[kind])
        i0, _ = g("I", "base_industry_sample", "22-25", "hyf_tau")
        i1, si = g("I", "industry_age_sex_month", "22-25", "hyf_tau")
        check("confound: industry effects remove the naive differential",
              i0 < -0.03 and abs(i1) < 0.03, f"base {i0:+.4f}, ind {i1:+.4f}")

print("\n--- main(), end to end, in the AI world ---")
install("ai")
s97.check = fx.load("97_headline_checks.py", "s97_fresh").check
s97.GATE, s97.SEX_GATE = AI_GATE, AI_SEX
s97.ROWS.clear(); s97.FAILURES.clear(); s97.NOTES.clear()
s97.DONE = s97.PLANNED = 0
_stdout = sys.stdout
rc = s97.main()
sys.stdout = _stdout
out = pd.read_csv(s97.OUT / "headline_checks.csv")
summ = (s97.OUT / "97_summary.txt").read_text()
check("main() returns 0", rc == 0, f"rc {rc}; {s97.FAILURES}")
check("every attempted fit came back", s97.DONE == s97.PLANNED,
      f"{s97.DONE} of {s97.PLANNED}")
check("the summary names what was recovered rather than refitted",
      "NOT REFITTED" in summ and "l58" in summ)
check("no employer count under the floor",
      bool(((out["n_firms"].isna()) | (out["n_firms"] >= 5)).all()))
check("no identifier column is exported",
      not any(c in out.columns for c in ("employer_id", "ind3", "fe_emp_t")))
check.done()
