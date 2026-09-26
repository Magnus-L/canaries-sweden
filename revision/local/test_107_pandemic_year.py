#!/usr/bin/env python3
"""
test_107_pandemic_year.py -- the dry run of script 107 (lane 39c).

THE MECHANISM PLANTED. 97's employer x age band x sex x month grid extended
to January 2019 (240 employers, four DAIOE tiers, tier 3 the top quartile),
in three worlds that share every uniform draw:

  A  young workers in top employers fall by FALL from January 2024 and
     young women there by FALL_F more; nothing before 2024 moves. The
     female placebos read their hand triple differences and are near zero;
     the female drift from 2019 is flat; the pooled trend with the
     pandemic-year term is flat and the term near zero.
  B  world A plus a permanent step from March 2021: the young at top
     employers DROP lower and young women there DROP_F lower still. The
     36-month placebo (2021-22 against 2020) reads both steps by hand; the
     24-month placebo (2022 against 2021) reads far less; the female drift
     from 2019 is negative and NOT FLAT.
  C  world A plus a dip of DIP in the young at top employers during March
     2020 to February 2021 only: the pandemic-year term reads the dip and
     the trend stays near zero.

The early sex counts: the first run finds no L_counts_sex_2019/_2020 cache
and must pull them through the replaceable pull_sex (the fake serves the
planted world) and cache them; a second run must read the caches and not
pull. Part R's counts reproduce the planted world's sums exactly. main()
runs end to end with no SQL, six fits, three export files, the summary's
blocks, no identifier out, no cell under the floor.

    python3 revision/local/test_107_pandemic_year.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _fixtures_trip37 as fx  # noqa: E402

mc, TMP = fx.sandbox("107", ("CANARIES_107_OUT",))
s107 = fx.load("107_pandemic_year.py", "s107")
s107.OUT = TMP / "out"
s107.CACHE = mc.CACHE_DIR
check = fx.Check()
SHARE = Path(mc.SHARE)

EMPS = list(range(1, 241))
tier = lambda e: e % 4                                          # noqa: E731
size = lambda e: 4.2 if tier(e) == 3 else 1 + (e // 4) % 5     # noqa: E731
fx.install_score(mc, EMPS, tier, size, SHARE)
HIGH = {e for e in EMPS if tier(e) == 3}
FALL, FALL_F = float(np.log(0.85)), float(np.log(0.88))
DROP, DROP_F, DIP = 0.06, 0.08, 0.10
MONTHS = [f"{y}-{m:02d}" for y in range(2019, 2026) for m in range(1, 13 if y < 2025 else 7)]
LAM = {"22-25": 8, "26-30": 9, "31-34": 7, "35-40": 8, "41-49": 10, "50+": 12}
OLD = ["31-34", "35-40", "41-49", "50+"]

E, Bd, S, M = np.meshgrid(EMPS, list(LAM), ["1", "2"], MONTHS, indexing="ij")
GRID = pd.DataFrame({"employer_id": E.ravel(), "age_group": Bd.ravel(),
                     "gender": S.ravel(), "year_month": M.ravel()})


def world(kind: str) -> pd.DataFrame:
    d = GRID
    lam = (d["age_group"].map(LAM) * d["employer_id"].map(size)).to_numpy(float)
    hi = d["employer_id"].isin(HIGH).to_numpy()
    y22 = (d["age_group"] == "22-25").to_numpy()
    fem = (d["gender"] == "2").to_numpy()
    ym = d["year_month"].to_numpy()
    later = ym >= "2024-01"
    lam = lam * np.exp(FALL * (hi & y22 & later)) * np.exp(FALL_F * (hi & y22 & fem & later))
    if kind == "B":
        step = ym >= "2021-03"
        lam = lam * np.exp(-DROP * (hi & y22 & step)) * np.exp(-DROP_F * (hi & y22 & fem & step))
    if kind == "C":
        dip = (ym >= "2020-03") & (ym <= "2021-02")
        lam = lam * np.exp(-DIP * (hi & y22 & dip))
    out = d.copy()
    out["n_emp"] = fx.poisson_same_noise(lam, 107)
    return out


def install(kind: str, early_caches: bool) -> tuple:
    sx = world(kind)
    for y in range(2019, 2026):
        part = sx[sx["year_month"].str.slice(0, 4) == str(y)]
        if y >= 2021 or early_caches:
            part.to_parquet(mc.CACHE_DIR / f"L_counts_sex_{y}.parquet", index=False)
        else:
            (mc.CACHE_DIR / f"L_counts_sex_{y}.parquet").unlink(missing_ok=True)
    st = sx.groupby(["employer_id", "year_month", "age_group"])["n_emp"].sum().reset_index()
    for y in range(2019, 2026):
        st[st["year_month"].str.slice(0, 4) == str(y)].to_parquet(
            mc.CACHE_DIR / f"L_counts_{y}.parquet", index=False)
    return sx, st


PULLS = []


def make_fake_pull(sx):
    def fake(year, conn, s67):
        PULLS.append(year)
        return sx[sx["year_month"].str.slice(0, 4) == str(year)][s107.SEX_COLS].copy()
    return fake


def hand(st, S, band="22-25") -> float:
    c = st[st["year_month"] < "2022-12"]
    return fx.triple_diff(c, HIGH, [band], OLD, post_from=s107.shift_ym("2024-01", S),
                          interim_from=s107.shift_ym("2022-12", S))


def hand_female(sx, S) -> float:
    out = {}
    c = sx[sx["year_month"] < "2022-12"]
    for g in ("1", "2"):
        out[g] = fx.triple_diff(c[c["gender"] == g], HIGH, ["22-25"], OLD,
                                post_from=s107.shift_ym("2024-01", S),
                                interim_from=s107.shift_ym("2022-12", S))
    return out["2"] - out["1"]


def reset() -> None:
    s107.ROWS.clear(); s107.PATH.clear(); s107.FAILURES.clear(); s107.NOTES.clear()
    s107.DONE = s107.PLANNED = 0
    PULLS.clear()


s107.open_conn = lambda: object()
s82, s61, s67, s78, l47, l70, j47 = s107.load_modules()
for m_ in (s82, s61, s67, s78, l47, l70, j47):
    m_.OUT, m_.CACHE = s107.OUT, mc.CACHE_DIR

R = {}
for name in ("A", "B"):
    print(f"\n=== world {name} ===")
    sx, st = install(name, early_caches=False)
    reset()
    s107.pull_sex = make_fake_pull(sx)
    EXPO = s82.build_exposure(l47, l70, j47, audit=False)["exposure"]
    check(f"{name}: the planted tier is the top quartile", set(EXPO.loc[EXPO["fq"] == 4, "employer_id"]) == HIGH)
    early = s107.early_sex_counts(s67)
    check(f"{name}: the early sex counts are pulled for 2019 and 2020 and cached",
          sorted(PULLS) == [2019, 2020] and all((mc.CACHE_DIR / f"L_counts_sex_{y}.parquet").exists() for y in (2019, 2020))
          and len(early) == len(sx[sx["year_month"] < "2021-01"]))
    PULLS.clear()
    early2 = s107.early_sex_counts(s67)
    check(f"{name}: a second call reads the caches and does not pull", PULLS == [] and len(early2) == len(early))
    sexc = s107.load_counts("L_counts_sex", s61.PANEL_YEARS, s107.SEX_COLS)
    if name == "A":
        try:
            s107.sex_gate(sexc, EXPO, s67, s78, j47)
            stopped = False
        except SystemExit:
            stopped = True
        check("the sex gate STOPS against Table 1", stopped)
        W_SEX = {"post": s107.get("G", "sex_gate", "22-25", "hyf_post"), "tau": s107.get("G", "sex_gate", "22-25", "hyf_tau")}
        reset()
    s107.check = lambda *a, **k: []
    sex_ext = pd.concat([early, sexc], ignore_index=True)
    b = s107.sex_pre_panel(sex_ext, EXPO, s67, s78, j47)
    s107.female_placebos(b, j47)
    s107.female_drift(b, j47)
    s107.female_path(b, j47)
    counts = s107.load_counts("L_counts", [2019, 2020] + s61.PANEL_YEARS, s107.COUNT_COLS)
    s107.pooled_pandemic(counts, EXPO, s78, j47)
    check(f"{name}: no fit failed", not s107.FAILURES, "; ".join(s107.FAILURES))
    g = lambda *k: s107.get(*k)                                  # noqa: E731
    r = {"f36": g("F", "placebo_shift_36", "22-25", "hyf_tau"), "f24": g("F", "placebo_shift_24", "22-25", "hyf_tau"),
         "y36": g("F", "placebo_shift_36", "22-25", "hy_tau"), "y24": g("F", "placebo_shift_24", "22-25", "hy_tau"),
         "fd": g("F", "drift_from_2019", "22-25", s107.FTREND), "yd": g("F", "drift_from_2019", "22-25", s107.TREND),
         "pt": g("P", "drift_from_2019_pandemic", "22-25", s107.TREND), "pp": g("P", "drift_from_2019_pandemic", "22-25", s107.PANT),
         "hf36": hand_female(sx, 36), "hf24": hand_female(sx, 24), "h36": hand(st, 36), "h24": hand(st, 24)}
    R[name] = r
    print(f"  female placebo S36 {r['f36'][0]:+.4f} ({r['f36'][1]:.4f}) hand {r['hf36']:+.4f}; S24 {r['f24'][0]:+.4f} hand {r['hf24']:+.4f}; "
          f"young men S36 {r['y36'][0]:+.4f} hand {r['h36']:+.4f}; female drift {r['fd'][0]:+.6f} ({r['fd'][1]:.6f}); "
          f"pooled trend {r['pt'][0]:+.6f} ({r['pt'][1]:.6f}) pandemic {r['pp'][0]:+.4f} ({r['pp'][1]:.4f})")
    check(f"{name}: the female placebos read their hand triple differences (within 0.04)",
          abs(r["f36"][0] - r["hf36"]) < 0.04 and abs(r["f24"][0] - r["hf24"]) < 0.04)
    check(f"{name}: the female path carries fifteen quarters x three terms plus the reference rows",
          len(s107.PATH) == 15 * 3 + 3 and {p["term"] for p in s107.PATH} == {"hy", "hf", "hyf"})
    if name == "A":
        check("A: the female placebos and drift are near zero (under 2.5 SE)",
              all(abs(c) < 2.5 * s for c, s in (r["f36"], r["f24"], r["fd"])))
        check("A: the pooled trend and the pandemic-year term are near zero (under 2.5 SE)",
              all(abs(c) < 2.5 * s for c, s in (r["pt"], r["pp"])))
    else:
        check("B: the 36-month female placebo reads the step (below -0.03) and the 24-month one far less",
              r["f36"][0] < -0.03 and abs(r["f24"][0]) < 0.5 * abs(r["f36"][0]), f"{r['f36'][0]:+.4f} vs {r['f24'][0]:+.4f}")
        check("B: the female drift from 2019 is negative and NOT FLAT",
              r["fd"][0] < 0 and abs(r["fd"][0]) > 2 * r["fd"][1], f"t {r['fd'][0] / r['fd'][1]:+.2f}")
        # A permanent step is read as a trend; the window just before the
        # step then sits above the fitted line, so the pandemic-year term is
        # positive, not a spurious dip. World C tests the term on a real dip.
        check("B: the pooled trend is negative and NOT FLAT, and the pandemic-year term does not manufacture a dip",
              r["pt"][0] < 0 and abs(r["pt"][0]) > 2 * r["pt"][1] and r["pp"][0] > -0.5 * DROP,
              f"trend t {r['pt'][0] / r['pt'][1]:+.2f}, pandemic {r['pp'][0]:+.4f}")
    # R: the counts reproduce the world
    skel = s61.build_skeleton(counts[counts["year_month"] >= s61.PANEL_FROM], "22-25", j47)
    employers = set(s78.with_exposure(skel, EXPO)["employer_id"].unique())
    q = s107.quarterly_counts(counts, sex_ext, EXPO, employers)
    def cell(quarter, band, group, gender):
        return float(q[(q.quarter == quarter) & (q.band == band) & (q.group == group) & (q.gender == gender)]["n_emp"].iloc[0])
    def hand_cell(quarter, bands, group, gender):
        src = sx if gender != "all" else st
        d = src[src["employer_id"].isin(employers) & src["age_group"].isin(bands)]
        d = d[d["employer_id"].isin(HIGH)] if group == "top" else d[~d["employer_id"].isin(HIGH)]
        if gender != "all":
            d = d[d["gender"] == ("2" if gender == "women" else "1")]
        yq = quarter[:4] + "-" + {"Q1": "01", "Q2": "04", "Q3": "07", "Q4": "10"}[quarter[4:]]
        months = [f"{quarter[:4]}-{int(yq[5:]) + k:02d}" for k in range(3)]
        return float(d[d["year_month"].isin(months)]["n_emp"].sum())
    check(f"{name}: Part R reproduces the world's quarterly sums exactly (six cells checked)",
          all(abs(cell(qq, b_, gp, gd) - hand_cell(qq, [b_] if b_ == "22-25" else OLD, gp, gd)) < 1e-9
              for qq, b_, gp, gd in (("2020Q2", "22-25", "top", "all"), ("2020Q2", "31-69", "rest", "all"),
                                      ("2021Q1", "22-25", "top", "women"), ("2021Q1", "22-25", "rest", "men"),
                                      ("2025Q2", "31-69", "top", "women"), ("2019Q4", "22-25", "rest", "all"))))
    check(f"{name}: no identifier in the counts export and no cell under the floor",
          "employer_id" not in q.columns and bool((q["n_employers"] >= 5).all()))

print("\n=== world C: a dip in the pandemic year only ===")
sx, st = install("C", early_caches=True)
reset()
s107.pull_sex = make_fake_pull(sx)
EXPO = s82.build_exposure(l47, l70, j47, audit=False)["exposure"]
counts = s107.load_counts("L_counts", [2019, 2020] + s61.PANEL_YEARS, s107.COUNT_COLS)
s107.pooled_pandemic(counts, EXPO, s78, j47)
pt = s107.get("P", "drift_from_2019_pandemic", "22-25", s107.TREND)
pp = s107.get("P", "drift_from_2019_pandemic", "22-25", s107.PANT)
print(f"  trend {pt[0]:+.6f} ({pt[1]:.6f}); pandemic-year term {pp[0]:+.4f} ({pp[1]:.4f}) vs planted {-DIP:+.4f}")
check("C: the pandemic-year term reads the dip (within 0.04) and the trend stays near zero (under 2.5 SE)",
      abs(pp[0] - (-DIP)) < 0.04 and abs(pt[0]) < 2.5 * pt[1])

print("\n--- main(), end to end, in world A, caches present ---")
sx, st = install("A", early_caches=True)
reset()
s107.pull_sex = lambda year, conn, s67: (_ for _ in ()).throw(RuntimeError("must not pull when cached"))
s107.check = fx.load("107_pandemic_year.py", "s107_fresh").check
s107.SEX_GATE = W_SEX
_stdout = sys.stdout
rc = s107.main()
sys.stdout = _stdout
out = pd.read_csv(s107.OUT / "pandemic_year.csv"); path = pd.read_csv(s107.OUT / "female_prepath.csv")
qc = pd.read_csv(s107.OUT / "quarterly_counts.csv"); summ = (s107.OUT / "107_summary.txt").read_text()
check("main() returns 0", rc == 0, f"rc {rc}; {s107.FAILURES}")
check("six fits attempted and returned", s107.DONE == s107.PLANNED == 6, f"{s107.DONE} of {s107.PLANNED}")
check("the summary prints the gate, F1 to F3, P and R with the first-quarter ratios",
      "female differential tau" in summ and "F1. THE FEMALE PLACEBO" in summ and summ.count("windows: reference") == 2
      and "F2. THE FEMALE DRIFT" in summ and "F3. THE FEMALE QUARTERLY PATH" in summ and "P. THE POOLED DRIFT" in summ
      and "pandemic-year term" in summ and "R. WHICH GROUP MOVED" in summ and "2020Q1:" in summ and "NO FIT" not in summ)
check("the exports hold G, F and P rows, the path's three terms and the counts by sex",
      set(out["part"]) == {"G", "F", "P"} and set(path["term"]) == {"hy", "hf", "hyf"}
      and set(qc["gender"]) == {"all", "women", "men"})
check("no employer count under the floor and no identifier out",
      bool(((out["n_firms"].isna()) | (out["n_firms"] >= 5)).all())
      and not any(c in f.columns for f in (out, path, qc) for c in ("employer_id", "fe_emp_t", "fq")))
check.done()
