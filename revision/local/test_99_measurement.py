#!/usr/bin/env python3
"""
test_99_measurement.py -- the dry run of script 99 (lane 37a, measurement).

THE MECHANISM PLANTED. The Editor's worry is that the headline counts pass
through an OCCUPATION-SELECTED intermediate: a panel that keeps only the
workers the occupation register codes. If exposed employers' young
workers were uncoded more often after 2023, such a panel would show a
decline that is not in the declarations. So the fixture draws the raw
declarations (employer x band x sex x month, with a true young decline of
FALL in top-quartile employers from January 2024), then an uncoded share
that is higher for exposed employers' young after 2023, and two worlds:

  CLEAN     the headline L_counts ARE the raw counts.
  SELECTED  the headline L_counts are the CODED counts only (the
            occupation-selected intermediate).

Part R must say "the headline counts are the raw counts" in the first and
refuse it in the second, with tau on the rebuilt counts equal to the
planted FALL in both and the selected headline tau more negative.

Part M: the raw counts are drawn as baseline incumbents (at the employer
in November 2022) plus others, with the planted decline carried entirely
by the others (fewer new matches). tau on the others must be the decline
and tau on the baseline incumbents about zero.

Part D cannot run its SQL here, so the test checks the SQL the builder
writes (the code-source order for a year with its own register and for
one without) and the Python that turns a reconciliation cache into the
export: the unlinked kept as their own category, shares summing to one
within month x band x status, the production non-match rate, the floor.

    python3 revision/local/test_99_measurement.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _fixtures_trip37 as fx  # noqa: E402

mc, TMP = fx.sandbox("99", ("CANARIES_99_OUT",))
s99 = fx.load("99_measurement.py", "s99")
s99.OUT = TMP / "out"
s99.CACHE = mc.CACHE_DIR
check = fx.Check()

EMPS = list(range(1, 241))
tier = lambda e: e % 4                                          # noqa: E731
size = lambda e: 4.2 if tier(e) == 3 else 1 + (e // 4) % 5     # noqa: E731
fx.install_score(mc, EMPS, tier, size, Path(mc.SHARE))
HIGH = {e for e in EMPS if tier(e) == 3}
FALL = float(np.log(0.85))
MONTHS = fx.months()
LAM = {"22-25": 8, "26-30": 9, "31-34": 7, "35-40": 8, "41-49": 10, "50+": 12}
E, B, S, M = np.meshgrid(EMPS, list(LAM), ["1", "2"], MONTHS, indexing="ij")
G = pd.DataFrame({"employer_id": E.ravel(), "age_group": B.ravel(),
                  "gender": S.ravel(), "year_month": M.ravel()})


def draw() -> pd.DataFrame:
    lam = (G["age_group"].map(LAM) * G["employer_id"].map(size)).to_numpy(float)
    hi = G["employer_id"].isin(HIGH).to_numpy()
    young = (G["age_group"] == "22-25").to_numpy()
    later = (G["year_month"] >= "2024-01").to_numpy()
    after_base = (G["year_month"] > "2022-11").to_numpy()
    # baseline incumbents: 70 per cent of the rate; the others carry the
    # whole planted decline (fewer new matches in exposed employers)
    share_base = np.where(after_base, 0.7, 0.6)
    d = G.copy()
    d["base"] = fx.poisson_same_noise(lam * share_base, 991)
    lo = lam * (1 - share_base)
    # the others' rate is cut so that the TOTAL young rate falls by FALL
    lo_planted = np.where(hi & young & later,
                          lam * np.exp(FALL) - lam * share_base, lo)
    d["other"] = fx.poisson_same_noise(lo_planted, 992)
    d["n_raw"] = d["base"] + d["other"]
    # uncoded workers: more of them among exposed employers' young in 2024-25
    p = np.where(hi & young & later, 0.20, 0.05)
    u = np.random.default_rng(993).binomial(d["n_raw"].to_numpy(), p)
    d["n_coded"] = d["n_raw"] - u
    return d


D = draw()
keys = ["employer_id", "year_month", "age_group"]


def install(world: str) -> None:
    col = "n_raw" if world == "clean" else "n_coded"
    fx.write_by_year(D[keys + ["gender"]].assign(n_emp=D["n_raw"]),
                     mc.CACHE_DIR, "R_counts_raw")
    hd = D.groupby(keys)[col].sum().reset_index().rename(columns={col: "n_emp"})
    fx.write_by_year(hd, mc.CACHE_DIR, "L_counts")
    hs = (D.groupby(keys + ["gender"])[col].sum().reset_index()
          .rename(columns={col: "n_emp"}))
    fx.write_by_year(hs, mc.CACHE_DIR, "L_counts_sex")
    bm = pd.concat([
        D.groupby(keys)["base"].sum().reset_index().rename(
            columns={"base": "n_emp"}).assign(at_base="baseline"),
        D.groupby(keys)["other"].sum().reset_index().rename(
            columns={"other": "n_emp"}).assign(at_base="other")])
    fx.write_by_year(bm[s99.MATCH_COLS], mc.CACHE_DIR, "L_counts_basematch")


old = ["31-34", "35-40", "41-49", "50+"]
raw_stock = D.groupby(keys)["n_raw"].sum().reset_index().rename(
    columns={"n_raw": "n_emp"})
sel_stock = D.groupby(keys)["n_coded"].sum().reset_index().rename(
    columns={"n_coded": "n_emp"})
h_raw = fx.triple_diff(raw_stock, HIGH, ["22-25"], old)
h_sel = fx.triple_diff(sel_stock, HIGH, ["22-25"], old)
check("by hand: the raw counts carry the planted fall",
      abs(h_raw - FALL) < 0.03, f"{h_raw:+.4f} against {FALL:+.4f}")
check("by hand: the occupation-selected counts overstate it",
      h_sel < h_raw - 0.08, f"{h_sel:+.4f}")

s82, s61, s78, l47, l70, j47 = s99.load_modules()
for m_ in (s82, s61, s78, l47, l70, j47):
    m_.OUT, m_.CACHE = s99.OUT, mc.CACHE_DIR
EXPO = s82.build_exposure(l47, l70, j47)["exposure"]


def world_gate() -> dict:
    """The gate pointed at this world's headline panel."""
    s99.EST.clear()
    lc = s99.load_cache("L_counts", s61.PANEL_YEARS, s99.COUNT_COLS)
    s99.headline_fit(lc, EXPO, s61, s78, j47, "G", "probe")
    return {"post": s99.get("G", "probe", "post"),
            "tau": s99.get("G", "probe", "tau")}


SUMM = {}
for world in ("clean", "selected"):
    print(f"\n=== the {world.upper()} world ===")
    install(world)
    s99.GATE = world_gate()
    s99.EST.clear(); s99.FAILURES.clear(); s99.NOTES.clear()
    s99.DONE = s99.PLANNED = 0
    s99.PARTS = "RM"
    _stdout = sys.stdout
    rc = s99.main()
    sys.stdout = _stdout
    SUMM[world] = (s99.OUT / "99_summary.txt").read_text()
    est = pd.read_csv(s99.OUT / "measurement_estimates.csv")
    check(f"{world}: main() returns 0", rc == 0, f"{rc}; {s99.FAILURES}")
    check(f"{world}: every attempted fit came back",
          s99.DONE == s99.PLANNED, f"{s99.DONE} of {s99.PLANNED}")
    t_r, s_r = s99.get("R", "rebuilt_22_25", "tau")
    check(f"{world}: tau on the rebuilt counts is the planted fall",
          abs(t_r - FALL) < 0.03, f"{t_r:+.4f} ({s_r:.4f})")
    share = s99.get("R", "all_sexes_2024", "share_identical")[0]
    mx = s99.get("R", "all_sexes_2024", "max_abs_diff")[0]
    tb, _ = s99.get("M", "baseline_22_25", "tau")
    to, so = s99.get("M", "other_22_25", "tau")
    check(f"{world}: M, the baseline incumbents do not carry the decline",
          abs(tb) < 0.03, f"{tb:+.4f}")
    check(f"{world}: M, the others carry it", to < -0.1 and to < -1.96 * so,
          f"{to:+.4f} ({so:.4f})")
    check(f"{world}: the score ignores post-2019 codes",
          s99.get("R", "score_without_post2019_codes",
                  "employers_changing_quartile")[0] == 0)
    check(f"{world}: no identifier column is exported",
          not any(c in est.columns for c in ("employer_id", "person_id")))
    tot = pd.read_csv(s99.OUT / "measurement_totals.csv")
    check(f"{world}: national totals by band and month from both sources",
          set(tot["source"]) == {"rebuilt", "headline"}
          and tot["age_group"].nunique() == 6)
    if world == "clean":
        check("clean: every cell identical", share == 1.0 and mx == 0,
              f"{share}, {mx}")
        check("clean: R1 passes",
              "THE HEADLINE COUNTS ARE THE RAW COUNTS" in SUMM[world])
    else:
        check("selected: the comparison finds the selected cells",
              share < 1.0 and mx > 0, f"share {share:.4f}, max {mx:.0f}")
        g_sel, _ = s99.get("G", "gate_22_25", "tau")
        check("selected: the headline tau is more negative than the rebuilt",
              g_sel < t_r - 0.05, f"{g_sel:+.4f} vs {t_r:+.4f}")
        check("selected: R1 is refused", "R1 NOT MET" in SUMM[world])

print("\n--- the gate stops the script on a miss ---")
s99.GATE = {"post": (-0.0578, 0.0155), "tau": (-0.0399, 0.0102)}
s99.EST.clear(); s99.FAILURES.clear()
_stdout = sys.stdout
try:
    s99.main(); stopped = False
except SystemExit:
    stopped = True
sys.stdout = _stdout
check("the gate STOPS against Table 1's numbers", stopped)

print("\n--- Part D: the SQL the builder writes ---")
plan = {y: ("Ssyk4_2012_J16", "SsykAr_J16") for y in s99.CODE_YEARS}
captured = {}
pd_read_sql = pd.read_sql


def _capture(q, conn):
    captured["q"] = q
    return pd.DataFrame()


s99.pd.read_sql = _capture
fake = object()
for y in (2020, 2024):
    captured.clear()
    s99.q_recon(y, fake, plan)
    q = captured["q"]
    if y == 2020:
        i_cur = q.find("THEN 'current'")
        i_19 = q.find("THEN 'carried_earlier'")
        check("2020: own register first, then 2019 as 'earlier'",
              0 < i_cur < i_19 and "carried_2023" not in q
              and "Arb_AGIIndivid201901" in q, "order in the CASE")
    else:
        i23, i22 = q.find("THEN 'carried_2023'"), q.find("THEN 'carried_2022'")
        check("2024: no 'current'; carried from 2023, then 2022",
              "'current'" not in q and 0 < i23 < i22
              and "Arb_AGIIndivid202301" in q)
    check(f"{y}: '****' and empty codes are treated as missing",
          "LEFT(LTRIM(CAST(" in q and "<> '*'" in q)
s99.pd.read_sql = pd_read_sql

print("\n--- Part D: the export from a reconciliation cache ---")
rows = []
for per in ("202401", "202402"):
    for band in ("22-25", "50-69", "unknown"):
        for st in ("incumbent", "new_match", "entrant"):
            for linked, cat, n in (("yes", "carried_2023", 800),
                                   ("yes", "carried_2022", 60),
                                   ("yes", "none", 90), ("no", "none", 40),
                                   ("yes", "carried_earlier", 3)):
                rows.append((per, band, st, linked, cat, "0", n))
rc_ = pd.DataFrame(rows, columns=s99.RECON_COLS)
for y in s99.RECON_YEARS:
    (rc_.assign(period=rc_["period"].str.replace("2024", str(y)))
     .to_parquet(mc.CACHE_DIR / f"R_recon_{y}.parquet", index=False))
s99.EST.clear(); s99.NOTES.clear(); s99.FAILURES.clear()
s99.part_d(s82)
R = pd.read_csv(s99.OUT / "measurement_reconciliation.csv")
sums = R.groupby(["period", "band", "status"])["share_of_month_band_status"].sum()
check("D: the unlinked are their own category",
      "no_age_sex" in set(R["category"]))
check("D: counts below five are suppressed with their share",
      R.loc[R["codecat"] == "carried_earlier", "person_months"].isna().all()
      and R.loc[R["codecat"] == "carried_earlier",
                "share_of_month_band_status"].isna().all())
check("D: the shown shares sum to one less the suppressed cells",
      bool(((sums > 0.99) & (sums <= 1.0 + 1e-9)).all()))
nm24 = s99.get("D", "year_2024", "production_nonmatch_share")[0]
check("D: production non-match 2024 = (none + unlinked none + earlier) / all",
      abs(nm24 - (90 + 40 + 3) / (800 + 60 + 90 + 40 + 3)) < 1e-12,
      f"{nm24:.4f}")
nm22 = s99.get("D", "year_2022", "production_nonmatch_share")[0]
check("D: in 2022 the production cascade counts only the own register",
      abs(nm22 - 1.0) < 1e-12, f"{nm22:.4f} (the cache has no 'current')")
check.done()
