#!/usr/bin/env python3
"""
test_101_count_provenance.py -- the dry run of script 101 (lane 38c).

THE MECHANISM PLANTED, at the level of PERSONS, because the question is
how one person is counted by two constructions. Ordinary persons fill an
employer x band x sex x month grid. Three kinds of anomalous person are
then added, each exactly as the LEFT JOINs to the Individ vintages would
present them:

  multisex     two register rows, sex 1 and sex 2, one birth year: counted
               once by 47L's distinct pooled count (L_counts), once as a
               man and once as a woman by 67 (L_counts_sex) and by 99
  multigender  sex 1 and an invalid code: once in L_counts and in
               L_counts_sex (the invalid row is filtered), twice in 99's
               raw counts (which keep a '0' category)
  multibirth   two birth years in different bands, sex 1: counted in both
               bands by every construction

CLEAN world: the independent pooled rebuild equals L_counts; every cell
where the sex-split sum exceeds the pooled count is explained exactly by
the multisex persons in it. R2 and R3 must pass; the gap in person-months
equals the multisex person-months exactly; the estimation-panel cells are
explained and retained. BROKEN world, the same draws plus ten planted
person-months in L_counts_sex that no anomalous person explains and seven
in the raw pooled rebuild: R2 and R3 must be refused, with exactly ten
unexplained cells "with no anomalous person", and the raw pooled rebuild
refitted. Part F runs its gates (pointed at the world, as test_99 does)
and the two corrected fits; Part I's SQL cannot run here and is not
called (no SQL may be issued in a test world).

    python3 revision/local/test_101_count_provenance.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _fixtures_trip37 as fx  # noqa: E402

mc, TMP = fx.sandbox("101", ("CANARIES_101_OUT",))
s101 = fx.load("101_count_provenance.py", "s101")
s101.OUT = TMP / "out"
s101.CACHE = mc.CACHE_DIR
check = fx.Check()

EMPS = list(range(1, 241))
tier = lambda e: e % 4                                          # noqa: E731
size = lambda e: 4.2 if tier(e) == 3 else 1 + (e // 4) % 5     # noqa: E731
fx.install_score(mc, EMPS, tier, size, Path(mc.SHARE))
HIGH = {e for e in EMPS if tier(e) == 3}
FALL = float(np.log(0.85))
MONTHS = fx.months()
BANDS = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]
LAM = dict(zip(BANDS, (8, 9, 7, 8, 10, 12)))
KEYS = ["employer_id", "year_month", "age_group"]
E, B, S, M = np.meshgrid(EMPS, BANDS, ["1", "2"], MONTHS, indexing="ij")
G = pd.DataFrame({"employer_id": E.ravel(), "age_group": B.ravel(),
                  "gender": S.ravel(), "year_month": M.ravel()})
hi = G["employer_id"].isin(HIGH).to_numpy()
lam = (G["age_group"].map(LAM) * G["employer_id"].map(size)).to_numpy(float).copy()
lam *= np.where(hi & (G["age_group"] == "22-25").to_numpy()
                & (G["year_month"] >= "2024-01").to_numpy(), np.exp(FALL), 1)
G["n"] = fx.poisson_same_noise(lam, 1011)          # ordinary persons

rng = np.random.default_rng(1012)
EM = G[["employer_id", "year_month"]].drop_duplicates().to_numpy()
pick = lambda k: EM[rng.choice(len(EM), k, replace=False)]  # noqa: E731
N_MS, N_MG, N_MB = 400, 150, 120
ms = pd.DataFrame(pick(N_MS), columns=["employer_id", "year_month"]).assign(
    age_group=rng.choice(BANDS, N_MS))
mg = pd.DataFrame(pick(N_MG), columns=["employer_id", "year_month"]).assign(
    age_group=rng.choice(BANDS, N_MG))
mb = pd.DataFrame(pick(N_MB), columns=["employer_id", "year_month"])
mb["a1"] = rng.choice(BANDS[:3], N_MB)
mb["a2"] = rng.choice(BANDS[3:], N_MB)


def add(d: pd.DataFrame, rows: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Add one person-month per row of `rows` to the matching cells."""
    k = rows.groupby(cols).size().rename("plus")
    out = d.set_index(cols).join(k, how="outer").fillna({"plus": 0})
    out["n_emp"] = out["n_emp"].fillna(0) + out["plus"]
    return out.drop(columns="plus").reset_index()


base_sex = G.rename(columns={"n": "n_emp"})[KEYS + ["gender", "n_emp"]]
# L_counts_sex (67): multisex twice, multigender once (as 1), multibirth
# in both bands (as 1)
Ssex = add(base_sex, ms.assign(gender="1"), KEYS + ["gender"])
Ssex = add(Ssex, ms.assign(gender="2"), KEYS + ["gender"])
Ssex = add(Ssex, mg.assign(gender="1"), KEYS + ["gender"])
Ssex = add(Ssex, mb.rename(columns={"a1": "age_group"})[KEYS].assign(
    gender="1"), KEYS + ["gender"])
Ssex = add(Ssex, mb.rename(columns={"a2": "age_group"})[KEYS].assign(
    gender="1"), KEYS + ["gender"])
# L_counts (47L): every person once per band
Pool = base_sex.groupby(KEYS)["n_emp"].sum().reset_index()
for r in (ms, mg, mb.rename(columns={"a1": "age_group"})[KEYS],
          mb.rename(columns={"a2": "age_group"})[KEYS]):
    Pool = add(Pool, r[KEYS], KEYS)
# 99's raw counts: 67's plus the invalid-sex row of the multigender persons
R99 = add(Ssex, mg.assign(gender="0"), KEYS + ["gender"])
# the anomaly table, as q_anom returns it
an = []
for g_ in ("1", "2", "all"):
    an.append(ms.assign(gender=g_, multisex=1, multigender=1, multibirth=0))
for g_ in ("1", "0", "all"):
    an.append(mg.assign(gender=g_, multisex=0, multigender=1, multibirth=0))
for col in ("a1", "a2"):
    for g_ in ("1", "all"):
        an.append(mb.rename(columns={col: "age_group"})[KEYS].assign(
            gender=g_, multisex=0, multigender=0, multibirth=1))
AN = (pd.concat(an).groupby(KEYS + ["gender", "multisex", "multigender",
                                    "multibirth"]).size().rename("n_emp")
      .reset_index())
EXTRA_MS_PM = N_MS                                   # the expected gap

s82, s61, s67, s78, l47, l70, j47 = s101.load_modules()
for m_ in (s82, s61, s67, s78, l47, l70, j47):
    m_.OUT, m_.CACHE = s101.OUT, mc.CACHE_DIR
EXPO = s82.build_exposure(l47, l70, j47)["exposure"]


def install(world: str) -> None:
    sx, rp = Ssex.copy(), Pool.copy()
    if world == "broken":
        cells = Ssex[Ssex["gender"] == "2"].sample(10, random_state=3)
        sx = add(sx, cells[KEYS + ["gender"]], KEYS + ["gender"])
        rp = add(rp, Pool.sample(7, random_state=4)[KEYS], KEYS)
    fx.write_by_year(Pool, mc.CACHE_DIR, "L_counts")
    fx.write_by_year(sx, mc.CACHE_DIR, "L_counts_sex")
    fx.write_by_year(R99, mc.CACHE_DIR, "R_counts_raw")
    fx.write_by_year(rp, mc.CACHE_DIR, "R_counts_rawpool")
    fx.write_by_year(AN, mc.CACHE_DIR, "P_anom")


def world_gates() -> tuple:
    s101.EST.clear()
    b = s78.with_exposure(s61.build_skeleton(Pool, "22-25", j47), EXPO)
    b, t = s78.eq2_terms(b)
    s101.fit_tau(b, "probe_p", t, j47.FES, "gp", 1, s101.POST, s101.INTERIM)
    bs = s78.with_exposure(s67.build_skeleton_sex(
        Ssex, "22-25", j47, "n_emp"), EXPO)
    bs, ts = s78.gender_eq2_terms(bs)
    s101.fit_tau(bs, "probe_s", ts, j47.FES, "gs", 1, s101.FPOST,
                 s101.FINTERIM)
    return ({"post": s101.get("F", "gp", "post"),
             "tau": s101.get("F", "gp", "tau")},
            {"post": s101.get("F", "gs", "post"),
             "tau": s101.get("F", "gs", "tau")})


install("clean")
GP, GS = world_gates()
SUMM = {}
for world in ("clean", "broken"):
    print(f"\n=== the {world.upper()} world ===")
    install(world)
    s101.GATE, s101.SEX_GATE = GP, GS
    s101.EST.clear(); s101.FAILURES.clear(); s101.NOTES.clear()
    s101.DONE = s101.PLANNED = 0
    s101.PARTS = "RF"                   # Part I is SQL on the register
    _stdout = sys.stdout
    rc = s101.main()
    sys.stdout = _stdout
    SUMM[world] = (s101.OUT / "101_summary.txt").read_text()
    check(f"{world}: main() returns 0", rc == 0, f"{rc}; {s101.FAILURES}")
    gap = sum(s101.get("R", f"sexsum_minus_pooled_{y}", "person_months_gap")[0]
              for y in range(2021, 2026))
    if world == "clean":
        check("clean: the sex-split gap is exactly the multisex person-months",
              gap == EXTRA_MS_PM, f"{gap:,.0f} vs {EXTRA_MS_PM:,}")
        check("clean: R2 passes",
              "THE INDEPENDENT POOLED REBUILD EQUALS THE HEADLINE COUNTS"
              in SUMM[world])
        check("clean: R3 passes",
              "99S POOLED GAP IS PERSONS RECORDED UNDER TWO SEXES"
              in SUMM[world])
        g99 = sum(s101.get("R", f"raw99sum_minus_pooled_{y}",
                           "share_explained")[0] for y in range(2021, 2026))
        check("clean: 99's comparison is explained by the multigender persons",
              g99 == 5.0, f"{g99}")
        n, _ = s101.get("R", "estimation_panel_22_25", "cells_with_gap")
        r_, _ = s101.get("R", "estimation_panel_22_25",
                         "differing_cells_retained_by_fit")
        z, _ = s101.get("R", "estimation_panel_22_25",
                        "differing_cells_headline_zero")
        check("clean: the panel's differing cells are all retained, none zero",
              n > 0 and r_ == n and z == 0, f"{n}, {r_}, {z}")
        check("clean: the raw pooled rebuild is not refitted (identical)",
              any("not refitted" in x for x in s101.NOTES))
        check("clean: 2 gates + 2 corrected fits",
              s101.DONE == s101.PLANNED == 4, f"{s101.DONE}/{s101.PLANNED}")
        check("clean: R4 passes for both corrected fits",
              SUMM[world].count("MOVES BY LESS THAN A TENTH OF AN SE") == 2)
        c, _ = s101.get("F", "sex_no_multisex", "tau")
        g, _ = s101.get("F", "gate_sex", "tau")
        check("clean: removing the multisex persons changes the sex fit",
              c != g, f"{c:+.6f} vs {g:+.6f}")
    else:
        check("broken: R2 refused", "R2 NOT MET" in SUMM[world])
        check("broken: R3 refused", "R3 NOT MET" in SUMM[world])
        un = sum(s101.get("R", f"sexsum_minus_pooled_{y}",
                          "unexplained_no_anomalous_person")[0] or 0
                 for y in range(2021, 2026)
                 if s101.get("R", f"sexsum_minus_pooled_{y}",
                             "unexplained_no_anomalous_person")[0] == s101.get(
                     "R", f"sexsum_minus_pooled_{y}",
                     "unexplained_no_anomalous_person")[0])
        check("broken: exactly the ten planted cells are unexplained, with no "
              "anomalous person in them", un == 10, f"{un}")
        check("broken: the raw pooled rebuild is refitted",
              any(r["spec"] == "rawpool_pooled" for r in s101.EST))
    ex = pd.read_csv(s101.OUT / "provenance.csv")
    check(f"{world}: no identifier leaves",
          not any(c in ex.columns for c in ("employer_id", "person_id")))
    small = ex[(ex["status"] == "count") & ex["coef"].between(1, 4)
               & (ex["term"] != "max_abs_diff")]
    check(f"{world}: counts of 1 to 4 are suppressed in the export",
          small.empty, f"{len(small)} rows")
check.done()
