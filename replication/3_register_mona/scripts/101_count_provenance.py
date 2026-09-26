#!/usr/bin/env python3
"""
101_count_provenance.py -- where 99's 657 pooled differences come from, and
                           an independent POOLED rebuild of the headline
                           counts.

======================================================================
  RUNS IN MONA (lane 38c). Output folder CANARIES_101_OUT (default
  output_101). SQL, read only, cached: R_counts_rawpool_YYYY (the raw
  pooled rebuild, 99's query without sex, ~3 min a year);
  P_anom_YYYY (the persons whose register rows disagree with each
  other, by cell, ~3-5 min a year); one aggregate probe of the three
  Individ vintages (~2 min). Reads 47L's L_counts, 67's L_counts_sex and,
  if present, 99's R_counts_raw.
======================================================================

QUESTION
Script 99 (lane 37a) rebuilt the counts from the raw declarations and
found them identical to L_counts_sex in every cell of every year, while
the POOLED comparison with L_counts differed in about 170 cells a year
(657 on the estimation panel), each by one person-month, the rebuild
higher. The external review of 25 Sep 2026 points out that a pooled count
built on the same population must equal the sum of its sex cells, so
something differs between the two constructions, and asks for its
provenance before the audit table is published.

THE HYPOTHESIS, FROM THE CODE (read locally, not assumed)
99's pooled rebuild is the SUM over sex of its sex-split counts
(`stock = raw.groupby([employer, month, band]).sum()` over `gender`),
while 47L's L_counts counts DISTINCT persons per employer, month and band
with no sex in the query. Both 47L and 67 attach birth year and sex by
LEFT JOINing the three Individ vintages row by row, so a person with two
rows in the first vintage that holds them, carrying different sex codes,
yields two sex values: 67 (and 99 by sex) count that person once as a man
and once as a woman, 47L once. Sum-of-sex minus distinct-pooled is then
exactly the number of such persons in the cell. 82's cascade audit
already reports that a vintage holds more than one row for some persons.
Competing explanations the script also tests: persons with two birth
years (two bands; these inflate BOTH constructions equally and cannot
make the pooled gap); sex codes outside 1/2 (a '0' category in 99's raw
counts, absent from L_counts_sex); rows without an employer id (dropped
before every comparison, 11, 11, 4, 72 and 36 rows in L_counts_2021-25).

PART I. THE REGISTER ITSELF (aggregate probe, no person out)
For Individ 2019, 2021, 2023: rows, distinct persons, persons with more
than one row, and among them those whose rows carry more than one sex
code, and more than one birth year.

PART R. THE CELL-BY-CELL PROVENANCE, 2021-2025
  (i)   P  = L_counts (47L, distinct persons, pooled)
  (ii)  S  = L_counts_sex summed over sex (67)
  (iii) RP = R_counts_rawpool: a new raw pooled rebuild, 99's query with
        sex removed from the SELECT and GROUP BY: distinct (employer,
        month, person) triples first, the linkage once per person, then
        distinct persons by employer, month and band
  (iv)  RS = 99's R_counts_raw summed over its sex categories (1, 2, 0),
        which is what 99 compared with P
and the anomaly table from the declarations, per employer x month x band
x sex: persons with two valid sex codes (multisex), with a valid and an
invalid one (multigender), with two birth years (multibirth). Expected
gaps: S - P = sum over sex 1/2 of multisex persons minus the multisex
persons counted once; RS - P the same over every sex category for the
multigender persons. Every cell where a gap is non-zero is classified:
explained exactly by those persons, or not; the unexplained are
cross-tabulated against the other candidate causes.
Then on the 22-25 ESTIMATION PANEL (the gate's skeleton): the differing
cells, the share explained, and whether they are RETAINED by the fit
(headline count positive and every fixed-effect group containing the cell
with a positive total, fixest's first-pass separation rule).

PART F. THE ESTIMATES WITHOUT THE ANOMALOUS PERSONS
  pooled tau with the multibirth persons removed from P (they are the
  only anomaly that touches the pooled counts: one person, two bands);
  the female differential with the multisex persons removed from both
  sex cells (they are counted as a man and as a woman).
RP needs no refit if it equals P in every cell: tau is then identical by
construction, and the summary says so instead of fitting.

THE GATES (hard stop)
  Pooled: Table 1 at 22-25 within 0.0005: -0.0578 (0.0155), tau -0.0399
  (0.0102). Sex: -0.0858 (0.0142), tau -0.0714 (0.0109).

READ RULES, FIXED BEFORE THE RUN
  R2. THE INDEPENDENT POOLED REBUILD EQUALS THE HEADLINE COUNTS if RP
      equals P in every cell of every year (share identical 1.000, max
      difference 0).
  R3. 99'S POOLED GAP IS PERSONS RECORDED UNDER TWO SEXES if every cell
      where S differs from P differs by exactly the multisex persons in
      it (share explained 1.000), in every year and on the estimation
      panel. Otherwise the unexplained cells are reported by cause.
  R4. The female differential does not rest on them if removing the
      multisex persons moves its tau by less than a tenth of its SE. The
      same rule for the pooled tau without the multibirth persons.

EXPORT (output_101/)
  provenance.csv         per year and on the panel: cells, differing
                         cells, explained, unexplained by cause,
                         person-months; the register probe; the fits
  101_summary.txt, 101_log.txt; vcov_s101_*.csv
Counts of cells and persons only; employer counts below 5 suppressed.

IN THE PAPER
OA "The counts rebuilt from the raw declarations" paragraph (the 657
cells and their cause), response A2d.

    python 101_count_provenance.py
"""

import gc
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mona_common as mc  # noqa: E402

OUT = HERE / os.environ.get("CANARIES_101_OUT", "output_101")
OUT.mkdir(exist_ok=True)
os.environ.setdefault("CANARIES_82_OUT", str(OUT))
PARTS = os.environ.get("CANARIES_101_PARTS", "IRF").upper()
CACHE = mc.CACHE_DIR

FLOOR = 5
BAND = "22-25"
GATE = {"post": (-0.0578, 0.0155), "tau": (-0.0399, 0.0102)}
SEX_GATE = {"post": (-0.0858, 0.0142), "tau": (-0.0714, 0.0109)}
GATE_TOL = 0.0005
MOVE_SE = 0.10
KEYS = ["employer_id", "year_month", "age_group"]
COUNT_COLS = KEYS + ["n_emp"]
SEX_COLS = KEYS + ["gender", "n_emp"]
ANOM_COLS = KEYS + ["gender", "multisex", "multigender", "multibirth",
                    "n_emp"]
VINTAGES = (2019, 2021, 2023)
POST, INTERIM = "post_x_high_x_young", "interim_x_high_x_young"
FPOST, FINTERIM = POST + "_x_female", INTERIM + "_x_female"

NOTES: list = []
FAILURES: list = []
EST: list = []
PLANNED = 0
DONE = 0
T0 = time.time()

READ_RULES = [
    "READ RULES, FIXED BEFORE THE RUN:",
    "  GATES. Pooled -0.0578 (0.0155), tau -0.0399 (0.0102); sex -0.0858",
    "  (0.0142), tau -0.0714 (0.0109); within 0.0005. A miss stops Part F.",
    "  R2. The independent pooled rebuild equals the headline counts if RP",
    "  equals L_counts in every cell of every year.",
    "  R3. 99's pooled gap is persons recorded under two sexes if every cell",
    "  where the sex-split sum differs from L_counts differs by exactly the",
    "  multisex persons in it, every year and on the estimation panel.",
    f"  R4. A tau moves by less than {MOVE_SE} SE when the anomalous persons",
    "  are removed.",
    f"  Counts below {FLOOR} are suppressed before anything leaves MONA.",
]


# ----------------------------------------------------------------------
# plumbing
# ----------------------------------------------------------------------

def _mod(fname: str, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def drain(mod, tag: str) -> None:
    for n in list(getattr(mod, "NOTES", [])):
        NOTES.append(f"{tag}: {n}")
    for f in list(getattr(mod, "FAILURES", [])):
        FAILURES.append(f"{tag}/{f}")
    for attr in ("NOTES", "FAILURES"):
        if hasattr(mod, attr):
            getattr(mod, attr).clear()


def load_modules():
    s82 = _mod("82_occupation_route.py", "s82")
    s82.OUT = OUT
    s61, s67, s74, s78, s80, l47, l70, j47 = s82.load_modules()
    s78.OUT, s78.CACHE = OUT, CACHE
    if s82.MAIN_LEVEL != "uniform3" or s82.MAIN_ARM != "backward" \
            or s82.FLOOR_MAIN != FLOOR:
        raise RuntimeError("82's primary arm is not the one the paper "
                           "reports; refusing to run.")
    if s78.POST_FROM != "2024-01":
        raise RuntimeError(f"78's adoption date is {s78.POST_FROM}")
    return s82, s61, s67, s78, l47, l70, j47


def tstat(c, s) -> float:
    return float(c / s) if s and s == s and s > 0 else float("nan")


def est(part, spec, term, coef, se=np.nan, n_obs=np.nan, n_firms=np.nan,
        status="ok", vp=np.nan, vi=np.nan, cpi=np.nan):
    EST.append({"part": part, "spec": spec, "term": term,
                "coef": float(coef) if coef == coef else np.nan,
                "se": float(se) if se == se else np.nan, "t": tstat(coef, se),
                "var_post": vp, "var_interim": vi, "cov_post_interim": cpi,
                "n_obs": n_obs, "n_firms": n_firms, "status": status})
    save_est()


def save_est() -> None:
    df = pd.DataFrame(EST)
    if not df.empty:
        had = df["n_firms"].notna()
        df = mc.enforce_min_cell(df, count_col="n_firms", floor=FLOOR)
        small = had & df["n_firms"].isna()
        df.loc[small, ["coef", "se", "t"]] = np.nan
        # a count of persons or cells between 1 and 4 is suppressed too
        cnt = (df["status"].eq("count") & df["coef"].between(1, FLOOR - 1)
               & ~df["term"].eq("max_abs_diff"))
        df.loc[cnt, "coef"] = np.nan
    df.to_csv(OUT / "provenance.csv", index=False)


def get(part, spec, term):
    for r in EST:
        if (r["part"], r["spec"], r["term"]) == (part, spec, term):
            return r["coef"], r["se"]
    return np.nan, np.nan


def fit(b: pd.DataFrame, tag: str, terms: list, fes: tuple):
    global DONE, PLANNED
    PLANNED += 1
    print(f"    {tag}: {len(b):,} rows, {b['employer_id'].nunique():,} firms"
          f"{mc.mem_line(' | ')}")
    t = time.time()
    try:
        r = mc.run_fepois_multi(b, OUT, tag=f"s101_{tag}", terms=terms,
                                fes=fes, cluster="employer_id")
    except BaseException as ex:
        print(f"    {tag} FAILED: {type(ex).__name__}: {ex}")
        traceback.print_exc()
        r = pd.DataFrame()
    if r.empty or r["coef"].isna().all():
        FAILURES.append(tag)
        return None, None
    g = r.set_index("term")
    v = None
    if "vcov" in r.attrs and Path(r.attrs["vcov"]).exists():
        v = pd.read_csv(r.attrs["vcov"]).set_index("term")
    DONE += 1
    print(f"    {tag}: done in {(time.time() - t) / 60:.1f} min")
    return g, v


def tau(g, v, post, interim) -> tuple:
    nan = (np.nan,) * 5
    if g is None or post not in g.index or interim not in g.index:
        return nan
    c = float(g.loc[post, "coef"]) - float(g.loc[interim, "coef"])
    if v is None or post not in v.index or interim not in v.index:
        return c, np.nan, np.nan, np.nan, np.nan
    vp, vi = float(v.loc[post, post]), float(v.loc[interim, interim])
    cpi = float(v.loc[post, interim])
    var = vp + vi - 2.0 * cpi
    return c, (float(np.sqrt(var)) if var > 0 else np.nan), vp, vi, cpi


def fit_tau(b, tag, terms, fes, spec, n_firms, post, interim) -> tuple:
    g, v = fit(b, tag, terms, fes)
    if g is None:
        return np.nan, np.nan
    n_obs = int(g["n_obs"].max())
    est("F", spec, "post", g.loc[post, "coef"], g.loc[post, "se"], n_obs,
        n_firms)
    c, s, vp, vi, cpi = tau(g, v, post, interim)
    est("F", spec, "tau", c, s, n_obs, n_firms, "derived", vp, vi, cpi)
    return c, s


def norm(d: pd.DataFrame, label: str) -> pd.DataFrame:
    """int64 employer, stripped text keys, on every side (failure class
    3); rows with no employer id are counted and dropped."""
    ids = pd.to_numeric(d["employer_id"], errors="coerce")
    miss = int(ids.isna().sum())
    if miss:
        NOTES.append(f"{label}: {miss:,} rows without an employer id dropped")
    out = d[ids.notna()].assign(employer_id=ids[ids.notna()].astype("int64"))
    for k in ("year_month", "age_group", "gender"):
        if k in out.columns:
            out[k] = out[k].astype(str).str.strip()
    return out


def cells(d: pd.DataFrame, label: str) -> pd.Series:
    return norm(d, label).groupby(KEYS, observed=True)["n_emp"].sum()


# ----------------------------------------------------------------------
# SQL
# ----------------------------------------------------------------------

def agi_union(year: int, cols: str) -> str:
    suffix, max_month = ("_def", 12) if year < 2025 else ("_prel", 6)
    return "\nUNION ALL\n".join(
        f"SELECT {cols} FROM dbo.Arb_AGIIndivid{year}{m:02d}{suffix}"
        for m in range(1, max_month + 1))


BAND_CASE = """CASE
             WHEN {y} - {f} BETWEEN 22 AND 25 THEN '22-25'
             WHEN {y} - {f} BETWEEN 26 AND 30 THEN '26-30'
             WHEN {y} - {f} BETWEEN 31 AND 34 THEN '31-34'
             WHEN {y} - {f} BETWEEN 35 AND 40 THEN '35-40'
             WHEN {y} - {f} BETWEEN 41 AND 49 THEN '41-49'
             WHEN {y} - {f} BETWEEN 50 AND 69 THEN '50+'
             ELSE NULL END"""
FOD = ("COALESCE(TRY_CAST(a.FodelseAr AS INT), TRY_CAST(b.FodelseAr AS INT), "
       "TRY_CAST(c.FodelseAr AS INT))")
KON = "LTRIM(RTRIM(COALESCE(a.Kon, b.Kon, c.Kon)))"
JOINS = """LEFT JOIN dbo.Individ_2023 a ON p.person_id = a.P1207_LopNr_PersonNr
        LEFT JOIN dbo.Individ_2021 b ON p.person_id = b.P1207_LopNr_PersonNr
        LEFT JOIN dbo.Individ_2019 c ON p.person_id = c.P1207_LopNr_PersonNr"""


def q_rawpool(year: int, conn) -> pd.DataFrame:
    """99's raw rebuild with sex removed: distinct triples first, every
    joined register row kept (as 47L keeps them), distinct persons by
    employer, month and band. No occupation column."""
    u = agi_union(year, "P1207_LOPNR_PEORGNR AS employer_id, PERIOD AS "
                        "period, P1207_LOPNR_PERSONNR AS person_id")
    bc = BAND_CASE.format(y=year, f="d.fodelse")
    q = f"""
    WITH trip AS (SELECT DISTINCT employer_id, period, person_id FROM ({u}) x),
    pers AS (SELECT DISTINCT person_id FROM trip),
    d AS (SELECT p.person_id, {FOD} AS fodelse FROM pers p
        {JOINS})
    SELECT t.employer_id,
           LEFT(t.period,4) + '-' + SUBSTRING(t.period,5,2) AS year_month,
           {bc} AS age_group, COUNT(DISTINCT t.person_id) AS n_emp
    FROM trip t JOIN d ON t.person_id = d.person_id
    WHERE d.fodelse IS NOT NULL AND {year} - d.fodelse BETWEEN 22 AND 69
    GROUP BY t.employer_id, t.period, {bc}
    """
    return pd.read_sql(q, conn)


def q_anom(year: int, conn) -> pd.DataFrame:
    """The persons whose joined register rows disagree: two valid sex
    codes, a valid and an invalid one, or two birth years. Counted by
    employer x month x band x sex category, with the pooled row
    (gender 'all') from the same GROUPING SETS so a person in two sex
    categories is counted once there."""
    u = agi_union(year, "P1207_LOPNR_PEORGNR AS employer_id, PERIOD AS "
                        "period, P1207_LOPNR_PERSONNR AS person_id")
    bc = BAND_CASE.format(y=year, f="j.fod")
    q = f"""
    WITH pers AS (SELECT DISTINCT person_id FROM ({u}) x),
    j AS (SELECT p.person_id, {FOD} AS fod, {KON} AS kon FROM pers p
        {JOINS}),
    pk AS (
        SELECT person_id,
               COUNT(DISTINCT CASE WHEN kon IN ('1', '2') THEN kon END) AS nkon,
               COUNT(DISTINCT CASE WHEN kon IN ('1', '2') THEN kon
                                   ELSE '0' END) AS ngen,
               COUNT(DISTINCT fod) AS nfod
        FROM j GROUP BY person_id
        HAVING COUNT(DISTINCT CASE WHEN kon IN ('1', '2') THEN kon
                                   ELSE '0' END) > 1
            OR COUNT(DISTINCT fod) > 1),
    trip AS (SELECT DISTINCT y.employer_id, y.period, y.person_id
             FROM ({u}) y WHERE y.person_id IN (SELECT person_id FROM pk)),
    xx AS (
        SELECT t.employer_id, t.period, t.person_id, {bc} AS age_group,
               CASE WHEN j.kon IN ('1', '2') THEN j.kon ELSE '0' END AS gender,
               CASE WHEN pk.nkon > 1 THEN 1 ELSE 0 END AS multisex,
               CASE WHEN pk.ngen > 1 THEN 1 ELSE 0 END AS multigender,
               CASE WHEN pk.nfod > 1 THEN 1 ELSE 0 END AS multibirth
        FROM trip t
        JOIN pk ON t.person_id = pk.person_id
        JOIN j ON t.person_id = j.person_id
        WHERE j.fod IS NOT NULL AND {year} - j.fod BETWEEN 22 AND 69)
    SELECT employer_id,
           LEFT(period,4) + '-' + SUBSTRING(period,5,2) AS year_month,
           age_group,
           CASE WHEN GROUPING(gender) = 1 THEN 'all' ELSE gender END AS gender,
           multisex, multigender, multibirth,
           COUNT(DISTINCT person_id) AS n_emp
    FROM xx
    GROUP BY GROUPING SETS (
        (employer_id, period, age_group, gender, multisex, multigender,
         multibirth),
        (employer_id, period, age_group, multisex, multigender, multibirth))
    """
    return pd.read_sql(q, conn)


def pulled(prefix, years, require, query) -> dict:
    out, conn = {}, None
    for y in years:
        cf = CACHE / f"{prefix}_{y}.parquet"
        c = mc.read_cache(cf, require=require)
        if c is None:
            if conn is None:
                conn = mc.connect()
            t = time.time()
            c = query(y, conn)
            mc.write_cache(c, cf)
            print(f"  {prefix} {y}: {len(c):,} rows "
                  f"({(time.time() - t) / 60:.1f} min)")
        else:
            print(f"  {prefix} {y}: cached ({len(c):,} rows)")
        out[y] = c
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
    return out


# ----------------------------------------------------------------------
# Part I: the register probe
# ----------------------------------------------------------------------

def part_i() -> None:
    print("\n  PART I, the register probe:")
    conn = mc.connect()
    try:
        for y in VINTAGES:
            q = f"""
            SELECT COUNT(*) AS persons_multirow,
                   SUM(CASE WHEN nk > 1 THEN 1 ELSE 0 END) AS persons_multisex,
                   SUM(CASE WHEN nf > 1 THEN 1 ELSE 0 END) AS persons_multibirth
            FROM (SELECT P1207_LopNr_PersonNr,
                         COUNT(DISTINCT CASE WHEN LTRIM(RTRIM(Kon)) IN ('1','2')
                                        THEN LTRIM(RTRIM(Kon)) END) AS nk,
                         COUNT(DISTINCT TRY_CAST(FodelseAr AS INT)) AS nf
                  FROM dbo.Individ_{y}
                  GROUP BY P1207_LopNr_PersonNr
                  HAVING COUNT(*) > 1) z"""
            r = pd.read_sql(q, conn).iloc[0]
            t = pd.read_sql(f"SELECT COUNT(*) AS n_rows, COUNT(DISTINCT "
                            f"P1207_LopNr_PersonNr) AS n_persons FROM "
                            f"dbo.Individ_{y}", conn).iloc[0]
            for k, v in (("rows", t["n_rows"]), ("persons", t["n_persons"]),
                         ("persons_multirow", r["persons_multirow"]),
                         ("persons_multisex", r["persons_multisex"]),
                         ("persons_multibirth", r["persons_multibirth"])):
                est("I", f"Individ_{y}", k, float(v or 0), status="count")
            print(f"  Individ_{y}: {int(t['n_rows']):,} rows, "
                  f"{int(t['n_persons']):,} persons; more than one row "
                  f"{int(r['persons_multirow'] or 0):,} (two sex codes "
                  f"{int(r['persons_multisex'] or 0):,}, two birth years "
                  f"{int(r['persons_multibirth'] or 0):,})")
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ----------------------------------------------------------------------
# Part R: the provenance
# ----------------------------------------------------------------------

def expected(an: pd.DataFrame, flag: str, genders: tuple) -> pd.Series:
    """Extra person-months a sex-split sum carries over a distinct pooled
    count: persons with `flag`, summed over `genders`, minus the same
    persons counted once (gender 'all')."""
    a = an[an[flag] == 1]
    split = (a[a["gender"].isin(genders)].groupby(KEYS, observed=True)
             ["n_emp"].sum())
    once = a[a["gender"] == "all"].groupby(KEYS, observed=True)["n_emp"].sum()
    return split.sub(once, fill_value=0)


def classify(label: str, gap: pd.Series, exp: pd.Series, an: pd.DataFrame,
             part: str = "R") -> dict:
    """Gap cells against the expected gap; the unexplained cross-tabbed
    against the other candidate causes. Aggregates only."""
    j = gap.rename("gap").to_frame().join(exp.rename("exp"), how="outer")
    j = j.fillna(0)
    diff = j[(j["gap"] != 0) | (j["exp"] != 0)]
    expl = diff[diff["gap"] == diff["exp"]]
    unex = diff[diff["gap"] != diff["exp"]]
    r = {"cells_with_gap": int((j["gap"] != 0).sum()),
         "cells_expected_gap": int((j["exp"] != 0).sum()),
         "cells_explained_exactly": int(len(expl[expl["gap"] != 0])),
         "cells_unexplained": int(len(unex)),
         "person_months_gap": float(j["gap"].sum()),
         "person_months_expected": float(j["exp"].sum())}
    r["share_explained"] = (r["cells_explained_exactly"]
                            / r["cells_with_gap"]) if r["cells_with_gap"] \
        else 1.0
    if len(unex):
        a = an.groupby(KEYS, observed=True)[["multibirth", "multigender"]] \
            .max()
        u = unex.join(a, how="left").fillna(0)
        r["unexplained_with_multibirth"] = int((u["multibirth"] > 0).sum())
        r["unexplained_with_multigender"] = int((u["multigender"] > 0).sum())
        r["unexplained_no_anomalous_person"] = int(
            ((u["multibirth"] == 0) & (u["multigender"] == 0)).sum())
    for k, v in r.items():
        est(part, label, k, v, status="share" if k.startswith("share")
            else "count")
    print(f"  {label}: {r['cells_with_gap']:,} cells with a gap, "
          f"{r['share_explained']:.6f} explained exactly; "
          f"{r['cells_unexplained']:,} unexplained; person-months "
          f"{r['person_months_gap']:,.0f} (expected "
          f"{r['person_months_expected']:,.0f})")
    return r


def compare(label: str, new: pd.Series, old: pd.Series) -> dict:
    j = new.rename("new").to_frame().join(old.rename("old"), how="outer")
    one = int(j["new"].isna().sum() + j["old"].isna().sum())
    j = j.fillna(0)
    d = (j["new"] - j["old"]).abs()
    r = {"cells": int(len(j)), "cells_identical": int((d == 0).sum()),
         "share_identical": float((d == 0).mean()) if len(j) else np.nan,
         "max_abs_diff": float(d.max()) if len(j) else np.nan,
         "cells_on_one_side": one,
         "person_months_new": float(j["new"].sum()),
         "person_months_old": float(j["old"].sum())}
    for k, v in r.items():
        est("R", label, k, v, status="share" if k.startswith("share")
            else "count")
    print(f"  {label}: {r['cells']:,} cells, {r['share_identical']:.6f} "
          f"identical, max |diff| {r['max_abs_diff']:.0f}")
    return r


def part_r(years, anoms: dict, rawpool: dict) -> None:
    print("\n  PART R, the provenance:")
    for y in years:
        P = mc.read_cache(CACHE / f"L_counts_{y}.parquet", require=COUNT_COLS)
        S = mc.read_cache(CACHE / f"L_counts_sex_{y}.parquet",
                          require=SEX_COLS)
        if P is None or S is None:
            FAILURES.append(f"R/{y}/L_counts or L_counts_sex missing")
            continue
        Pc = cells(P, f"L_counts_{y}")
        S = norm(S, f"L_counts_sex_{y}")
        Sc = S[S["gender"].isin(["1", "2"])].groupby(
            KEYS, observed=True)["n_emp"].sum()
        an = norm(anoms[y], f"P_anom_{y}")
        for f in ("multisex", "multigender", "multibirth"):
            an[f] = pd.to_numeric(an[f], errors="coerce").fillna(0).astype(int)
        est("R", f"anomalous_{y}", "multisex_person_months",
            float(an.loc[(an["multisex"] == 1) & (an["gender"] == "all"),
                         "n_emp"].sum()), status="count")
        est("R", f"anomalous_{y}", "multibirth_person_months",
            float(an.loc[(an["multibirth"] == 1) & (an["gender"] == "all"),
                         "n_emp"].sum()), status="count")
        # (ii) - (i): the sex-split sum against the distinct pooled count
        classify(f"sexsum_minus_pooled_{y}", Sc.sub(Pc, fill_value=0),
                 expected(an, "multisex", ("1", "2")), an)
        # (iii) against (i): the independent pooled rebuild
        compare(f"rawpool_vs_pooled_{y}", cells(rawpool[y], f"rawpool_{y}"),
                Pc)
        # (iv) against (i): what 99 compared
        R99 = mc.read_cache(CACHE / f"R_counts_raw_{y}.parquet",
                            require=SEX_COLS)
        if R99 is not None:
            classify(f"raw99sum_minus_pooled_{y}",
                     cells(R99, f"R_counts_raw_{y}").sub(Pc, fill_value=0),
                     expected(an, "multigender", ("0", "1", "2")), an)
        del P, S, Pc, Sc
        gc.collect()


def panel_check(expo, anoms, s61, s78, j47) -> None:
    """The 22-25 estimation panel: differing cells, explained, retained."""
    print("\n  PART R on the estimation panel:")
    P = pd.concat([mc.read_cache(CACHE / f"L_counts_{y}.parquet",
                                 require=COUNT_COLS) for y in s61.PANEL_YEARS])
    S = pd.concat([mc.read_cache(CACHE / f"L_counts_sex_{y}.parquet",
                                 require=SEX_COLS) for y in s61.PANEL_YEARS])
    S = norm(S, "L_counts_sex")
    S = (S[S["gender"].isin(["1", "2"])].groupby(KEYS, observed=True)
         ["n_emp"].sum().reset_index())
    bp = s78.with_exposure(s61.build_skeleton(norm(P, "L_counts"), BAND, j47),
                           expo)
    bs = s78.with_exposure(s61.build_skeleton(S, BAND, j47), expo)
    for d in (bp, bs):
        d["employer_id"] = pd.to_numeric(d["employer_id"]).astype("int64")
        d["year_month"] = d["year_month"].astype(str)
        d["age_group"] = d["age_group"].astype(str)
    a = bp.set_index(KEYS)["n_emp"]
    gap = bs.set_index(KEYS)["n_emp"].sub(a, fill_value=0)
    an = pd.concat([norm(anoms[y], f"P_anom_{y}") for y in s61.PANEL_YEARS])
    for f in ("multisex", "multigender", "multibirth"):
        an[f] = pd.to_numeric(an[f], errors="coerce").fillna(0).astype(int)
    exp = expected(an, "multisex", ("1", "2")).reindex(gap.index).fillna(0)
    classify("estimation_panel_22_25", gap, exp, an)
    # retention: headline count positive and every FE group with a
    # positive total (fixest's first-pass separation rule)
    ok = np.ones(len(bp), dtype=bool)
    for fe in ("fe_emp_t", "fe_emp_age", "fe_t_age"):
        tot = bp.groupby(fe, observed=True)["n_emp"].transform("sum")
        ok &= (tot > 0).to_numpy()
    ok &= (bp["n_emp"] > 0).to_numpy()
    ret = pd.Series(ok, index=bp.set_index(KEYS).index)
    dcells = gap[gap != 0].index
    inp = dcells.isin(ret.index)
    n_ret = int(ret.reindex(dcells).fillna(False).astype(bool).sum())
    est("R", "estimation_panel_22_25", "differing_cells_in_headline_panel",
        int(inp.sum()), status="count")
    est("R", "estimation_panel_22_25", "differing_cells_retained_by_fit",
        n_ret, status="count")
    zero = int((a.reindex(dcells).fillna(0) == 0).sum())
    est("R", "estimation_panel_22_25", "differing_cells_headline_zero", zero,
        status="count")
    print(f"  panel: {len(dcells):,} differing cells, {int(inp.sum()):,} in "
          f"the headline panel, {n_ret:,} retained by the fit's first-pass "
          f"rule, {zero:,} with a headline count of zero")
    del bp, bs, P, S
    gc.collect()


# ----------------------------------------------------------------------
# Part F: the estimates without the anomalous persons
# ----------------------------------------------------------------------

def gate_check(label, got, want) -> list:
    return [f"{label} {k}: this run {x:+.4f} ({y:.4f}), Table 1 "
            f"{want[k][0]:+.4f} ({want[k][1]:.4f})"
            for k, (x, y) in got.items()
            if not (abs(x - want[k][0]) <= GATE_TOL
                    and abs(y - want[k][1]) <= GATE_TOL)]


def part_f(expo, anoms, rawpool, s61, s67, s78, j47) -> None:
    print("\n  PART F, the estimates without the anomalous persons:")
    years = s61.PANEL_YEARS
    an = pd.concat([norm(anoms[y], f"P_anom_{y}") for y in years])
    for f in ("multisex", "multigender", "multibirth"):
        an[f] = pd.to_numeric(an[f], errors="coerce").fillna(0).astype(int)
    P = norm(pd.concat([mc.read_cache(CACHE / f"L_counts_{y}.parquet",
                                      require=COUNT_COLS) for y in years]),
             "L_counts")
    # the pooled gate
    b = s78.with_exposure(s61.build_skeleton(P, BAND, j47), expo)
    b, terms = s78.eq2_terms(b)
    n = int(b["employer_id"].nunique())
    c, s = fit_tau(b, "gate_pooled", terms, j47.FES, "gate_pooled", n, POST,
                   INTERIM)
    bad = gate_check("pooled", {"post": get("F", "gate_pooled", "post"),
                                "tau": (c, s)}, GATE)
    if bad:
        FAILURES.append("THE POOLED GATE FAILED: " + "; ".join(bad))
        return
    del b
    gc.collect()
    # RP: refit only if it is not identical to P in every cell
    same = all(get("R", f"rawpool_vs_pooled_{y}", "max_abs_diff")[0] == 0
               and get("R", f"rawpool_vs_pooled_{y}", "cells_on_one_side")[0]
               == 0 for y in years)
    if same:
        NOTES.append("F: the raw pooled rebuild equals L_counts in every "
                     "cell of every year, so its tau is Table 1's by "
                     "construction; not refitted")
    else:
        RP = norm(pd.concat([rawpool[y] for y in years]), "rawpool")
        b = s78.with_exposure(s61.build_skeleton(RP, BAND, j47), expo)
        b, terms = s78.eq2_terms(b)
        fit_tau(b, "rawpool", terms, j47.FES, "rawpool_pooled",
                int(b["employer_id"].nunique()), POST, INTERIM)
        del b, RP
        gc.collect()
    # the pooled counts without the multibirth persons
    mb = (an[(an["multibirth"] == 1) & (an["gender"] == "all")]
          .groupby(KEYS, observed=True)["n_emp"].sum())
    if mb.sum() > 0:
        Pi = P.groupby(KEYS, observed=True)["n_emp"].sum()
        P2 = Pi.sub(mb, fill_value=0).clip(lower=0).reset_index()
        b = s78.with_exposure(s61.build_skeleton(P2, BAND, j47), expo)
        b, terms = s78.eq2_terms(b)
        fit_tau(b, "no_multibirth", terms, j47.FES, "pooled_no_multibirth",
                int(b["employer_id"].nunique()), POST, INTERIM)
        del b, P2, Pi
        gc.collect()
    else:
        NOTES.append("F: no multibirth person in the panel years; the pooled "
                     "counts need no correction")
    del P
    gc.collect()
    # the sex gate, then the sex counts without the multisex persons
    S = norm(pd.concat([mc.read_cache(CACHE / f"L_counts_sex_{y}.parquet",
                                      require=SEX_COLS) for y in years]),
             "L_counts_sex")
    b = s78.with_exposure(s67.build_skeleton_sex(S, BAND, j47, "n_emp"), expo)
    b, terms = s78.gender_eq2_terms(b)
    n = int(b["employer_id"].nunique())
    c, s = fit_tau(b, "gate_sex", terms, j47.FES, "gate_sex", n, FPOST,
                   FINTERIM)
    bad = gate_check("sex", {"post": get("F", "gate_sex", "post"),
                             "tau": (c, s)}, SEX_GATE)
    if bad:
        FAILURES.append("THE SEX GATE FAILED: " + "; ".join(bad))
        return
    del b
    gc.collect()
    ms = (an[(an["multisex"] == 1) & an["gender"].isin(["1", "2"])]
          .groupby(KEYS + ["gender"], observed=True)["n_emp"].sum())
    if ms.sum() > 0:
        Si = S.groupby(KEYS + ["gender"], observed=True)["n_emp"].sum()
        S2 = Si.sub(ms, fill_value=0).clip(lower=0).reset_index()
        b = s78.with_exposure(s67.build_skeleton_sex(S2, BAND, j47, "n_emp"),
                              expo)
        b, terms = s78.gender_eq2_terms(b)
        fit_tau(b, "no_multisex", terms, j47.FES, "sex_no_multisex",
                int(b["employer_id"].nunique()), FPOST, FINTERIM)
        del b, S2, Si
        gc.collect()
    else:
        NOTES.append("F: no multisex person in the panel years")
    del S
    gc.collect()


# ----------------------------------------------------------------------
# summary and main
# ----------------------------------------------------------------------

def verdicts(years) -> list:
    L = []
    rp = [get("R", f"rawpool_vs_pooled_{y}", "max_abs_diff")[0] for y in years]
    one = [get("R", f"rawpool_vs_pooled_{y}", "cells_on_one_side")[0]
           for y in years]
    if all(x == x for x in rp):
        ok = all(x == 0 for x in rp) and all(x == 0 for x in one)
        L.append(f"  R2: {'THE INDEPENDENT POOLED REBUILD EQUALS THE HEADLINE COUNTS' if ok else 'R2 NOT MET'}"
                 f"; max |diff| by year {rp}")
    else:
        L.append("  R2: NO VERDICT, a year is missing")
    sh = [get("R", f"sexsum_minus_pooled_{y}", "share_explained")[0]
          for y in years]
    shp, _ = get("R", "estimation_panel_22_25", "share_explained")
    if all(x == x for x in sh) and shp == shp:
        ok = all(x == 1.0 for x in sh) and shp == 1.0
        L.append(f"  R3: {'99S POOLED GAP IS PERSONS RECORDED UNDER TWO SEXES' if ok else 'R3 NOT MET: see the unexplained cells by cause'}"
                 f"; share explained by year {[round(x, 6) for x in sh]}, "
                 f"panel {shp:.6f}")
    else:
        L.append("  R3: NO VERDICT, a comparison is missing")
    for spec, gate, lab in (("pooled_no_multibirth", "gate_pooled",
                             "pooled tau without multibirth persons"),
                            ("sex_no_multisex", "gate_sex",
                             "female differential without multisex persons")):
        c, s = get("F", spec, "tau")
        g, gs = get("F", gate, "tau")
        if c == c and g == g:
            ok = abs(c - g) < MOVE_SE * gs
            L.append(f"  R4 ({lab}): {'MOVES BY LESS THAN A TENTH OF AN SE' if ok else 'R4 NOT MET'}"
                     f"; {c:+.5f} ({s:.5f}) against {g:+.5f} ({gs:.5f})")
    return L


def write_summary(years) -> None:
    L = ["PROVENANCE OF THE POOLED DIFFERENCES, AND THE RAW POOLED REBUILD",
         "=" * 64, ""]
    for spec, lab in (("gate_pooled", "pooled"), ("gate_sex", "sex")):
        c, s = get("F", spec, "tau")
        if c == c:
            L.append(f"GATE {lab}: tau {c:+.4f} ({s:.4f})")
    i = [r for r in EST if r["part"] == "I"]
    if i:
        L += ["", "I. THE REGISTER VINTAGES:"]
        for y in VINTAGES:
            vals = {r["term"]: r["coef"] for r in i
                    if r["spec"] == f"Individ_{y}"}
            if vals:
                L.append(f"  Individ_{y}: " + ", ".join(
                    f"{k} {v:,.0f}" for k, v in vals.items()))
    rr = [r for r in EST if r["part"] == "R"
          and r["term"] in ("share_identical", "share_explained",
                            "cells_with_gap", "cells_unexplained",
                            "max_abs_diff", "person_months_gap",
                            "multisex_person_months",
                            "multibirth_person_months",
                            "differing_cells_retained_by_fit",
                            "differing_cells_headline_zero",
                            "unexplained_with_multibirth",
                            "unexplained_with_multigender",
                            "unexplained_no_anomalous_person")]
    if rr:
        L += ["", "R. CELL BY CELL:"]
        for r in rr:
            v = r["coef"]
            L.append(f"  {r['spec']:<34} {r['term']:<34} "
                     + (f"{v:.6f}" if r["term"].startswith("share")
                        else f"{v:,.0f}" if v == v else "<5 (suppressed)"))
    f = [r for r in EST if r["part"] == "F" and r["term"] == "tau"]
    if f:
        L += ["", "F. TAU:"]
        L += [f"  {r['spec']:<24} {r['coef']:+.5f} ({r['se']:.5f})" for r in f]
    L += ["", "VERDICTS:"] + verdicts(years)
    L += ["", f"FITS: {DONE} of {PLANNED} attempted came back. A run far "
          "shorter than the estimate (1 to 1.5 hours) is a failure."]
    if NOTES:
        L += ["", "NOTES:"] + [f"  {n}" for n in NOTES]
    if FAILURES:
        L += ["", "FAILED: " + " | ".join(FAILURES),
              "A missing row is a missing fit, never a zero."]
    L += [""] + READ_RULES + ["", f"Runtime {(time.time() - T0) / 60:.1f} "
                              "min. " + mc.mem_line("")]
    (OUT / "101_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))


def run_part(name, fn, *args) -> None:
    try:
        fn(*args)
    except BaseException as ex:
        if isinstance(ex, SystemExit):
            raise
        print(f"  Part {name} FAILED ({type(ex).__name__}: {ex})")
        traceback.print_exc()
        FAILURES.append(f"{name}/{type(ex).__name__}: {ex}")


def main() -> int:
    global T0
    mc.Tee(OUT / "101_log.txt")
    T0 = time.time()
    print("=" * 70)
    print(f"101: COUNT PROVENANCE   parts {PARTS}")
    print("=" * 70)
    print("\n".join(READ_RULES))
    print(mc.mem_line("  "))
    rc = 0
    years = []
    try:
        s82, s61, s67, s78, l47, l70, j47 = load_modules()
        years = list(s61.PANEL_YEARS)
        built = s82.build_exposure(l47, l70, j47)
        drain(s82, "82")
        expo = built["exposure"]
        del built
        gc.collect()
        if "I" in PARTS:
            run_part("I", part_i)
        anoms = pulled("P_anom", years, ANOM_COLS, q_anom)
        rawpool = pulled("R_counts_rawpool", years, COUNT_COLS, q_rawpool)
        if "R" in PARTS:
            run_part("R", part_r, years, anoms, rawpool)
            run_part("R-panel", panel_check, expo, anoms, s61, s78, j47)
        if "F" in PARTS:
            run_part("F", part_f, expo, anoms, rawpool, s61, s67, s78, j47)
        drain(s78, "78")
    except BaseException as ex:
        if isinstance(ex, SystemExit):
            raise
        print(f"101 FAILED: {type(ex).__name__}: {ex}")
        traceback.print_exc()
        FAILURES.append(f"main/{type(ex).__name__}: {ex}")
        rc = 1
    save_est()
    write_summary(years)
    rc = rc or (1 if FAILURES else 0)
    mc.runlog("101_count_provenance", rc, (time.time() - T0) / 60)
    print("\n101 done.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
