#!/usr/bin/env python3
"""
99_measurement.py -- the measurement checks the Editor made a condition of
                     the R&R: the headline counts rebuilt from the raw
                     declarations, the denominator reconciled from one
                     starting population, and the young's employment split
                     into baseline incumbents and new matches.

======================================================================
  RUNS IN MONA (lane 37a). Output folder CANARIES_99_OUT (default
  output_99); parts with CANARIES_99_PARTS (default RDM). SQL, all read
  only, every pull cached so a rerun issues none:
    R  the raw rebuild, 2021-2025, cached R_counts_raw_YYYY (~3 min/yr)
    D  the reconciliation, 2020-2025, cached R_recon_YYYY (one query a
       year joining the year's declarations to the previous year's; the
       heaviest pull of the trip, budget 10-20 min/yr)
    M  counts split by presence at the employer in November 2022,
       2021-2025, cached L_counts_basematch_YYYY (~2 min/yr)
  Plus one INFORMATION_SCHEMA probe of the Individ tables.
======================================================================

QUESTION
The external review of 25 Sep 2026, relaying the Editor: the headline
counts must be shown to come from the raw employer declarations and the
demographic linkage alone, with no occupation file after 2019 and no
occupation-selected intermediate dataset anywhere on the path; the
coverage of the occupation register must be reconciled from ONE starting
population, all declared person-months, before any exclusion; and the
young's decline must be split into the workers already at the employer
and new person-employer matches.

THE GATE (hard stop, before anything is varied)
The paper's panel, from 47L's L_counts caches with script 82's score,
must reproduce Table 1 at 22-25 within 0.0005 on coefficient and SE:
adoption -0.0578 (0.0155), tau -0.0399 (0.0102).

PART R. THE INDEPENDENT RAW REBUILD
A query written afresh, not 47L's: the monthly declarations are reduced
to distinct (employer, month, person) triples FIRST; the demographic
linkage (birth year and sex from Individ 2023, then 2021, then 2019, the
panel's rule) is attached once per PERSON; the triples are counted by
employer x month x age band (calendar year minus birth year, 22-69) x
sex. No occupation column of any year is read, and nothing passes through
a cache another script wrote. Compared with the counts the headline uses
(47L's L_counts for all sexes, 67's L_counts_sex by sex), cell by cell,
year by year: cells in either, cells identical, the share identical, the
maximum absolute difference, total person-months on each side; and the
national totals by age band and month from both. Then on the estimation
panel itself (the 22-25 skeleton, scored employers, balanced and zero
filled): the same statistics. Then tau at 22-25 refitted on the rebuilt
counts, beside -0.0399.
The score is the fixed 2019 employer score (82's build_exposure, the
backward cascade 2015-2019). The script also refits the quartile with
every code recorded after 2019 removed from the cascade frame and reports
whether any employer's quartile moves (it must not: the forward years are
a separate arm that is never the score).

PART D. THE DENOMINATOR RECONCILIATION (2020-2025)
Starting population: every declared person-month (distinct employer,
month and person), all ages, before any exclusion. Each is classified by
  age band      under 22, the six panel bands, over 69, unknown
  status        incumbent (the same employer in any month of the previous
                calendar year), new match (in the declarations the
                previous year, not with this employer), entrant (not in
                the declarations the previous year). The declarations
                start in January 2019, so "entrant" means first observed
                in the declarations within a year; true labour-market
                entry cannot be established here, and 2019 has no status.
  linkage       whether a birth year and a sex coded 1 or 2 are found in
                Individ 2023, 2021 or 2019 (the panel's rule); the
                unlinked are kept in the denominator as their own group
  code          current (the declaration year's own register, to 2023);
                carried from the 2023, 2022 or 2021 register; carried
                from an earlier one (2020, 2019); none. The first
                register holding a valid code wins, own year first, then
                the most recent earlier one. '****', empty and NULL are
                missing.
  observed lag  the source register's year minus its SsykAr column (the
                year the code was observed), 0, 1, 2 or more, or n/a if
                the column is absent (probed, not assumed)
The export carries counts and shares within month x band x status. The
summary prints, by year, the production non-match rate (the cascade the
coverage appendix uses: own register to 2022, then 2023, 2022 and 2021)
over ALL declared person-months, to reconcile with the appendix's 9.2 to
10.5 per cent, and the latest Individ vintage with its catalogue dates.

PART M. BASELINE INCUMBENTS AND NEW MATCHES (22-25)
Each worker-employer pair in a month is split by whether the pair was in
the declarations of November 2022, the last month before the launch: the
baseline incumbents, and everyone else (new matches and returns). Both
counts come from one pull and must sum to L_counts cell by cell. tau at
22-25 is fitted on each count separately, on the headline's terms and
effects (three dimensions, as the headline: within the fepois ceiling).

READ RULES, FIXED BEFORE THE RUN
  R1. THE HEADLINE COUNTS ARE THE RAW COUNTS if every cell of the rebuilt
      counts equals L_counts (share identical 1.000, maximum difference 0)
      AND tau on the rebuilt counts equals -0.0399 (0.0102) within 0.0005.
      Anything else is reported with the arithmetic, and the paper must
      then say which counts it uses.
  D and M carry no verdict: they describe and decompose.

EXPORT (output_99/)
  measurement_estimates.csv       tau rows (with var_post, var_interim,
                                  cov_post_interim), the reconstruction
                                  statistics
  measurement_totals.csv          national person-months by age band x
                                  month, rebuilt and headline
  measurement_reconciliation.csv  month x band x status x linkage x code
                                  x lag: person-months and shares (cells
                                  below five suppressed with their share)
  99_summary.txt, 99_log.txt; vcov_s99_*.csv
Identifier-level comparisons stay inside MONA; only aggregates leave.

IN THE PAPER
R: Section 2's data paragraph and OA Part I (the counts need no
occupation file); the response letter's answer to the Editor. D: OA Part
IV, tab:iv_coverage and the "Three quantities are easily confused"
paragraph (appendix_v3 ~l.1890). M: OA III.2 beside the hires and
separations rows.

    python 99_measurement.py
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

OUT = HERE / os.environ.get("CANARIES_99_OUT", "output_99")
OUT.mkdir(exist_ok=True)
os.environ.setdefault("CANARIES_82_OUT", str(OUT))
PARTS = os.environ.get("CANARIES_99_PARTS", "RDM").upper()
CACHE = mc.CACHE_DIR

FLOOR = 5
BAND = "22-25"
GATE = {"post": (-0.0578, 0.0155), "tau": (-0.0399, 0.0102)}
GATE_TOL = 0.0005
POST, INTERIM = "post_x_high_x_young", "interim_x_high_x_young"
COUNT_COLS = ["employer_id", "year_month", "age_group", "n_emp"]
SEX_COLS = ["employer_id", "year_month", "age_group", "gender", "n_emp"]
RECON_YEARS = list(range(2020, 2026))
CODE_YEARS = [2019, 2020, 2021, 2022, 2023]
BASE_MONTH = "202211"                      # the baseline for Part M
RECON_COLS = ["period", "band", "status", "linked", "codecat", "lag", "n"]
MATCH_COLS = ["employer_id", "year_month", "age_group", "at_base", "n_emp"]

NOTES: list = []
FAILURES: list = []
EST: list = []
PLANNED = 0
DONE = 0
T0 = time.time()

READ_RULES = [
    "READ RULES, FIXED BEFORE THE RUN:",
    "  GATE. The headline panel reproduces Table 1 at 22-25 within 0.0005:",
    "  adoption -0.0578 (0.0155), tau -0.0399 (0.0102); a miss stops.",
    "  R1. The headline counts are the raw counts if every rebuilt cell",
    "  equals L_counts (share identical 1.000, max difference 0) AND tau on",
    "  the rebuilt counts is -0.0399 (0.0102) within 0.0005.",
    "  D and M carry no verdict.",
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
    return s82, s61, s78, l47, l70, j47


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
    df.to_csv(OUT / "measurement_estimates.csv", index=False)


def get(part, spec, term):
    for r in EST:
        if (r["part"], r["spec"], r["term"]) == (part, spec, term):
            return r["coef"], r["se"]
    return np.nan, np.nan


def fit(b: pd.DataFrame, tag: str, terms: list, fes: tuple):
    """One Poisson fit; (coefficients, clustered vcov or None). A failure
    returns (None, None) and is recorded; R's stderr is written in full by
    mona_common._r_failed, never truncated here."""
    global DONE, PLANNED
    PLANNED += 1
    print(f"    {tag}: {len(b):,} rows, {b['employer_id'].nunique():,} firms"
          f"{mc.mem_line(' | ')}")
    t = time.time()
    try:
        r = mc.run_fepois_multi(b, OUT, tag=f"s99_{tag}", terms=terms,
                                fes=fes, cluster="employer_id")
    except BaseException as ex:
        print(f"    {tag} FAILED: {type(ex).__name__}: {ex}")
        traceback.print_exc()
        r = pd.DataFrame()
    if r.empty:
        FAILURES.append(tag)
        return None, None
    g = r.set_index("term")
    v = None
    if "vcov" in r.attrs and Path(r.attrs["vcov"]).exists():
        v = pd.read_csv(r.attrs["vcov"]).set_index("term")
    DONE += 1
    print(f"    {tag}: done in {(time.time() - t) / 60:.1f} min")
    return g, v


def tau(g, v) -> tuple:
    """(tau, se, V_pp, V_ii, V_pi); Var = V_pp + V_ii - 2 V_pi."""
    if g is None or POST not in g.index or INTERIM not in g.index:
        return (np.nan,) * 5
    c = float(g.loc[POST, "coef"]) - float(g.loc[INTERIM, "coef"])
    if v is None or POST not in v.index or INTERIM not in v.index:
        return c, np.nan, np.nan, np.nan, np.nan
    vp, vi = float(v.loc[POST, POST]), float(v.loc[INTERIM, INTERIM])
    cpi = float(v.loc[POST, INTERIM])
    var = vp + vi - 2.0 * cpi
    return c, (float(np.sqrt(var)) if var > 0 else np.nan), vp, vi, cpi


def headline_fit(counts, expo, s61, s78, j47, part, spec) -> tuple:
    """Equation (2) at 22-25 on a counts frame; records post and tau."""
    skel = s61.build_skeleton(counts, BAND, j47)
    b = s78.with_exposure(skel, expo)
    del skel
    gc.collect()
    if b.empty:
        FAILURES.append(f"{part}/{spec}/empty panel")
        return None, None
    n = int(b["employer_id"].nunique())
    b, terms = s78.eq2_terms(b)
    g, v = fit(b, spec, terms, j47.FES)
    del b
    gc.collect()
    if g is None:
        return None, None
    n_obs = int(g["n_obs"].max())
    est(part, spec, "post", g.loc[POST, "coef"], g.loc[POST, "se"], n_obs, n)
    c, s, vp, vi, cpi = tau(g, v)
    est(part, spec, "tau", c, s, n_obs, n, "derived", vp, vi, cpi)
    return c, s


def load_cache(prefix, years, require):
    out = []
    for y in years:
        c = mc.read_cache(CACHE / f"{prefix}_{y}.parquet", require=require)
        if c is None:
            return None
        out.append(c)
    return pd.concat(out, ignore_index=True)


def norm(d: pd.DataFrame, keys: list) -> pd.DataFrame:
    """Identifier and key dtypes normalised identically on every side of a
    comparison (failure class 3): int64 employer, text keys."""
    out = d.assign(employer_id=pd.to_numeric(d["employer_id"]).astype("int64"))
    for k in keys:
        if k != "employer_id":
            out[k] = out[k].astype(str).str.strip()
    return out


def compare_cells(new: pd.DataFrame, old: pd.DataFrame, keys: list,
                  label: str) -> dict:
    """Cell-by-cell comparison; returns only aggregates."""
    a = norm(new, keys).groupby(keys, observed=True)["n_emp"].sum()
    b = norm(old, keys).groupby(keys, observed=True)["n_emp"].sum()
    j = a.rename("new").to_frame().join(b.rename("old"), how="outer")
    only_new = int(j["old"].isna().sum())
    only_old = int(j["new"].isna().sum())
    j = j.fillna(0)
    diff = (j["new"] - j["old"]).abs()
    r = {"label": label, "cells": int(len(j)),
         "cells_identical": int((diff == 0).sum()),
         "share_identical": float((diff == 0).mean()) if len(j) else np.nan,
         "max_abs_diff": float(diff.max()) if len(j) else np.nan,
         "cells_only_rebuilt": only_new, "cells_only_headline": only_old,
         "person_months_rebuilt": float(j["new"].sum()),
         "person_months_headline": float(j["old"].sum())}
    print(f"  {label}: {r['cells']:,} cells, {r['share_identical']:.6f} "
          f"identical, max |diff| {r['max_abs_diff']:.0f}, person-months "
          f"{r['person_months_rebuilt']:,.0f} rebuilt vs "
          f"{r['person_months_headline']:,.0f}")
    for k in ("cells", "cells_identical", "share_identical", "max_abs_diff",
              "cells_only_rebuilt", "cells_only_headline",
              "person_months_rebuilt", "person_months_headline"):
        est("R", label, k, r[k], status="count")
    return r


# ----------------------------------------------------------------------
# Part R: the raw rebuild
# ----------------------------------------------------------------------

BAND_CASE = """CASE
             WHEN {y} - d.fodelse BETWEEN 22 AND 25 THEN '22-25'
             WHEN {y} - d.fodelse BETWEEN 26 AND 30 THEN '26-30'
             WHEN {y} - d.fodelse BETWEEN 31 AND 34 THEN '31-34'
             WHEN {y} - d.fodelse BETWEEN 35 AND 40 THEN '35-40'
             WHEN {y} - d.fodelse BETWEEN 41 AND 49 THEN '41-49'
             WHEN {y} - d.fodelse BETWEEN 50 AND 69 THEN '50+'
             ELSE NULL END"""
SEX_CASE = "CASE WHEN d.kon IN ('1', '2') THEN d.kon ELSE '0' END"


def agi_union(year: int, cols: str) -> str:
    suffix, max_month = ("_def", 12) if year < 2025 else ("_prel", 6)
    if year == 2019:
        suffix = "_def"
    return "\nUNION ALL\n".join(
        f"SELECT {cols} FROM dbo.Arb_AGIIndivid{year}{m:02d}{suffix}"
        for m in range(1, max_month + 1))


def q_counts_raw(year: int, conn) -> pd.DataFrame:
    """Written afresh (not 47L's query): distinct (employer, month,
    person) triples first, the demographic linkage once per person, then
    counts by employer x month x band x sex. No occupation column."""
    u = agi_union(year, "P1207_LOPNR_PEORGNR AS employer_id, PERIOD AS "
                        "period, P1207_LOPNR_PERSONNR AS person_id")
    bc = BAND_CASE.format(y=year)
    q = f"""
    WITH trip AS (SELECT DISTINCT employer_id, period, person_id FROM ({u}) x),
    pers AS (SELECT DISTINCT person_id FROM trip),
    d AS (
        SELECT p.person_id,
               COALESCE(TRY_CAST(a.FodelseAr AS INT), TRY_CAST(b.FodelseAr AS INT),
                        TRY_CAST(c.FodelseAr AS INT)) AS fodelse,
               LTRIM(RTRIM(COALESCE(a.Kon, b.Kon, c.Kon))) AS kon
        FROM pers p
        LEFT JOIN dbo.Individ_2023 a ON p.person_id = a.P1207_LopNr_PersonNr
        LEFT JOIN dbo.Individ_2021 b ON p.person_id = b.P1207_LopNr_PersonNr
        LEFT JOIN dbo.Individ_2019 c ON p.person_id = c.P1207_LopNr_PersonNr)
    SELECT t.employer_id,
           LEFT(t.period,4) + '-' + SUBSTRING(t.period,5,2) AS year_month,
           {bc} AS age_group, {SEX_CASE} AS gender,
           COUNT(DISTINCT t.person_id) AS n_emp
    FROM trip t JOIN d ON t.person_id = d.person_id
    WHERE d.fodelse IS NOT NULL AND {year} - d.fodelse BETWEEN 22 AND 69
    GROUP BY t.employer_id, t.period, {bc}, {SEX_CASE}
    """
    return pd.read_sql(q, conn)


def pulled(prefix: str, years, require, query) -> pd.DataFrame:
    out, conn = [], None
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
        out.append(c)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
    return pd.concat(out, ignore_index=True)


def score_check(built: dict, s82) -> None:
    """The quartile refitted with every code recorded after 2019 removed
    from the incumbent frame: no employer's quartile may move."""
    inc = built["inc"]
    yrs = inc["source_year"].astype(str)
    late = ~(yrs.isin([str(y) for y in s82.CASCADE_BACK]) | (yrs == "none"))
    e2 = s82.occ_route_exposure(inc[~late], built["nfloor"], s82.FLOOR_MAIN,
                                s82.ARM_YEARS[s82.MAIN_ARM])
    m = built["exposure"][["employer_id", "fq"]].merge(
        e2[["employer_id", "fq"]], on="employer_id", how="outer",
        suffixes=("", "_pre2020"))
    moved = int((m["fq"] != m["fq_pre2020"]).sum())
    msg = (f"score: {int(late.sum()):,} incumbent-frame rows carry a code "
           f"recorded after 2019 (the forward arm); with them removed "
           f"{moved:,} of {len(m):,} employers change quartile")
    NOTES.append(msg)
    print(f"  {msg}")
    est("R", "score_without_post2019_codes", "employers_changing_quartile",
        moved, status="count")


def part_r(expo, s61, s78, j47) -> None:
    print("\n  PART R, the raw rebuild:")
    raw = pulled("R_counts_raw", s61.PANEL_YEARS,
                 ["employer_id", "year_month", "age_group", "gender", "n_emp"],
                 q_counts_raw)
    raw["gender"] = raw["gender"].astype(str).str.strip()
    stock = (raw.groupby(["employer_id", "year_month", "age_group"],
                         observed=True)["n_emp"].sum().reset_index())
    tot = []
    for y in s61.PANEL_YEARS:
        old = mc.read_cache(CACHE / f"L_counts_{y}.parquet", require=COUNT_COLS)
        if old is None:
            FAILURES.append(f"R/L_counts_{y} missing")
            continue
        new = stock[stock["year_month"].astype(str).str.slice(0, 4) == str(y)]
        compare_cells(new, old, ["employer_id", "year_month", "age_group"],
                      f"all_sexes_{y}")
        for src, d in (("rebuilt", new), ("headline", old)):
            t = (d.assign(year_month=d["year_month"].astype(str),
                          age_group=d["age_group"].astype(str))
                 .groupby(["year_month", "age_group"])["n_emp"].sum()
                 .reset_index().assign(source=src))
            tot.append(t)
        olds = mc.read_cache(CACHE / f"L_counts_sex_{y}.parquet",
                             require=SEX_COLS)
        if olds is not None:
            news = raw[(raw["year_month"].astype(str).str.slice(0, 4) == str(y))
                       & raw["gender"].isin(["1", "2"])]
            compare_cells(news, olds, ["employer_id", "year_month",
                                       "age_group", "gender"], f"by_sex_{y}")
        del old, new
        gc.collect()
    if tot:
        T = pd.concat(tot, ignore_index=True)
        T = mc.enforce_min_cell(T, count_col="n_emp", floor=FLOOR)
        T.to_csv(OUT / "measurement_totals.csv", index=False)
    del raw
    gc.collect()
    # the estimation panel itself, headline counts against rebuilt ones
    lc = load_cache("L_counts", s61.PANEL_YEARS, COUNT_COLS)
    sk_old = s78.with_exposure(s61.build_skeleton(lc, BAND, j47), expo)
    del lc
    gc.collect()
    sk_new = s78.with_exposure(s61.build_skeleton(stock, BAND, j47), expo)
    k = ["employer_id", "age_group", "year_month"]
    compare_cells(sk_new[k + ["n_emp"]], sk_old[k + ["n_emp"]], k,
                  "estimation_panel_22_25")
    del sk_old, sk_new
    gc.collect()
    c, s = headline_fit(stock, expo, s61, s78, j47, "R", "rebuilt_22_25")
    del stock
    gc.collect()
    if c == c:
        print(f"  tau on the rebuilt counts {c:+.4f} ({s:.4f}); Table 1 "
              f"{GATE['tau'][0]:+.4f} ({GATE['tau'][1]:.4f})")


# ----------------------------------------------------------------------
# Part D: the reconciliation
# ----------------------------------------------------------------------

def probe_individ(conn, s82) -> dict:
    """Which code and observed-year columns each Individ table carries,
    and what the catalogue says about the vintages."""
    cat = s82.individ_catalogue(conn, CODE_YEARS)
    plan = {}
    for y in CODE_YEARS:
        have = cat.get(y, {})
        code = s82.pick_col(have, "Ssyk4_2012_J16", "Ssyk4_2012")
        ar = s82.pick_col(have, "SsykAr_J16", "SsykAr")
        if code is None:
            raise RuntimeError(f"Individ_{y}: no SSYK 2012 four-digit column "
                               f"({len(have)} columns); the reconciliation "
                               f"would misclassify every worker")
        plan[y] = (code, ar)
    NOTES.append("D: code columns " + ", ".join(
        f"{y} {c}/{a or 'no SsykAr'}" for y, (c, a) in plan.items()))
    try:
        t = pd.read_sql("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
                        "WHERE TABLE_NAME LIKE 'Individ[_]%'", conn)
        yrs = sorted(int(x[-4:]) for x in t["TABLE_NAME"].astype(str)
                     if x[-4:].isdigit())
        NOTES.append(f"D: Individ vintages in the extract {yrs[0]}-{yrs[-1]}; "
                     f"latest Individ_{yrs[-1]}")
        try:
            d = pd.read_sql(
                "SELECT name, create_date, modify_date FROM sys.objects "
                f"WHERE name = 'Individ_{yrs[-1]}'", conn)
            if len(d):
                NOTES.append(f"D: Individ_{yrs[-1]} created "
                             f"{d['create_date'].iloc[0]}, modified "
                             f"{d['modify_date'].iloc[0]} (catalogue dates; "
                             f"the delivery's own extraction date is not "
                             f"recorded in the database)")
        except Exception as ex:
            NOTES.append(f"D: catalogue dates not readable "
                         f"({type(ex).__name__})")
    except Exception as ex:
        NOTES.append(f"D: vintage listing failed ({type(ex).__name__})")
    return plan


def valid(col: str) -> str:
    v = f"LTRIM(CAST({col} AS VARCHAR(8)))"
    return f"({col} IS NOT NULL AND {v} <> '' AND LEFT({v}, 1) <> '*')"


def q_recon(year: int, conn, plan: dict) -> pd.DataFrame:
    """All declared person-months of `year`, classified by band, status
    against the previous year's declarations, linkage and code source."""
    alias = {y: f"i{str(y)[2:]}" for y in CODE_YEARS}
    order = ([year] if year <= 2023 else []) + \
        [y for y in sorted(CODE_YEARS, reverse=True) if y < year]
    label = {}
    for y in order:
        label[y] = ("current" if y == year else
                    f"carried_{y}" if y >= 2021 else "carried_earlier")
    code_case, lag_case = [], []
    for y in order:
        code, ar = plan[y]
        col = f"{alias[y]}.[{code}]"
        code_case.append(f"WHEN {valid(col)} THEN '{label[y]}'")
        if ar:
            lagv = f"({y} - TRY_CAST({alias[y]}.[{ar}] AS INT))"
            lag_case.append(f"WHEN {valid(col)} THEN CASE WHEN {lagv} IS NULL "
                            f"THEN 'na' WHEN {lagv} <= 0 THEN '0' WHEN {lagv} = 1 "
                            f"THEN '1' ELSE '2+' END")
        else:
            lag_case.append(f"WHEN {valid(col)} THEN 'na'")
    joins = "\n".join(
        f"LEFT JOIN dbo.Individ_{y} {alias[y]} "
        f"ON c.per = {alias[y]}.P1207_LopNr_PersonNr" for y in CODE_YEARS)
    fod = ("COALESCE(TRY_CAST(i23.FodelseAr AS INT), TRY_CAST(i21.FodelseAr AS "
           "INT), TRY_CAST(i19.FodelseAr AS INT))")
    kon = "LTRIM(RTRIM(COALESCE(i23.Kon, i21.Kon, i19.Kon)))"
    band = f"""CASE WHEN {fod} IS NULL THEN 'unknown'
                    WHEN {year} - {fod} < 22 THEN 'under22'
                    WHEN {year} - {fod} <= 25 THEN '22-25'
                    WHEN {year} - {fod} <= 30 THEN '26-30'
                    WHEN {year} - {fod} <= 34 THEN '31-34'
                    WHEN {year} - {fod} <= 40 THEN '35-40'
                    WHEN {year} - {fod} <= 49 THEN '41-49'
                    WHEN {year} - {fod} <= 69 THEN '50-69'
                    ELSE 'over69' END"""
    cur = agi_union(year, "P1207_LOPNR_PEORGNR AS emp, PERIOD AS period, "
                          "P1207_LOPNR_PERSONNR AS per")
    prv = agi_union(year - 1, "P1207_LOPNR_PEORGNR AS emp, "
                              "P1207_LOPNR_PERSONNR AS per")
    q = f"""
    WITH c0 AS (SELECT DISTINCT emp, period, per FROM ({cur}) x),
    pv AS (SELECT DISTINCT emp, per FROM ({prv}) y),
    pp AS (SELECT DISTINCT per FROM pv),
    cls AS (
        SELECT c.emp, c.period, c.per,
               CASE WHEN pv.per IS NOT NULL THEN 'incumbent'
                    WHEN pp.per IS NOT NULL THEN 'new_match'
                    ELSE 'entrant' END AS status,
               {band} AS band,
               CASE WHEN {fod} IS NOT NULL AND {kon} IN ('1', '2')
                    THEN 'yes' ELSE 'no' END AS linked,
               CASE {' '.join(code_case)} ELSE 'none' END AS codecat,
               CASE {' '.join(lag_case)} ELSE 'na' END AS lag
        FROM c0 c
        LEFT JOIN pv ON pv.per = c.per AND pv.emp = c.emp
        LEFT JOIN pp ON pp.per = c.per
        {joins}),
    percell AS (
        SELECT emp, period, band, status, linked, codecat, lag,
               COUNT(DISTINCT per) AS n
        FROM cls GROUP BY emp, period, band, status, linked, codecat, lag)
    SELECT period, band, status, linked, codecat, lag, SUM(n) AS n
    FROM percell GROUP BY period, band, status, linked, codecat, lag
    """
    return pd.read_sql(q, conn)


def production_coded(codecat: pd.Series, year: int) -> pd.Series:
    """The production cascade the coverage appendix uses: the year's own
    register to 2022; the 2023, 2022 or 2021 register from 2023."""
    if year <= 2022:
        return codecat == "current"
    return codecat.isin(["current", "carried_2023", "carried_2022",
                         "carried_2021"])


def part_d(s82) -> None:
    print("\n  PART D, the reconciliation:")
    conn = None
    need = [y for y in RECON_YEARS
            if mc.read_cache(CACHE / f"R_recon_{y}.parquet",
                             require=RECON_COLS) is None]
    plan = None
    if need:
        conn = mc.connect()
        plan = probe_individ(conn, s82)
    out, lines = [], []
    for y in RECON_YEARS:
        cf = CACHE / f"R_recon_{y}.parquet"
        d = mc.read_cache(cf, require=RECON_COLS)
        if d is None:
            t = time.time()
            try:
                d = q_recon(y, conn, plan)
            except BaseException as ex:
                print(f"  D {y} FAILED: {type(ex).__name__}: {ex}")
                traceback.print_exc()
                FAILURES.append(f"D/{y}/{type(ex).__name__}")
                continue
            mc.write_cache(d, cf)
            print(f"  D {y}: {len(d):,} rows ({(time.time() - t) / 60:.1f} min)")
        d = d.assign(year=y, n=pd.to_numeric(d["n"]))
        for c in ("period", "band", "status", "linked", "codecat", "lag"):
            d[c] = d[c].astype(str).str.strip()
        out.append(d)
        allpm = float(d["n"].sum())
        nm = float(d.loc[~production_coded(d["codecat"], y), "n"].sum())
        nm_l = float(d.loc[(d["linked"] == "yes")
                           & ~production_coded(d["codecat"], y), "n"].sum())
        lines.append(f"  {y}: {allpm:,.0f} declared person-months; production "
                     f"non-match {nm / allpm:.2%} of all "
                     f"({nm_l / allpm:.2%} among the linked); unlinked "
                     f"{float(d.loc[d['linked'] == 'no', 'n'].sum()) / allpm:.2%}")
        est("D", f"year_{y}", "production_nonmatch_share", nm / allpm,
            status="share")
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
    if not out:
        return
    D = pd.concat(out, ignore_index=True)
    # the category the review asked for: the unlinked are their own group
    D["category"] = np.where(D["linked"] == "no", "no_age_sex", D["codecat"])
    D["share_of_month_band_status"] = D["n"] / D.groupby(
        ["period", "band", "status"])["n"].transform("sum")
    D = D.rename(columns={"n": "person_months"})
    had = D["person_months"].notna()
    D = mc.enforce_min_cell(D, count_col="person_months", floor=FLOOR)
    D.loc[had & D["person_months"].isna(), "share_of_month_band_status"] = np.nan
    D.to_csv(OUT / "measurement_reconciliation.csv", index=False)
    NOTES.append("D: " + "; ".join(x.strip() for x in lines))
    for x in lines:
        print(x)


# ----------------------------------------------------------------------
# Part M: baseline incumbents and new matches
# ----------------------------------------------------------------------

def q_basematch(year: int, conn) -> pd.DataFrame:
    """47L's counts with one column added: whether the worker-employer
    pair was in the declarations of November 2022."""
    suffix, max_month = ("_def", 12) if year < 2025 else ("_prel", 6)
    monthly = "\nUNION ALL\n".join(f"""
        SELECT agi.P1207_LOPNR_PEORGNR AS employer_id,
               agi.PERIOD AS period, agi.P1207_LOPNR_PERSONNR AS person_id,
               COALESCE(TRY_CAST(a.FodelseAr AS INT), TRY_CAST(b.FodelseAr AS INT),
                        TRY_CAST(c.FodelseAr AS INT)) AS fodelse
        FROM dbo.Arb_AGIIndivid{year}{m:02d}{suffix} agi
        LEFT JOIN dbo.Individ_2023 a ON agi.P1207_LOPNR_PERSONNR = a.P1207_LopNr_PersonNr
        LEFT JOIN dbo.Individ_2021 b ON agi.P1207_LOPNR_PERSONNR = b.P1207_LopNr_PersonNr
        LEFT JOIN dbo.Individ_2019 c ON agi.P1207_LOPNR_PERSONNR = c.P1207_LopNr_PersonNr
        """ for m in range(1, max_month + 1))
    band = """CASE
             WHEN age BETWEEN 22 AND 25 THEN '22-25'
             WHEN age BETWEEN 26 AND 30 THEN '26-30'
             WHEN age BETWEEN 31 AND 34 THEN '31-34'
             WHEN age BETWEEN 35 AND 40 THEN '35-40'
             WHEN age BETWEEN 41 AND 49 THEN '41-49'
             WHEN age BETWEEN 50 AND 69 THEN '50+'
             ELSE NULL END"""
    q = f"""
    WITH base AS ({monthly}),
    bp AS (SELECT DISTINCT P1207_LOPNR_PEORGNR AS emp,
                  P1207_LOPNR_PERSONNR AS per
           FROM dbo.Arb_AGIIndivid{BASE_MONTH}_def),
    aged AS (
        SELECT x.employer_id, x.period, x.person_id, {year} - x.fodelse AS age,
               CASE WHEN bp.per IS NOT NULL THEN 'baseline' ELSE 'other' END
                   AS at_base
        FROM base x
        LEFT JOIN bp ON bp.emp = x.employer_id AND bp.per = x.person_id
        WHERE x.fodelse IS NOT NULL)
    SELECT employer_id,
           LEFT(period,4) + '-' + SUBSTRING(period,5,2) AS year_month,
           {band} AS age_group, at_base,
           COUNT(DISTINCT person_id) AS n_emp
    FROM aged
    WHERE age BETWEEN 22 AND 69
    GROUP BY employer_id, period, {band}, at_base
    """
    return pd.read_sql(q, conn)


def part_m(expo, s61, s78, j47) -> None:
    print("\n  PART M, baseline incumbents and new matches:")
    bm = pulled("L_counts_basematch", s61.PANEL_YEARS, MATCH_COLS, q_basematch)
    bm["at_base"] = bm["at_base"].astype(str).str.strip()
    lc = load_cache("L_counts", s61.PANEL_YEARS, COUNT_COLS)
    if lc is not None:
        compare_cells(bm, lc, ["employer_id", "year_month", "age_group"],
                      "basematch_sum_vs_headline")
    del lc
    gc.collect()
    for grp in ("baseline", "other"):
        c = bm[bm["at_base"] == grp].drop(columns="at_base")
        tv, sv = headline_fit(c, expo, s61, s78, j47, "M", f"{grp}_22_25")
        print(f"  {grp}: tau {tv:+.4f} ({sv:.4f})")
        del c
        gc.collect()


# ----------------------------------------------------------------------
# summary and main
# ----------------------------------------------------------------------

def write_summary() -> None:
    L = ["MEASUREMENT: RAW REBUILD, DENOMINATOR, INCUMBENTS AND NEW MATCHES",
         "=" * 66, ""]
    c, s = get("G", "gate_22_25", "tau")
    if c == c:
        L.append(f"GATE (L_counts): tau {c:+.4f} ({s:.4f}); Table 1 "
                 f"{GATE['tau'][0]:+.4f} ({GATE['tau'][1]:.4f})")
    r = [x for x in EST if x["part"] == "R" and x["term"] == "share_identical"]
    if r:
        L += ["", "R. THE RAW REBUILD AGAINST THE HEADLINE COUNTS:"]
        for x in r:
            mx = next((y["coef"] for y in EST if y["spec"] == x["spec"]
                       and y["term"] == "max_abs_diff"), np.nan)
            L.append(f"  {x['spec']:<28} share identical {x['coef']:.6f}, "
                     f"max |diff| {mx:.0f}")
        c, s = get("R", "rebuilt_22_25", "tau")
        L.append(f"  tau on the rebuilt counts {c:+.4f} ({s:.4f})")
        allid = all(x["coef"] == 1.0 for x in r
                    if x["spec"].startswith("all_sexes"))
        ok = allid and c == c and abs(c - GATE["tau"][0]) <= GATE_TOL \
            and abs(s - GATE["tau"][1]) <= GATE_TOL
        L.append(f"  R1: {'THE HEADLINE COUNTS ARE THE RAW COUNTS' if ok else 'R1 NOT MET: see the arithmetic above'}")
    d = [x for x in EST if x["part"] == "D"]
    if d:
        L += ["", "D. PRODUCTION NON-MATCH OVER ALL DECLARED PERSON-MONTHS "
              "(appendix: 9.2 to 10.5 per cent):"]
        L += [f"  {x['spec'][5:]}: {x['coef']:.2%}" for x in d]
    m = [x for x in EST if x["part"] == "M" and x["term"] == "tau"]
    if m:
        L += ["", "M. TAU AT 22-25 BY PRESENCE AT THE EMPLOYER IN NOVEMBER 2022:"]
        L += [f"  {x['spec']:<16} {x['coef']:+.4f} ({x['se']:.4f})" for x in m]
    L += ["", f"FITS: {DONE} of {PLANNED} attempted came back. A run far "
          "shorter than the estimate (2.5 to 3 hours) is a failure."]
    if NOTES:
        L += ["", "NOTES:"] + [f"  {n}" for n in NOTES]
    if FAILURES:
        L += ["", "FAILED: " + " | ".join(FAILURES),
              "A missing row is a missing fit, never a zero."]
    L += [""] + READ_RULES + ["", f"Runtime {(time.time() - T0) / 60:.1f} "
                              "min. " + mc.mem_line("")]
    (OUT / "99_summary.txt").write_text("\n".join(L), encoding="utf-8")
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
    mc.Tee(OUT / "99_log.txt")
    T0 = time.time()
    print("=" * 70)
    print(f"99: MEASUREMENT   parts {PARTS}")
    print("=" * 70)
    print("\n".join(READ_RULES))
    print(mc.mem_line("  "))
    rc = 0
    try:
        s82, s61, s78, l47, l70, j47 = load_modules()
        built = s82.build_exposure(l47, l70, j47)
        drain(s82, "82")
        expo = built["exposure"]
        score_check(built, s82)
        del built
        gc.collect()
        lc = load_cache("L_counts", s61.PANEL_YEARS, COUNT_COLS)
        if lc is None:
            raise RuntimeError("L_counts_2021-2025 missing; run 47L")
        print("\n  GATE on the headline panel:")
        c, s = headline_fit(lc, expo, s61, s78, j47, "G", "gate_22_25")
        del lc
        gc.collect()
        p, ps = get("G", "gate_22_25", "post")
        bad = [f"{k}: this run {x:+.4f} ({y:.4f}), Table 1 {GATE[k][0]:+.4f} "
               f"({GATE[k][1]:.4f})" for k, (x, y) in
               (("post", (p, ps)), ("tau", (c, s)))
               if not (abs(x - GATE[k][0]) <= GATE_TOL
                       and abs(y - GATE[k][1]) <= GATE_TOL)]
        if bad:
            FAILURES.append("THE GATE FAILED: " + "; ".join(bad))
            write_summary()
            raise SystemExit("99: the gate failed; stopping.")
        print(f"  THE GATE PASSES: tau {c:+.4f} ({s:.4f})")
        if "R" in PARTS:
            run_part("R", part_r, expo, s61, s78, j47)
        if "D" in PARTS:
            run_part("D", part_d, s82)
        if "M" in PARTS:
            run_part("M", part_m, expo, s61, s78, j47)
        drain(s78, "78")
    except SystemExit:
        mc.runlog("99_measurement", 2, (time.time() - T0) / 60)
        raise
    except BaseException as ex:
        print(f"99 FAILED: {type(ex).__name__}: {ex}")
        traceback.print_exc()
        FAILURES.append(f"main/{type(ex).__name__}: {ex}")
        rc = 1
    save_est()
    write_summary()
    rc = rc or (1 if FAILURES else 0)
    mc.runlog("99_measurement", rc, (time.time() - T0) / 60)
    print("\n99 done.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
