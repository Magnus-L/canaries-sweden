#!/usr/bin/env python3
"""
50_sim_moments.py -- calibration moments for the simulation study, all
aggregates, all export-safe (cells >= 5), Individ + HREG + one AGI month.

======================================================================
  RUNS IN SCB's MONA SECURE ENVIRONMENT ONLY. Standalone: submit THIS
  file. Writes output_50/. About 15 minutes. Needs nothing from 47h,
  but reads 47h's weight and collapse caches if they exist.
  Local end-to-end test: revision/local/test_50_synthetic.py
======================================================================

WHY (19 Sep 2026). The simulation study (notes/simulation-study-spec_
2026-09-19.md) needs a data-generating process calibrated to MEASURED
moments, not to our beliefs: 47b failed because the mechanism (22-25s
still completing the education the register records) was not in our
model. ML's further point: the occupations reachable from an education
drift with years since completion in education-specific ways (nurse ->
care-unit manager, guard -> police), so exposure is a function of
(education, experience, cohort). M6 measures exactly that matrix across
the five SUN 2020 cross-sections, where the same experience band in
2019 and 2023 is a different cohort.

MOMENTS
  M1a completion timing: age at ExamAr by SUN level, per year
  M1b level change 22-35 between Individ_t and Individ_t+1 (the hazard
      behind the 47b artefact), by age band
  M2  occupation change among freshly coded in t and t+1: by age band and
      experience band, code change / quartile change / entry into 1xxx
  M3  staleness: year - SsykAr_J16 by age band and year, and SsykStatus
  M4a enrolment prevalence among 22-30 by tertiary status, 2021-2023
  M4b field switch: last registration field vs later ExamAr field
  M5  employer size bands (AGI 2019-11) and, from 47h caches if present,
      the balanced-panel retention share for 22-25 by size band
  M6  the (group x experience band x ssyk4) matrix per year 2019-2023,
      one file per year (export cap 5 MB per file)

EXPORT SAFETY: every table passes floor5() before writing; no identifiers;
no raw rows. Cells 1-4 are suppressed (NaN), zeros stay.
"""

import gc
import hashlib
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import mona_common as mc

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_50"
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR

YEARS = [2019, 2020, 2021, 2022, 2023]
KEY_PATH = str(Path(mc.SHARE) / "utb_grupp2_sun2020_niva3_inr4_nyckel.dta")
KEY_SHA256 = "c760361ba21554951a0744ee00de2f02f22f2e021b87f0863d9ece049e786637"
DAIOE_PATH = str(Path(mc.SHARE) / "daioe_quartiles.dta")
FLOOR = 5

AGE_CASE = """CASE
        WHEN {y} - TRY_CAST(FodelseAr AS INT) BETWEEN 22 AND 25 THEN '22-25'
        WHEN {y} - TRY_CAST(FodelseAr AS INT) BETWEEN 26 AND 30 THEN '26-30'
        WHEN {y} - TRY_CAST(FodelseAr AS INT) BETWEEN 31 AND 34 THEN '31-34'
        WHEN {y} - TRY_CAST(FodelseAr AS INT) BETWEEN 35 AND 40 THEN '35-40'
        WHEN {y} - TRY_CAST(FodelseAr AS INT) BETWEEN 41 AND 49 THEN '41-49'
        WHEN {y} - TRY_CAST(FodelseAr AS INT) BETWEEN 50 AND 69 THEN '50+'
        ELSE NULL END"""
EXP_CASE = """CASE WHEN TRY_CAST({a}.ExamAr AS INT) IS NULL THEN 'na'
        WHEN {y} - TRY_CAST({a}.ExamAr AS INT) <= 2  THEN '0-2'
        WHEN {y} - TRY_CAST({a}.ExamAr AS INT) <= 5  THEN '3-5'
        WHEN {y} - TRY_CAST({a}.ExamAr AS INT) <= 10 THEN '6-10'
        WHEN {y} - TRY_CAST({a}.ExamAr AS INT) <= 20 THEN '11-20'
        ELSE '21+' END"""
CODED = ("{a}.Ssyk4_2012_J16 IS NOT NULL AND LTRIM({a}.Ssyk4_2012_J16) <> '' "
         "AND LEFT(LTRIM({a}.Ssyk4_2012_J16), 1) <> '*'")
SSYK4 = "RIGHT('0000' + CAST({a}.Ssyk4_2012_J16 AS VARCHAR(4)), 4)"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def norm_code(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip().str.lower()
    return out.where(out.notna() & (out != ""), other=pd.NA)


def floor5(df: pd.DataFrame, col: str = "n") -> pd.DataFrame:
    """Suppress counts 1-4; zeros stay. Applied to EVERY export."""
    out = df.copy()
    small = (out[col] > 0) & (out[col] < FLOOR)
    if small.any():
        print(f"    floor: {int(small.sum()):,} cells < {FLOOR} suppressed")
        out.loc[small, col] = np.nan
    return out


def export(df: pd.DataFrame, name: str, col: str = "n"):
    df = floor5(df, col)
    df.to_csv(OUT / name, index=False)
    print(f"  wrote {name}: {len(df):,} rows")


def load_key() -> pd.DataFrame:
    got = hashlib.sha256(Path(KEY_PATH).read_bytes()).hexdigest()
    if got != KEY_SHA256:
        raise RuntimeError(f"key hash mismatch: {got[:16]}...")
    key = pd.read_stata(KEY_PATH).rename(columns={
        "sun2020niva_3_kod": "niva", "sun2020inr_4_kod": "inr", "utb_grupp2": "grp"})
    key["niva"], key["inr"] = norm_code(key["niva"]), norm_code(key["inr"])
    return key.dropna(subset=["niva", "inr"])[["niva", "inr", "grp"]]


def load_daioe() -> pd.DataFrame:
    got = hashlib.sha256(Path(DAIOE_PATH).read_bytes()).hexdigest()
    if got != mc.DAIOE_SHA256:
        raise RuntimeError(f"daioe hash mismatch: {got[:16]}...")
    d = pd.read_stata(DAIOE_PATH)
    d["ssyk4"] = d["ssyk4"].astype(str).str.zfill(4)
    q = d["exposure_quartile"]
    if not pd.api.types.is_numeric_dtype(q):
        q = q.astype(str).str.extract(r"(\d)")[0].astype(int)
    d["q"] = q.astype(int)
    return d[["ssyk4", "q"]]


# ----------------------------------------------------------------------
# SQL pulls (patched out by the local test)
# ----------------------------------------------------------------------

def q_completion_age(y: int, conn) -> pd.DataFrame:
    """M1a: age at completion of the highest education, by SUN level digit."""
    q = f"""
    SELECT LEFT(LTRIM(Sun2020Niva), 1) AS level,
           TRY_CAST(ExamAr AS INT) - TRY_CAST(FodelseAr AS INT) AS exam_age,
           COUNT(*) AS n
    FROM dbo.Individ_{y}
    WHERE TRY_CAST(ExamAr AS INT) IS NOT NULL AND TRY_CAST(FodelseAr AS INT) IS NOT NULL
      AND Sun2020Niva IS NOT NULL AND LTRIM(Sun2020Niva) <> ''
    GROUP BY LEFT(LTRIM(Sun2020Niva), 1),
             TRY_CAST(ExamAr AS INT) - TRY_CAST(FodelseAr AS INT)
    """
    return pd.read_sql(q, conn)


def q_level_change(t: int, conn) -> pd.DataFrame:
    """M1b: highest-education level in t and t+1 for the same person, 22-35."""
    q = f"""
    SELECT {AGE_CASE.format(y=t).replace('FodelseAr', 'a.FodelseAr')} AS age_group,
           LEFT(LTRIM(a.Sun2020Niva), 1) AS level_t,
           LEFT(LTRIM(b.Sun2020Niva), 1) AS level_t1,
           COUNT(*) AS n
    FROM dbo.Individ_{t} a
    JOIN dbo.Individ_{t + 1} b ON a.P1207_LopNr_PersonNr = b.P1207_LopNr_PersonNr
    WHERE {t} - TRY_CAST(a.FodelseAr AS INT) BETWEEN 22 AND 35
    GROUP BY {AGE_CASE.format(y=t).replace('FodelseAr', 'a.FodelseAr')},
             LEFT(LTRIM(a.Sun2020Niva), 1), LEFT(LTRIM(b.Sun2020Niva), 1)
    """
    return pd.read_sql(q, conn)


def q_occ_change(t: int, conn) -> pd.DataFrame:
    """M2: code in t and t+1 among persons FRESHLY coded in both years, by
    age band and experience band. Aggregated at ssyk4 x ssyk4 (internal);
    collapsed to quartile transitions before export."""
    age = AGE_CASE.format(y=t).replace("FodelseAr", "a.FodelseAr")
    q = f"""
    SELECT {age} AS age_group, {EXP_CASE.format(a='a', y=t)} AS expband,
           {SSYK4.format(a='a')} AS ssyk4_t, {SSYK4.format(a='b')} AS ssyk4_t1,
           COUNT(*) AS n
    FROM dbo.Individ_{t} a
    JOIN dbo.Individ_{t + 1} b ON a.P1207_LopNr_PersonNr = b.P1207_LopNr_PersonNr
    WHERE {CODED.format(a='a')} AND {CODED.format(a='b')}
      AND TRY_CAST(a.SsykAr_J16 AS INT) = {t} AND TRY_CAST(b.SsykAr_J16 AS INT) = {t + 1}
      AND {t} - TRY_CAST(a.FodelseAr AS INT) BETWEEN 22 AND 69
    GROUP BY {age}, {EXP_CASE.format(a='a', y=t)},
             {SSYK4.format(a='a')}, {SSYK4.format(a='b')}
    """
    return pd.read_sql(q, conn)


def q_staleness(y: int, conn) -> pd.DataFrame:
    """M3: years since the occupation code was assigned, by age band."""
    age = AGE_CASE.format(y=y)
    q = f"""
    SELECT {age} AS age_group,
           {y} - TRY_CAST(SsykAr_J16 AS INT) AS stale_years,
           LTRIM(RTRIM(SsykStatus_J16)) AS ssyk_status,
           COUNT(*) AS n
    FROM dbo.Individ_{y} a
    WHERE {CODED.format(a='a')} AND {y} - TRY_CAST(FodelseAr AS INT) BETWEEN 22 AND 69
    GROUP BY {age}, {y} - TRY_CAST(SsykAr_J16 AS INT), LTRIM(RTRIM(SsykStatus_J16))
    """
    return pd.read_sql(q, conn)


def q_enrolment(y: int, conn) -> pd.DataFrame:
    """M4a: among 22-30 in Individ_y, tertiary completed or not, and whether a
    registration exists within the three academic years before y."""
    age = AGE_CASE.format(y=y).replace("FodelseAr", "i.FodelseAr")
    q = f"""
    WITH reg AS (
        SELECT DISTINCT P1207_Lopnr_Personnr AS person_id
        FROM dbo.HREG_AKTIVITET_1971_2021
        WHERE TRY_CAST(LEFT(ARTERMIN, 4) AS INT) BETWEEN {min(y, 2021) - 3} AND {min(y, 2021)}
    )
    SELECT {age} AS age_group,
           CASE WHEN LEFT(LTRIM(i.Sun2020Niva), 1) IN ('4','5','6') THEN 1 ELSE 0 END AS tertiary,
           CASE WHEN r.person_id IS NULL THEN 0 ELSE 1 END AS registered,
           COUNT(*) AS n
    FROM dbo.Individ_{y} i
    LEFT JOIN reg r ON i.P1207_LopNr_PersonNr = r.person_id
    WHERE {y} - TRY_CAST(i.FodelseAr AS INT) BETWEEN 22 AND 30
    GROUP BY {age},
             CASE WHEN LEFT(LTRIM(i.Sun2020Niva), 1) IN ('4','5','6') THEN 1 ELSE 0 END,
             CASE WHEN r.person_id IS NULL THEN 0 ELSE 1 END
    """
    return pd.read_sql(q, conn)


def q_field_switch(conn) -> pd.DataFrame:
    """M4b: for tertiary completers 2019-2023 (Individ_2023 record), the field
    of the last registration before the completion year vs the completed
    field: same / different / no registration found."""
    q = """
    WITH comp AS (
        SELECT P1207_LopNr_PersonNr AS person_id,
               TRY_CAST(ExamAr AS INT) AS exam_year,
               NULLIF(LTRIM(RTRIM(Sun2020Inr)), '') AS inr_done
        FROM dbo.Individ_2023
        WHERE LEFT(LTRIM(Sun2020Niva), 1) IN ('4','5','6')
          AND TRY_CAST(ExamAr AS INT) BETWEEN 2019 AND 2023
    ),
    reg AS (
        SELECT c.person_id, c.exam_year, c.inr_done,
               NULLIF(LTRIM(RTRIM(h.SUN2020INR)), '') AS inr_reg,
               ROW_NUMBER() OVER (PARTITION BY c.person_id
                                  ORDER BY h.ARTERMIN DESC, ISNULL(h.AKTPROC, 0) DESC) AS rn
        FROM comp c
        JOIN dbo.HREG_AKTIVITET_1971_2021 h ON c.person_id = h.P1207_Lopnr_Personnr
        WHERE TRY_CAST(LEFT(h.ARTERMIN, 4) AS INT) BETWEEN c.exam_year - 6 AND c.exam_year - 1
    )
    SELECT c.exam_year,
           CASE WHEN r.person_id IS NULL THEN 'none'
                WHEN r.inr_reg = c.inr_done THEN 'same'
                WHEN LEFT(r.inr_reg, 3) = LEFT(c.inr_done, 3) THEN 'same3'
                ELSE 'different' END AS match,
           COUNT(*) AS n
    FROM comp c
    LEFT JOIN reg r ON c.person_id = r.person_id AND r.rn = 1
    GROUP BY c.exam_year,
             CASE WHEN r.person_id IS NULL THEN 'none'
                  WHEN r.inr_reg = c.inr_done THEN 'same'
                  WHEN LEFT(r.inr_reg, 3) = LEFT(c.inr_done, 3) THEN 'same3'
                  ELSE 'different' END
    """
    return pd.read_sql(q, conn)


def q_employer_size(conn) -> pd.DataFrame:
    """M5: employers by size band, AGI November 2019."""
    q = """
    WITH e AS (
        SELECT P1207_LOPNR_PEORGNR AS employer_id,
               COUNT(DISTINCT P1207_LOPNR_PERSONNR) AS size
        FROM dbo.Arb_AGIIndivid201911_def
        GROUP BY P1207_LOPNR_PEORGNR
    )
    SELECT CASE WHEN size < 5 THEN '1-4' WHEN size < 10 THEN '5-9'
                WHEN size < 20 THEN '10-19' WHEN size < 50 THEN '20-49'
                WHEN size < 100 THEN '50-99' WHEN size < 250 THEN '100-249'
                WHEN size < 1000 THEN '250-999' ELSE '1000+' END AS size_band,
           COUNT(*) AS n, SUM(size) AS persons
    FROM e
    GROUP BY CASE WHEN size < 5 THEN '1-4' WHEN size < 10 THEN '5-9'
                  WHEN size < 20 THEN '10-19' WHEN size < 50 THEN '20-49'
                  WHEN size < 100 THEN '50-99' WHEN size < 250 THEN '100-249'
                  WHEN size < 1000 THEN '250-999' ELSE '1000+' END
    """
    return pd.read_sql(q, conn)


def q_matrix(y: int, conn) -> pd.DataFrame:
    """M6: counts per (niva, inr, ssyk4, expband, fresh) for one Individ year;
    the same shape as 47h.pull_weights without the young flag."""
    q = f"""
    SELECT NULLIF(LTRIM(RTRIM(Sun2020Niva)),'') AS niva,
           NULLIF(LTRIM(RTRIM(Sun2020Inr)),'')  AS inr,
           {SSYK4.format(a='a')} AS ssyk4,
           CASE WHEN TRY_CAST(SsykAr_J16 AS INT) = {y} THEN 1 ELSE 0 END AS fresh,
           {EXP_CASE.format(a='a', y=y)} AS expband,
           COUNT(*) AS n
    FROM dbo.Individ_{y} a
    WHERE Sun2020Niva IS NOT NULL AND LTRIM(Sun2020Niva) <> ''
      AND Sun2020Inr  IS NOT NULL AND LTRIM(Sun2020Inr)  <> ''
      AND {CODED.format(a='a')}
    GROUP BY NULLIF(LTRIM(RTRIM(Sun2020Niva)),''), NULLIF(LTRIM(RTRIM(Sun2020Inr)),''),
             {SSYK4.format(a='a')},
             CASE WHEN TRY_CAST(SsykAr_J16 AS INT) = {y} THEN 1 ELSE 0 END,
             {EXP_CASE.format(a='a', y=y)}
    """
    return pd.read_sql(q, conn)


# ----------------------------------------------------------------------
# Moment builders (pure pandas; tested locally)
# ----------------------------------------------------------------------

def m2_collapse(raw: pd.DataFrame, daioe: pd.DataFrame) -> pd.DataFrame:
    d = raw.copy()
    d = d.merge(daioe.rename(columns={"ssyk4": "ssyk4_t", "q": "q_t"}), on="ssyk4_t", how="left")
    d = d.merge(daioe.rename(columns={"ssyk4": "ssyk4_t1", "q": "q_t1"}), on="ssyk4_t1", how="left")
    d["same_code"] = (d["ssyk4_t"] == d["ssyk4_t1"]).astype(int)
    d["enters_mgr"] = ((d["ssyk4_t"].str[:1] != "1") & (d["ssyk4_t1"].str[:1] == "1")).astype(int)
    d["q_t"] = d["q_t"].fillna(0).astype(int)
    d["q_t1"] = d["q_t1"].fillna(0).astype(int)
    return (d.groupby(["age_group", "expband", "q_t", "q_t1", "same_code", "enters_mgr"],
                      observed=True)["n"].sum().reset_index())


def m6_collapse(raw: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    d = raw.copy()
    d["niva"], d["inr"] = norm_code(d["niva"]), norm_code(d["inr"])
    d = d.merge(key, on=["niva", "inr"], how="left")
    d["grp"] = d["grp"].astype("string").fillna("unmatched")
    return (d.groupby(["grp", "expband", "fresh", "ssyk4"], observed=True)["n"]
            .sum().reset_index())


def m5_retention_from_cache() -> "pd.DataFrame | None":
    """Balanced-panel retention for 22-25 from a 47h collapse cache, if any:
    share of employers with both a Q4 and a below-Q4 cell, by size band."""
    f = CACHE / "edu_hr_coll_OL_daioe_true_T2021_2019.parquet"
    if not f.exists():
        print("  M5b: no 47h collapse cache; skipped")
        return None
    c = pd.read_parquet(f)
    c = c[c["age_group"].astype(str) == "22-25"]
    size = c.groupby("employer_id")["n_emp"].sum() / 12.0
    q4 = c[c["exposure_quartile"] == 4].groupby("employer_id").size()
    lo = c[c["exposure_quartile"] < 4].groupby("employer_id").size()
    df = pd.DataFrame({"size": size})
    df["both"] = df.index.isin(q4.index) & df.index.isin(lo.index)
    df["size_band"] = pd.cut(df["size"], [0, 1, 3, 10, 30, 100, np.inf],
                             labels=["<1", "1-3", "3-10", "10-30", "30-100", "100+"])
    out = df.groupby("size_band", observed=True).agg(n=("both", "size"), n_both=("both", "sum")).reset_index()
    return out


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    mc.Tee(OUT / "50_log.txt")
    sys.excepthook = lambda et, ev, tb: print(
        "\nUNCAUGHT EXCEPTION\n" + "".join(traceback.format_exception(et, ev, tb)))
    t0 = time.time()
    print("=" * 70)
    print("50: CALIBRATION MOMENTS FOR THE SIMULATION STUDY")
    print("=" * 70)
    print(mc.mem_line("  "))
    key, daioe = load_key(), load_daioe()
    conn = mc.connect()

    def stage(label, fn):
        t = time.time()
        try:
            r = fn()
            print(f"  {label}: ok ({time.time()-t:.0f}s)")
            return r
        except Exception as ex:
            # one bad moment costs that moment, never the job
            print(f"  {label}: FAILED ({type(ex).__name__}): {str(ex)[:300]}")
            return None

    print("\nM1 completion timing and level change")
    r = stage("M1a", lambda: pd.concat([q_completion_age(y, conn).assign(year=y) for y in YEARS]))
    if r is not None:
        export(r, "m1a_completion_age.csv")
    r = stage("M1b", lambda: pd.concat([q_level_change(t, conn).assign(year=t) for t in YEARS[:-1]]))
    if r is not None:
        export(r, "m1b_level_change.csv")

    print("\nM2 occupation change among freshly coded")
    frames = []
    for t in YEARS[:-1]:
        raw = stage(f"M2 {t}->{t+1}", lambda t=t: q_occ_change(t, conn))
        if raw is not None:
            frames.append(m2_collapse(raw, daioe).assign(year=t))
            del raw
            gc.collect()
    if frames:
        export(pd.concat(frames, ignore_index=True), "m2_occ_change.csv")

    print("\nM3 staleness")
    r = stage("M3", lambda: pd.concat([q_staleness(y, conn).assign(year=y) for y in YEARS]))
    if r is not None:
        export(r, "m3_staleness.csv")

    print("\nM4 enrolment")
    r = stage("M4a", lambda: pd.concat([q_enrolment(y, conn).assign(year=y) for y in (2021, 2022, 2023)]))
    if r is not None:
        export(r, "m4a_enrolment_prevalence.csv")
    r = stage("M4b", lambda: q_field_switch(conn))
    if r is not None:
        export(r, "m4b_field_switch.csv")

    print("\nM5 employers")
    r = stage("M5a", lambda: q_employer_size(conn))
    if r is not None:
        export(r, "m5a_employer_size.csv")
    r = stage("M5b", m5_retention_from_cache)
    if r is not None:
        export(r, "m5b_retention_22_25.csv")

    print("\nM6 group x experience x occupation matrix, per year")
    for y in YEARS:
        raw = stage(f"M6 {y}", lambda y=y: q_matrix(y, conn))
        if raw is not None:
            export(m6_collapse(raw, key), f"m6_matrix_{y}.csv")
            del raw
            gc.collect()

    lines = [f"50 done in {(time.time()-t0)/60:.1f} min. " + mc.mem_line(),
             "Files: " + ", ".join(sorted(p.name for p in OUT.glob("*.csv")))]
    (OUT / "50_summary.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
