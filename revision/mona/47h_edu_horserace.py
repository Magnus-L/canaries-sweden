#!/usr/bin/env python3
"""
47h_edu_horserace.py: the education-to-exposure bridge, and a comparison
of eight rules for building it, each judged by an as-of backtest.

QUESTION
The paper scores an employer's exposure to generative AI from the
education of its workers rather than from their occupations, because the
occupation register is published with a lag and the education register is
a census. Which rule should turn an education into an exposure score, and
which rule should assign a worker to an education? This script builds the
score books and tests every candidate the same way: by re-estimating on
years where the truth is observable, with the education register truncated
as it will be truncated for 2024 and 2025.

DESIGN
A score rule gives each education (SUN 2020 level and field, mapped to an
education group through the delivered key) the employment-weighted mean
exposure of the four-digit occupations its holders worked in during a
weight year; the exposure is the DAIOE generative-AI percentile, or the
Eloundou et al. score, or the share of holders in top-quartile
occupations. An assignment rule gives each worker an education from the
register as it stands in a given year. The eight designs are:

  OL_exact       Eloundou score, 2019 stock of all coded workers, group
                 detail (the mapping of Nordstrom Skans and Sokolow Romin
                 on this data).
  OL_daioe       the same with the DAIOE percentile. This is the rule the
                 paper uses: script 47j takes its score book.
  fresh_stock    OL_daioe with weights from codes assigned in 2019 only.
  entrant        weights from recent completers (0 to 5 years since the
                 exam year, aged 35 or under), 2019 to 2021 pooled, with
                 the 2019 stock as fallback below MIN_CELL completers.
  entrant_share  the share of the group's entrants in top-quartile
                 occupations.
  expband        a score per (group, years since completion) cell.
  enrol          entrant, with workers under 30 who hold no tertiary
                 degree in the truncated register scored by the field of
                 their latest registration in the enrolment register.
  full           entrant_share with level-by-field detail, expband and
                 enrol combined.

Every design is estimated on the withdrawn design's unit: employer by
exposure quartile by month cells, Poisson pseudo-maximum likelihood,
PostRB x High and PostGPT x High, employer-by-quartile and
employer-by-month effects, standard errors clustered by employer,
employers with a cumulative count of at least five, on 2019 to 2023. Each
design is fitted twice: with each year's own education register (the true
arm) and with the register truncated at 2021 or at 2022 and carried
forward (the as-of arm). The artefact is the as-of coefficient minus the
true one. Tier A runs every design at 22-25; Tier B runs 26-30 and 50-69
for the reference designs and for every design that clears the rule; Tier
C runs 31-34, 35-40 and 41-49 for the reference designs. A gate first
re-estimates OL_daioe under the cascade of the earlier script 47b, as a
check that the pull reproduces it; a discrepancy on the true arm stops the
run, one on the legacy arm is reported.

Read rule, fixed before the run: a design whose 22-25 artefact is below
0.05 in absolute value at both truncations can carry register evidence;
between 0.05 and half the occupation artefact of script 45 (0.153 at
2021, 0.081 at 2022) it is usable only with the artefact stated beside
every estimate; at or above half it is closed. The preferred design is
the one with the smallest 22-25 artefact and a near-zero artefact at
50-69, never the one with the largest coefficient.

INPUTS AND OUTPUTS
Reads, in MONA, Individ_YYYY for 2019 to 2021 (education, occupation,
year of the code, exam year, birth year) for the weights; the monthly
employer declarations for 2019 to 2023 joined to the education registers
of 2019 to 2022 for the true and as-of records; HREG_AKTIVITET for the
enrolment designs; and the input files
utb_grupp2_sun2020_niva3_inr4_nyckel.dta (the education key),
daioe_quartiles.dta and eloundou_ssyk4.dta, each verified by hash. Caches
edu_hr_weights_YYYY.parquet, edu_hr_YYYY.parquet and the collapsed
edu_hr_coll_<design>_<arm>_T<year>_<year>.parquet pieces. Writes to
output_47h/: score_<design>.csv, score_diagnostics.csv,
anchoring_rates.csv, horserace_estimates.csv, gate_decomposition.csv and
47h_summary.txt. Completed fits are reused on a resubmission when the
design list is unchanged; the environment variable CANARIES_47H_FRESH=1
discards them.

IN THE PAPER
Section 2, the exposure definition: each education group is given the
employment-weighted mean generative-AI exposure of the occupations its
holders worked in during 2019 (the OL_daioe score book, built by
ScoreBook.build from the 2019 weights and the key). The cached 2019 year
frame and the score book are what scripts 47j, 61, 66, 67, 68, 70 to 77
use to score employers. The estimates in horserace_estimates.csv concern
worker-level education designs that the paper does not report and are not
quoted.
"""

import gc
import hashlib
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import mona_common as mc

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_47h"
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR
CACHE.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
YEARS = list(range(2019, 2024))          # backtest window: every year coded
TRUNCATIONS = (2021, 2022)
WEIGHT_YEARS = (2019, 2020, 2021)        # all pre-ChatGPT; 2019 alone = O&L
ARMS = ("true", "asof")
GATE_ARMS = ("true", "asof", "asof_legacy")   # the gate alone adds the legacy arm
AGES_A = ["22-25"]
AGES_B = ["26-30", "50+"]
# Tier C: if the youngest band cannot be classified, does an age gradient
# survive among bands where the classification is predetermined? Reference
# designs only, one truncation, three bands; the artefact is reported for
# each, so the fallback gets the same test.
AGES_C = ["31-34", "35-40", "41-49"]
GRADIENT_TIER = True
GRADIENT_T = 2022
MIN_CELL = 200                           # scoring floor for a niva x inr cell
                                         # or a (group, exp band) cell
EXPORT_FLOOR = 5
ENTRANT_MAX_EXP = 5                      # years since completion
ENTRANT_MAX_AGE = 35
ENROL_MAX_AGE = 29                       # anchoring applies below 30
ENROL_WINDOW = 3                         # academic years before T
EXP_BANDS = ["0-2", "3-5", "6-10", "11-20", "21+", "na"]
STEP1_MIN_CUMULATIVE = 5                 # as 47: employer cumulative floor
GATE_47B = {"asof": -0.3695, "true": -0.0099}   # OL_daioe, T=2021, 22-25
GATE_WARN, GATE_HALT = 0.005, 0.05
OCC_ARTEFACT = {2021: -0.3068, 2022: -0.1627}   # script 45
REFERENCE_DESIGNS = ("OL_exact", "OL_daioe")

# Path(...) / name resolves on the UNC share and on POSIX alike, so the
# local end-to-end test can run.
KEY_PATH = str(Path(mc.SHARE) / "utb_grupp2_sun2020_niva3_inr4_nyckel.dta")
KEY_SHA256 = "c760361ba21554951a0744ee00de2f02f22f2e021b87f0863d9ece049e786637"
ELOUNDOU_PATH = str(Path(mc.SHARE) / "eloundou_ssyk4.dta")
DAIOE_PATH = str(Path(mc.SHARE) / "daioe_quartiles.dta")
ELOUNDOU_SHA256 = "d47b771e2f75a6166f3a855a3ba181e51a8510d8c9d07c82ebceb9e2f68eda93"

AGE_CASE = """CASE
        WHEN age BETWEEN 22 AND 25 THEN '22-25'
        WHEN age BETWEEN 26 AND 30 THEN '26-30'
        WHEN age BETWEEN 31 AND 34 THEN '31-34'
        WHEN age BETWEEN 35 AND 40 THEN '35-40'
        WHEN age BETWEEN 41 AND 49 THEN '41-49'
        WHEN age BETWEEN 50 AND 69 THEN '50+'
        ELSE NULL END"""

# Design registry. Keys: measure (daioe|eloundou), score (mean|share),
# weights (stock|entrant), fresh (codes assigned in the weight year only),
# detail (group|tier), expband, enrol, years (weight years).
DESIGNS = {
    "OL_exact":      dict(measure="eloundou", score="mean",  weights="stock",   fresh=False, detail="group", expband=False, enrol=False, years=(2019,)),
    "OL_daioe":      dict(measure="daioe",    score="mean",  weights="stock",   fresh=False, detail="group", expband=False, enrol=False, years=(2019,)),
    "fresh_stock":   dict(measure="daioe",    score="mean",  weights="stock",   fresh=True,  detail="group", expband=False, enrol=False, years=(2019,)),
    "entrant":       dict(measure="daioe",    score="mean",  weights="entrant", fresh=True,  detail="group", expband=False, enrol=False, years=WEIGHT_YEARS),
    "entrant_share": dict(measure="daioe",    score="share", weights="entrant", fresh=True,  detail="group", expband=False, enrol=False, years=WEIGHT_YEARS),
    "expband":       dict(measure="daioe",    score="mean",  weights="stock",   fresh=True,  detail="group", expband=True,  enrol=False, years=WEIGHT_YEARS),
    "enrol":         dict(measure="daioe",    score="mean",  weights="entrant", fresh=True,  detail="group", expband=False, enrol=True,  years=WEIGHT_YEARS),
    "full":          dict(measure="daioe",    score="share", weights="entrant", fresh=True,  detail="tier",  expband=True,  enrol=True,  years=WEIGHT_YEARS),
}

# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------

def norm_code(s: pd.Series) -> pd.Series:
    """Trim + lowercase; NULL and '' (2019 vs 2021+ encodings) -> <NA>."""
    out = s.astype("string").str.strip().str.lower()
    return out.where(out.notna() & (out != ""), other=pd.NA)


def sha256_of(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_key() -> pd.DataFrame:
    got = sha256_of(KEY_PATH)
    if got != KEY_SHA256:
        raise RuntimeError(f"key hash mismatch: {got[:16]}... is not the "
                           f"delivered key {KEY_SHA256[:16]}...")
    key = pd.read_stata(KEY_PATH).rename(columns={
        "sun2020niva_3_kod": "niva", "sun2020inr_4_kod": "inr",
        "utb_grupp2": "grp"})
    key["niva"] = norm_code(key["niva"])
    key["inr"] = norm_code(key["inr"])
    key = key.dropna(subset=["niva", "inr"])
    assert not key.duplicated(["niva", "inr"]).any(), "key not unique on niva x inr"
    return key[["niva", "inr", "grp"]].reset_index(drop=True)


def load_scores() -> pd.DataFrame:
    """ssyk4 -> daioe percentile, daioe high flag, eloundou beta, eloundou
    high flag. Both files hash-checked; a mismatch stops the run before
    any SQL is spent."""
    got = sha256_of(DAIOE_PATH)
    if got != mc.DAIOE_SHA256:
        raise RuntimeError(f"daioe hash mismatch: {got[:16]}...")
    d = pd.read_stata(DAIOE_PATH)
    d["ssyk4"] = d["ssyk4"].astype(str).str.zfill(4)
    d = d[["ssyk4", "pctl_rank_genai", "high_exposure"]].rename(columns={
        "pctl_rank_genai": "daioe_score", "high_exposure": "daioe_high"})
    got = sha256_of(ELOUNDOU_PATH)
    if got != ELOUNDOU_SHA256:
        raise RuntimeError(f"eloundou hash mismatch: {got[:16]}...")
    e = pd.read_stata(ELOUNDOU_PATH)
    e["ssyk4"] = e["ssyk4"].astype(str).str.zfill(4)
    e = e[["ssyk4", "eloundou_score", "high_exposure_eloundou"]].rename(
        columns={"high_exposure_eloundou": "eloundou_high"})
    s = d.merge(e, on="ssyk4", how="outer")
    for c in ("daioe_score", "daioe_high", "eloundou_score", "eloundou_high"):
        s[c] = pd.to_numeric(s[c], errors="coerce")
    print(f"  scores: {s['daioe_score'].notna().sum()} DAIOE codes, "
          f"{s['eloundou_score'].notna().sum()} Eloundou codes, "
          f"{s['daioe_score'].notna().mul(s['eloundou_score'].notna()).sum()} in both")
    return s


def is_tertiary(niva: pd.Series) -> pd.Series:
    """SUN 2020 niva first digit 4, 5 or 6 = post-secondary or research."""
    return (niva.astype("string").str[:1].isin(["4", "5", "6"])
            .fillna(False).astype(bool))


def wquantile_cutoffs(values: np.ndarray, weights: np.ndarray):
    """Employment-weighted 25/50/75 cut points of a worker-level score."""
    o = np.argsort(values, kind="stable")
    v, w = values[o], weights[o]
    cum = np.cumsum(w) / w.sum()
    return [float(v[np.searchsorted(cum, q, side="left")]) for q in (0.25, 0.5, 0.75)]


def to_quartile(score: np.ndarray, cuts) -> np.ndarray:
    """1..4 from cut points; NaN score -> 0 (unclassified, dropped later)."""
    q = np.searchsorted(np.asarray(cuts), score, side="right") + 1
    q = q.astype(np.int8)
    q[np.isnan(score)] = 0
    return q


# ----------------------------------------------------------------------
# SQL pulls (both patched out by the local test)
# ----------------------------------------------------------------------

def pull_weights(year: int, conn) -> pd.DataFrame:
    """
    One Individ year -> counts per (niva, inr, ssyk4, fresh, expband, young).
    Aggregated in SQL: a few hundred thousand rows at most.
      fresh   = the occupation code was assigned in this year (SsykAr_J16)
      expband = years since completion of the highest education (ExamAr)
      young   = age <= ENTRANT_MAX_AGE
    TRY_CAST guards the char columns that carry '****' or blanks.
    """
    q = f"""
    SELECT NULLIF(LTRIM(RTRIM(Sun2020Niva)),'') AS niva,
           NULLIF(LTRIM(RTRIM(Sun2020Inr)),'')  AS inr,
           RIGHT('0000' + CAST(Ssyk4_2012_J16 AS VARCHAR(4)), 4) AS ssyk4,
           CASE WHEN TRY_CAST(SsykAr_J16 AS INT) = {year} THEN 1 ELSE 0 END AS fresh,
           CASE WHEN TRY_CAST(ExamAr AS INT) IS NULL THEN 'na'
                WHEN {year} - TRY_CAST(ExamAr AS INT) <= 2  THEN '0-2'
                WHEN {year} - TRY_CAST(ExamAr AS INT) <= 5  THEN '3-5'
                WHEN {year} - TRY_CAST(ExamAr AS INT) <= 10 THEN '6-10'
                WHEN {year} - TRY_CAST(ExamAr AS INT) <= 20 THEN '11-20'
                ELSE '21+' END AS expband,
           CASE WHEN {year} - TRY_CAST(FodelseAr AS INT) <= {ENTRANT_MAX_AGE}
                THEN 1 ELSE 0 END AS young,
           COUNT(*) AS n
    FROM dbo.Individ_{year}
    WHERE Sun2020Niva IS NOT NULL AND LTRIM(Sun2020Niva) <> ''
      AND Sun2020Inr  IS NOT NULL AND LTRIM(Sun2020Inr)  <> ''
      AND Ssyk4_2012_J16 IS NOT NULL AND LTRIM(Ssyk4_2012_J16) <> ''
    GROUP BY NULLIF(LTRIM(RTRIM(Sun2020Niva)),''), NULLIF(LTRIM(RTRIM(Sun2020Inr)),''),
             Ssyk4_2012_J16,
             CASE WHEN TRY_CAST(SsykAr_J16 AS INT) = {year} THEN 1 ELSE 0 END,
             CASE WHEN TRY_CAST(ExamAr AS INT) IS NULL THEN 'na'
                  WHEN {year} - TRY_CAST(ExamAr AS INT) <= 2  THEN '0-2'
                  WHEN {year} - TRY_CAST(ExamAr AS INT) <= 5  THEN '3-5'
                  WHEN {year} - TRY_CAST(ExamAr AS INT) <= 10 THEN '6-10'
                  WHEN {year} - TRY_CAST(ExamAr AS INT) <= 20 THEN '11-20'
                  ELSE '21+' END,
             CASE WHEN {year} - TRY_CAST(FodelseAr AS INT) <= {ENTRANT_MAX_AGE}
                  THEN 1 ELSE 0 END
    """
    return pd.read_sql(q, conn)


def probe_enrolment(conn) -> bool:
    """Cheap probe of the enrolment register and of the SQL features the
    year pull relies on (ROW_NUMBER, TRY_CAST). False -> the enrolment
    designs degrade to their unanchored twins, loudly."""
    q = """
    SELECT TOP 5 person_id, enr_inr FROM (
        SELECT P1207_Lopnr_Personnr AS person_id, SUN2020INR AS enr_inr,
               ROW_NUMBER() OVER (PARTITION BY P1207_Lopnr_Personnr
                                  ORDER BY ARTERMIN DESC, ISNULL(AKTPROC, 0) DESC) AS rn
        FROM dbo.HREG_AKTIVITET_1971_2021
        WHERE SUN2020INR IS NOT NULL AND LTRIM(SUN2020INR) <> ''
          AND TRY_CAST(LEFT(ARTERMIN, 4) AS INT) BETWEEN 2018 AND 2021) x
    WHERE rn = 1
    """
    try:
        r = pd.read_sql(q, conn)
        return len(r) > 0
    except Exception as ex:
        print(f"  enrolment probe FAILED ({type(ex).__name__}): "
              f"{str(ex)[:300]}")
        return False


def _legacy_cols(prefix: str, vintages, field: str) -> str:
    """
    47b's cascade, reproduced verbatim: a plain COALESCE over the raw column,
    PER FIELD and independently. Two consequences we believe matter, and this
    arm exists to MEASURE rather than assert them:
      - '' (the 2021+ encoding for missing) is not NULL, so COALESCE returns
        it and the row fails the key join instead of falling through;
      - niva and inr are resolved separately, so a record can be assembled
        from two different vintages.
    Kept only for the gate. Nothing downstream uses it.
    """
    return ("COALESCE("
            + ", ".join(f"{prefix}{i}.{field}" for i in range(1, len(vintages) + 1))
            + ")")


def _asof_cols(prefix: str, vintages, field: str) -> str:
    """First vintage with a non-empty niva supplies the whole record
    (niva, inr, ExamAr together), so a record is never assembled across
    vintages. NULLIF makes '' (the 2021+ missing encoding) fall through
    the cascade, which COALESCE alone did not in 47/47b."""
    branches = " ".join(
        f"WHEN NULLIF(LTRIM(RTRIM({prefix}{i}.Sun2020Niva)),'') IS NOT NULL "
        f"THEN {field.format(a=f'{prefix}{i}')}"
        for i in range(1, len(vintages) + 1))
    return f"CASE {branches} ELSE NULL END"


def pull_year(year: int, conn, enrol_ok: bool) -> pd.DataFrame:
    """
    One AGI year -> employer x month x age x
        (niva_t, inr_t, expb_t)            the year's own register
        (niva_21, inr_21, expb_21, enr_21) the register truncated at 2021
        (niva_22, inr_22, expb_22, enr_22) the register truncated at 2022
    -> n_emp (distinct persons). Read in chunks and compacted to
    categoricals as they arrive, so a 50M-row year never sits in memory as
    Python strings.
    enr_T = SUN2020INR of the latest registration within ENROL_WINDOW
    academic years before T, from HREG_AKTIVITET (ends 2021, so T=2022 is
    served by registrations to 2021: a conservative test for that arm).
    """
    suffix, max_month = ("_def", 12) if year < 2025 else ("_prel", 6)
    own = min(year, 2023)
    v21 = [y for y in (2021, 2020, 2019)]
    v22 = [y for y in (2022, 2021, 2020)]
    joins = [f"LEFT JOIN dbo.Individ_{own} t "
             f"ON agi.P1207_LOPNR_PERSONNR = t.P1207_LopNr_PersonNr"]
    joins += [f"LEFT JOIN dbo.Individ_{y} b{i} "
              f"ON agi.P1207_LOPNR_PERSONNR = b{i}.P1207_LopNr_PersonNr"
              for i, y in enumerate(v21, 1)]
    joins += [f"LEFT JOIN dbo.Individ_{y} c{i} "
              f"ON agi.P1207_LOPNR_PERSONNR = c{i}.P1207_LopNr_PersonNr"
              for i, y in enumerate(v22, 1)]
    enr_ctes, enr_joins, enr_sel = "", "", ""
    if enrol_ok:
        for tag, T in (("e21", 2021), ("e22", 2022)):
            enr_ctes += f"""
    {tag} AS (
        SELECT person_id, enr_inr FROM (
            SELECT P1207_Lopnr_Personnr AS person_id,
                   NULLIF(LTRIM(RTRIM(SUN2020INR)),'') AS enr_inr,
                   ROW_NUMBER() OVER (PARTITION BY P1207_Lopnr_Personnr
                       ORDER BY ARTERMIN DESC, ISNULL(AKTPROC, 0) DESC) AS rn
            FROM dbo.HREG_AKTIVITET_1971_2021
            WHERE SUN2020INR IS NOT NULL AND LTRIM(SUN2020INR) <> ''
              AND TRY_CAST(LEFT(ARTERMIN, 4) AS INT)
                  BETWEEN {T - ENROL_WINDOW} AND {T}) x
        WHERE rn = 1),"""
            enr_joins += (f" LEFT JOIN {tag} ON agi.P1207_LOPNR_PERSONNR "
                          f"= {tag}.person_id")
        enr_sel = "e21.enr_inr AS enr_21, e22.enr_inr AS enr_22,"
    else:
        enr_sel = "CAST(NULL AS VARCHAR(4)) AS enr_21, CAST(NULL AS VARCHAR(4)) AS enr_22,"

    exam_expr = "TRY_CAST({a}.ExamAr AS INT)"
    born = ("COALESCE(TRY_CAST(t.FodelseAr AS INT), "
            + ", ".join(f"TRY_CAST(b{i}.FodelseAr AS INT)" for i in range(1, 4))
            + ")")
    monthly = "\nUNION ALL\n".join(f"""
        SELECT agi.P1207_LOPNR_PEORGNR AS employer_id,
               agi.PERIOD AS period,
               agi.P1207_LOPNR_PERSONNR AS person_id,
               NULLIF(LTRIM(RTRIM(t.Sun2020Niva)),'') AS niva_t,
               NULLIF(LTRIM(RTRIM(t.Sun2020Inr)),'')  AS inr_t,
               TRY_CAST(t.ExamAr AS INT) AS exam_t,
               {_asof_cols('b', v21, "NULLIF(LTRIM(RTRIM({a}.Sun2020Niva)),'')")} AS niva_21,
               {_asof_cols('b', v21, "NULLIF(LTRIM(RTRIM({a}.Sun2020Inr)),'')")}  AS inr_21,
               {_asof_cols('b', v21, exam_expr)} AS exam_21,
               {_legacy_cols('b', v21, 'Sun2020Niva')} AS niva_21g,
               {_legacy_cols('b', v21, 'Sun2020Inr')}  AS inr_21g,
               {_asof_cols('c', v22, "NULLIF(LTRIM(RTRIM({a}.Sun2020Niva)),'')")} AS niva_22,
               {_asof_cols('c', v22, "NULLIF(LTRIM(RTRIM({a}.Sun2020Inr)),'')")}  AS inr_22,
               {_asof_cols('c', v22, exam_expr)} AS exam_22,
               {enr_sel}
               {born} AS birth_year
        FROM dbo.Arb_AGIIndivid{year}{m:02d}{suffix} agi
        {' '.join(joins)}{enr_joins}""" for m in range(1, max_month + 1))

    def band(col):
        return f"""CASE WHEN {col} IS NULL THEN 'na'
                WHEN {year} - {col} <= 2  THEN '0-2'
                WHEN {year} - {col} <= 5  THEN '3-5'
                WHEN {year} - {col} <= 10 THEN '6-10'
                WHEN {year} - {col} <= 20 THEN '11-20'
                ELSE '21+' END"""
    q = f"""
    WITH {enr_ctes}
    base AS ({monthly}),
    age_calc AS (
        SELECT employer_id, period, person_id,
               niva_t, inr_t, {band('exam_t')} AS expb_t,
               niva_21, inr_21, {band('exam_21')} AS expb_21, enr_21,
               niva_21g, inr_21g,
               niva_22, inr_22, {band('exam_22')} AS expb_22, enr_22,
               CAST(LEFT(period,4) AS INT) - birth_year AS age
        FROM base WHERE birth_year IS NOT NULL
    )
    SELECT employer_id,
           LEFT(period,4) + '-' + SUBSTRING(period,5,2) AS year_month,
           niva_t, inr_t, expb_t, niva_21, inr_21, expb_21, enr_21,
           niva_21g, inr_21g,
           niva_22, inr_22, expb_22, enr_22,
           {AGE_CASE} AS age_group, COUNT(DISTINCT person_id) AS n_emp
    FROM age_calc WHERE age BETWEEN 22 AND 69
    GROUP BY employer_id, period, niva_t, inr_t, expb_t, niva_21, inr_21,
             expb_21, enr_21, niva_21g, inr_21g,
             niva_22, inr_22, expb_22, enr_22, {AGE_CASE}
    """
    chunks = []
    for ch in pd.read_sql(q, conn, chunksize=2_000_000):
        chunks.append(compact(ch))
    if not chunks:
        return compact(pd.DataFrame(columns=YEAR_COLS + ["n_emp"]))
    out = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    return out


WEIGHT_COLS = ["niva", "inr", "ssyk4", "fresh", "expband", "young", "n"]
YEAR_COLS = ["employer_id", "year_month", "niva_t", "inr_t", "expb_t",
             "niva_21", "inr_21", "expb_21", "enr_21",
             "niva_21g", "inr_21g",
             "niva_22", "inr_22", "expb_22", "enr_22", "age_group"]
CODE_COLS = ["niva_t", "inr_t", "niva_21", "inr_21", "enr_21",
             "niva_21g", "inr_21g",
             "niva_22", "inr_22", "enr_22"]


def compact(df: pd.DataFrame) -> pd.DataFrame:
    """Strings -> normalised categoricals, counts -> int32. Applied per
    chunk so the peak is one chunk of Python strings, not one year."""
    df = df.copy()
    for c in CODE_COLS:
        df[c] = norm_code(df[c]).astype("category")
    for c in ("expb_t", "expb_21", "expb_22", "age_group", "year_month"):
        df[c] = df[c].astype("string").astype("category")
    emp = pd.to_numeric(df["employer_id"], errors="coerce")
    bad = emp.isna()
    if bad.any():
        print(f"  compact: dropping {int(bad.sum()):,} rows with no employer id")
        df = df[~bad]
        emp = emp[~bad]
    df["employer_id"] = emp.astype("int64")
    df["n_emp"] = pd.to_numeric(df["n_emp"], errors="coerce").fillna(0).astype("int32")
    return df


# ----------------------------------------------------------------------
# Scores: education -> exposure, one table per design
# ----------------------------------------------------------------------

class ScoreBook:
    """
    Every lookup a design needs, built once from the weight counts.
      grp_score[design]     : grp -> score
      tier_score[design]    : (niva, inr) -> score   (detail == tier)
      band_score[design]    : (grp, expband) -> score (expband designs)
      inr_score[design]     : inr -> score over tertiary rows (enrol designs)
      cuts[design]          : quartile cut points on the 2019 stock
    Fallbacks are explicit: a band cell below MIN_CELL takes the group
    score; a tier cell below MIN_CELL takes the group score; an inr with
    fewer than MIN_CELL tertiary completers anchors nobody.
    """

    def __init__(self, counts: dict, key: pd.DataFrame, scores: pd.DataFrame):
        self.key = key
        self.scores = scores
        self.counts = counts            # year -> raw weight frame
        self.grp_score, self.tier_score = {}, {}
        self.band_score, self.inr_score, self.cuts = {}, {}, {}
        self.diag = {}
        # the binning population: 2019 stock, every coded worker, keyed
        base = self._prep(counts[2019])
        self.base_pop = base

    def _prep(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        df["niva"] = norm_code(df["niva"])
        df["inr"] = norm_code(df["inr"])
        df["ssyk4"] = df["ssyk4"].astype(str).str.zfill(4)
        df["n"] = pd.to_numeric(df["n"], errors="coerce").fillna(0).astype(int)
        df = df[df["n"] > 0]
        df = df.merge(self.key, on=["niva", "inr"], how="left")
        df = df.merge(self.scores, on="ssyk4", how="left")
        df["expband"] = df["expband"].astype("string").fillna("na")
        return df

    def _population(self, spec: dict) -> pd.DataFrame:
        frames = []
        for y in spec["years"]:
            df = self._prep(self.counts[y])
            if spec["fresh"]:
                df = df[df["fresh"] == 1]
            if spec["weights"] == "entrant":
                df = df[(df["young"] == 1)
                        & df["expband"].isin(["0-2", "3-5"])]
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _wmean(df, col, w="n"):
        d = df.dropna(subset=[col])
        if d.empty or d[w].sum() == 0:
            return np.nan
        return float(np.average(d[col], weights=d[w]))

    def _score_col(self, spec):
        m = spec["measure"]
        return f"{m}_high" if spec["score"] == "share" else f"{m}_score"

    def _agg(self, df: pd.DataFrame, by, col: str, floor: int) -> pd.DataFrame:
        d = df.dropna(subset=[col]).copy()
        d["_ws"] = d[col] * d["n"]
        g = d.groupby(by, observed=True).agg(n=("n", "sum"), _ws=("_ws", "sum")).reset_index()
        g["score"] = g["_ws"] / g["n"]
        g = g[g["n"] >= floor].drop(columns="_ws")
        return g

    def build(self, name: str, spec: dict):
        col = self._score_col(spec)
        pop = self._population(spec)
        keyed = pop[pop["grp"].notna()]
        grp = self._agg(keyed, ["grp"], col, EXPORT_FLOOR)
        # stock fallback for an entrant population that is thin somewhere
        if spec["weights"] == "entrant":
            stock = self._agg(self._prep(self.counts[2019])[lambda d: d["grp"].notna()],
                              ["grp"], col, EXPORT_FLOOR)
            thin = grp[grp["n"] < MIN_CELL]["grp"]
            missing = set(stock["grp"]) - set(grp["grp"])
            fill = stock[stock["grp"].isin(set(thin) | missing)]
            grp = pd.concat([grp[~grp["grp"].isin(thin)], fill], ignore_index=True)
            self.diag[f"{name}_stock_fallback_groups"] = int(len(fill))
        self.grp_score[name] = dict(zip(grp["grp"], grp["score"]))
        if spec["detail"] == "tier":
            t = self._agg(keyed, ["niva", "inr"], col, MIN_CELL)
            self.tier_score[name] = {(a, b): s for a, b, s in
                                     zip(t["niva"], t["inr"], t["score"])}
            self.diag[f"{name}_tier1_cells"] = int(len(t))
        if spec["expband"]:
            b = self._agg(keyed, ["grp", "expband"], col, MIN_CELL)
            self.band_score[name] = {(g, e): s for g, e, s in
                                     zip(b["grp"], b["expband"], b["score"])}
            self.diag[f"{name}_band_cells"] = int(len(b))
        if spec["enrol"]:
            ter = pop[is_tertiary(pop["niva"])]
            i = self._agg(ter, ["inr"], col, MIN_CELL)
            self.inr_score[name] = dict(zip(i["inr"], i["score"]))
            self.diag[f"{name}_anchor_fields"] = int(len(i))
        # cut points on the 2019 stock, each worker at the design's score
        base = self.base_pop
        s = self.score_frame(name, spec, base["niva"], base["inr"],
                             base["expband"], None)
        ok = ~np.isnan(s)
        self.cuts[name] = wquantile_cutoffs(s[ok], base["n"].to_numpy()[ok])
        # export table
        exp = grp.rename(columns={"n": "n_workers"}).copy()
        exp["quartile"] = to_quartile(exp["score"].to_numpy(), self.cuts[name])
        exp.sort_values("score").to_csv(OUT / f"score_{name}.csv", index=False)
        print(f"  {name:<14} groups {len(grp):>3}  cuts "
              f"{' / '.join(f'{c:.3f}' for c in self.cuts[name])}"
              + "".join(f"  {k.split('_', 1)[1]} {v}" for k, v in self.diag.items()
                        if k.startswith(name + "_")))

    def score_frame(self, name, spec, niva, inr, expb, enr_inr,
                    age_group=None) -> np.ndarray:
        """Vectorised score for arrays of worker attributes. Order of
        precedence: enrolment anchor (if applicable) > tier-1 cell > band
        cell > group score > NaN."""
        niva = pd.Series(np.asarray(niva, dtype=object)).astype("string")
        inr = pd.Series(np.asarray(inr, dtype=object)).astype("string")
        grp = (pd.DataFrame({"niva": niva, "inr": inr})
               .merge(self.key, on=["niva", "inr"], how="left")["grp"]
               .astype("string"))
        out = grp.map(self.grp_score[name]).astype(float).to_numpy()
        if spec["expband"]:
            eb = pd.Series(np.asarray(expb, dtype=object)).astype("string").fillna("na")
            k = list(zip(grp.fillna(""), eb))
            band = pd.Series([self.band_score[name].get(t, np.nan) for t in k], dtype=float)
            out = np.where(np.isnan(band.to_numpy()), out, band.to_numpy())
        if spec["detail"] == "tier":
            k = list(zip(niva.fillna(""), inr.fillna("")))
            tier = pd.Series([self.tier_score[name].get(t, np.nan) for t in k], dtype=float)
            out = np.where(np.isnan(tier.to_numpy()), out, tier.to_numpy())
        if spec["enrol"] and enr_inr is not None:
            e = pd.Series(np.asarray(enr_inr, dtype=object)).astype("string")
            anchor = e.map(self.inr_score[name]).astype(float).to_numpy()
            young = (np.ones(len(out), dtype=bool) if age_group is None
                     else pd.Series(age_group).astype("string")
                     .isin(["22-25", "26-30"]).to_numpy())
            eligible = young & ~is_tertiary(niva).to_numpy() & ~np.isnan(anchor)
            out = np.where(eligible, anchor, out)
        return out


# ----------------------------------------------------------------------
# Assignment on a year frame, collapsed to the estimation cell
# ----------------------------------------------------------------------

def collapse_year(year: int, frame: pd.DataFrame, book: ScoreBook,
                  designs: dict, ages: list) -> dict:
    """
    For every (design, arm, T): quartile per row -> employer x month x
    quartile x age cells for the requested ages. Scores are computed once
    per distinct attribute combination (a few thousand), then broadcast
    through integer codes, so nothing touches the 50M-row frame more than
    once per design.
    Returns {(design, arm, T): collapsed frame} and the anchoring rates.
    """
    f = frame[frame["age_group"].isin(ages)]
    out, rates = {}, []
    for name, spec in designs.items():
        for T in TRUNCATIONS:
            for arm in ARMS:
                if arm == "asof_legacy" and T != 2021:
                    continue          # the legacy columns exist for T=2021 only
                if arm == "true":
                    cols = ["niva_t", "inr_t", "expb_t"]
                    enr = None
                elif arm == "asof_legacy":
                    # 47b's own cascade. The experience band comes from the
                    # corrected arm: 47b had none of its own there, and the
                    # band is not what this arm tests.
                    cols = ["niva_21g", "inr_21g", "expb_21"]
                    enr = None
                else:
                    cols = [f"niva_{T % 100}", f"inr_{T % 100}", f"expb_{T % 100}"]
                    enr = f"enr_{T % 100}" if spec["enrol"] else None
                keycols = cols + ([enr] if enr else []) + ["age_group"]
                # distinct attribute combinations -> score -> broadcast
                combos = f[keycols].drop_duplicates().reset_index(drop=True)
                s = book.score_frame(
                    name, spec, combos[cols[0]], combos[cols[1]], combos[cols[2]],
                    combos[enr] if enr else None, combos["age_group"])
                combos["_q"] = to_quartile(s, book.cuts[name])
                if enr:
                    # anchoring rate: share of as-of person-months whose score
                    # came from the enrolment field (young, non-tertiary, anchored)
                    anch = (combos["age_group"].astype("string").isin(["22-25", "26-30"])
                            & ~is_tertiary(combos[cols[0]])
                            & combos[enr].notna()
                            & combos[enr].astype("string").map(
                                lambda v: v in book.inr_score[name]).fillna(False))
                    combos["_anch"] = anch.to_numpy()
                g = f.merge(combos, on=keycols, how="left")
                if enr:
                    r = (g.groupby("age_group", observed=True)
                         .apply(lambda d: pd.Series({
                             "n_emp": int(d["n_emp"].sum()),
                             "n_anchored": int(d.loc[d["_anch"].fillna(False), "n_emp"].sum())}),
                             include_groups=False).reset_index())
                    r["design"], r["T"], r["year"] = name, T, year
                    rates.append(r)
                g = g[g["_q"] > 0]
                coll = (g.groupby(["employer_id", "year_month", "_q", "age_group"],
                                  observed=True)["n_emp"].sum().reset_index()
                        .rename(columns={"_q": "exposure_quartile"}))
                coll["exposure_quartile"] = coll["exposure_quartile"].astype("int8")
                out[(name, arm, T)] = coll
                del g, combos
        gc.collect()
    return out, (pd.concat(rates, ignore_index=True) if rates else pd.DataFrame())


# ----------------------------------------------------------------------
# Fast balanced panel (same semantics as mc.balance_panel, integer codes)
# ----------------------------------------------------------------------

def fast_balance(sub: pd.DataFrame, all_months) -> pd.DataFrame:
    """
    Same result as mona_common.balance_panel (asserted equal in the local
    test), built with integer codes and a single reindex rather than three
    merges over string keys. Employers with both a Q4 cell and a below-Q4
    cell in ANY month; every (employer, quartile) they have crossed with
    every month; missing cells zero.
    """
    months = pd.Index(sorted(all_months))
    emp_q = sub[["employer_id", "exposure_quartile"]].drop_duplicates()
    q4 = set(emp_q.loc[emp_q["exposure_quartile"] == 4, "employer_id"])
    lo = set(emp_q.loc[emp_q["exposure_quartile"] < 4, "employer_id"])
    keep = q4 & lo
    emp_q = emp_q[emp_q["employer_id"].isin(keep)].sort_values(
        ["employer_id", "exposure_quartile"]).reset_index(drop=True)
    if emp_q.empty:
        return pd.DataFrame(columns=["employer_id", "exposure_quartile",
                                     "year_month", "n_emp"])
    s2 = sub[sub["employer_id"].isin(keep)].copy()
    s2["year_month"] = s2["year_month"].astype(str)
    s2["exposure_quartile"] = s2["exposure_quartile"].astype(int)
    cell = (s2.groupby(["employer_id", "exposure_quartile", "year_month"])
            ["n_emp"].sum())
    P, M = len(emp_q), len(months)
    full = pd.MultiIndex.from_arrays(
        [np.repeat(emp_q["employer_id"].to_numpy(), M),
         np.repeat(emp_q["exposure_quartile"].astype(int).to_numpy(), M),
         np.tile(months.to_numpy().astype(str), P)],
        names=["employer_id", "exposure_quartile", "year_month"])
    bal = cell.reindex(full, fill_value=0).reset_index()
    bal["n_emp"] = bal["n_emp"].astype(int)
    return bal


def estimate(coll: pd.DataFrame, age: str, tag: str) -> dict:
    """Balance, treat, fit; return the post_gpt_x_high row as a dict."""
    sub = coll[coll["age_group"].astype(str) == age].copy()
    sub["year_month"] = sub["year_month"].astype(str)
    sub["exposure_quartile"] = sub["exposure_quartile"].astype(int)
    cum = sub.groupby("employer_id")["n_emp"].sum()
    sub = sub[sub["employer_id"].isin(cum[cum >= STEP1_MIN_CUMULATIVE].index)]
    months = sorted(coll["year_month"].astype(str).unique())
    t0 = time.time()
    bal = mc.add_treatment(fast_balance(sub, months))
    n = len(bal)
    r = mc.run_fepois(bal, OUT, tag=tag)
    del bal
    gc.collect()
    row = {"gamma2": np.nan, "se": np.nan, "p": np.nan, "n_obs": n,
           "status": "no_output", "elapsed_s": round(time.time() - t0, 1)}
    if not r.empty and (r["term"] == "post_gpt_x_high").any():
        g = r[r["term"] == "post_gpt_x_high"].iloc[0]
        row.update(gamma2=float(g["coef"]), se=float(g["se"]),
                   p=float(g["pvalue"]), status=str(g.get("status", "ok")))
    return row


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def _append_row(path: Path, row: dict):
    """Results go to disk as they are produced."""
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", index=False, header=not path.exists())


def main():
    mc.Tee(OUT / "47h_log.txt")
    sys.excepthook = lambda et, ev, tb: print(
        "\nUNCAUGHT EXCEPTION\n" + "".join(traceback.format_exception(et, ev, tb)))
    t_start = time.time()
    print("=" * 70)
    print("47h: EDUCATION-EXPOSURE HORSE RACE, JUDGED BY THE AS-OF BACKTEST")
    print("=" * 70)
    print(mc.mem_line("  "))
    print("  plan: weights 3 x ~1 min | year pulls 5 x 10-15 min | "
          "collapses 5 x ~5 min | Tier A 32 fits x ~6 min | Tier B ~1 h")
    print(f"  designs: {', '.join(DESIGNS)}")

    # ---- inputs, hash-checked before any SQL is spent ----
    key = load_key()
    scores = load_scores()
    print(f"  key: {len(key):,} niva x inr cells, {key['grp'].nunique()} groups")
    conn = mc.connect()

    # ---- cheap probes first: a failure here costs seconds, not hours ----
    enrol_ok = probe_enrolment(conn)
    print(f"  enrolment register usable: {enrol_ok}")
    designs = dict(DESIGNS)
    if not enrol_ok:
        for name in ("enrol", "full"):
            designs[name] = dict(designs[name], enrol=False)
            print(f"  {name}: enrolment anchoring DISABLED (register probe failed); "
                  f"runs as its unanchored twin")

    # ---- weights ----
    counts = {}
    for y in WEIGHT_YEARS:
        cf = CACHE / f"edu_hr_weights_{y}.parquet"
        w = mc.read_cache(cf, require=WEIGHT_COLS)
        if w is None:
            t0 = time.time()
            w = pull_weights(y, conn)
            mc.write_cache(w, cf)
            print(f"  weights {y}: {len(w):,} cells ({time.time()-t0:.0f}s)")
        else:
            print(f"  weights {y}: cached ({len(w):,} cells)")
        counts[y] = w
    book = ScoreBook(counts, key, scores)
    print("\nSCORES")
    for name, spec in designs.items():
        book.build(name, spec)
    pd.DataFrame([{"k": k, "v": v} for k, v in book.diag.items()]).to_csv(
        OUT / "score_diagnostics.csv", index=False)

    # ---- year pulls and collapses ----
    print("\nYEAR PULLS AND COLLAPSES")
    ages_all = AGES_A + AGES_B + (AGES_C if GRADIENT_TIER else [])
    rate_frames = []
    def coll_path(name, arm, T, y):
        return CACHE / f"edu_hr_coll_{name}_{arm}_T{T}_{y}.parquet"

    def wanted_pieces(y):
        """Every collapse piece this run will later load, as (key, path)."""
        want = [((n, a, T), coll_path(n, a, T, y))
                for n in designs for a in ("true", "asof") for T in TRUNCATIONS]
        # the legacy arm is T=2021 only: niva_21g/inr_21g are the 2021
        # vintage, and collapse_year skips it for any other truncation.
        # Asking for more than exists here would leave `missing` permanently
        # non-empty and defeat the cache on every run.
        want += [(("OL_daioe", "asof_legacy", 2021),
                  coll_path("OL_daioe", "asof_legacy", 2021, y))]
        return want

    for y in YEARS:
        # The collapse costs about half an hour a year and the pull about
        # twenty minutes, so a re-run that recomputes both is hours before the
        # first fit. Both are pure functions of the year frame and the score
        # book, so a year whose pieces are all on disk needs neither: skip it
        # entirely and never touch the frame. This is what makes adding one
        # arm cheap instead of a full rebuild.
        missing = [(k, pth) for k, pth in wanted_pieces(y) if not pth.exists()]
        if not missing:
            print(f"  {y}: all {len(wanted_pieces(y))} collapse pieces cached, "
                  f"no pull needed")
            continue
        cf = CACHE / f"edu_hr_{y}.parquet"
        # require= is what stops a cache written before a change to the pull
        # from being reused.
        frame = mc.read_cache(cf, require=YEAR_COLS + ["n_emp"])
        if frame is None:
            t0 = time.time()
            frame = pull_year(y, conn, enrol_ok)
            mc.write_cache(frame, cf)
            print(f"  {y}: {len(frame):,} cells pulled ({time.time()-t0:.0f}s)")
        else:
            print(f"  {y}: cached ({len(frame):,} cells)")
        t0 = time.time()
        # Collapse only the designs that are actually missing a piece.
        need_main = sorted({k[0] for k, _ in missing if k[1] != "asof_legacy"})
        need_legacy = any(k[1] == "asof_legacy" for k, _ in missing)
        pieces, rates = ({}, pd.DataFrame())
        if need_main:
            pieces, rates = collapse_year(
                y, frame, book, {n: designs[n] for n in need_main}, ages_all)
        # one extra pass for the gate's legacy arm, so the gate can measure
        # 47b's own cascade rather than assume it. Doing this inside the main
        # pass would collapse every design x 3 arms instead of x 2, for a
        # comparison only one design needs.
        if need_legacy:
            globals()["ARMS"] = ("asof_legacy",)
            gpieces, _ = collapse_year(y, frame, book,
                                       {"OL_daioe": designs["OL_daioe"]}, ages_all)
            globals()["ARMS"] = ("true", "asof")
            pieces.update(gpieces)
        for (name, arm, T), coll in pieces.items():
            mc.write_cache(coll, coll_path(name, arm, T, y))
        if not rates.empty:
            rate_frames.append(rates)
        print(f"  {y}: collapsed {len(pieces)} design-arm-T pieces "
              f"({len(missing)} were missing) ({time.time()-t0:.0f}s)  "
              + mc.mem_line())
        del frame, pieces
        gc.collect()
    if rate_frames:
        r = pd.concat(rate_frames, ignore_index=True)
        r["anchored_share"] = r["n_anchored"] / r["n_emp"]
        mc.enforce_min_cell(r, count_col="n_emp").to_csv(
            OUT / "anchoring_rates.csv", index=False)

    def load_coll(name, arm, T):
        return pd.concat([pd.read_parquet(CACHE / f"edu_hr_coll_{name}_{arm}_T{T}_{y}.parquet")
                          for y in YEARS], ignore_index=True)

    # ---- resume, rather than start the fits again from nothing ----
    # Each cell is a pure function of its cached collapse pieces, so a
    # completed fit can be read back instead of recomputed when a job is
    # resubmitted. The guard is a code tag: reusing a coefficient computed
    # under a different set of designs, arms or truncations would mix two
    # versions of the script in one table, so a row is reused only when the
    # tag matches exactly.
    results_path = OUT / "horserace_estimates.csv"
    code_tag = hashlib.sha256(
        repr((sorted(designs), tuple(ARMS), tuple(GATE_ARMS),
              tuple(TRUNCATIONS), tuple(ages_all))).encode()).hexdigest()[:12]
    done = {}
    if os.environ.get("CANARIES_47H_FRESH") == "1":
        results_path.unlink(missing_ok=True)
        print("\n  CANARIES_47H_FRESH=1: previous fits discarded")
    elif results_path.exists():
        try:
            prev = pd.read_csv(results_path)
            if "code_tag" in prev.columns:
                keep = prev[prev["code_tag"].astype(str) == code_tag]
                dropped = len(prev) - len(keep)
                for _, r in keep.iterrows():
                    done[(r["design"], r["arm"], int(r["trunc"]),
                          r["age_group"])] = r.to_dict()
                print(f"\n  RESUMING: {len(done)} fits reused from an earlier "
                      f"run" + (f", {dropped} discarded (different code tag)"
                                if dropped else ""))
                if dropped:
                    keep.to_csv(results_path, index=False)
            else:
                results_path.unlink()
                print("\n  previous results carry no code tag; starting the "
                      "fits again")
        except BaseException as ex:
            print(f"\n  could not read previous results "
                  f"({type(ex).__name__}); starting the fits again")
            results_path.unlink(missing_ok=True)
            done = {}

    def run_cell(name, arm, T, age, tier):
        key = (name, arm, int(T), age)
        if key in done:
            row = done[key]
            print(f"  [{tier}] {name:<14} {arm:<4} T{T} {age:<5} gamma2 "
                  f"{row['gamma2']:+.4f} (SE {row['se']:.4f}) reused")
            return row
        row = estimate(load_coll(name, arm, T), age,
                       tag=f"hr_{name}_{arm}_T{T}_{age.replace('-', '_').replace('+', 'p')}")
        row = dict(design=name, arm=arm, trunc=T, age_group=age, tier=tier,
                   code_tag=code_tag, **row)
        _append_row(results_path, row)
        done[key] = row
        print(f"  [{tier}] {name:<14} {arm:<4} T{T} {age:<5} gamma2 {row['gamma2']:+.4f} "
              f"(SE {row['se']:.4f}) n {row['n_obs']:,} {row['elapsed_s']:.0f}s "
              f"{row['status']}")
        # Each fit concatenates five years of collapse pieces and builds two
        # string fixed-effect columns over ten million rows; collect between
        # fits.
        gc.collect()
        return row

    # ---- gate: does the pull reproduce 47b, and if not, why ----
    # The as-of arm here differs from script 47b's in two deliberate ways
    # (NULLIF, so an empty string falls through the cascade, and whole
    # records rather than per-field COALESCE). Rather than assume that those
    # explain any difference, the gate estimates 47b's exact cascade as a
    # third arm and decides on that one:
    #   legacy reproduces 47b -> the pull is verified, the gap between the two
    #                            as-of arms is the cascade change, proceed
    #   legacy does not       -> the pull differs for an unknown reason
    print("\nGATE (OL_daioe = script 47b's design, three arms)")
    globals()["ARMS"] = GATE_ARMS
    gate = {arm: run_cell("OL_daioe", arm, 2021, "22-25", "gate")["gamma2"]
            for arm in GATE_ARMS}
    globals()["ARMS"] = ("true", "asof")
    d_true = abs(gate["true"] - GATE_47B["true"])
    d_leg = abs(gate["asof_legacy"] - GATE_47B["asof"])
    print(f"  true        {gate['true']:+.4f} vs 47b {GATE_47B['true']:+.4f}"
          f"   |d| {d_true:.4f}")
    print(f"  asof_legacy {gate['asof_legacy']:+.4f} vs 47b "
          f"{GATE_47B['asof']:+.4f}   |d| {d_leg:.4f}   <- the gate")
    print(f"  asof        {gate['asof']:+.4f}   (corrected cascade; not gated)")
    print(f"  ARTEFACT legacy {gate['asof_legacy'] - gate['true']:+.4f}"
          f"   corrected {gate['asof'] - gate['true']:+.4f}")
    try:
        pd.DataFrame([{"arm": a, "gamma2": v, "ref_47b": GATE_47B.get(a, np.nan)}
                      for a, v in gate.items()]).to_csv(
            OUT / "gate_decomposition.csv", index=False)
    except BaseException as ex:
        print(f"  [optional] gate_decomposition.csv FAILED ({type(ex).__name__})")
    # The two gates are treated differently. The true arm halts: if this
    # script cannot reproduce 47b where the two should agree exactly, the
    # pull itself is suspect and nothing downstream is worth computing. The
    # legacy arm warns: script 47b carried a defect of its own (its collapse
    # emitted `edu_quartile` where the caller expected `exposure_quartile`),
    # so a discrepancy on the as-of arm is recorded and reported beside every
    # estimate rather than treated as a reason to refuse the other designs.
    if d_true > GATE_HALT:
        raise SystemExit(
            f"GATE FAILED on the TRUE arm: {gate['true']:+.4f} against 47b's "
            f"{GATE_47B['true']:+.4f} (|d| {d_true:.4f}). The two should agree "
            f"almost exactly here, so the pull itself is in doubt. Stopping "
            f"before Tier A.")
    gate_note = ""
    if d_leg > GATE_HALT:
        gate_note = (
            f"UNRECONCILED: the legacy arm gives {gate['asof_legacy']:+.4f} "
            f"against 47b's {GATE_47B['asof']:+.4f} (|d| {d_leg:.4f}), and it "
            f"is identical to the corrected arm, so 47b's cascade is not the "
            f"difference. Cause not established; see "
            f"notes/47h-gate-diagnosis_2026-09-20.md. Every number below is "
            f"47h's own and must be reported with this discrepancy stated.")
        print("\n  *** " + gate_note + "\n")
    else:
        print("  GATE PASS: the legacy arm reproduces 47b, so the gap between "
              "the two as-of arms is the cascade fix and nothing else.")

    # ---- Tier A ----
    print("\nTIER A: every design, 22-25, both arms, both truncations")
    art = {}
    for name in designs:
        for T in TRUNCATIONS:
            rows = {}
            for arm in ("true", "asof"):
                if name == "OL_daioe" and T == 2021 and arm in gate:
                    rows[arm] = gate[arm]
                    continue
                rows[arm] = run_cell(name, arm, T, "22-25", "A")["gamma2"]
            art[(name, T)] = rows["asof"] - rows["true"]
            print(f"  ==> {name:<14} T{T}: true {rows['true']:+.4f} as-of "
                  f"{rows['asof']:+.4f} ARTEFACT {art[(name, T)]:+.4f}")

    # ---- verdicts and Tier B ----
    def verdict(name):
        a = [abs(art.get((name, T), np.nan)) for T in TRUNCATIONS]
        half = [abs(OCC_ARTEFACT[T]) / 2 for T in TRUNCATIONS]
        if any(np.isnan(a)):
            return "no estimate"
        if all(x < 0.05 for x in a):
            return "CLEAN: carries register evidence"
        if all(x < h for x, h in zip(a, half)):
            return "USABLE only with the artefact beside every estimate"
        return "CLOSED: at or above half the occupation artefact"

    verdicts = {name: verdict(name) for name in designs}
    finalists = [n for n, v in verdicts.items() if v.startswith(("CLEAN", "USABLE"))]
    tier_b = list(dict.fromkeys(list(REFERENCE_DESIGNS) + finalists))
    print("\nTIER B: 26-30 and 50+ for " + ", ".join(tier_b))
    for name in tier_b:
        for T in TRUNCATIONS:
            for age in AGES_B:
                rows = {arm: run_cell(name, arm, T, age, "B")["gamma2"]
                        for arm in ("true", "asof")}
                art[(name, T, age)] = rows["asof"] - rows["true"]

    # ---- Tier C: the age gradient where classification is predetermined ----
    if GRADIENT_TIER:
        print(f"\nTIER C: gradient bands {', '.join(AGES_C)} at T={GRADIENT_T}, "
              f"reference designs")
        for name in REFERENCE_DESIGNS:
            for age in AGES_C:
                rows = {arm: run_cell(name, arm, GRADIENT_T, age, "C")["gamma2"]
                        for arm in ("true", "asof")}
                art[(name, GRADIENT_T, age)] = rows["asof"] - rows["true"]
                art[("true", name, GRADIENT_T, age)] = rows["true"]

    # ---- summary ----
    lines = ["EDUCATION-EXPOSURE HORSE RACE: ARTEFACT BY DESIGN", "=" * 60,
             "Occupation design (script 45), 22-25: "
             + "  ".join(f"T{T} {v:+.4f}" for T, v in OCC_ARTEFACT.items()),
             "Read rule: clean < 0.05 at both T; usable < half the occupation "
             "artefact (0.153 / 0.081); else closed.", ""]
    if gate_note:
        lines += ["", "*** " + gate_note, ""]
    lines.append(f"{'design':<14} {'T2021':>9} {'T2022':>9}   verdict")
    order = sorted(designs, key=lambda n: max(abs(art.get((n, T), np.inf)) for T in TRUNCATIONS))
    for name in order:
        a = [art.get((name, T), np.nan) for T in TRUNCATIONS]
        lines.append(f"{name:<14} {a[0]:>+9.4f} {a[1]:>+9.4f}   {verdicts[name]}")
    lines += ["", "Tier B artefacts (26-30 | 50+ placebo, must be near zero):"]
    for name in tier_b:
        for T in TRUNCATIONS:
            v = [art.get((name, T, age), np.nan) for age in AGES_B]
            lines.append(f"  {name:<14} T{T}: 26-30 {v[0]:+.4f}   50+ {v[1]:+.4f}")
    if GRADIENT_TIER:
        lines += ["", f"Tier C, T={GRADIENT_T}: true gamma2 and artefact by band "
                  "(the fallback: a gradient among predetermined bands):"]
        for name in REFERENCE_DESIGNS:
            for age in AGES_C:
                tr = art.get(("true", name, GRADIENT_T, age), np.nan)
                a = art.get((name, GRADIENT_T, age), np.nan)
                lines.append(f"  {name:<14} {age:<5} true {tr:+.4f}  artefact {a:+.4f}")
    if not enrol_ok:
        lines += ["", "NOTE: enrolment register probe failed; 'enrol' and 'full' ran "
                  "WITHOUT anchoring and are not the designs their names claim."]
    lines += ["", "The winner is the smallest 22-25 artefact with a near-zero 50+ "
              "artefact, never the largest coefficient.",
              f"Total runtime {(time.time()-t_start)/60:.1f} min. " + mc.mem_line()]
    (OUT / "47h_summary.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print("\n47h done.")


if __name__ == "__main__":
    main()
