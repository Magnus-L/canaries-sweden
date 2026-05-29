#!/usr/bin/env python3
"""
35_mona_predicted_quartile.py -- PredQuartile robustness for the canaries finding.

======================================================================
  THIS SCRIPT IS DESIGNED TO RUN IN SCB's MONA SECURE ENVIRONMENT
  Do NOT run outside MONA -- the data is not available externally.
======================================================================

WHY THIS SCRIPT EXISTS
======================
Script 34 (EduQuartile) tests the canaries finding using
predetermined-by-construction education exposure. This script (PredQuartile)
provides an independent triangulation: it predicts the DAIOE quartile
of each 2024-2025 worker-month from current features (FDB NACE, AGI
compensation/benefits structure, demographics) and re-runs the
employer-level Poisson DiD with the predicted quartile.

The classifier is trained on 2019-2022 worker-months WITH same-year
LISA SSYK4 (no cascade fallback in training). 2023 is held out for
validation. Predictions are made for 2024-2025 (also for any
2019-2023 cell that had to use cascade fallback originally; ~10 % of
the panel).

Replacement strategy:
  - 2019-2023 cells with same-year LISA: keep ground-truth quartile.
  - 2019-2023 cells using cascade: replace with predicted quartile.
  - 2024-2025 cells: replace with predicted quartile (all are stale).

This anchors the test in current-period features rather than
year-old occupation codes, addressing the LISA staleness concern from
a different angle than EduQuartile.

WHAT THIS SCRIPT DOES
=====================
Stage 1: Pull worker-employer-months 2019-2025 with full AGI feature
set (compensation breakdown, benefit indicators, country codes,
employer info) and join FDB_JE NACE-2 + firm size + sector. Pull the
ground-truth LISA SSYK4 + DAIOE quartile for 2019-2023; flag whether
each cell was "current-year" (training-eligible) or "cascade".

Stage 2: Train a classifier on 2019-2022 current-year cells. Default
attempts scikit-learn HistGradientBoostingClassifier; falls back to a
cell-level empirical-Bayes lookup if sklearn is unavailable. Validate
on 2023 current-year cells. Log per-age accuracy and Q4 recall.

Stage 3: Predict the DAIOE quartile for all 2024-2025 cells and for
2019-2023 cascade-fallback cells. Combine into a panel where each
worker-month has a final quartile (ground-truth or predicted).

Stage 4: Aggregate to the employer x quartile x age x month level,
re-run Step 1 Poisson DiD using the same R+fixest subprocess as
script 32. Report gamma_2 by age group and compare with the original
Step 1 (OccQuartile) estimates.

USAGE IN MONA
=============
1. Upload as 35_mona_predicted_quartile.txt; rename to .py.
2. Required co-located files:
   - 32_mona_kauhanen_robustness.py  (R subprocess helpers)
   - 33_step4_fdb_patch.py           (FDB pull, optional but useful)
3. Run:
       python 35_mona_predicted_quartile.py
4. Outputs (under output_35/):
   - features_train.parquet           Training features (cached)
   - features_predict.parquet         Prediction features (cached)
   - validation_metrics.txt           Accuracy by age group
   - predicted_quartile.parquet       Lopnr x ym -> predicted quartile
   - panel_with_predicted_q.parquet   Employer-cell panel for Poisson
   - step1_predicted_quartile.csv     Poisson coefficients by age group
   - predicted_summary.txt            Prose summary

EXPORT-SAFETY
=============
All exported files are aggregated counts and regression coefficients.
Predicted-quartile parquet is intermediate (not exported), used only
for the downstream cell-level aggregation.

EXPECTED RUNTIME
================
- AGI feature pull: 30-45 min (one-off, cached)
- FDB NACE pull: 1 min
- Training + validation: 10-30 min (sklearn) or 5 min (cell-lookup)
- Prediction: 5-10 min
- Poisson PML (6 age groups): 15-30 min
- Total: 60-120 min on first run; ~30 min on re-runs from cache

CRASH RECOVERY
==============
- All caches are parquet files with explicit existence checks.
- Per-age Poisson outputs saved incrementally so partial completion
  is preserved.
"""

import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pyodbc

# ----------------------------------------------------------------------
# Script 32 import
# ----------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
KAUHANEN_PY = SCRIPT_DIR / "32_mona_kauhanen_robustness.py"

if not KAUHANEN_PY.exists():
    print(f"FATAL: 32_mona_kauhanen_robustness.py not found at {KAUHANEN_PY}")
    sys.exit(1)

_spec = importlib.util.spec_from_file_location("kauhanen", str(KAUHANEN_PY))
kauhanen = importlib.util.module_from_spec(_spec)
sys.modules["kauhanen"] = kauhanen
_spec.loader.exec_module(kauhanen)

SQL_CONN_STRING = kauhanen.SQL_CONN_STRING
AGE_GROUPS = kauhanen.AGE_GROUPS
DAIOE_PATH = kauhanen.DAIOE_PATH

# ----------------------------------------------------------------------
# Local config
# ----------------------------------------------------------------------
OUTPUT_DIR = SCRIPT_DIR / "output_35"
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_PATH = OUTPUT_DIR / "35_mona_predicted_quartile_log.txt"

FEATURES_TRAIN_CACHE = OUTPUT_DIR / "features_train.parquet"
FEATURES_PREDICT_CACHE = OUTPUT_DIR / "features_predict.parquet"
# Enriched caches (post demographics + FDB + LISA + features). The
# 2026-04-29 run died holding both df_train_valid (~316M rows post-attaches)
# and df_predict (~97M rows) in memory simultaneously; with these
# checkpoints the next stage can free everything and reload only what's
# needed.
ENRICHED_TRAIN_CACHE   = OUTPUT_DIR / "enriched_train.parquet"
ENRICHED_PREDICT_CACHE = OUTPUT_DIR / "enriched_predict.parquet"

# Cap classifier training rows. A 4-class classifier on stable worker-level
# features does not need 200M rows of training data; 10M (stratified by
# truth quartile) is plenty and keeps sklearn memory bounded.
MAX_TRAIN_ROWS = 10_000_000
PREDICTED_Q_CACHE = OUTPUT_DIR / "predicted_quartile.parquet"
PANEL_FINAL_CACHE = OUTPUT_DIR / "panel_with_predicted_q.parquet"

TRAIN_YEARS = [2019, 2020, 2021, 2022]
VALIDATE_YEARS = [2023]
PREDICT_YEARS = [2024, 2025]

# 2025 _Prel only available through 202506 per Magnus 2026-04-29
PREDICT_MAX_MONTH_2025 = 6

# AGI features to pull
AGI_FEATURE_COLS = [
    "P1207_LOPNR_PERSONNR",      # lopnr
    "P1207_LOPNR_PEORGNR",       # employer_id
    "PERIOD",                     # YYYYMM
    "ASTNR",                      # workplace
    "KONTANT_ERSATTNING_ULAG_AG", # primary cash compensation
    "SP_OVRIGA_FORMANER_ULAG_AG", # other taxable benefits
    "SP_BILFORMAN_ULAG_AG",       # car benefit
    "TJANSTEPENSION",             # pension
    "TRAKTAMENTE",                # per diem
    "BILERSATTNING",              # car compensation
    "PERSONALOPTION",             # stock options
    "FORSKAR_SKATTE_NAMNDEN",     # researcher tax status
    "FORSTA_ANSTALLD",            # first employee
    "LANDSKOD_ARBETSLAND",        # country of work
    "LOKALANSTALLD",              # locally employed
]


# ======================================================================
#   LOGGING
# ======================================================================

class _Tee:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w", encoding="ascii", errors="replace")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ======================================================================
#   STAGE 1: FEATURE PULL
# ======================================================================

def _table_for(year, month):
    """Return AGI table name. _Prel for 2025; _Def for 2019-2024."""
    ym = f"{year}{month:02d}"
    if year == 2025:
        return f"dbo.Arb_AGIIndivid{ym}_Prel"
    return f"dbo.Arb_AGIIndivid{ym}_Def"


def _agi_select_clause(available_cols=None):
    """
    SELECT clause picking the AGI feature columns.

    If `available_cols` (set/list) is provided, only columns in that set
    are included; absent columns are skipped silently. Required columns
    (P1207_LOPNR_PERSONNR, P1207_LOPNR_PEORGNR, KONTANT_ERSATTNING_ULAG_AG)
    are always included; if any is missing, this raises so the caller can
    skip the table cleanly.
    """
    REQUIRED = {
        "P1207_LOPNR_PERSONNR",
        "P1207_LOPNR_PEORGNR",
        "KONTANT_ERSATTNING_ULAG_AG",
    }
    if available_cols is not None:
        avail_upper = {c.upper() for c in available_cols}
        missing_required = REQUIRED - avail_upper
        if missing_required:
            raise RuntimeError(
                f"AGI table missing required columns: {sorted(missing_required)}"
            )
    cols = []
    for c in AGI_FEATURE_COLS:
        if available_cols is not None and c.upper() not in avail_upper:
            continue
        if c == "PERIOD":
            cols.append(f"{c} AS period")
        elif c == "P1207_LOPNR_PERSONNR":
            cols.append(f"{c} AS lopnr")
        elif c == "P1207_LOPNR_PEORGNR":
            cols.append(f"{c} AS employer_id")
        else:
            cols.append(f"{c} AS {c.lower()}")
    return ",\n            ".join(cols)


def pull_agi_features(years):
    """
    Pull AGI worker-employer-months for the given years with the full
    feature set. Returns one row per (lopnr, employer_id, year_month).

    Per-month INFORMATION_SCHEMA.COLUMNS discovery: SQL Server validates
    all column refs at parse time, so a single missing optional feature
    column blows up the whole SELECT for that month. Discovery costs
    milliseconds and keeps the pull resilient across schema variation
    between Def and Prel and across years.

    Cached to FEATURES_TRAIN_CACHE for years <= 2023, FEATURES_PREDICT_CACHE
    for years >= 2024.
    """
    is_train = max(years) <= max(VALIDATE_YEARS)
    cache = FEATURES_TRAIN_CACHE if is_train else FEATURES_PREDICT_CACHE

    if cache.exists():
        print(f"  Cache exists: {cache.name}; loading.")
        return pd.read_parquet(cache)

    print(f"  Pulling AGI features for years {years[0]}-{years[-1]}")
    conn = pyodbc.connect(SQL_CONN_STRING)
    frames = []
    used_cols_first_month = None

    for year in years:
        for month in range(1, 13):
            if year == 2025 and month > PREDICT_MAX_MONTH_2025:
                continue
            table = _table_for(year, month)
            table_bare = table.split(".", 1)[1]  # drop "dbo."
            try:
                cols_q = f"""
                    SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = '{table_bare}'
                """
                avail = set(pd.read_sql(cols_q, conn)["COLUMN_NAME"].tolist())
            except BaseException as e:
                print(f"    {year}-{month:02d}: schema discovery failed ({e}); skipping")
                continue
            if not avail:
                # Table doesn't exist (e.g. _Prel month not yet released)
                continue
            try:
                select_clause = _agi_select_clause(avail)
            except RuntimeError as e:
                print(f"    {year}-{month:02d}: {e}; skipping")
                continue
            if month == 1 and used_cols_first_month is None:
                pulled_cols = [c for c in AGI_FEATURE_COLS if c.upper() in {x.upper() for x in avail}]
                missing = [c for c in AGI_FEATURE_COLS if c.upper() not in {x.upper() for x in avail}]
                print(f"    Schema for {table_bare}: {len(pulled_cols)}/{len(AGI_FEATURE_COLS)} feature cols present.")
                if missing:
                    print(f"      Missing (will be NaN downstream): {missing}")
                used_cols_first_month = pulled_cols

            query = f"""
                SELECT
            {select_clause}
                FROM {table}
                WHERE P1207_LOPNR_PEORGNR IS NOT NULL
                  AND KONTANT_ERSATTNING_ULAG_AG > 0
            """
            try:
                t0 = time.time()
                df = pd.read_sql(query, conn)
                if len(df) == 0:
                    continue
                df["year_month"] = f"{year}-{month:02d}"
                frames.append(df)
                if month == 1:
                    print(f"    {year}-{month:02d}: {len(df):,} rows in {time.time()-t0:.0f}s")
            except BaseException as e:
                print(f"    {year}-{month:02d}: pull failed ({e})")
                continue

    conn.close()
    if not frames:
        raise RuntimeError(f"No AGI data pulled for years {years}")

    df = pd.concat(frames, ignore_index=True)
    print(f"  Total rows: {len(df):,}")
    df.to_parquet(cache, index=False)
    return df


def attach_demographics(df):
    """Attach age + gender from Population_PersonNr."""
    print("  Attaching demographics from Population_PersonNr...")
    conn = pyodbc.connect(SQL_CONN_STRING)
    pop = pd.read_sql(
        """SELECT PersonLopNr AS lopnr, FodelseAr AS birth_year, Kon AS gender
           FROM dbo.Population_PersonNr
           WHERE FodelseAr IS NOT NULL""",
        conn,
    )
    conn.close()
    df = df.merge(pop, on="lopnr", how="left")
    df["panel_year"] = df["year_month"].str.slice(0, 4).astype(int)
    # FodelseAr can come back as int or string from MONA; coerce.
    df["birth_year"] = pd.to_numeric(df["birth_year"], errors="coerce")
    df["age"] = df["panel_year"] - df["birth_year"]
    df["age_group"] = pd.cut(
        df["age"],
        bins=[21, 25, 30, 34, 40, 49, 69],
        labels=["22-25", "26-30", "31-34", "35-40", "41-49", "50+"],
        right=True, include_lowest=False,
    )
    df = df[df["age_group"].notna()].copy()
    df["age_group"] = df["age_group"].astype(str)
    return df


def attach_fdb_nace(df):
    """Attach NACE-2 and firm size from FDB_JE_2014_2021 + FDB_JE_2022_2024."""
    print("  Attaching FDB NACE-2 + firm size...")
    conn = pyodbc.connect(SQL_CONN_STRING)
    query = """
        SELECT P1207_Lopnr_peorgnr AS employer_id,
               LEFT(REPLACE(REPLACE(ng1, '.', ''), ' ', ''), 2) AS nace2,
               anst,
               CAST(ar AS INT) AS year
        FROM dbo.FDB_JE_2022_2024
        WHERE ng1 IS NOT NULL
        UNION ALL
        SELECT P1207_Lopnr_peorgnr AS employer_id,
               LEFT(REPLACE(REPLACE(ng1, '.', ''), ' ', ''), 2) AS nace2,
               anst,
               CAST(ar AS INT) AS year
        FROM dbo.FDB_JE_2014_2021
        WHERE ng1 IS NOT NULL AND ar >= '2018'
    """
    fdb = pd.read_sql(query, conn)
    conn.close()
    print(f"    Pulled {len(fdb):,} firm-year rows")
    fdb = fdb.sort_values(["employer_id", "year"], ascending=[True, False])
    fdb = fdb.drop_duplicates(subset=["employer_id"], keep="first")
    fdb["nace2"] = fdb["nace2"].astype(str).str.zfill(2)
    fdb = fdb[fdb["nace2"].str.match(r"^\d{2}$")]
    fdb["log_anst"] = np.log1p(pd.to_numeric(fdb["anst"], errors="coerce").fillna(0))

    df = df.merge(
        fdb[["employer_id", "nace2", "log_anst"]],
        on="employer_id", how="left",
    )
    df["nace2"] = df["nace2"].fillna("00")
    df["log_anst"] = df["log_anst"].fillna(0.0)
    return df


def attach_lisa_quartile(df, year_label):
    """
    For training, attach the same-year LISA SSYK4 -> DAIOE quartile.
    Flags whether the cell is "current-year" (training-eligible) or
    "cascade" (LISA missing for the panel year, used fallback).
    """
    print(f"  Attaching LISA SSYK4 -> DAIOE quartile (year sample: {year_label})...")

    daioe = kauhanen.load_daioe_quartiles()
    daioe["ssyk4"] = daioe["ssyk4"].astype(str).str.zfill(4)

    panel_years = sorted(df["panel_year"].unique().tolist())
    conn = pyodbc.connect(SQL_CONN_STRING)
    frames = []
    for year in panel_years:
        if year > 2023:
            continue
        # Per-year SSYK column discovery: SQL Server parses all column refs
        # at parse time, so a missing column in a hardcoded COALESCE fails
        # the whole query. Discover which Ssyk4 variants actually exist.
        try:
            cols_q = f"""
                SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = 'Individ_{year}'
            """
            cols = set(pd.read_sql(cols_q, conn)["COLUMN_NAME"].tolist())
        except BaseException as e:
            print(f"    LISA {year}: schema discovery failed ({e}); skipping")
            continue

        if "Ssyk4_2012_J16" in cols and "Ssyk4_2012" in cols:
            ssyk_expr = "COALESCE(Ssyk4_2012_J16, Ssyk4_2012)"
        elif "Ssyk4_2012_J16" in cols:
            ssyk_expr = "Ssyk4_2012_J16"
        elif "Ssyk4_2012" in cols:
            ssyk_expr = "Ssyk4_2012"
        else:
            print(f"    LISA {year}: no Ssyk4 column found; skipping")
            continue

        # Same-year LISA only; for cascade-flagging we look at availability
        query = f"""
            SELECT P1207_LopNr_PersonNr AS lopnr,
                   {year} AS lisa_year,
                   {ssyk_expr} AS ssyk4
            FROM dbo.Individ_{year}
            WHERE {ssyk_expr} IS NOT NULL
        """
        try:
            f = pd.read_sql(query, conn)
            f["ssyk4"] = f["ssyk4"].astype(str).str.zfill(4)
            frames.append(f)
            print(f"    LISA {year}: {len(f):,} rows (ssyk={ssyk_expr})")
        except BaseException as e:
            print(f"    LISA {year} failed: {e}")
    conn.close()

    if not frames:
        df["daioe_quartile_truth"] = np.nan
        df["is_current_year"] = False
        return df

    lisa = pd.concat(frames, ignore_index=True)
    lisa = lisa.merge(daioe, on="ssyk4", how="left")
    lisa = lisa.rename(columns={"daioe_quartile": "daioe_quartile_truth"})
    lisa = lisa[["lopnr", "lisa_year", "daioe_quartile_truth"]]

    df["lisa_year"] = df["panel_year"]
    df = df.merge(lisa, on=["lopnr", "lisa_year"], how="left")
    df["is_current_year"] = df["daioe_quartile_truth"].notna()
    return df


def _col_or_zero(df, col):
    """Return a Series for `col` if present, else a zero Series of df's length."""
    if col in df.columns:
        return df[col].fillna(0)
    return pd.Series(0, index=df.index)


def _col_or_const(df, col, fill_val):
    if col in df.columns:
        return df[col].fillna(fill_val)
    return pd.Series(fill_val, index=df.index)


def build_features(df):
    """
    Engineer the modelling feature columns.

    Defensive: any AGI feature column that was missing from the source
    table (per per-month INFORMATION_SCHEMA discovery) silently
    contributes a zero/default series rather than raising KeyError.
    """
    print("  Engineering features...")
    eps = 1.0
    df["log_cash"] = np.log(_col_or_zero(df, "kontant_ersattning_ulag_ag") + eps)
    df["log_other_benefits"] = np.log(_col_or_zero(df, "sp_ovriga_formaner_ulag_ag") + eps)
    df["pension_rate"] = (
        _col_or_zero(df, "tjanstepension")
        / (_col_or_zero(df, "kontant_ersattning_ulag_ag") + 1.0)
    ).clip(0, 2)
    df["car_benefit"] = (_col_or_zero(df, "sp_bilforman_ulag_ag") > 0).astype(int)
    df["per_diem"] = _col_or_zero(df, "traktamente").astype(int)
    df["car_compensation"] = _col_or_zero(df, "bilersattning").astype(int)
    df["stock_option"] = _col_or_zero(df, "personaloption").astype(int)
    df["researcher_tax"] = (_col_or_zero(df, "forskar_skatte_namnden") > 0).astype(int)
    df["first_employee"] = _col_or_zero(df, "forsta_anstalld").astype(int)
    df["intl_work"] = (
        _col_or_const(df, "landskod_arbetsland", "SE").str.upper() != "SE"
    ).astype(int)
    df["local_employed"] = _col_or_zero(df, "lokalanstalld").astype(int)
    df["male"] = (df["gender"] == 1).astype(int)
    return df


# ======================================================================
#   STAGE 2: TRAIN AND VALIDATE
# ======================================================================

FEATURE_COLS_NUM = [
    "log_cash", "log_other_benefits", "pension_rate", "log_anst",
    "car_benefit", "per_diem", "car_compensation", "stock_option",
    "researcher_tax", "first_employee", "intl_work", "local_employed",
    "male", "age",
]
FEATURE_COLS_CAT = ["nace2", "age_group"]


def _try_sklearn_classifier():
    """Return a fitted classifier class if sklearn is available, else None."""
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier
    except BaseException:
        try:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier
        except BaseException:
            return None


def _encode_features(df, cat_cols, fitted_codes=None):
    """One-hot encode categoricals via simple integer codes; return X array."""
    df = df.copy()
    codes = {}
    for c in cat_cols:
        if fitted_codes is not None and c in fitted_codes:
            mapping = fitted_codes[c]
            df[c + "_code"] = df[c].astype(str).map(mapping).fillna(-1).astype(int)
        else:
            uniques = sorted(df[c].astype(str).unique().tolist())
            mapping = {v: i for i, v in enumerate(uniques)}
            codes[c] = mapping
            df[c + "_code"] = df[c].astype(str).map(mapping).astype(int)

    X = df[FEATURE_COLS_NUM + [c + "_code" for c in cat_cols]].astype(float).values
    return X, codes if fitted_codes is None else fitted_codes


def train_classifier(train_df, valid_df):
    """
    Train DAIOE quartile classifier on train_df, validate on valid_df.
    Returns (predict_fn, validation_metrics).

    predict_fn: callable taking a DataFrame (with feature columns) and
    returning an array of predicted quartiles (1..4).

    Memory note: training rows are capped at MAX_TRAIN_ROWS via stratified
    sampling on the truth quartile. A four-class classifier on worker-level
    features does not need 200M training rows; 10M is empirically saturating.
    """
    print("\n  STAGE 2: train classifier")
    train_df = train_df[train_df["is_current_year"]].copy()
    valid_df = valid_df[valid_df["is_current_year"]].copy()
    print(f"    Training rows (full): {len(train_df):,}")
    print(f"    Validation rows: {len(valid_df):,}")

    if len(train_df) > MAX_TRAIN_ROWS:
        per_class = MAX_TRAIN_ROWS // 4
        train_df = (
            train_df.groupby("daioe_quartile_truth", group_keys=False)
                    .apply(lambda x: x.sample(min(len(x), per_class), random_state=42))
                    .reset_index(drop=True)
        )
        print(f"    Subsampled training to {len(train_df):,} rows (~{per_class:,}/class)")

    cls = _try_sklearn_classifier()
    if cls is not None:
        print(f"    Using sklearn: {cls.__name__}")
        X_train, codes = _encode_features(train_df, FEATURE_COLS_CAT)
        y_train = train_df["daioe_quartile_truth"].astype(int).values
        clf = cls(max_iter=200) if cls.__name__ == "HistGradientBoostingClassifier" \
              else cls(n_estimators=100, n_jobs=-1, max_depth=20)
        t0 = time.time()
        clf.fit(X_train, y_train)
        print(f"    Trained in {time.time()-t0:.0f}s")

        X_valid, _ = _encode_features(valid_df, FEATURE_COLS_CAT, fitted_codes=codes)
        y_valid = valid_df["daioe_quartile_truth"].astype(int).values
        y_pred = clf.predict(X_valid).astype(int)

        acc = (y_pred == y_valid).mean()
        print(f"    Validation accuracy (overall): {acc:.3f}")

        # Per-age accuracy
        per_age = {}
        for age in ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]:
            m = (valid_df["age_group"] == age).values
            if m.sum() == 0:
                continue
            per_age[age] = float((y_pred[m] == y_valid[m]).mean())
            print(f"    Validation accuracy ({age}): {per_age[age]:.3f}")

        # Q4 recall (most important: how well do we identify Q4 workers?)
        q4_truth = (y_valid == 4)
        q4_pred = (y_pred == 4)
        if q4_truth.sum() > 0:
            recall_q4 = (q4_pred & q4_truth).sum() / q4_truth.sum()
            precision_q4 = ((q4_pred & q4_truth).sum() / q4_pred.sum()
                            if q4_pred.sum() > 0 else float("nan"))
            print(f"    Q4 recall: {recall_q4:.3f}; Q4 precision: {precision_q4:.3f}")
        else:
            recall_q4 = float("nan")
            precision_q4 = float("nan")

        metrics = {
            "method": cls.__name__,
            "overall_accuracy": float(acc),
            "per_age_accuracy": per_age,
            "q4_recall": float(recall_q4) if not np.isnan(recall_q4) else None,
            "q4_precision": float(precision_q4) if not np.isnan(precision_q4) else None,
            "n_train": int(len(train_df)),
            "n_valid": int(len(valid_df)),
        }

        def predict_fn(df_to_predict):
            X, _ = _encode_features(df_to_predict, FEATURE_COLS_CAT, fitted_codes=codes)
            return clf.predict(X).astype(int)

        return predict_fn, metrics

    # Fallback: cell-level empirical Bayes
    print("    sklearn unavailable; using cell-level lookup")
    train_df["log_cash_q5"] = pd.qcut(
        train_df["log_cash"], 5, labels=False, duplicates="drop"
    ).fillna(-1).astype(int)
    cell = ["nace2", "age_group", "male", "log_cash_q5"]
    cell_dist = (
        train_df.groupby(cell)["daioe_quartile_truth"]
        .value_counts()
        .unstack(fill_value=0)
    )
    cell_dist.columns = [int(c) for c in cell_dist.columns]
    cell_modal = cell_dist.idxmax(axis=1).rename("predicted_q").reset_index()

    # Compute log_cash_q5 cut points from training
    cut_points = (
        train_df.groupby(pd.qcut(train_df["log_cash"], 5, duplicates="drop"))
        ["log_cash"].max()
        .sort_values().tolist()
    )

    def _q5_assign(s):
        out = np.full(len(s), -1, dtype=int)
        for i, v in enumerate(s):
            for k, c in enumerate(cut_points):
                if v <= c:
                    out[i] = k
                    break
            else:
                out[i] = len(cut_points) - 1
        return out

    valid_df = valid_df.copy()
    valid_df["log_cash_q5"] = _q5_assign(valid_df["log_cash"].values)
    valid_df = valid_df.merge(cell_modal, on=cell, how="left")
    # Backstop for cells not present in train
    backstop = train_df["daioe_quartile_truth"].mode().iloc[0]
    valid_df["predicted_q"] = valid_df["predicted_q"].fillna(backstop).astype(int)

    y_valid = valid_df["daioe_quartile_truth"].astype(int).values
    y_pred = valid_df["predicted_q"].astype(int).values
    acc = (y_pred == y_valid).mean()
    print(f"    Validation accuracy (overall): {acc:.3f}")

    per_age = {}
    for age in ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]:
        m = (valid_df["age_group"] == age).values
        if m.sum() == 0:
            continue
        per_age[age] = float((y_pred[m] == y_valid[m]).mean())
        print(f"    Validation accuracy ({age}): {per_age[age]:.3f}")

    metrics = {
        "method": "cell_lookup",
        "overall_accuracy": float(acc),
        "per_age_accuracy": per_age,
        "q4_recall": None,
        "q4_precision": None,
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
    }

    def predict_fn(df_to_predict):
        d = df_to_predict.copy()
        d["log_cash_q5"] = _q5_assign(d["log_cash"].values)
        d = d.merge(cell_modal, on=cell, how="left")
        return d["predicted_q"].fillna(backstop).astype(int).values

    return predict_fn, metrics


# ======================================================================
#   STAGE 3: PREDICT AND BUILD FINAL PANEL
# ======================================================================

def _aggregate_segment_to_cells(df_seg, source_label, predict_fn=None,
                                 use_truth=False):
    """
    Aggregate one worker-month segment to cell level. Cells are
    (employer_id, final_quartile, age_group, year_month). Counting
    semantics: distinct lopnr per cell, via drop_duplicates+size.

    df_seg is mutated and freed; caller should let it go out of scope.
    """
    if use_truth:
        df_seg["final_quartile"] = df_seg["daioe_quartile_truth"].astype(int)
    else:
        df_seg["final_quartile"] = predict_fn(df_seg)
    keys = ["employer_id", "final_quartile", "age_group", "year_month"]
    df_seg = df_seg[keys + ["lopnr"]]
    seg_unique = df_seg.drop_duplicates(subset=keys + ["lopnr"])
    cells = (
        seg_unique.groupby(keys, as_index=False, observed=True)
                  .size()
                  .rename(columns={"size": "n_emp"})
    )
    cells["source"] = source_label
    return cells


def build_panel_with_predicted_quartile_streaming(predict_fn):
    """
    Memory-bounded version of build_panel_with_predicted_quartile.

    Reads enriched parquets, processes one segment at a time, aggregates
    each segment to cell level (cells are tiny relative to worker-months),
    frees the segment, and only concatenates at the cell level.

    Avoids the ~400M-row pd.concat that killed the previous run.
    """
    import gc
    print("\n  STAGE 3: build final panel with predicted quartiles (streaming)")

    cell_parts = []

    # ----- Training + validation segment (2019-2023) -----
    print("    Loading enriched train cache...")
    df = pd.read_parquet(ENRICHED_TRAIN_CACHE)
    print(f"      Rows: {len(df):,}")

    gt = df[df["is_current_year"]].copy()
    print(f"      Ground-truth rows (LISA same-year present): {len(gt):,}")
    cells_gt = _aggregate_segment_to_cells(gt, "ground_truth", use_truth=True)
    print(f"      gt cells: {len(cells_gt):,}")
    cell_parts.append(cells_gt)
    del gt; gc.collect()

    cas = df[~df["is_current_year"]].copy()
    if len(cas) > 0:
        print(f"      Cascade rows (LISA missing, predicted): {len(cas):,}")
        cells_cas = _aggregate_segment_to_cells(
            cas, "predicted_train", predict_fn=predict_fn
        )
        print(f"      cascade cells: {len(cells_cas):,}")
        cell_parts.append(cells_cas)
    del cas, df; gc.collect()

    # ----- Prediction segment (2024-2025) -----
    print("    Loading enriched predict cache...")
    df = pd.read_parquet(ENRICHED_PREDICT_CACHE)
    print(f"      Rows: {len(df):,}")
    cells_pred = _aggregate_segment_to_cells(
        df, "predicted", predict_fn=predict_fn
    )
    print(f"      predict cells: {len(cells_pred):,}")
    cell_parts.append(cells_pred)
    del df; gc.collect()

    # ----- Combine cell-level (small) -----
    cell = pd.concat(cell_parts, ignore_index=True)
    # The same (employer, quartile, age, ym) cell can receive workers from
    # multiple segments (gt vs cascade). Sum n_emp across segments and pick
    # the dominant source label for the cell.
    keys = ["employer_id", "final_quartile", "age_group", "year_month"]
    cell_summed = (
        cell.groupby(keys, as_index=False, observed=True)["n_emp"].sum()
    )
    print(f"    Final cell count: {len(cell_summed):,}")

    # Source breakdown (informational)
    src_breakdown = cell.groupby("source")["n_emp"].sum().to_dict()
    print(f"    Source breakdown (worker-months): {src_breakdown}")

    cell_summed.to_parquet(PANEL_FINAL_CACHE, index=False)
    return cell_summed


# ======================================================================
#   STAGE 4: POISSON DiD
# ======================================================================

def estimate_poisson_predicted(cell_panel):
    """Run Step-1-equivalent Poisson DiD with predicted quartile."""
    print("\n  STAGE 4: Poisson PML with predicted quartile")
    results = []

    for age_label in AGE_GROUPS:
        print(f"\n  Age group: {age_label}")
        sub = cell_panel[cell_panel["age_group"] == age_label].copy()
        sub = sub.rename(columns={"final_quartile": "daioe_quartile"})
        sub_filtered = kauhanen.filter_step1(sub, age_label)
        if len(sub_filtered) == 0:
            print(f"    No employers; skip")
            continue
        balanced = kauhanen.build_balanced_panel(
            sub_filtered, "daioe_quartile", n_bins=4
        )
        if len(balanced) == 0:
            print(f"    No employers span Q4 and Q1-Q3; skip")
            continue
        result = kauhanen.estimate_poisson(
            balanced, "daioe_quartile", n_bins=4,
            age_label=age_label, spec_label="PRED_QUARTILE",
        )
        if result is not None:
            results.append(result)
            pd.DataFrame(results).to_csv(
                OUTPUT_DIR / "step1_predicted_quartile.csv", index=False
            )

    return pd.DataFrame(results)


# ======================================================================
#   SUMMARY
# ======================================================================

def write_summary(metrics, results, occq):
    out = OUTPUT_DIR / "predicted_summary.txt"
    lines = []
    lines.append("PREDICTED-QUARTILE ROBUSTNESS -- SUMMARY")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Classifier: {metrics['method']}")
    lines.append(f"  Training rows: {metrics['n_train']:,}")
    lines.append(f"  Validation rows: {metrics['n_valid']:,}")
    lines.append(f"  Overall validation accuracy: {metrics['overall_accuracy']:.3f}")
    if metrics.get("q4_recall") is not None:
        lines.append(f"  Q4 recall: {metrics['q4_recall']:.3f}")
        lines.append(f"  Q4 precision: {metrics['q4_precision']:.3f}")
    lines.append("  Per-age accuracy:")
    for age, acc in metrics["per_age_accuracy"].items():
        lines.append(f"    {age:>6}: {acc:.3f}")
    lines.append("")

    if len(results) == 0:
        lines.append("Poisson estimation produced no results.")
    else:
        lines.append("PredQuartile vs OccQuartile (Step 1) gamma_2 by age group:")
        lines.append("")
        lines.append(f"  {'Age':>8}  {'OccQ g2':>10}  {'PredQ g2':>10}  {'PredQ p':>10}")
        lines.append(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}")
        for age in ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]:
            occ_row = occq[occq["age_group"] == age] if len(occq) else pd.DataFrame()
            pre_row = results[results["age_group"] == age]
            occ_g2 = occ_row["gamma2"].iloc[0] if len(occ_row) else float("nan")
            pre_g2 = pre_row["gamma2"].iloc[0] if len(pre_row) else float("nan")
            pre_p = pre_row["p2"].iloc[0] if len(pre_row) else float("nan")
            lines.append(f"  {age:>8}  {occ_g2:>10.4f}  {pre_g2:>10.4f}  {pre_p:>10.4f}")

    lines.append("")
    lines.append("Interpretation:")
    lines.append("  - PredQ g2 ~ OccQ g2: canaries effect robust to staleness.")
    lines.append("  - PredQ g2 attenuated: real but partly stale-code-amplified.")
    lines.append("  - PredQ g2 null: OccQ likely measurement-error driven.")

    out.write_text("\n".join(lines), encoding="ascii", errors="replace")
    print(f"\n  Summary written to {out.name}")


# ======================================================================
#   MAIN
# ======================================================================

def _build_enriched_train_cache():
    """Stage 1a: pull, attach, engineer features for 2019-2023 train+valid.
    Saves enriched_train.parquet and frees memory before returning."""
    import gc
    print("\nSTAGE 1a: enrich train+valid (2019-2023)")
    df = pull_agi_features(TRAIN_YEARS + VALIDATE_YEARS)
    df = attach_demographics(df)
    df = attach_fdb_nace(df)
    df = attach_lisa_quartile(df, "2019-2023")
    df = build_features(df)
    print(f"  Enriched train+valid: {len(df):,} rows")
    print(f"  Saving {ENRICHED_TRAIN_CACHE.name}")
    df.to_parquet(ENRICHED_TRAIN_CACHE, index=False)
    del df; gc.collect()


def _build_enriched_predict_cache():
    """Stage 1b: pull, attach, engineer features for 2024-2025 predict.
    Saves enriched_predict.parquet and frees memory before returning."""
    import gc
    print("\nSTAGE 1b: enrich predict (2024-2025)")
    df = pull_agi_features(PREDICT_YEARS)
    df = attach_demographics(df)
    df = attach_fdb_nace(df)
    df["daioe_quartile_truth"] = np.nan
    df["is_current_year"] = False
    df = build_features(df)
    print(f"  Enriched predict: {len(df):,} rows")
    print(f"  Saving {ENRICHED_PREDICT_CACHE.name}")
    df.to_parquet(ENRICHED_PREDICT_CACHE, index=False)
    del df; gc.collect()


def main():
    import gc
    sys.stdout = _Tee(LOG_PATH)
    print("=" * 70)
    print("35_mona_predicted_quartile.py")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Stage 1: build enriched parquet caches (train and predict).
    # Each is built and then memory is freed before the next stage.
    # ------------------------------------------------------------------
    if not ENRICHED_TRAIN_CACHE.exists():
        _build_enriched_train_cache()
    else:
        print(f"\nSTAGE 1a: enriched train cache exists; skipping "
              f"({ENRICHED_TRAIN_CACHE.name})")

    if not ENRICHED_PREDICT_CACHE.exists():
        _build_enriched_predict_cache()
    else:
        print(f"\nSTAGE 1b: enriched predict cache exists; skipping "
              f"({ENRICHED_PREDICT_CACHE.name})")

    # ------------------------------------------------------------------
    # Stage 2: load training+validation subsets only, train classifier,
    # save predict_fn (via metrics + serialised model), free.
    # ------------------------------------------------------------------
    print("\nSTAGE 2: load training subsets and train classifier")
    df_tv = pd.read_parquet(ENRICHED_TRAIN_CACHE)
    train_mask = df_tv["panel_year"].isin(TRAIN_YEARS)
    train_df = df_tv[train_mask].copy()
    valid_df = df_tv[~train_mask].copy()
    del df_tv; gc.collect()
    print(f"  train_df rows (pre-current-year filter): {len(train_df):,}")
    print(f"  valid_df rows (pre-current-year filter): {len(valid_df):,}")

    predict_fn, metrics = train_classifier(train_df, valid_df)
    (OUTPUT_DIR / "validation_metrics.txt").write_text(
        json.dumps(metrics, indent=2), encoding="ascii"
    )
    del train_df, valid_df; gc.collect()

    # ------------------------------------------------------------------
    # Stage 3: build cell panel by streaming through each segment
    # (gt train, cascade train, predict). Never holds the full
    # ~400M-row combined panel in memory.
    # ------------------------------------------------------------------
    cell_panel = build_panel_with_predicted_quartile_streaming(predict_fn)

    # ------------------------------------------------------------------
    # Stage 4: Poisson DiD.
    # ------------------------------------------------------------------
    results = estimate_poisson_predicted(cell_panel)

    # Comparison vs OccQuartile Step 1
    occq_path = kauhanen.OUTPUT_DIR / "step1_poisson_current.csv"
    occq = pd.read_csv(occq_path) if occq_path.exists() else pd.DataFrame()

    write_summary(metrics, results, occq)
    print("\nDone. Outputs in output_35/.")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        print(f"\nFATAL: {e}")
        traceback.print_exc()
        sys.exit(2)
