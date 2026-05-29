"""
35b_classify_and_aggregate.py
=============================
Stage 2-4 of the PredQuartile pipeline, implemented as a separate
script that starts from `enriched_train.parquet` and `enriched_predict.parquet`
already on disk. Designed for memory-bounded execution.

Why a separate script
---------------------
Python's allocator does not return memory to the OS even after `del +
gc.collect()`. Running Stages 1-4 in one process accumulates ~325 GB at
the OS level (each stage's transient peak persists). Splitting Stage 1
(35) and Stages 2-4 (this script) starts each process at a fresh memory
baseline.

The 2026-04-29 evening run successfully wrote the enriched parquets
in Stage 1 then was killed at 335 GB during Stage 2's classifier-training
sub-step. This script picks up from those parquets.

Memory architecture
-------------------
1. **pyarrow predicate pushdown** for every parquet read. Never instantiate
   the full 292M-row enriched_train DataFrame.
2. **Stratified subsample to 10M rows** before classifier training.
3. **Per-segment cell aggregation** with `drop_duplicates+size`. Each
   segment is loaded, aggregated to cells, freed, repeated. The full
   ~400M-row combined panel is never held.
4. **Intermediate cell parquets** (cells_gt.parquet, cells_cas.parquet,
   cells_pred.parquet) saved between stages so reruns can resume.

Estimated peak: 50-70 GB. Estimated wall: 30-45 min.

Inputs (on MONA, written by 35):
  - enriched_train.parquet   (292.7M rows, 2019-2023)
  - enriched_predict.parquet (89.4M rows, 2024-2025)

Outputs (in output_35/):
  - validation_metrics.txt
  - cells_gt.parquet, cells_cas.parquet, cells_pred.parquet
  - panel_with_predicted_q.parquet (final cell panel; consumed by 36)
  - step1_predicted_quartile.csv
  - predicted_summary.txt
"""

import gc
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Reuse kauhanen's filter / build_balanced_panel / estimate_poisson + R subprocess
import importlib.util
_KAUH_SPEC = importlib.util.spec_from_file_location(
    "kauhanen", Path(__file__).parent / "32_mona_kauhanen_robustness.py"
)
kauhanen = importlib.util.module_from_spec(_KAUH_SPEC)
_KAUH_SPEC.loader.exec_module(kauhanen)


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output_35"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ENRICHED_TRAIN_CACHE   = OUTPUT_DIR / "enriched_train.parquet"
ENRICHED_PREDICT_CACHE = OUTPUT_DIR / "enriched_predict.parquet"

CELLS_GT   = OUTPUT_DIR / "cells_gt.parquet"
CELLS_CAS  = OUTPUT_DIR / "cells_cas.parquet"
CELLS_PRED = OUTPUT_DIR / "cells_pred.parquet"
PANEL_FINAL_CACHE = OUTPUT_DIR / "panel_with_predicted_q.parquet"

LOG_PATH = OUTPUT_DIR / "35b_classify_and_aggregate_log.txt"


# ----------------------------------------------------------------------
# Constants (must match 35)
# ----------------------------------------------------------------------

TRAIN_YEARS    = [2019, 2020, 2021, 2022]
VALIDATE_YEARS = [2023]
MAX_TRAIN_ROWS = 10_000_000

FEATURE_COLS_NUM = [
    "log_cash", "log_other_benefits", "pension_rate", "log_anst",
    "car_benefit", "per_diem", "car_compensation", "stock_option",
    "researcher_tax", "first_employee", "intl_work", "local_employed",
    "male", "age",
]
FEATURE_COLS_CAT = ["nace2", "age_group"]

# Columns we need to read from the enriched parquets. Only these.
NEEDED_COLS = list(set(
    FEATURE_COLS_NUM + FEATURE_COLS_CAT +
    ["lopnr", "employer_id", "year_month", "panel_year",
     "is_current_year", "daioe_quartile_truth"]
))

KEY_COLS = ["employer_id", "final_quartile", "age_group", "year_month"]


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

class _Tee:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w", encoding="ascii", errors="replace")
    def write(self, msg):
        self.terminal.write(msg); self.log.write(msg); self.log.flush()
    def flush(self):
        self.terminal.flush(); self.log.flush()


# ----------------------------------------------------------------------
# Parquet read with predicate pushdown
# ----------------------------------------------------------------------

def read_filtered(parquet_path, filters=None, columns=None):
    """
    Read parquet with column projection and row filtering at the parquet
    layer. pyarrow only materialises the matching rows / requested cols,
    not the full table. Filters use pyarrow's predicate-pushdown syntax.
    """
    print(f"    Reading {parquet_path.name} (filters={filters}, "
          f"cols={len(columns) if columns else 'all'})")
    t0 = time.time()
    table = pq.read_table(
        str(parquet_path),
        columns=columns,
        filters=filters,
    )
    df = table.to_pandas()
    del table
    gc.collect()
    print(f"      -> {len(df):,} rows in {time.time()-t0:.0f}s")
    return df


# ----------------------------------------------------------------------
# Encoding (must match 35's _encode_features so the trained classifier
# can be reused)
# ----------------------------------------------------------------------

def encode_features(df, fitted_codes=None):
    """One-hot-style integer encoding for categorical cols. Returns (X, codes)."""
    df = df.copy()
    codes = {} if fitted_codes is None else fitted_codes
    for c in FEATURE_COLS_CAT:
        if fitted_codes is not None and c in fitted_codes:
            mapping = fitted_codes[c]
            df[c + "_code"] = df[c].astype(str).map(mapping).fillna(-1).astype(int)
        else:
            uniq = df[c].astype(str).unique().tolist()
            mapping = {v: i for i, v in enumerate(uniq)}
            df[c + "_code"] = df[c].astype(str).map(mapping).astype(int)
            codes[c] = mapping
    X = df[FEATURE_COLS_NUM + [c + "_code" for c in FEATURE_COLS_CAT]].astype(float).values
    return X, codes


# ----------------------------------------------------------------------
# Stage 2: train classifier on subsample
# ----------------------------------------------------------------------

def stage2_train_classifier():
    """
    Load the gt-subset of enriched_train (is_current_year=True, train years),
    stratified-subsample to MAX_TRAIN_ROWS, train classifier, validate on 2023.
    Returns (predict_fn, codes, metrics).
    """
    print("\n=== STAGE 2: train classifier ===")

    # --- Training data: is_current_year=True AND panel_year in TRAIN_YEARS ---
    train_filters = [
        ("is_current_year", "==", True),
        ("panel_year", "in", TRAIN_YEARS),
    ]
    train_df = read_filtered(ENRICHED_TRAIN_CACHE, filters=train_filters,
                              columns=NEEDED_COLS)

    print(f"  Full training rows: {len(train_df):,}")
    if len(train_df) > MAX_TRAIN_ROWS:
        per_class = MAX_TRAIN_ROWS // 4
        train_df = (
            train_df.groupby("daioe_quartile_truth", group_keys=False)
                    .apply(lambda x: x.sample(min(len(x), per_class), random_state=42))
                    .reset_index(drop=True)
        )
        print(f"  Subsampled training to {len(train_df):,} rows (~{per_class:,}/class)")

    # --- Validation data: is_current_year=True AND panel_year in VALIDATE_YEARS ---
    valid_filters = [
        ("is_current_year", "==", True),
        ("panel_year", "in", VALIDATE_YEARS),
    ]
    valid_df = read_filtered(ENRICHED_TRAIN_CACHE, filters=valid_filters,
                              columns=NEEDED_COLS)

    # --- Train ---
    print("\n  Training classifier")
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        cls_name = "HistGradientBoostingClassifier"
        cls = HistGradientBoostingClassifier(max_iter=200)
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        cls_name = "RandomForestClassifier"
        cls = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_depth=20)
    print(f"    Using {cls_name}")

    X_train, codes = encode_features(train_df)
    y_train = train_df["daioe_quartile_truth"].astype(int).values
    t0 = time.time()
    cls.fit(X_train, y_train)
    print(f"    Trained in {time.time()-t0:.0f}s")
    del X_train, y_train; gc.collect()

    X_valid, _ = encode_features(valid_df, fitted_codes=codes)
    y_valid = valid_df["daioe_quartile_truth"].astype(int).values
    y_pred = cls.predict(X_valid).astype(int)

    acc = float((y_pred == y_valid).mean())
    print(f"    Validation accuracy (overall): {acc:.3f}")

    per_age = {}
    for age in ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]:
        m = (valid_df["age_group"] == age).values
        if m.sum() == 0:
            continue
        per_age[age] = float((y_pred[m] == y_valid[m]).mean())
        print(f"    Validation accuracy ({age}): {per_age[age]:.3f}")

    q4_truth = (y_valid == 4)
    q4_pred  = (y_pred == 4)
    if q4_truth.sum() > 0:
        recall_q4 = float((q4_pred & q4_truth).sum() / q4_truth.sum())
        precision_q4 = float(
            (q4_pred & q4_truth).sum() / q4_pred.sum() if q4_pred.sum() > 0 else float("nan")
        )
    else:
        recall_q4 = float("nan"); precision_q4 = float("nan")
    print(f"    Q4 recall: {recall_q4:.3f}; Q4 precision: {precision_q4:.3f}")

    metrics = {
        "method": cls_name,
        "overall_accuracy": acc,
        "per_age_accuracy": per_age,
        "q4_recall": recall_q4,
        "q4_precision": precision_q4,
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
    }
    (OUTPUT_DIR / "validation_metrics.txt").write_text(
        json.dumps(metrics, indent=2), encoding="ascii"
    )

    # Free training/validation data; keep classifier + codes
    del train_df, valid_df, X_valid, y_valid, y_pred, q4_truth, q4_pred
    gc.collect()

    def predict_fn(df_to_predict):
        X, _ = encode_features(df_to_predict, fitted_codes=codes)
        return cls.predict(X).astype(int)

    return predict_fn, codes, metrics


# ----------------------------------------------------------------------
# Stage 3: per-segment cell aggregation
# ----------------------------------------------------------------------

def aggregate_segment_to_cells(df_seg, predict_fn=None, use_truth=False):
    """One worker-month segment -> cell-level (employer x quartile x age x ym)
    counts. Drops big DataFrame after aggregation."""
    if use_truth:
        df_seg["final_quartile"] = df_seg["daioe_quartile_truth"].astype(int)
    else:
        df_seg["final_quartile"] = predict_fn(df_seg)
    df_seg = df_seg[KEY_COLS + ["lopnr"]]
    df_seg = df_seg.drop_duplicates(subset=KEY_COLS + ["lopnr"])
    cells = (
        df_seg.groupby(KEY_COLS, as_index=False, observed=True)
              .size()
              .rename(columns={"size": "n_emp"})
    )
    return cells


def stage3a_cells_gt():
    """Ground-truth cells: is_current_year=True from enriched_train."""
    if CELLS_GT.exists():
        print(f"  cells_gt cache exists; skipping ({CELLS_GT.name})")
        return
    print("\n=== STAGE 3a: gt cells (is_current_year=True from train) ===")
    df = read_filtered(
        ENRICHED_TRAIN_CACHE,
        filters=[("is_current_year", "==", True)],
        columns=["employer_id", "age_group", "year_month",
                 "lopnr", "daioe_quartile_truth"],
    )
    cells = aggregate_segment_to_cells(df, use_truth=True)
    print(f"    gt cells: {len(cells):,}")
    cells.to_parquet(CELLS_GT, index=False)
    del df, cells; gc.collect()


def stage3b_cells_cas(predict_fn):
    """Cascade cells: is_current_year=False from enriched_train."""
    if CELLS_CAS.exists():
        print(f"  cells_cas cache exists; skipping ({CELLS_CAS.name})")
        return
    print("\n=== STAGE 3b: cascade cells (is_current_year=False from train) ===")
    df = read_filtered(
        ENRICHED_TRAIN_CACHE,
        filters=[("is_current_year", "==", False)],
        columns=NEEDED_COLS,  # need feature cols for predict_fn
    )
    if len(df) == 0:
        print("    no cascade rows; skipping")
        return
    cells = aggregate_segment_to_cells(df, predict_fn=predict_fn)
    print(f"    cas cells: {len(cells):,}")
    cells.to_parquet(CELLS_CAS, index=False)
    del df, cells; gc.collect()


def stage3c_cells_pred(predict_fn):
    """Predict cells: 2024-2025 from enriched_predict."""
    if CELLS_PRED.exists():
        print(f"  cells_pred cache exists; skipping ({CELLS_PRED.name})")
        return
    print("\n=== STAGE 3c: predict cells (2024-2025) ===")
    df = read_filtered(
        ENRICHED_PREDICT_CACHE,
        columns=NEEDED_COLS,
    )
    cells = aggregate_segment_to_cells(df, predict_fn=predict_fn)
    print(f"    pred cells: {len(cells):,}")
    cells.to_parquet(CELLS_PRED, index=False)
    del df, cells; gc.collect()


def stage3d_combine():
    """Sum n_emp across segments for any (employer, quartile, age, ym) cell
    that appears in multiple segments."""
    print("\n=== STAGE 3d: combine cells ===")
    parts = []
    for path in [CELLS_GT, CELLS_CAS, CELLS_PRED]:
        if path.exists():
            parts.append(pd.read_parquet(path))
            print(f"    Loaded {path.name}: {len(parts[-1]):,} rows")
    if not parts:
        raise RuntimeError("No cell parquets found; run Stage 3 first")
    cell = pd.concat(parts, ignore_index=True)
    cell = (
        cell.groupby(KEY_COLS, as_index=False, observed=True)["n_emp"].sum()
    )
    print(f"    Final cell count: {len(cell):,}")
    cell.to_parquet(PANEL_FINAL_CACHE, index=False)
    return cell


# ----------------------------------------------------------------------
# Stage 4: Poisson DiD
# ----------------------------------------------------------------------

def stage4_poisson(cell_panel):
    print("\n=== STAGE 4: Poisson DiD by age group ===")
    AGE_GROUPS = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]
    results = []
    for age_label in AGE_GROUPS:
        print(f"\n  Age group: {age_label}")
        sub = cell_panel[cell_panel["age_group"] == age_label].copy()
        sub = sub.rename(columns={"final_quartile": "daioe_quartile"})
        sub_filtered = kauhanen.filter_step1(sub, age_label)
        if len(sub_filtered) == 0:
            print(f"    no employers; skip")
            continue
        balanced = kauhanen.build_balanced_panel(
            sub_filtered, "daioe_quartile", n_bins=4
        )
        if len(balanced) == 0:
            print(f"    no employers span Q4 and Q1-Q3; skip")
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


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    sys.stdout = _Tee(LOG_PATH)
    print("=" * 70)
    print("35b_classify_and_aggregate.py")
    print("=" * 70)

    if not ENRICHED_TRAIN_CACHE.exists():
        raise RuntimeError(
            f"Missing {ENRICHED_TRAIN_CACHE}; run 35 first to generate it"
        )
    if not ENRICHED_PREDICT_CACHE.exists():
        raise RuntimeError(
            f"Missing {ENRICHED_PREDICT_CACHE}; run 35 first to generate it"
        )

    # Stage 2: train classifier (returns predict_fn for Stage 3)
    predict_fn, codes, metrics = stage2_train_classifier()

    # Stage 3: per-segment cell aggregation
    stage3a_cells_gt()
    stage3b_cells_cas(predict_fn)
    stage3c_cells_pred(predict_fn)
    cell_panel = stage3d_combine()

    # Stage 4: Poisson DiD
    results = stage4_poisson(cell_panel)
    if not results.empty:
        results.to_csv(OUTPUT_DIR / "step1_predicted_quartile.csv", index=False)
        print(f"\nFinal results -> step1_predicted_quartile.csv")

    print("\nDONE.")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        print(f"\nFATAL: {e}")
        traceback.print_exc()
        sys.exit(2)
