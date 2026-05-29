"""
36_eventstudy_eduq_predq.py
===========================
Event-study Poisson PML for the EduQuartile (script 34) and PredQuartile
(script 35) cell panels. Produces per-half-year coefficients (relative
to 2022H1) for each age group, which the laptop side will plot.

Why a separate script (not Stage 4 inside 34/35)
------------------------------------------------
The pooled PostGPT dummy in 34 / 35 averages ~30 months of post-ChatGPT
time, mixing a near-zero 2023 effect with a sharp late-2024/2025 effect.
The pooled coefficient is therefore mechanically attenuated. Event
studies show how the effect develops over time, which is what we want
to defend in the appendix.

Inputs (from MONA cache, written by 34 / 35 in their final stages):
  - panel_with_eduquartile.parquet      (script 34, Stage 2 cache)
  - predicted_quartile_panel.parquet    (script 35, Stage 3 cache)

Outputs (one CSV per spec):
  - eventstudy_eduquartile.csv
  - eventstudy_predquartile.csv

Each output has columns:
  spec, age_group, period, coef, se, pvalue, n_obs, n_emp_total, converged

Per-age-group runtime: ~1-3 min (Poisson with ~12 half-year x high
interactions on a balanced cell panel of ~10-30M rows). Total run:
~30-60 min for both specs across 6 age groups.

This script does NOT touch the SQL server. It reads cached parquet,
runs R subprocess, writes CSV. Memory peak: small (cell panels are
already aggregated, so much smaller than the worker-month panel).
"""

import os
import sys
import time
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Paths and constants
# ----------------------------------------------------------------------

import importlib.util
_KAUH_SPEC = importlib.util.spec_from_file_location(
    "kauhanen", Path(__file__).parent / "32_mona_kauhanen_robustness.py"
)
kauhanen = importlib.util.module_from_spec(_KAUH_SPEC)
_KAUH_SPEC.loader.exec_module(kauhanen)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR_34 = SCRIPT_DIR / "output_34"
OUTPUT_DIR_35 = SCRIPT_DIR / "output_35"
OUTPUT_DIR_36 = SCRIPT_DIR / "output_36"
OUTPUT_DIR_36.mkdir(parents=True, exist_ok=True)

EDUQ_PANEL_PATH  = OUTPUT_DIR_34 / "panel_with_eduquartile.parquet"
PREDQ_PANEL_PATH = OUTPUT_DIR_35 / "panel_with_predicted_q.parquet"

R_FEPOIS_ES_PATH = Path(__file__).parent / "r_fepois_es.R"
R_FEOLS_ES_PATH  = Path(__file__).parent / "r_feols_es.R"

REF_HALFYEAR = "2022H1"  # match main-text figure; same as script 18

AGE_GROUPS = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]

# Engine order: OLS+1 first (workhorse spec, faster, matches Figure 3
# units), Poisson second (magnitude-comparable to staged appendix). For
# each (spec, engine) pair we write a separate CSV.
ENGINES = [
    ("ols",     R_FEOLS_ES_PATH),   # OLS+1: log(n_emp + 1) ~ i(...)
    ("poisson", R_FEPOIS_ES_PATH),  # Poisson PML: n_emp ~ i(...)
]


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
# Helpers
# ----------------------------------------------------------------------

def assign_halfyear(year_month):
    """'2022-03' -> '2022H1'; '2024-09' -> '2024H2'."""
    y, m = year_month.split("-")
    return f"{y}H{'1' if int(m) <= 6 else '2'}"


def _find_rscript():
    """Reuse kauhanen's Rscript discovery."""
    return kauhanen._find_rscript() if hasattr(kauhanen, "_find_rscript") else "Rscript"


def run_event_study_for_age(panel, bin_col, age_label, spec_label, engine,
                            r_script_path, out_dir):
    """
    Run event study for a single (age group, engine) pair.

    panel : DataFrame with [employer_id, bin_col, age_group, year_month, n_emp]
    bin_col : 'edu_quartile' or 'final_quartile' (the quartile column)
    age_label : '22-25' / '26-30' / ...
    spec_label : 'EDU_QUARTILE' or 'PRED_QUARTILE'
    engine : 'ols' or 'poisson' -- determines the R subprocess script
    r_script_path : path to r_feols_es.R or r_fepois_es.R

    Returns DataFrame with one row per half-year, or None on failure.
    """
    print(f"\n  [{spec_label} / {age_label}] preparing panel")
    sub = panel[panel["age_group"] == age_label].copy()
    if len(sub) == 0:
        print(f"    no rows; skip")
        return None

    # Use the same filter / balance helpers as Step 1 to keep specs comparable
    sub = sub.rename(columns={bin_col: "daioe_quartile"})
    sub = kauhanen.filter_step1(sub, age_label)
    if len(sub) == 0:
        print(f"    filter_step1 returned empty; skip")
        return None
    balanced = kauhanen.build_balanced_panel(sub, "daioe_quartile", n_bins=4)
    if len(balanced) == 0:
        print(f"    no employers span Q4 and Q1-Q3; skip")
        return None

    # Construct event-study columns
    panel_es = balanced.copy()
    panel_es["high"]      = (panel_es["daioe_quartile"] == 4).astype(int)
    panel_es["halfyear"]  = panel_es["year_month"].apply(assign_halfyear)
    panel_es["fe_emp_bin"] = (
        panel_es["employer_id"].astype(str) + "_" + panel_es["daioe_quartile"].astype(str)
    )
    panel_es["fe_emp_t"]   = (
        panel_es["employer_id"].astype(str) + "_" + panel_es["year_month"]
    )

    cols = ["n_emp", "high", "halfyear", "fe_emp_bin", "fe_emp_t", "employer_id"]
    tag = f"{spec_label}_{engine}_{age_label}".replace(" ", "_").replace("+", "plus")
    in_path  = out_dir / f"_es_in_{tag}_{int(time.time()*1000)}.csv"
    out_path = out_dir / f"_es_out_{tag}_{int(time.time()*1000)}.csv"

    n_obs = len(panel_es)
    n_emp_total = panel_es["n_emp"].sum()
    n_employers = panel_es["employer_id"].nunique()
    print(f"    N obs = {n_obs:,}, N employers = {n_employers:,}, "
          f"sum n_emp = {n_emp_total:,}")
    print(f"    Half-years: {sorted(panel_es['halfyear'].unique())}")

    try:
        panel_es[cols].to_csv(in_path, index=False)
        cmd = [
            _find_rscript(), str(r_script_path),
            "--input", str(in_path),
            "--output", str(out_path),
            "--cluster", "employer_id",
            "--ref", REF_HALFYEAR,
        ]
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        elapsed = time.time() - t0
        if proc.stdout:
            for line in proc.stdout.strip().splitlines():
                print(f"      [R] {line}")
        if proc.returncode != 0:
            print(f"      [R] non-zero exit {proc.returncode}")
            if proc.stderr:
                for line in proc.stderr.strip().splitlines():
                    print(f"      [R-stderr] {line}")
        if not out_path.exists():
            print(f"      [R] no output file written")
            return None
        result = pd.read_csv(out_path)
        result["spec"] = spec_label
        result["engine"] = engine
        result["age_group"] = age_label
        print(f"    [{elapsed:.0f}s, {engine}] {len(result)} half-year coefficients")
        return result
    finally:
        for p in (in_path, out_path):
            try:
                if p.exists(): p.unlink()
            except Exception:
                pass


def run_event_study_for_panel(panel_path, bin_col, spec_label):
    """
    Run event studies across all age groups and engines for one cached
    cell panel. Both OLS+1 and Poisson are run; each is saved to its own
    CSV (eventstudy_<spec>_<engine>.csv).
    """
    if not panel_path.exists():
        print(f"  Panel cache not found: {panel_path}; skip {spec_label}")
        return pd.DataFrame()
    print(f"\n=== {spec_label}: reading {panel_path.name} ===")
    panel = pd.read_parquet(panel_path)
    print(f"  Panel rows: {len(panel):,}, cols: {list(panel.columns)}")

    all_results = []
    for engine, r_path in ENGINES:
        print(f"\n  --- engine: {engine} ---")
        engine_parts = []
        for age in AGE_GROUPS:
            res = run_event_study_for_age(
                panel, bin_col, age, spec_label, engine, r_path, OUTPUT_DIR_36
            )
            if res is not None:
                engine_parts.append(res)
                # Save incrementally per (spec, engine) so partial progress
                # is recoverable even on a long run
                pd.concat(engine_parts, ignore_index=True).to_csv(
                    OUTPUT_DIR_36 / f"eventstudy_{spec_label.lower()}_{engine}.csv",
                    index=False
                )
        if engine_parts:
            all_results.extend(engine_parts)

    if not all_results:
        return pd.DataFrame()
    return pd.concat(all_results, ignore_index=True)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    log_path = OUTPUT_DIR_36 / "36_eventstudy_log.txt"
    sys.stdout = _Tee(log_path)

    print("=" * 60)
    print("36_eventstudy_eduq_predq.py -- Poisson event study")
    print("=" * 60)

    eduq_results  = run_event_study_for_panel(
        EDUQ_PANEL_PATH, "edu_quartile", "EDU_QUARTILE"
    )
    predq_results = run_event_study_for_panel(
        PREDQ_PANEL_PATH, "final_quartile", "PRED_QUARTILE"
    )

    if not eduq_results.empty:
        out = OUTPUT_DIR_36 / "eventstudy_edu_quartile.csv"
        eduq_results.to_csv(out, index=False)
        print(f"\n  EduQuartile event study: {len(eduq_results)} rows -> {out.name}")
    if not predq_results.empty:
        out = OUTPUT_DIR_36 / "eventstudy_pred_quartile.csv"
        predq_results.to_csv(out, index=False)
        print(f"  PredQuartile event study: {len(predq_results)} rows -> {out.name}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
